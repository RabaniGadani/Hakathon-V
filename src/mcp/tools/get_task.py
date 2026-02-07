"""
Get Task MCP Tool
Retrieves a specific task by ID
"""
from pydantic import Field

from ..server import mcp
from ...services.task_service import TaskService
from ...database.database import get_session
from ...utils.errors import TaskNotFoundException


@mcp.tool()
def get_task(
    task_id: int = Field(..., description="Task ID to retrieve"),
    user_id: int = Field(..., description="User ID (injected by system)")
) -> dict:
    """Retrieve a specific task by ID for the authenticated user."""
    session_gen = get_session()
    db = next(session_gen)
    try:
        task = TaskService.get_task_by_id(db=db, task_id=task_id, user_id=user_id)
        return {
            "id": task.id,
            "title": task.title,
            "description": task.description or "",
            "priority": task.priority.value,
            "status": task.status.value,
            "due_date": task.due_date.isoformat() if task.due_date else None,
            "category": task.category,
            "completed": task.completed,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
        }
    except TaskNotFoundException:
        return {"error": "not_found", "message": f"Task with ID {task_id} not found"}
    except Exception as e:
        return {"error": "database_error", "message": str(e)}
    finally:
        try:
            next(session_gen)
        except StopIteration:
            pass
