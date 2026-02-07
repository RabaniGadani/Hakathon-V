"""
Complete Task MCP Tool
Marks a task as completed (idempotent)
"""
from datetime import datetime
from pydantic import Field

from ..server import mcp
from ...services.task_service import TaskService
from ...database.database import get_session
from ...utils.errors import TaskNotFoundException


@mcp.tool()
def complete_task(
    task_id: int = Field(..., description="Task ID to complete"),
    user_id: int = Field(..., description="User ID (injected by system)")
) -> dict:
    """Mark a task as completed (idempotent operation)."""
    session_gen = get_session()
    db = next(session_gen)
    try:
        # Get the task first to check current status
        task = TaskService.get_task_by_id(db=db, task_id=task_id, user_id=user_id)
        already_completed = task.completed

        if not already_completed:
            # Toggle to complete
            task = TaskService.toggle_task_completion(db=db, task_id=task_id, user_id=user_id)

        return {
            "id": task.id,
            "title": task.title,
            "completed": True,
            "status": "done",
            "already_completed": already_completed,
            "completed_at": datetime.utcnow().isoformat(),
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
