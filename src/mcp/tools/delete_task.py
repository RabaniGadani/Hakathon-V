"""
Delete Task MCP Tool
Permanently deletes a task
"""
from pydantic import Field

from ..server import mcp
from ...services.task_service import TaskService
from ...database.database import get_session
from ...utils.errors import TaskNotFoundException


@mcp.tool()
def delete_task(
    task_id: int = Field(..., description="Task ID to delete"),
    user_id: int = Field(..., description="User ID (injected by system)")
) -> dict:
    """Permanently delete a task."""
    session_gen = get_session()
    db = next(session_gen)
    try:
        # Get task first to return title in response
        task = TaskService.get_task_by_id(db=db, task_id=task_id, user_id=user_id)
        task_title = task.title

        # Delete the task
        TaskService.delete_task(db=db, task_id=task_id, user_id=user_id)

        return {
            "deleted": True,
            "task_id": task_id,
            "title": task_title,
        }
    except TaskNotFoundException:
        return {"error": "not_found", "message": f"Task with ID {task_id} not found", "deleted": False}
    except Exception as e:
        return {"error": "database_error", "message": str(e), "deleted": False}
    finally:
        try:
            next(session_gen)
        except StopIteration:
            pass
