"""
Update Task MCP Tool
Updates task attributes
"""
from typing import Optional
from datetime import date
from pydantic import Field

from ..server import mcp
from ...models.task import TaskUpdate, Priority, Status
from ...services.task_service import TaskService
from ...database.database import get_session
from ...utils.errors import TaskNotFoundException


@mcp.tool()
def update_task(
    task_id: int = Field(..., description="Task ID to update"),
    title: Optional[str] = Field(None, description="New title"),
    description: Optional[str] = Field(None, description="New description"),
    priority: Optional[str] = Field(None, description="New priority: low, medium, or high"),
    due_date: Optional[str] = Field(None, description="New due date (YYYY-MM-DD) or 'null' to clear"),
    category: Optional[str] = Field(None, description="New category or 'null' to clear"),
    status: Optional[str] = Field(None, description="New status: todo, in_progress, or done"),
    user_id: int = Field(..., description="User ID (injected by system)")
) -> dict:
    """Update an existing task's attributes."""
    session_gen = get_session()
    db = next(session_gen)
    try:
        # Build update data
        update_data = {}
        changes = []

        if title is not None:
            update_data["title"] = title
            changes.append("title")

        if description is not None:
            update_data["description"] = description
            changes.append("description")

        if priority is not None:
            try:
                update_data["priority"] = Priority(priority.lower())
                changes.append("priority")
            except ValueError:
                pass

        if due_date is not None:
            if due_date.lower() == "null":
                update_data["due_date"] = None
                changes.append("due_date")
            else:
                try:
                    update_data["due_date"] = date.fromisoformat(due_date)
                    changes.append("due_date")
                except ValueError:
                    pass

        if category is not None:
            if category.lower() == "null":
                update_data["category"] = None
            else:
                update_data["category"] = category
            changes.append("category")

        if status is not None:
            try:
                update_data["status"] = Status(status.lower())
                changes.append("status")
            except ValueError:
                pass

        if not update_data:
            return {"error": "validation_error", "message": "No valid fields to update"}

        task_update = TaskUpdate(**update_data)
        task = TaskService.update_task(db=db, task_id=task_id, task_data=task_update, user_id=user_id)

        return {
            "id": task.id,
            "title": task.title,
            "description": task.description or "",
            "priority": task.priority.value,
            "status": task.status.value,
            "due_date": task.due_date.isoformat() if task.due_date else None,
            "updated_at": task.updated_at.isoformat(),
            "changes": changes,
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
