"""
Create Task MCP Tool
Creates a new task for the authenticated user
"""
from typing import Optional
from datetime import date
from pydantic import Field

from ..server import mcp
from ...models.task import TaskCreate, Priority
from ...services.task_service import TaskService
from ...database.database import get_session


@mcp.tool()
def create_task(
    title: str = Field(..., description="Task title (1-200 chars)"),
    description: str = Field("", description="Detailed task description (max 1000 chars)"),
    priority: str = Field("medium", description="Task priority: low, medium, or high"),
    due_date: Optional[str] = Field(None, description="Due date in ISO format (YYYY-MM-DD)"),
    category: Optional[str] = Field(None, description="Task category (max 100 chars)"),
    user_id: int = Field(..., description="User ID (injected by system)")
) -> dict:
    """Create a new todo task for the authenticated user."""
    # Parse priority
    try:
        priority_enum = Priority(priority.lower())
    except ValueError:
        priority_enum = Priority.MEDIUM

    # Parse due date
    parsed_due_date = None
    if due_date:
        try:
            parsed_due_date = date.fromisoformat(due_date)
        except ValueError:
            pass  # Invalid date format, leave as None

    # Create task data
    task_data = TaskCreate(
        title=title,
        description=description if description else None,
        priority=priority_enum,
        due_date=parsed_due_date,
        category=category,
    )

    # Get database session and create task
    session_gen = get_session()
    db = next(session_gen)
    try:
        task = TaskService.create_task(db=db, task_data=task_data, user_id=user_id)
        return {
            "id": task.id,
            "title": task.title,
            "description": task.description or "",
            "priority": task.priority.value,
            "status": task.status.value,
            "due_date": task.due_date.isoformat() if task.due_date else None,
            "completed": task.completed,
            "created_at": task.created_at.isoformat(),
        }
    except Exception as e:
        return {"error": "database_error", "message": str(e)}
    finally:
        try:
            next(session_gen)
        except StopIteration:
            pass
