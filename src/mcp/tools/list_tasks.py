"""
List Tasks MCP Tool
Lists tasks for the authenticated user with optional filtering
"""
from typing import Optional, Literal
from datetime import date, timedelta
from pydantic import Field

from ..server import mcp
from ...services.task_service import TaskService
from ...database.database import get_session


@mcp.tool()
def list_tasks(
    status: Literal["all", "pending", "completed"] = Field("all", description="Filter by status"),
    priority: Literal["all", "low", "medium", "high"] = Field("all", description="Filter by priority"),
    due_date_filter: Literal["all", "today", "this_week", "overdue"] = Field("all", description="Filter by due date"),
    limit: int = Field(50, description="Maximum number of tasks to return (1-100)"),
    user_id: int = Field(..., description="User ID (injected by system)")
) -> dict:
    """List all tasks for the authenticated user with optional filtering."""
    # Clamp limit
    limit = max(1, min(100, limit))

    # Get database session
    session_gen = get_session()
    db = next(session_gen)
    try:
        # Get base tasks
        tasks = TaskService.get_tasks_by_user(
            db=db,
            user_id=user_id,
            status=status if status != "all" else None,
        )

        # Apply priority filter
        if priority != "all":
            tasks = [t for t in tasks if t.priority.value == priority]

        # Apply due date filter
        today = date.today()
        if due_date_filter == "today":
            tasks = [t for t in tasks if t.due_date == today]
        elif due_date_filter == "this_week":
            week_end = today + timedelta(days=7)
            tasks = [t for t in tasks if t.due_date and today <= t.due_date <= week_end]
        elif due_date_filter == "overdue":
            tasks = [t for t in tasks if t.due_date and t.due_date < today and not t.completed]

        # Apply limit
        tasks = tasks[:limit]

        return {
            "tasks": [
                {
                    "id": task.id,
                    "title": task.title,
                    "priority": task.priority.value,
                    "status": task.status.value,
                    "due_date": task.due_date.isoformat() if task.due_date else None,
                    "completed": task.completed,
                }
                for task in tasks
            ],
            "total": len(tasks),
            "filters_applied": {
                "status": status,
                "priority": priority,
                "due_date": due_date_filter,
            },
        }
    except Exception as e:
        return {"error": "database_error", "message": str(e)}
    finally:
        try:
            next(session_gen)
        except StopIteration:
            pass
