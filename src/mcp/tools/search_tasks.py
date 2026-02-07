"""
Search Tasks MCP Tool
Searches tasks by keyword
"""
from pydantic import Field

from ..server import mcp
from ...services.task_service import TaskService
from ...database.database import get_session


@mcp.tool()
def search_tasks(
    query: str = Field(..., description="Search term (minimum 2 characters)"),
    limit: int = Field(20, description="Maximum results (1-50)"),
    user_id: int = Field(..., description="User ID (injected by system)")
) -> dict:
    """Search tasks by keyword in title and description."""
    # Validate query length
    if len(query) < 2:
        return {"error": "validation_error", "message": "Query must be at least 2 characters"}

    # Clamp limit
    limit = max(1, min(50, limit))

    session_gen = get_session()
    db = next(session_gen)
    try:
        # Get all user tasks
        tasks = TaskService.get_tasks_by_user(db=db, user_id=user_id)

        # Search in title and description
        query_lower = query.lower()
        matching_tasks = []
        for task in tasks:
            relevance = 0.0
            match_field = None

            # Check title (higher relevance)
            if query_lower in task.title.lower():
                relevance = 1.0
                match_field = "title"
            # Check description (lower relevance)
            elif task.description and query_lower in task.description.lower():
                relevance = 0.5
                match_field = "description"

            if match_field:
                matching_tasks.append({
                    "id": task.id,
                    "title": task.title,
                    "match_field": match_field,
                    "relevance": relevance,
                })

        # Sort by relevance (descending) and apply limit
        matching_tasks.sort(key=lambda x: x["relevance"], reverse=True)
        matching_tasks = matching_tasks[:limit]

        return {
            "tasks": matching_tasks,
            "query": query,
            "total": len(matching_tasks),
        }
    except Exception as e:
        return {"error": "database_error", "message": str(e)}
    finally:
        try:
            next(session_gen)
        except StopIteration:
            pass
