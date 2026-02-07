"""
MCP Server setup for the Todo App
Uses FastMCP pattern to expose task operations as tools
"""
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP(name="Todo Tasks MCP Server")

__all__ = ["mcp"]


def register_tools():
    """Register all MCP tools with the server. Call after server is created."""
    # Import tools here to avoid circular imports
    # Tools use @mcp.tool() decorator which registers them
    from .tools import (
        create_task,
        list_tasks,
        get_task,
        complete_task,
        update_task,
        delete_task,
        search_tasks,
    )
    return [
        create_task,
        list_tasks,
        get_task,
        complete_task,
        update_task,
        delete_task,
        search_tasks,
    ]
