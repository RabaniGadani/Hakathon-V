"""
MCP (Model Context Protocol) module for the Todo App
Exposes task operations as MCP tools for AI agent consumption
"""
from .server import mcp, register_tools

__all__ = [
    "mcp",
    "register_tools",
]
