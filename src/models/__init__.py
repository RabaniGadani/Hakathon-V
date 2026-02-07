"""
Models module for the Todo App
Contains all database models and their relationships
"""
from sqlmodel import SQLModel
from .user import User
from .task import Task
from .conversation import Conversation, ConversationCreate, ConversationUpdate, ConversationPublic
from .message import Message, MessageCreate, MessagePublic, MessageRole
from .mcp_tool_call import MCPToolCall, MCPToolCallCreate, MCPToolCallUpdate, MCPToolCallPublic, ToolCallStatus

__all__ = [
    "SQLModel",
    "User",
    "Task",
    "Conversation",
    "ConversationCreate",
    "ConversationUpdate",
    "ConversationPublic",
    "Message",
    "MessageCreate",
    "MessagePublic",
    "MessageRole",
    "MCPToolCall",
    "MCPToolCallCreate",
    "MCPToolCallUpdate",
    "MCPToolCallPublic",
    "ToolCallStatus",
]