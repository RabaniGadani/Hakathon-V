"""
MCP Tool Call model for the Todo App
Audit log of MCP tool invocations for traceability
"""
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING
from enum import Enum
from sqlmodel import Field, SQLModel, Relationship, Column
from sqlalchemy import JSON

if TYPE_CHECKING:
    from .message import Message


class ToolCallStatus(str, Enum):
    """Tool call execution status"""
    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"


class MCPToolCallBase(SQLModel):
    """Base model for MCP tool call with common fields"""
    tool_name: str = Field(max_length=100, nullable=False)
    status: ToolCallStatus = Field(default=ToolCallStatus.PENDING)
    error_message: Optional[str] = Field(default=None)
    execution_time_ms: Optional[int] = Field(default=None)


class MCPToolCall(MCPToolCallBase, table=True):
    """MCP Tool Call model for database table"""
    __tablename__ = "mcp_tool_call"

    id: Optional[int] = Field(default=None, primary_key=True)
    message_id: int = Field(foreign_key="message.id", nullable=False, index=True)
    parameters: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    result: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    message: Optional["Message"] = Relationship(back_populates="tool_calls")


class MCPToolCallCreate(SQLModel):
    """Schema for creating a new tool call record"""
    tool_name: str
    parameters: Dict[str, Any] = {}


class MCPToolCallUpdate(SQLModel):
    """Schema for updating tool call after execution"""
    status: Optional[ToolCallStatus] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: Optional[int] = None


class MCPToolCallPublic(MCPToolCallBase):
    """Public representation of tool call"""
    id: int
    message_id: int
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]]
    created_at: datetime
