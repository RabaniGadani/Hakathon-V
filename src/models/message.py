"""
Message model for the Todo App
Represents individual messages within a conversation
"""
from datetime import datetime
from typing import Optional, List, TYPE_CHECKING
from enum import Enum
from sqlmodel import Field, SQLModel, Relationship

if TYPE_CHECKING:
    from .conversation import Conversation
    from .mcp_tool_call import MCPToolCall


class MessageRole(str, Enum):
    """Message role options"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageBase(SQLModel):
    """Base model for message with common fields"""
    role: MessageRole = Field(nullable=False)
    content: str = Field(nullable=False)
    token_count: Optional[int] = Field(default=None)


class Message(MessageBase, table=True):
    """Message model for database table"""
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id", nullable=False, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    conversation: Optional["Conversation"] = Relationship(back_populates="messages")
    tool_calls: List["MCPToolCall"] = Relationship(back_populates="message")


class MessageCreate(SQLModel):
    """Schema for creating a new message"""
    role: MessageRole
    content: str
    token_count: Optional[int] = None


class MessagePublic(MessageBase):
    """Public representation of message"""
    id: int
    conversation_id: int
    created_at: datetime
