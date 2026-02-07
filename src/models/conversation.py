"""
Conversation model for the Todo App
Represents a chat session between a user and the AI assistant
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from sqlmodel import Field, SQLModel, Relationship, Column
from sqlalchemy import JSON

if TYPE_CHECKING:
    from .message import Message
    from .user import User


class ConversationBase(SQLModel):
    """Base model for conversation with common fields"""
    title: Optional[str] = Field(default=None, max_length=200)
    is_active: bool = Field(default=True)


class Conversation(ConversationBase, table=True):
    """Conversation model for database table"""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", nullable=False, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(
        default_factory=datetime.utcnow,
        sa_column_kwargs={"onupdate": datetime.utcnow}
    )
    # Context data for storing context like last referenced task
    # Note: 'metadata' is reserved in SQLAlchemy, using 'context_data' instead
    context_data: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # Relationships
    messages: List["Message"] = Relationship(back_populates="conversation")


class ConversationCreate(SQLModel):
    """Schema for creating a new conversation"""
    title: Optional[str] = None


class ConversationUpdate(SQLModel):
    """Schema for updating conversation"""
    title: Optional[str] = None
    is_active: Optional[bool] = None


class ConversationPublic(ConversationBase):
    """Public representation of conversation"""
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    message_count: Optional[int] = None
