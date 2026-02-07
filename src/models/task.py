"""
Task model for the Todo App
Defines the task entity with all required fields and relationships
"""
from datetime import datetime, date
from typing import Optional
from enum import Enum
from sqlmodel import Field, SQLModel
from pydantic import BaseModel


class Priority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Status(str, Enum):
    """Task status options"""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class TaskBase(SQLModel):
    """Base model for task with common fields"""
    title: str = Field(min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=1000)
    completed: bool = Field(default=False)
    priority: Priority = Field(default=Priority.MEDIUM)
    status: Status = Field(default=Status.TODO)
    due_date: Optional[date] = Field(default=None)
    category: Optional[str] = Field(default=None, max_length=100)


class Task(TaskBase, table=True):
    """Task model for database table"""
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"onupdate": datetime.utcnow})
    user_id: int = Field(foreign_key="user.id", nullable=False)  # References the User model


class TaskCreate(TaskBase):
    """Schema for creating a new task"""
    pass


class TaskUpdate(SQLModel):
    """Schema for updating task information"""
    title: Optional[str] = None
    description: Optional[str] = None
    completed: Optional[bool] = None
    priority: Optional[Priority] = None
    status: Optional[Status] = None
    due_date: Optional[date] = None
    category: Optional[str] = None


class TaskPublic(TaskBase):
    """Public representation of task (without sensitive data)"""
    id: int
    created_at: datetime
    updated_at: datetime
    user_id: int  # Included for reference but will be filtered out in API responses for security