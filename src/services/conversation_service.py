"""
Conversation service module for the Todo App
Handles business logic for conversation and message operations
"""
from typing import List, Optional
from sqlmodel import Session, select, func
from datetime import datetime

from ..models.conversation import Conversation, ConversationCreate, ConversationUpdate
from ..models.message import Message, MessageCreate, MessageRole
from ..models.mcp_tool_call import MCPToolCall, MCPToolCallCreate, MCPToolCallUpdate, ToolCallStatus
from ..utils.errors import TaskNotFoundException
from ..utils.logging import log_error


class ConversationNotFoundException(Exception):
    """Raised when a conversation is not found"""
    def __init__(self, conversation_id: int):
        self.conversation_id = conversation_id
        super().__init__(f"Conversation with ID {conversation_id} not found")


class ConversationService:
    """Service class for conversation and message operations"""

    @staticmethod
    def get_conversations_by_user(
        db: Session,
        user_id: int,
        limit: int = 20,
        offset: int = 0,
        active_only: bool = True
    ) -> List[Conversation]:
        """
        Get all conversations for a specific user with pagination.

        Args:
            db: Database session
            user_id: User ID to filter by
            limit: Maximum number of conversations to return
            offset: Number of conversations to skip
            active_only: If True, only return active conversations

        Returns:
            List of Conversation objects
        """
        try:
            statement = select(Conversation).where(Conversation.user_id == user_id)

            if active_only:
                statement = statement.where(Conversation.is_active == True)

            statement = statement.order_by(Conversation.updated_at.desc())
            statement = statement.offset(offset).limit(limit)

            conversations = db.exec(statement).all()
            return list(conversations)
        except Exception as e:
            log_error(e, "ConversationService.get_conversations_by_user", user_id)
            raise

    @staticmethod
    def get_conversation_by_id(
        db: Session,
        conversation_id: int,
        user_id: int
    ) -> Conversation:
        """
        Get a specific conversation by ID for a specific user.

        Args:
            db: Database session
            conversation_id: Conversation ID to retrieve
            user_id: User ID for authorization check

        Returns:
            Conversation object

        Raises:
            ConversationNotFoundException: If conversation not found or unauthorized
        """
        try:
            statement = select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id
            )
            conversation = db.exec(statement).first()

            if not conversation:
                raise ConversationNotFoundException(conversation_id)

            return conversation
        except ConversationNotFoundException:
            raise
        except Exception as e:
            log_error(e, f"ConversationService.get_conversation_by_id (id={conversation_id})", user_id)
            raise

    @staticmethod
    def create_conversation(
        db: Session,
        user_id: int,
        title: Optional[str] = None
    ) -> Conversation:
        """
        Create a new conversation for a user.

        Args:
            db: Database session
            user_id: User ID who owns the conversation
            title: Optional title for the conversation

        Returns:
            Created Conversation object
        """
        try:
            conversation = Conversation(
                user_id=user_id,
                title=title,
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )

            db.add(conversation)
            db.commit()
            db.refresh(conversation)

            return conversation
        except Exception as e:
            log_error(e, "ConversationService.create_conversation", user_id)
            db.rollback()
            raise

    @staticmethod
    def update_conversation(
        db: Session,
        conversation_id: int,
        user_id: int,
        update_data: ConversationUpdate
    ) -> Conversation:
        """
        Update a conversation's attributes.

        Args:
            db: Database session
            conversation_id: Conversation ID to update
            user_id: User ID for authorization check
            update_data: Data to update

        Returns:
            Updated Conversation object
        """
        try:
            conversation = ConversationService.get_conversation_by_id(db, conversation_id, user_id)

            update_dict = update_data.model_dump(exclude_unset=True)
            for field, value in update_dict.items():
                setattr(conversation, field, value)

            conversation.updated_at = datetime.utcnow()

            db.add(conversation)
            db.commit()
            db.refresh(conversation)

            return conversation
        except ConversationNotFoundException:
            raise
        except Exception as e:
            log_error(e, f"ConversationService.update_conversation (id={conversation_id})", user_id)
            db.rollback()
            raise

    @staticmethod
    def delete_conversation(
        db: Session,
        conversation_id: int,
        user_id: int,
        soft_delete: bool = True
    ) -> bool:
        """
        Delete a conversation (soft delete by default).

        Args:
            db: Database session
            conversation_id: Conversation ID to delete
            user_id: User ID for authorization check
            soft_delete: If True, mark as inactive; if False, permanently delete

        Returns:
            True if successful
        """
        try:
            conversation = ConversationService.get_conversation_by_id(db, conversation_id, user_id)

            if soft_delete:
                conversation.is_active = False
                conversation.updated_at = datetime.utcnow()
                db.add(conversation)
            else:
                db.delete(conversation)

            db.commit()
            return True
        except ConversationNotFoundException:
            raise
        except Exception as e:
            log_error(e, f"ConversationService.delete_conversation (id={conversation_id})", user_id)
            db.rollback()
            raise

    # Message operations

    @staticmethod
    def get_messages_by_conversation(
        db: Session,
        conversation_id: int,
        user_id: int,
        limit: int = 100,
        offset: int = 0
    ) -> List[Message]:
        """
        Get all messages for a conversation.

        Args:
            db: Database session
            conversation_id: Conversation ID to get messages for
            user_id: User ID for authorization check
            limit: Maximum number of messages to return
            offset: Number of messages to skip

        Returns:
            List of Message objects ordered by creation time
        """
        try:
            # Verify user owns the conversation
            ConversationService.get_conversation_by_id(db, conversation_id, user_id)

            statement = select(Message).where(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at.asc())
            statement = statement.offset(offset).limit(limit)

            messages = db.exec(statement).all()
            return list(messages)
        except ConversationNotFoundException:
            raise
        except Exception as e:
            log_error(e, f"ConversationService.get_messages_by_conversation (id={conversation_id})", user_id)
            raise

    @staticmethod
    def add_message(
        db: Session,
        conversation_id: int,
        user_id: int,
        role: MessageRole,
        content: str,
        token_count: Optional[int] = None
    ) -> Message:
        """
        Add a message to a conversation.

        Args:
            db: Database session
            conversation_id: Conversation ID to add message to
            user_id: User ID for authorization check
            role: Message role (user, assistant, system)
            content: Message content
            token_count: Optional token count for the message

        Returns:
            Created Message object
        """
        try:
            # Verify user owns the conversation and update its timestamp
            conversation = ConversationService.get_conversation_by_id(db, conversation_id, user_id)
            conversation.updated_at = datetime.utcnow()

            message = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                token_count=token_count,
                created_at=datetime.utcnow()
            )

            db.add(conversation)
            db.add(message)
            db.commit()
            db.refresh(message)

            return message
        except ConversationNotFoundException:
            raise
        except Exception as e:
            log_error(e, f"ConversationService.add_message (conversation_id={conversation_id})", user_id)
            db.rollback()
            raise

    # Tool call operations

    @staticmethod
    def record_tool_call(
        db: Session,
        message_id: int,
        tool_name: str,
        parameters: dict
    ) -> MCPToolCall:
        """
        Record a tool call initiated from a message.

        Args:
            db: Database session
            message_id: Message ID that initiated the tool call
            tool_name: Name of the MCP tool being called
            parameters: Parameters passed to the tool

        Returns:
            Created MCPToolCall object with pending status
        """
        try:
            tool_call = MCPToolCall(
                message_id=message_id,
                tool_name=tool_name,
                parameters=parameters,
                status=ToolCallStatus.PENDING,
                created_at=datetime.utcnow()
            )

            db.add(tool_call)
            db.commit()
            db.refresh(tool_call)

            return tool_call
        except Exception as e:
            log_error(e, f"ConversationService.record_tool_call (message_id={message_id})", None)
            db.rollback()
            raise

    @staticmethod
    def update_tool_call_result(
        db: Session,
        tool_call_id: int,
        status: ToolCallStatus,
        result: Optional[dict] = None,
        error_message: Optional[str] = None,
        execution_time_ms: Optional[int] = None
    ) -> MCPToolCall:
        """
        Update a tool call with its execution result.

        Args:
            db: Database session
            tool_call_id: Tool call ID to update
            status: Execution status (success or error)
            result: Tool execution result (if successful)
            error_message: Error message (if failed)
            execution_time_ms: Execution time in milliseconds

        Returns:
            Updated MCPToolCall object
        """
        try:
            statement = select(MCPToolCall).where(MCPToolCall.id == tool_call_id)
            tool_call = db.exec(statement).first()

            if not tool_call:
                raise ValueError(f"Tool call with ID {tool_call_id} not found")

            tool_call.status = status
            tool_call.result = result
            tool_call.error_message = error_message
            tool_call.execution_time_ms = execution_time_ms

            db.add(tool_call)
            db.commit()
            db.refresh(tool_call)

            return tool_call
        except Exception as e:
            log_error(e, f"ConversationService.update_tool_call_result (id={tool_call_id})", None)
            db.rollback()
            raise

    @staticmethod
    def get_conversation_count_by_user(db: Session, user_id: int) -> int:
        """Get total count of active conversations for a user."""
        try:
            statement = select(func.count(Conversation.id)).where(
                Conversation.user_id == user_id,
                Conversation.is_active == True
            )
            count = db.exec(statement).one()
            return count
        except Exception as e:
            log_error(e, "ConversationService.get_conversation_count_by_user", user_id)
            raise

    @staticmethod
    def update_metadata(
        db: Session,
        conversation_id: int,
        user_id: int,
        metadata_update: dict
    ) -> Conversation:
        """
        Update conversation metadata with new values.
        Merges new metadata with existing metadata.

        Args:
            db: Database session
            conversation_id: Conversation ID to update
            user_id: User ID for authorization check
            metadata_update: Dictionary of metadata to merge

        Returns:
            Updated Conversation object
        """
        try:
            conversation = ConversationService.get_conversation_by_id(db, conversation_id, user_id)

            # Merge metadata
            current_metadata = conversation.context_data or {}
            current_metadata.update(metadata_update)
            conversation.context_data = current_metadata
            conversation.updated_at = datetime.utcnow()

            db.add(conversation)
            db.commit()
            db.refresh(conversation)

            return conversation
        except ConversationNotFoundException:
            raise
        except Exception as e:
            log_error(e, f"ConversationService.update_metadata (id={conversation_id})", user_id)
            db.rollback()
            raise

    @staticmethod
    def set_last_referenced_task(
        db: Session,
        conversation_id: int,
        user_id: int,
        task_id: int,
        task_title: str
    ) -> Conversation:
        """
        Store the last referenced task in conversation metadata.
        Used for context awareness (e.g., "complete it", "delete that task").

        Args:
            db: Database session
            conversation_id: Conversation ID to update
            user_id: User ID for authorization check
            task_id: ID of the last referenced task
            task_title: Title of the last referenced task

        Returns:
            Updated Conversation object
        """
        return ConversationService.update_metadata(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            metadata_update={
                "last_referenced_task": {
                    "id": task_id,
                    "title": task_title,
                    "referenced_at": datetime.utcnow().isoformat()
                }
            }
        )

    @staticmethod
    def get_last_referenced_task(
        db: Session,
        conversation_id: int,
        user_id: int
    ) -> dict | None:
        """
        Get the last referenced task from conversation metadata.

        Args:
            db: Database session
            conversation_id: Conversation ID to query
            user_id: User ID for authorization check

        Returns:
            Dict with task id and title, or None if not set
        """
        try:
            conversation = ConversationService.get_conversation_by_id(db, conversation_id, user_id)
            metadata = conversation.context_data or {}
            return metadata.get("last_referenced_task")
        except ConversationNotFoundException:
            raise
        except Exception as e:
            log_error(e, f"ConversationService.get_last_referenced_task (id={conversation_id})", user_id)
            return None


__all__ = [
    "ConversationService",
    "ConversationNotFoundException",
]
