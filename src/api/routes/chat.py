"""
Chat API routes for the Todo App
Handles conversation and message endpoints for AI assistant interactions
"""
from fastapi import APIRouter, Depends, HTTPException, status, Query, Request
from fastapi.responses import StreamingResponse
from typing import List, Optional, AsyncGenerator
from sqlmodel import Session
from pydantic import BaseModel, field_validator
from collections import defaultdict
from datetime import datetime, timedelta
import threading
import json
import asyncio

from ...database.database import get_session
from ...models.conversation import ConversationPublic, ConversationCreate, ConversationUpdate
from ...models.message import MessagePublic, MessageRole
from ...services.conversation_service import ConversationService, ConversationNotFoundException
from ...services.chat_service import ChatService, ChatServiceError, AgentExecutionError
from ...api.deps import get_current_user
from ...models.user import User
from ...models.mcp_tool_call import ToolCallStatus


router = APIRouter()


# ============ Rate Limiting ============

class RateLimiter:
    """Simple in-memory rate limiter for chat endpoints."""

    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[int, list[datetime]] = defaultdict(list)
        self.lock = threading.Lock()

    def is_allowed(self, user_id: int) -> bool:
        """Check if user is allowed to make a request."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=self.window_seconds)

        with self.lock:
            # Clean old requests
            self.requests[user_id] = [
                req_time for req_time in self.requests[user_id]
                if req_time > window_start
            ]

            # Check rate limit
            if len(self.requests[user_id]) >= self.max_requests:
                return False

            # Record this request
            self.requests[user_id].append(now)
            return True

    def get_retry_after(self, user_id: int) -> int:
        """Get seconds until next allowed request."""
        if not self.requests[user_id]:
            return 0

        oldest_request = min(self.requests[user_id])
        retry_after = (oldest_request + timedelta(seconds=self.window_seconds) - datetime.utcnow()).seconds
        return max(0, retry_after)


# Initialize rate limiter (20 requests per minute per user)
chat_rate_limiter = RateLimiter(max_requests=20, window_seconds=60)


# ============ Request/Response schemas ============

# Message validation constants
MAX_MESSAGE_LENGTH = 4000  # Characters
MIN_MESSAGE_LENGTH = 1


class SendMessageRequest(BaseModel):
    """Request body for sending a message"""
    message: str

    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        # Strip whitespace
        v = v.strip()

        # Check minimum length
        if len(v) < MIN_MESSAGE_LENGTH:
            raise ValueError('Message cannot be empty')

        # Check maximum length
        if len(v) > MAX_MESSAGE_LENGTH:
            raise ValueError(f'Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters')

        return v


class SendMessageResponse(BaseModel):
    """Response from sending a message"""
    conversation_id: int
    message: MessagePublic
    tool_calls: Optional[List[dict]] = None


class CreateConversationRequest(BaseModel):
    """Request body for creating a conversation"""
    title: Optional[str] = None


# Conversation endpoints

@router.get("/{user_id}/conversations", response_model=List[ConversationPublic])
async def get_conversations(
    user_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Get all conversations for the authenticated user.

    Args:
        user_id: ID of the user requesting conversations
        skip: Number of conversations to skip for pagination
        limit: Maximum number of conversations to return
        current_user: Currently authenticated user
        session: Database session

    Returns:
        List of conversations ordered by most recently updated
    """
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own conversations"
        )

    try:
        conversations = ConversationService.get_conversations_by_user(
            db=session,
            user_id=user_id,
            limit=limit,
            offset=skip,
            active_only=True
        )
        return conversations
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversations: {str(e)}"
        )


@router.post("/{user_id}/conversations", response_model=ConversationPublic, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    user_id: int,
    request: CreateConversationRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Create a new conversation for the authenticated user.

    Args:
        user_id: ID of the user creating the conversation
        request: Conversation creation data
        current_user: Currently authenticated user
        session: Database session

    Returns:
        Created conversation
    """
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only create conversations for yourself"
        )

    try:
        conversation = ConversationService.create_conversation(
            db=session,
            user_id=user_id,
            title=request.title
        )
        return conversation
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create conversation: {str(e)}"
        )


@router.get("/{user_id}/conversations/{conversation_id}", response_model=ConversationPublic)
async def get_conversation(
    user_id: int,
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Get a specific conversation by ID.

    Args:
        user_id: ID of the user requesting the conversation
        conversation_id: ID of the conversation to retrieve
        current_user: Currently authenticated user
        session: Database session

    Returns:
        The requested conversation
    """
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own conversations"
        )

    try:
        conversation = ConversationService.get_conversation_by_id(
            db=session,
            conversation_id=conversation_id,
            user_id=user_id
        )
        return conversation
    except ConversationNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation with ID {conversation_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve conversation: {str(e)}"
        )


@router.delete("/{user_id}/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    user_id: int,
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Delete a conversation (soft delete).

    Args:
        user_id: ID of the user deleting the conversation
        conversation_id: ID of the conversation to delete
        current_user: Currently authenticated user
        session: Database session
    """
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only delete your own conversations"
        )

    try:
        ConversationService.delete_conversation(
            db=session,
            conversation_id=conversation_id,
            user_id=user_id,
            soft_delete=True
        )
    except ConversationNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation with ID {conversation_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )


# Message endpoints

@router.get("/{user_id}/conversations/{conversation_id}/messages", response_model=List[MessagePublic])
async def get_messages(
    user_id: int,
    conversation_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Get all messages for a conversation.

    Args:
        user_id: ID of the user requesting messages
        conversation_id: ID of the conversation
        skip: Number of messages to skip
        limit: Maximum number of messages to return
        current_user: Currently authenticated user
        session: Database session

    Returns:
        List of messages ordered by creation time (oldest first)
    """
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only access your own conversations"
        )

    try:
        messages = ConversationService.get_messages_by_conversation(
            db=session,
            conversation_id=conversation_id,
            user_id=user_id,
            limit=limit,
            offset=skip
        )
        return messages
    except ConversationNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation with ID {conversation_id} not found"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve messages: {str(e)}"
        )


@router.post("/{user_id}/conversations/{conversation_id}/messages", response_model=SendMessageResponse)
async def send_message(
    user_id: int,
    conversation_id: int,
    request: SendMessageRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Send a message to the AI assistant and get a response.

    This endpoint:
    1. Validates the message and checks rate limits
    2. Stores the user's message
    3. Sends it to the AI agent with conversation context
    4. Stores and returns the assistant's response
    5. Records any tool calls made by the agent

    Args:
        user_id: ID of the user sending the message
        conversation_id: ID of the conversation
        request: Message content (validated by Pydantic)
        current_user: Currently authenticated user
        session: Database session

    Returns:
        The assistant's response message and any tool calls

    Raises:
        HTTPException 403: User not authorized
        HTTPException 429: Rate limit exceeded
        HTTPException 503: AI service unavailable
    """
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only send messages in your own conversations"
        )

    # Check rate limit
    if not chat_rate_limiter.is_allowed(user_id):
        retry_after = chat_rate_limiter.get_retry_after(user_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Please wait {retry_after} seconds before sending another message.",
            headers={"Retry-After": str(retry_after)}
        )

    try:
        result = ChatService.process_message(
            db=session,
            conversation_id=conversation_id,
            user_id=user_id,
            user_message=request.message.strip()
        )

        # Get the actual assistant message from database
        from datetime import datetime
        messages = ConversationService.get_messages_by_conversation(
            db=session,
            conversation_id=conversation_id,
            user_id=user_id,
            limit=1,
            offset=0
        )

        # Find the assistant message we just created
        assistant_msg = None
        all_messages = ConversationService.get_messages_by_conversation(
            db=session,
            conversation_id=conversation_id,
            user_id=user_id,
            limit=100
        )
        for msg in reversed(all_messages):
            if msg.role == MessageRole.ASSISTANT:
                assistant_msg = msg
                break

        response_message = MessagePublic(
            id=result["message_id"],
            conversation_id=conversation_id,
            role=MessageRole.ASSISTANT,
            content=result["response"],
            token_count=assistant_msg.token_count if assistant_msg else None,
            created_at=assistant_msg.created_at if assistant_msg else datetime.utcnow()
        )

        return SendMessageResponse(
            conversation_id=conversation_id,
            message=response_message,
            tool_calls=result.get("tool_calls")
        )

    except ConversationNotFoundException:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation with ID {conversation_id} not found"
        )
    except AgentExecutionError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI service temporarily unavailable: {str(e)}"
        )
    except ChatServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat service error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process message: {str(e)}"
        )


@router.post("/{user_id}/conversations/{conversation_id}/messages/stream")
async def stream_message(
    user_id: int,
    conversation_id: int,
    request: SendMessageRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Send a message and stream the AI assistant's response using Server-Sent Events.

    This endpoint provides real-time streaming of the AI response as it's generated.
    The stream sends events in the following format:
    - data: {"type": "start", "message_id": int} - Response started
    - data: {"type": "chunk", "content": str} - Text chunk
    - data: {"type": "tool_call", "name": str, "status": str} - Tool execution
    - data: {"type": "done", "message_id": int, "tool_calls": list} - Response complete
    - data: {"type": "error", "message": str} - Error occurred

    Args:
        user_id: ID of the user sending the message
        conversation_id: ID of the conversation
        request: Message content
        current_user: Currently authenticated user
        session: Database session

    Returns:
        StreamingResponse with Server-Sent Events
    """
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You can only send messages in your own conversations"
        )

    # Check rate limit
    if not chat_rate_limiter.is_allowed(user_id):
        retry_after = chat_rate_limiter.get_retry_after(user_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Please wait {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)}
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for the streaming response."""
        try:
            # Validate conversation ownership first
            try:
                ConversationService.get_conversation_by_id(
                    session, conversation_id, user_id
                )
            except ConversationNotFoundException:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Conversation not found'})}\n\n"
                return

            # Store user message
            user_msg = ConversationService.add_message(
                db=session,
                conversation_id=conversation_id,
                user_id=user_id,
                role=MessageRole.USER,
                content=request.message.strip()
            )

            # Send start event
            yield f"data: {json.dumps({'type': 'start', 'user_message_id': user_msg.id})}\n\n"

            # Get conversation history
            history = ConversationService.get_messages_by_conversation(
                db=session,
                conversation_id=conversation_id,
                user_id=user_id,
                limit=50
            )

            # Build context messages
            context_messages = ChatService._build_context_messages(history)

            # Stream the response from the agent
            full_response = ""
            tool_calls = []

            async for event in ChatService.stream_agent_response(
                user_id=user_id,
                messages=context_messages
            ):
                if event["type"] == "chunk":
                    full_response += event["content"]
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["type"] == "tool_call":
                    tool_calls.append(event)
                    yield f"data: {json.dumps(event)}\n\n"
                elif event["type"] == "error":
                    yield f"data: {json.dumps(event)}\n\n"
                    return

            # Store assistant response
            assistant_msg = ConversationService.add_message(
                db=session,
                conversation_id=conversation_id,
                user_id=user_id,
                role=MessageRole.ASSISTANT,
                content=full_response
            )

            # Record tool calls
            tool_calls_recorded = []
            for tc in tool_calls:
                if "name" in tc:
                    recorded = ConversationService.record_tool_call(
                        db=session,
                        message_id=assistant_msg.id,
                        tool_name=tc["name"],
                        parameters=tc.get("parameters", {})
                    )
                    ConversationService.update_tool_call_result(
                        db=session,
                        tool_call_id=recorded.id,
                        status=ToolCallStatus.SUCCESS if tc.get("success", True) else ToolCallStatus.ERROR,
                        result=tc.get("result"),
                        error_message=tc.get("error")
                    )
                    tool_calls_recorded.append({
                        "id": recorded.id,
                        "name": tc["name"],
                        "success": tc.get("success", True)
                    })

            # Send done event
            yield f"data: {json.dumps({'type': 'done', 'message_id': assistant_msg.id, 'tool_calls': tool_calls_recorded})}\n\n"

        except AgentExecutionError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to process message: {str(e)}'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
