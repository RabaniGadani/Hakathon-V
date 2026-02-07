"""
Chat service module for the Todo App
Orchestrates AI agent interactions for natural language task management
"""
from typing import Optional, List, Dict, Any, AsyncGenerator
from datetime import datetime
import time
import os
import asyncio

from sqlmodel import Session
from agents import Agent, Runner

from ..config import settings
from ..agents.todo_agent import create_todo_agent, get_max_turns
from ..models.message import Message, MessageRole
from ..models.mcp_tool_call import ToolCallStatus
from .conversation_service import ConversationService, ConversationNotFoundException
from ..utils.logging import log_error


class ChatServiceError(Exception):
    """Base exception for chat service errors"""
    pass


class AgentExecutionError(ChatServiceError):
    """Raised when agent execution fails"""
    pass


class ChatService:
    """
    Service class for AI-powered chat interactions.

    Handles:
    - Processing user messages through the AI agent
    - Managing conversation context
    - Recording tool calls for audit
    - Streaming responses (optional)
    """

    @staticmethod
    def process_message(
        db: Session,
        conversation_id: int,
        user_id: int,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Process a user message and generate an AI response.

        This is the main entry point for chat interactions. It:
        1. Validates the conversation belongs to the user
        2. Stores the user message
        3. Retrieves conversation history for context
        4. Runs the AI agent with MCP tools
        5. Stores the assistant response
        6. Records any tool calls made

        Args:
            db: Database session
            conversation_id: Conversation to add message to
            user_id: Authenticated user's ID
            user_message: The user's input message

        Returns:
            Dict containing:
                - response: The assistant's text response
                - message_id: ID of the stored assistant message
                - tool_calls: List of tool calls made (if any)
                - conversation_id: The conversation ID

        Raises:
            ConversationNotFoundException: If conversation not found or unauthorized
            AgentExecutionError: If agent execution fails
        """
        try:
            # 1. Validate conversation ownership
            conversation = ConversationService.get_conversation_by_id(
                db, conversation_id, user_id
            )

            # 2. Store user message
            user_msg = ConversationService.add_message(
                db=db,
                conversation_id=conversation_id,
                user_id=user_id,
                role=MessageRole.USER,
                content=user_message
            )

            # 3. Get conversation history for context
            history = ConversationService.get_messages_by_conversation(
                db=db,
                conversation_id=conversation_id,
                user_id=user_id,
                limit=50  # Last 50 messages for context
            )

            # 4. Build context messages for agent
            context_messages = ChatService._build_context_messages(history)

            # 5. Run the AI agent
            agent_result = ChatService._run_agent(
                user_id=user_id,
                messages=context_messages
            )

            # 6. Store assistant response
            assistant_msg = ConversationService.add_message(
                db=db,
                conversation_id=conversation_id,
                user_id=user_id,
                role=MessageRole.ASSISTANT,
                content=agent_result["response"],
                token_count=agent_result.get("token_count")
            )

            # 7. Record tool calls if any
            tool_calls_recorded = []
            for tool_call in agent_result.get("tool_calls", []):
                recorded = ConversationService.record_tool_call(
                    db=db,
                    message_id=assistant_msg.id,
                    tool_name=tool_call["name"],
                    parameters=tool_call.get("parameters", {})
                )

                # Update with result
                ConversationService.update_tool_call_result(
                    db=db,
                    tool_call_id=recorded.id,
                    status=ToolCallStatus.SUCCESS if tool_call.get("success", True) else ToolCallStatus.ERROR,
                    result=tool_call.get("result"),
                    error_message=tool_call.get("error"),
                    execution_time_ms=tool_call.get("execution_time_ms")
                )

                tool_calls_recorded.append({
                    "id": recorded.id,
                    "name": tool_call["name"],
                    "success": tool_call.get("success", True)
                })

            return {
                "response": agent_result["response"],
                "message_id": assistant_msg.id,
                "tool_calls": tool_calls_recorded,
                "conversation_id": conversation_id
            }

        except ConversationNotFoundException:
            raise
        except Exception as e:
            log_error(e, f"ChatService.process_message (conversation_id={conversation_id})", user_id)
            raise AgentExecutionError(f"Failed to process message: {str(e)}")

    @staticmethod
    def start_conversation(
        db: Session,
        user_id: int,
        initial_message: Optional[str] = None,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Start a new conversation, optionally with an initial message.

        Args:
            db: Database session
            user_id: User ID to create conversation for
            initial_message: Optional first message to process
            title: Optional conversation title

        Returns:
            Dict containing:
                - conversation_id: The new conversation ID
                - response: Assistant response (if initial_message provided)
                - message_id: Assistant message ID (if initial_message provided)
        """
        try:
            # Create the conversation
            conversation = ConversationService.create_conversation(
                db=db,
                user_id=user_id,
                title=title
            )

            result = {
                "conversation_id": conversation.id,
                "created_at": conversation.created_at.isoformat()
            }

            # Process initial message if provided
            if initial_message:
                chat_result = ChatService.process_message(
                    db=db,
                    conversation_id=conversation.id,
                    user_id=user_id,
                    user_message=initial_message
                )
                result.update(chat_result)

            return result

        except Exception as e:
            log_error(e, "ChatService.start_conversation", user_id)
            raise

    @staticmethod
    def _build_context_messages(history: List[Message]) -> List[Dict[str, str]]:
        """
        Build context messages for the agent from conversation history.

        Args:
            history: List of Message objects in chronological order

        Returns:
            List of dicts with 'role' and 'content' keys
        """
        messages = []
        for msg in history:
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        return messages

    @staticmethod
    def _run_agent(
        user_id: int,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Run the AI agent with the given context.
        Supports both OpenAI and Gemini providers based on AI_PROVIDER setting.

        Args:
            user_id: User ID for tool authorization
            messages: Conversation history as context

        Returns:
            Dict containing:
                - response: Agent's text response
                - tool_calls: List of tool calls made
                - token_count: Approximate token usage
        """
        provider = settings.ai_provider.lower()

        if provider == "gemini":
            return ChatService._run_gemini_agent(user_id, messages)
        else:
            return ChatService._run_openai_agent(user_id, messages)

    @staticmethod
    def _run_gemini_agent(
        user_id: int,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Run the Gemini-based agent."""
        from ..agents.gemini_agent import create_gemini_agent

        if not settings.gemini_api_key:
            raise AgentExecutionError("Gemini API key not configured. Set GEMINI_API_KEY in your .env file.")

        try:
            agent = create_gemini_agent(user_id)
            result = agent.run(messages=messages, max_turns=get_max_turns())

            return {
                "response": result.get("response", ""),
                "tool_calls": result.get("tool_calls", []),
                "token_count": result.get("token_count")
            }

        except Exception as e:
            error_str = str(e).lower()
            log_error(e, f"ChatService._run_gemini_agent (user_id={user_id})", user_id)

            if "api key" in error_str or "invalid" in error_str:
                raise AgentExecutionError("Gemini API key is invalid. Please check your configuration.")
            elif "quota" in error_str or "rate" in error_str:
                raise AgentExecutionError("Gemini API quota exceeded. Please try again later.")
            else:
                raise AgentExecutionError(f"Gemini agent execution failed: {str(e)}")

    @staticmethod
    def _run_openai_agent(
        user_id: int,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Run the OpenAI-based agent."""
        if not settings.openai_api_key:
            raise AgentExecutionError("OpenAI API key not configured. Set OPENAI_API_KEY in your .env file.")

        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

        try:
            # Create agent for this user
            agent = create_todo_agent(user_id)

            # Run the agent with conversation context
            runner = Runner(
                agent=agent,
                context={"user_id": user_id}
            )

            # Execute with the latest user message
            # The agent will use MCP tools as needed
            result = runner.run(
                messages=messages,
                max_turns=get_max_turns()
            )

            # Extract response and tool calls
            response_text = result.final_output if hasattr(result, 'final_output') else str(result)

            tool_calls = []
            if hasattr(result, 'tool_calls'):
                for tc in result.tool_calls:
                    tool_calls.append({
                        "name": tc.name,
                        "parameters": tc.parameters if hasattr(tc, 'parameters') else {},
                        "result": tc.result if hasattr(tc, 'result') else None,
                        "success": not hasattr(tc, 'error') or tc.error is None,
                        "error": tc.error if hasattr(tc, 'error') else None
                    })

            return {
                "response": response_text,
                "tool_calls": tool_calls,
                "token_count": None  # Token counting would require additional tracking
            }

        except ConnectionError as e:
            log_error(e, f"ChatService._run_openai_agent - Connection error (user_id={user_id})", user_id)
            raise AgentExecutionError("Unable to connect to AI service. Please check your internet connection and try again.")

        except TimeoutError as e:
            log_error(e, f"ChatService._run_openai_agent - Timeout (user_id={user_id})", user_id)
            raise AgentExecutionError("AI service request timed out. Please try again.")

        except Exception as e:
            error_str = str(e).lower()
            log_error(e, f"ChatService._run_openai_agent (user_id={user_id})", user_id)

            # Check for common OpenAI API errors
            if "rate limit" in error_str or "429" in error_str:
                raise AgentExecutionError("AI service rate limit exceeded. Please wait a moment and try again.")
            elif "invalid api key" in error_str or "authentication" in error_str or "401" in error_str:
                raise AgentExecutionError("AI service authentication failed. Please contact support.")
            elif "insufficient quota" in error_str or "quota" in error_str:
                raise AgentExecutionError("AI service quota exceeded. Please contact support.")
            elif "model not found" in error_str or "404" in error_str:
                raise AgentExecutionError("AI model not available. Please contact support.")
            elif "server error" in error_str or "500" in error_str or "502" in error_str or "503" in error_str:
                raise AgentExecutionError("AI service is temporarily unavailable. Please try again later.")
            else:
                raise AgentExecutionError(f"Agent execution failed: {str(e)}")

    @staticmethod
    async def stream_agent_response(
        user_id: int,
        messages: List[Dict[str, str]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream the AI agent response as it's generated.

        This method yields events in the following format:
        - {"type": "chunk", "content": str} - Text chunk
        - {"type": "tool_call", "name": str, "parameters": dict, "result": any, "success": bool}
        - {"type": "error", "message": str}

        Args:
            user_id: User ID for tool authorization
            messages: Conversation history as context

        Yields:
            Dict events for streaming response
        """
        provider = settings.ai_provider.lower()

        if provider == "gemini":
            async for event in ChatService._stream_gemini_agent(user_id, messages):
                yield event
        else:
            async for event in ChatService._stream_openai_agent(user_id, messages):
                yield event

    @staticmethod
    async def _stream_gemini_agent(
        user_id: int,
        messages: List[Dict[str, str]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream responses from Gemini agent."""
        from ..agents.gemini_agent import (
            GeminiAgent, get_gemini_tools, execute_tool, GEMINI_AGENT_INSTRUCTIONS
        )
        import google.generativeai as genai

        if not settings.gemini_api_key:
            yield {"type": "error", "message": "Gemini API key not configured"}
            return

        try:
            # Configure Gemini
            genai.configure(api_key=settings.gemini_api_key)

            # Create model with tools
            model = genai.GenerativeModel(
                model_name=settings.gemini_model,
                tools=get_gemini_tools(),
                system_instruction=GEMINI_AGENT_INSTRUCTIONS.replace("{{USER_ID}}", str(user_id)),
            )

            # Convert messages to Gemini format
            gemini_history = []
            for msg in messages[:-1]:
                role = "user" if msg["role"] == "user" else "model"
                gemini_history.append({"role": role, "parts": [msg["content"]]})

            # Start chat
            chat = model.start_chat(history=gemini_history)

            # Get the last user message
            last_message = messages[-1]["content"] if messages else ""

            max_turns = get_max_turns()
            turns = 0

            # Send message with streaming
            response = chat.send_message(last_message, stream=True)

            # Process streaming response
            accumulated_text = ""
            function_calls = []

            for chunk in response:
                # Check for text content
                if hasattr(chunk, 'text') and chunk.text:
                    accumulated_text += chunk.text
                    yield {"type": "chunk", "content": chunk.text}

                # Check for function calls in the chunk
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for part in chunk.candidates[0].content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_calls.append(part.function_call)

            # Handle function calls after streaming completes
            while function_calls and turns < max_turns:
                function_responses = []

                for fc in function_calls:
                    func_name = fc.name
                    func_args = dict(fc.args) if fc.args else {}

                    # Yield tool call event
                    yield {
                        "type": "tool_call",
                        "name": func_name,
                        "parameters": func_args,
                        "status": "executing"
                    }

                    # Execute the tool
                    result = execute_tool(func_name, func_args)

                    # Yield tool result
                    yield {
                        "type": "tool_call",
                        "name": func_name,
                        "parameters": func_args,
                        "result": result,
                        "success": "error" not in result
                    }

                    function_responses.append(
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=func_name,
                                response={"result": result}
                            )
                        )
                    )

                # Send function results back with streaming
                function_calls = []
                response = chat.send_message(function_responses, stream=True)

                for chunk in response:
                    if hasattr(chunk, 'text') and chunk.text:
                        yield {"type": "chunk", "content": chunk.text}

                    if hasattr(chunk, 'candidates') and chunk.candidates:
                        for part in chunk.candidates[0].content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                function_calls.append(part.function_call)

                turns += 1

        except Exception as e:
            error_str = str(e).lower()
            log_error(e, f"ChatService._stream_gemini_agent (user_id={user_id})", user_id)

            if "api key" in error_str or "invalid" in error_str:
                yield {"type": "error", "message": "Gemini API key is invalid"}
            elif "quota" in error_str or "rate" in error_str:
                yield {"type": "error", "message": "Gemini API quota exceeded"}
            else:
                yield {"type": "error", "message": f"Gemini agent error: {str(e)}"}

    @staticmethod
    async def _stream_openai_agent(
        user_id: int,
        messages: List[Dict[str, str]]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream responses from OpenAI agent."""
        if not settings.openai_api_key:
            yield {"type": "error", "message": "OpenAI API key not configured"}
            return

        os.environ["OPENAI_API_KEY"] = settings.openai_api_key

        try:
            # For OpenAI, we'll use the synchronous agent but simulate streaming
            # by yielding the full response in chunks
            # A proper implementation would use OpenAI's streaming API directly

            agent = create_todo_agent(user_id)
            runner = Runner(
                agent=agent,
                context={"user_id": user_id}
            )

            # Run synchronously in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: runner.run(messages=messages, max_turns=get_max_turns())
            )

            # Extract response
            response_text = result.final_output if hasattr(result, 'final_output') else str(result)

            # Extract tool calls
            if hasattr(result, 'tool_calls'):
                for tc in result.tool_calls:
                    yield {
                        "type": "tool_call",
                        "name": tc.name,
                        "parameters": tc.parameters if hasattr(tc, 'parameters') else {},
                        "result": tc.result if hasattr(tc, 'result') else None,
                        "success": not hasattr(tc, 'error') or tc.error is None
                    }

            # Simulate streaming by yielding chunks of the response
            # This provides a better UX than waiting for the full response
            chunk_size = 20  # Characters per chunk
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                yield {"type": "chunk", "content": chunk}
                await asyncio.sleep(0.02)  # Small delay for visual effect

        except Exception as e:
            error_str = str(e).lower()
            log_error(e, f"ChatService._stream_openai_agent (user_id={user_id})", user_id)

            if "rate limit" in error_str or "429" in error_str:
                yield {"type": "error", "message": "Rate limit exceeded"}
            elif "invalid api key" in error_str or "authentication" in error_str:
                yield {"type": "error", "message": "API authentication failed"}
            else:
                yield {"type": "error", "message": f"Agent error: {str(e)}"}

    @staticmethod
    def get_conversation_history(
        db: Session,
        conversation_id: int,
        user_id: int,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get formatted conversation history.

        Args:
            db: Database session
            conversation_id: Conversation to retrieve
            user_id: User ID for authorization
            limit: Maximum messages to return

        Returns:
            List of message dicts with role, content, and timestamp
        """
        messages = ConversationService.get_messages_by_conversation(
            db=db,
            conversation_id=conversation_id,
            user_id=user_id,
            limit=limit
        )

        return [
            {
                "id": msg.id,
                "role": msg.role.value,
                "content": msg.content,
                "created_at": msg.created_at.isoformat(),
                "token_count": msg.token_count
            }
            for msg in messages
        ]


__all__ = [
    "ChatService",
    "ChatServiceError",
    "AgentExecutionError",
]
