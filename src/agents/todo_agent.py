"""
Todo Agent configuration for the Todo App
Uses OpenAI Agents SDK with MCP tools for natural language task management
"""
from typing import Optional
from agents import Agent, ModelSettings
from agents.mcp import MCPServerStdio

from ..config import settings
from ..mcp.server import mcp, register_tools


# Base agent instructions for task management (user_id will be appended)
TODO_AGENT_INSTRUCTIONS_BASE = """You are a helpful task management assistant for a todo application.
Your role is to help users manage their tasks through natural language conversation.

## CRITICAL: User ID Requirement
When calling ANY tool, you MUST include the user_id parameter that was provided to you.
This is required for security and to ensure tasks are associated with the correct user.
The user_id for this session is: {{USER_ID}}

## Capabilities
You can help users:
- Create new tasks with titles, descriptions, priorities, due dates, and categories
- List and view their existing tasks with various filters
- Mark tasks as completed
- Update task details (title, description, priority, due date, category, status)
- Delete tasks they no longer need (with confirmation)
- Search for tasks by keywords

## Guidelines
1. Always be helpful and confirm actions you've taken
2. When creating tasks, ask for clarification if the user's intent is unclear
3. When listing tasks, summarize the results in a user-friendly way
4. Provide helpful suggestions when appropriate (e.g., "Would you like me to set a due date?")
5. Handle errors gracefully and explain what went wrong in simple terms
6. Remember context from the conversation to provide better assistance

## Task Priorities
- low: For tasks that can wait
- medium: For regular tasks (default)
- high: For urgent or important tasks

## Task Statuses
- todo: Not started yet (default for new tasks)
- in_progress: Currently being worked on
- done: Completed

## Delete Confirmation Flow
When a user requests to delete a task:
1. FIRST, use the get_task or list_tasks tool to find and confirm which task(s) match
2. Tell the user which task you found and ask: "Are you sure you want to delete '[task title]'? This cannot be undone. Reply 'yes' to confirm or 'no' to cancel."
3. WAIT for the user's confirmation response
4. If they confirm with "yes", "confirm", "delete it", "go ahead", or similar affirmative:
   - Use the delete_task tool to permanently remove the task
   - Confirm the deletion was successful
5. If they say "no", "cancel", "nevermind", or similar negative:
   - Acknowledge and do NOT delete the task
   - Offer to help with something else

IMPORTANT: Never delete a task without explicit user confirmation. Always ask first.

## Response Format
- Be concise but friendly
- Use bullet points for lists of tasks
- Confirm successful actions clearly
- If an error occurs, explain what happened and suggest alternatives

## Context Awareness
- Remember what tasks were recently discussed
- When users say "it", "that task", or "the one I just mentioned", refer to the most recently discussed task
- Use task IDs when available to avoid ambiguity
"""


def create_todo_agent(user_id: int) -> Agent:
    """
    Create a todo agent configured for a specific user.

    The agent uses MCP tools to manage tasks, with user_id injected
    into the instructions for proper authorization.

    Args:
        user_id: The authenticated user's ID

    Returns:
        Configured Agent instance ready for conversation
    """
    # Register MCP tools (lazy loading to avoid circular imports)
    register_tools()

    # Inject user_id into instructions
    instructions = TODO_AGENT_INSTRUCTIONS_BASE.replace("{{USER_ID}}", str(user_id))

    # Create agent with MCP server
    agent = Agent(
        name="TodoAssistant",
        instructions=instructions,
        model=settings.openai_model,
        model_settings=ModelSettings(
            temperature=0.7,
            max_tokens=1024,
        ),
        mcp_servers=[mcp],
    )

    # Store user_id in agent context for reference
    agent.context = {"user_id": user_id}

    return agent


def get_agent_model() -> str:
    """Get the configured OpenAI model for the agent."""
    return settings.openai_model


def get_max_turns() -> int:
    """Get the maximum number of agent turns per request."""
    return settings.agent_max_turns


# Alias for backward compatibility
TODO_AGENT_INSTRUCTIONS = TODO_AGENT_INSTRUCTIONS_BASE

__all__ = [
    "create_todo_agent",
    "get_agent_model",
    "get_max_turns",
    "TODO_AGENT_INSTRUCTIONS_BASE",
    "TODO_AGENT_INSTRUCTIONS",  # Alias for backward compatibility
]
