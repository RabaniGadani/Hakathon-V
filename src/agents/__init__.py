"""
AI Agents module for the Todo App
Contains OpenAI and Gemini agent configurations for natural language task management
"""
from .todo_agent import (
    create_todo_agent,
    get_agent_model,
    get_max_turns,
    TODO_AGENT_INSTRUCTIONS,
)
from .gemini_agent import (
    create_gemini_agent,
    GeminiAgent,
    GEMINI_AGENT_INSTRUCTIONS,
)

__all__ = [
    # OpenAI Agent
    "create_todo_agent",
    "get_agent_model",
    "get_max_turns",
    "TODO_AGENT_INSTRUCTIONS",
    # Gemini Agent
    "create_gemini_agent",
    "GeminiAgent",
    "GEMINI_AGENT_INSTRUCTIONS",
]
