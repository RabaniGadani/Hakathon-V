"""
Gemini Agent configuration for the Todo App
Uses Google's Generative AI SDK with function calling for task management
"""
import json
from typing import Optional, List, Dict, Any
import google.generativeai as genai

from ..config import settings
from ..mcp.server import register_tools
from ..database.database import get_session
from ..services.task_service import TaskService
from ..models.task import TaskCreate, TaskUpdate, Priority, Status
from ..utils.errors import TaskNotFoundException
from datetime import date


# Base agent instructions (same as OpenAI agent)
GEMINI_AGENT_INSTRUCTIONS = """You are a helpful task management assistant for a todo application.
Your role is to help users manage their tasks through natural language conversation.

## CRITICAL: User ID Requirement
When calling ANY function, you MUST include the user_id parameter that was provided to you.
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
1. FIRST, use get_task or list_tasks to find and confirm which task(s) match
2. Tell the user which task you found and ask for confirmation
3. WAIT for the user's confirmation response
4. Only delete after explicit confirmation

## Response Format
- Be concise but friendly
- Use bullet points for lists of tasks
- Confirm successful actions clearly
- If an error occurs, explain what happened and suggest alternatives
"""


# Define tools/functions for Gemini
def get_gemini_tools():
    """Define the function declarations for Gemini function calling."""
    return [
        genai.protos.Tool(
            function_declarations=[
                genai.protos.FunctionDeclaration(
                    name="create_task",
                    description="Create a new todo task for the user",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "title": genai.protos.Schema(type=genai.protos.Type.STRING, description="Task title (1-200 chars)"),
                            "description": genai.protos.Schema(type=genai.protos.Type.STRING, description="Detailed task description"),
                            "priority": genai.protos.Schema(type=genai.protos.Type.STRING, description="Task priority: low, medium, or high"),
                            "due_date": genai.protos.Schema(type=genai.protos.Type.STRING, description="Due date in ISO format (YYYY-MM-DD)"),
                            "category": genai.protos.Schema(type=genai.protos.Type.STRING, description="Task category"),
                            "user_id": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="User ID (required)"),
                        },
                        required=["title", "user_id"],
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="list_tasks",
                    description="List all tasks for the user with optional filtering",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "status": genai.protos.Schema(type=genai.protos.Type.STRING, description="Filter by status: all, pending, completed"),
                            "priority": genai.protos.Schema(type=genai.protos.Type.STRING, description="Filter by priority: all, low, medium, high"),
                            "limit": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Maximum number of tasks to return"),
                            "user_id": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="User ID (required)"),
                        },
                        required=["user_id"],
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="get_task",
                    description="Get a specific task by ID",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "task_id": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Task ID to retrieve"),
                            "user_id": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="User ID (required)"),
                        },
                        required=["task_id", "user_id"],
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="complete_task",
                    description="Mark a task as completed",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "task_id": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Task ID to complete"),
                            "user_id": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="User ID (required)"),
                        },
                        required=["task_id", "user_id"],
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="update_task",
                    description="Update an existing task's attributes",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "task_id": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Task ID to update"),
                            "title": genai.protos.Schema(type=genai.protos.Type.STRING, description="New title"),
                            "description": genai.protos.Schema(type=genai.protos.Type.STRING, description="New description"),
                            "priority": genai.protos.Schema(type=genai.protos.Type.STRING, description="New priority: low, medium, high"),
                            "due_date": genai.protos.Schema(type=genai.protos.Type.STRING, description="New due date (YYYY-MM-DD) or 'null' to clear"),
                            "category": genai.protos.Schema(type=genai.protos.Type.STRING, description="New category"),
                            "status": genai.protos.Schema(type=genai.protos.Type.STRING, description="New status: todo, in_progress, done"),
                            "user_id": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="User ID (required)"),
                        },
                        required=["task_id", "user_id"],
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="delete_task",
                    description="Permanently delete a task",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "task_id": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Task ID to delete"),
                            "user_id": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="User ID (required)"),
                        },
                        required=["task_id", "user_id"],
                    ),
                ),
                genai.protos.FunctionDeclaration(
                    name="search_tasks",
                    description="Search tasks by keyword in title and description",
                    parameters=genai.protos.Schema(
                        type=genai.protos.Type.OBJECT,
                        properties={
                            "query": genai.protos.Schema(type=genai.protos.Type.STRING, description="Search term (minimum 2 characters)"),
                            "limit": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="Maximum results (1-50)"),
                            "user_id": genai.protos.Schema(type=genai.protos.Type.INTEGER, description="User ID (required)"),
                        },
                        required=["query", "user_id"],
                    ),
                ),
            ]
        )
    ]


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool/function and return the result."""
    session_gen = get_session()
    db = next(session_gen)

    try:
        if name == "create_task":
            return _create_task(db, args)
        elif name == "list_tasks":
            return _list_tasks(db, args)
        elif name == "get_task":
            return _get_task(db, args)
        elif name == "complete_task":
            return _complete_task(db, args)
        elif name == "update_task":
            return _update_task(db, args)
        elif name == "delete_task":
            return _delete_task(db, args)
        elif name == "search_tasks":
            return _search_tasks(db, args)
        else:
            return {"error": "unknown_function", "message": f"Unknown function: {name}"}
    except Exception as e:
        return {"error": "execution_error", "message": str(e)}
    finally:
        try:
            next(session_gen)
        except StopIteration:
            pass


def _create_task(db, args: Dict) -> Dict:
    """Create a new task."""
    try:
        priority_enum = Priority(args.get("priority", "medium").lower())
    except ValueError:
        priority_enum = Priority.MEDIUM

    parsed_due_date = None
    if args.get("due_date"):
        try:
            parsed_due_date = date.fromisoformat(args["due_date"])
        except ValueError:
            pass

    task_data = TaskCreate(
        title=args["title"],
        description=args.get("description"),
        priority=priority_enum,
        due_date=parsed_due_date,
        category=args.get("category"),
    )

    task = TaskService.create_task(db=db, task_data=task_data, user_id=args["user_id"])
    return {
        "id": task.id,
        "title": task.title,
        "description": task.description or "",
        "priority": task.priority.value,
        "status": task.status.value,
        "due_date": task.due_date.isoformat() if task.due_date else None,
        "completed": task.completed,
        "created_at": task.created_at.isoformat(),
    }


def _list_tasks(db, args: Dict) -> Dict:
    """List tasks with filtering."""
    status = args.get("status", "all")
    priority = args.get("priority", "all")
    limit = min(100, max(1, args.get("limit", 50)))

    tasks = TaskService.get_tasks_by_user(
        db=db,
        user_id=args["user_id"],
        status=status if status != "all" else None,
    )

    if priority != "all":
        tasks = [t for t in tasks if t.priority.value == priority]

    tasks = tasks[:limit]

    return {
        "tasks": [
            {
                "id": task.id,
                "title": task.title,
                "priority": task.priority.value,
                "status": task.status.value,
                "due_date": task.due_date.isoformat() if task.due_date else None,
                "completed": task.completed,
            }
            for task in tasks
        ],
        "total": len(tasks),
    }


def _get_task(db, args: Dict) -> Dict:
    """Get a specific task."""
    try:
        task = TaskService.get_task_by_id(db=db, task_id=args["task_id"], user_id=args["user_id"])
        return {
            "id": task.id,
            "title": task.title,
            "description": task.description or "",
            "priority": task.priority.value,
            "status": task.status.value,
            "due_date": task.due_date.isoformat() if task.due_date else None,
            "category": task.category,
            "completed": task.completed,
            "created_at": task.created_at.isoformat(),
        }
    except TaskNotFoundException:
        return {"error": "not_found", "message": f"Task with ID {args['task_id']} not found"}


def _complete_task(db, args: Dict) -> Dict:
    """Mark task as completed."""
    try:
        task = TaskService.get_task_by_id(db=db, task_id=args["task_id"], user_id=args["user_id"])
        already_completed = task.completed

        if not already_completed:
            task = TaskService.toggle_task_completion(db=db, task_id=args["task_id"], user_id=args["user_id"])

        return {
            "id": task.id,
            "title": task.title,
            "completed": True,
            "already_completed": already_completed,
        }
    except TaskNotFoundException:
        return {"error": "not_found", "message": f"Task with ID {args['task_id']} not found"}


def _update_task(db, args: Dict) -> Dict:
    """Update task attributes."""
    try:
        update_data = {}
        changes = []

        if "title" in args and args["title"]:
            update_data["title"] = args["title"]
            changes.append("title")

        if "description" in args:
            update_data["description"] = args["description"]
            changes.append("description")

        if "priority" in args and args["priority"]:
            try:
                update_data["priority"] = Priority(args["priority"].lower())
                changes.append("priority")
            except ValueError:
                pass

        if "due_date" in args:
            if args["due_date"] and args["due_date"].lower() == "null":
                update_data["due_date"] = None
                changes.append("due_date")
            elif args["due_date"]:
                try:
                    update_data["due_date"] = date.fromisoformat(args["due_date"])
                    changes.append("due_date")
                except ValueError:
                    pass

        if "category" in args:
            update_data["category"] = None if args["category"] and args["category"].lower() == "null" else args["category"]
            changes.append("category")

        if "status" in args and args["status"]:
            try:
                update_data["status"] = Status(args["status"].lower())
                changes.append("status")
            except ValueError:
                pass

        if not update_data:
            return {"error": "validation_error", "message": "No valid fields to update"}

        task_update = TaskUpdate(**update_data)
        task = TaskService.update_task(db=db, task_id=args["task_id"], task_data=task_update, user_id=args["user_id"])

        return {
            "id": task.id,
            "title": task.title,
            "updated": True,
            "changes": changes,
        }
    except TaskNotFoundException:
        return {"error": "not_found", "message": f"Task with ID {args['task_id']} not found"}


def _delete_task(db, args: Dict) -> Dict:
    """Delete a task."""
    try:
        task = TaskService.get_task_by_id(db=db, task_id=args["task_id"], user_id=args["user_id"])
        task_title = task.title

        TaskService.delete_task(db=db, task_id=args["task_id"], user_id=args["user_id"])

        return {
            "deleted": True,
            "task_id": args["task_id"],
            "title": task_title,
        }
    except TaskNotFoundException:
        return {"error": "not_found", "message": f"Task with ID {args['task_id']} not found"}


def _search_tasks(db, args: Dict) -> Dict:
    """Search tasks by keyword."""
    query = args.get("query", "")
    if len(query) < 2:
        return {"error": "validation_error", "message": "Query must be at least 2 characters"}

    limit = min(50, max(1, args.get("limit", 20)))
    tasks = TaskService.get_tasks_by_user(db=db, user_id=args["user_id"])

    query_lower = query.lower()
    matching_tasks = []

    for task in tasks:
        if query_lower in task.title.lower():
            matching_tasks.append({"id": task.id, "title": task.title, "match_field": "title"})
        elif task.description and query_lower in task.description.lower():
            matching_tasks.append({"id": task.id, "title": task.title, "match_field": "description"})

    return {
        "tasks": matching_tasks[:limit],
        "query": query,
        "total": len(matching_tasks[:limit]),
    }


class GeminiAgent:
    """Gemini-based agent for task management."""

    def __init__(self, user_id: int):
        self.user_id = user_id
        self.model_name = settings.gemini_model

        # Configure Gemini
        genai.configure(api_key=settings.gemini_api_key)

        # Create model with tools
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            tools=get_gemini_tools(),
            system_instruction=GEMINI_AGENT_INSTRUCTIONS.replace("{{USER_ID}}", str(user_id)),
        )

        self.chat = None

    def run(self, messages: List[Dict[str, str]], max_turns: int = 10) -> Dict[str, Any]:
        """
        Run the agent with conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_turns: Maximum function calling turns

        Returns:
            Dict with response text and tool calls
        """
        # Convert messages to Gemini format
        gemini_history = []
        for msg in messages[:-1]:  # All but the last message go to history
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        # Start or continue chat
        self.chat = self.model.start_chat(history=gemini_history)

        # Get the last user message
        last_message = messages[-1]["content"] if messages else ""

        tool_calls = []
        turns = 0

        try:
            # Send message and handle function calls
            response = self.chat.send_message(last_message)

            while turns < max_turns:
                # Check if there are function calls
                function_calls = []
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        function_calls.append(part.function_call)

                if not function_calls:
                    # No more function calls, we're done
                    break

                # Execute function calls
                function_responses = []
                for fc in function_calls:
                    func_name = fc.name
                    func_args = dict(fc.args) if fc.args else {}

                    # Execute the tool
                    result = execute_tool(func_name, func_args)

                    tool_calls.append({
                        "name": func_name,
                        "parameters": func_args,
                        "result": result,
                        "success": "error" not in result,
                    })

                    function_responses.append(
                        genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=func_name,
                                response={"result": result}
                            )
                        )
                    )

                # Send function results back
                response = self.chat.send_message(function_responses)
                turns += 1

            # Extract final text response
            response_text = ""
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    response_text += part.text

            return {
                "response": response_text,
                "tool_calls": tool_calls,
                "token_count": None,  # Gemini doesn't easily expose token counts
            }

        except Exception as e:
            return {
                "response": f"I encountered an error processing your request: {str(e)}",
                "tool_calls": tool_calls,
                "token_count": None,
            }


def create_gemini_agent(user_id: int) -> GeminiAgent:
    """Create a Gemini agent for the specified user."""
    return GeminiAgent(user_id)


__all__ = [
    "GeminiAgent",
    "create_gemini_agent",
    "GEMINI_AGENT_INSTRUCTIONS",
]
