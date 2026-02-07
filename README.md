# Todo App Backend API

A FastAPI backend for the Todo App with AI-powered chat functionality for task management.

## Features

- **User Authentication**: JWT-based authentication with Better Auth integration
- **Task Management**: Full CRUD operations for tasks with priorities, due dates, and categories
- **AI Chat Assistant**: Natural language task management via OpenAI Agents SDK with MCP tools

## Prerequisites

- Python 3.12+
- PostgreSQL (Neon Serverless recommended)
- OpenAI API key (for chat feature)

## Installation

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt

# Or with pyproject.toml
pip install -e .
```

### 2. Environment Variables

Create a `.env` file in the backend directory:

```env
# Database
DATABASE_URL=postgresql://user:password@host:5432/database

# Authentication
BETTER_AUTH_SECRET=your-secret-key

# AI Chat (required for chat feature)
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MODEL=gpt-4o-mini  # Optional, defaults to gpt-4o-mini
AGENT_MAX_TURNS=10         # Optional, defaults to 10

# Frontend
FRONTEND_ORIGIN=http://localhost:3000
```

### 3. Database Migration

Run the migration script to create/update tables:

```bash
python migrate.py
```

### 4. Start the Server

```bash
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/register` | POST | Register a new user |
| `/api/auth/login` | POST | Login and get JWT token |
| `/api/auth/me` | GET | Get current user info |

### Tasks

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/{user_id}/tasks` | GET | List user's tasks |
| `/api/{user_id}/tasks` | POST | Create a new task |
| `/api/{user_id}/tasks/{task_id}` | GET | Get a specific task |
| `/api/{user_id}/tasks/{task_id}` | PUT | Update a task |
| `/api/{user_id}/tasks/{task_id}` | DELETE | Delete a task |
| `/api/{user_id}/tasks/{task_id}/toggle` | PATCH | Toggle task completion |

### Chat (AI Assistant)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/{user_id}/conversations` | GET | List user's conversations |
| `/api/{user_id}/conversations` | POST | Create a new conversation |
| `/api/{user_id}/conversations/{id}` | GET | Get conversation details |
| `/api/{user_id}/conversations/{id}` | DELETE | Delete a conversation |
| `/api/{user_id}/conversations/{id}/messages` | GET | Get conversation messages |
| `/api/{user_id}/conversations/{id}/messages` | POST | Send a message to AI |

## AI Chat Feature

The AI chat assistant uses OpenAI Agents SDK with MCP (Model Context Protocol) tools to manage tasks through natural language.

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `create_task` | Create a new task |
| `list_tasks` | List tasks with filters |
| `get_task` | Get task details by ID |
| `complete_task` | Mark a task as done |
| `update_task` | Update task properties |
| `delete_task` | Delete a task |
| `search_tasks` | Search tasks by keyword |

### Usage Examples

```
User: "Create a task to buy groceries tomorrow"
AI: "I've created a task: Buy groceries (due: 2026-01-20)"

User: "Show my high priority tasks"
AI: "Here are your high priority tasks:
- Submit report (due: Jan 21)
- Call client (due: Jan 22)"

User: "Mark the report task as done"
AI: "Done! I've marked 'Submit report' as complete."

User: "Delete the groceries task"
AI: "Are you sure you want to delete 'Buy groceries'? Reply 'yes' to confirm."
User: "yes"
AI: "The task 'Buy groceries' has been deleted."
```

### Rate Limiting

The chat endpoint is rate limited to **20 requests per minute** per user to prevent abuse.

## Project Structure

```
backend/
├── src/
│   ├── api/
│   │   ├── main.py         # FastAPI app initialization
│   │   ├── deps.py         # Dependency injection
│   │   └── routes/
│   │       ├── auth.py     # Authentication endpoints
│   │       ├── tasks.py    # Task CRUD endpoints
│   │       └── chat.py     # Chat/conversation endpoints
│   ├── agents/
│   │   └── todo_agent.py   # AI agent configuration
│   ├── database/
│   │   └── database.py     # Database connection
│   ├── mcp/
│   │   ├── server.py       # MCP server setup
│   │   └── tools/          # MCP tool implementations
│   ├── models/
│   │   ├── user.py         # User model
│   │   ├── task.py         # Task model
│   │   ├── conversation.py # Conversation model
│   │   ├── message.py      # Message model
│   │   └── mcp_tool_call.py # Tool call audit log
│   ├── services/
│   │   ├── task_service.py         # Task business logic
│   │   ├── conversation_service.py # Conversation logic
│   │   └── chat_service.py         # AI chat orchestration
│   └── utils/
│       ├── auth.py         # Auth utilities
│       ├── errors.py       # Custom exceptions
│       └── logging.py      # Logging utilities
├── migrate.py              # Database migration script
├── main.py                 # Alternative entry point
└── pyproject.toml          # Project dependencies
```

## Development

### Running Tests

```bash
pytest
```

### API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## License

MIT
