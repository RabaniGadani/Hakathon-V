"""
Database migration script for the Todo App
Run this script from the backend directory: python migrate.py
"""
import os
import psycopg2

# Read DATABASE_URL from .env file manually
DATABASE_URL = None
env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            if line.startswith('DATABASE_URL='):
                DATABASE_URL = line.strip().split('=', 1)[1]
                break

if not DATABASE_URL:
    print("ERROR: DATABASE_URL not found in environment variables")
    exit(1)

print(f"Connecting to database...")

try:
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cursor = conn.cursor()

    # Check existing tables
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
    """)
    existing_tables = [row[0] for row in cursor.fetchall()]
    print(f"Existing tables: {existing_tables}")

    # ========== User table ==========
    if 'user' not in existing_tables:
        print("\nCreating user table...")
        cursor.execute("""
            CREATE TABLE "user" (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                name VARCHAR(255),
                hashed_password VARCHAR NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
            )
        """)
        print("  [OK] Created user table")
    else:
        print("\n  - User table already exists, skipping")

    # ========== Task table migrations ==========
    if 'task' in existing_tables:
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'task'
        """)
        task_columns = [row[0] for row in cursor.fetchall()]
        print(f"\nTask table columns: {task_columns}")

        task_migrations = [
            ("priority", "ALTER TABLE task ADD COLUMN priority VARCHAR(10) DEFAULT 'medium' NOT NULL"),
            ("status", "ALTER TABLE task ADD COLUMN status VARCHAR(20) DEFAULT 'todo' NOT NULL"),
            ("due_date", "ALTER TABLE task ADD COLUMN due_date DATE"),
            ("category", "ALTER TABLE task ADD COLUMN category VARCHAR(100)"),
        ]

        for column_name, sql in task_migrations:
            if column_name not in task_columns:
                print(f"Adding column to task: {column_name}")
                try:
                    cursor.execute(sql)
                    print(f"  [OK] Added {column_name}")
                except Exception as e:
                    print(f"  [ERROR] Error adding {column_name}: {e}")
            else:
                print(f"  - Column {column_name} already exists, skipping")

    # ========== Conversation table ==========
    if 'conversation' not in existing_tables:
        print("\nCreating conversation table...")
        cursor.execute("""
            CREATE TABLE conversation (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES "user"(id),
                title VARCHAR(200),
                is_active BOOLEAN DEFAULT TRUE NOT NULL,
                context_data JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX idx_conversation_user_id ON conversation(user_id)")
        print("  [OK] Created conversation table")
    else:
        print("\n  - Conversation table already exists, checking for context_data column...")
        cursor.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'conversation'
        """)
        conversation_columns = [row[0] for row in cursor.fetchall()]
        # Rename metadata to context_data if old column exists
        if 'metadata' in conversation_columns and 'context_data' not in conversation_columns:
            print("  Renaming metadata column to context_data...")
            cursor.execute("ALTER TABLE conversation RENAME COLUMN metadata TO context_data")
            print("  [OK] Renamed metadata to context_data")
        elif 'context_data' not in conversation_columns:
            print("  Adding context_data column to conversation...")
            cursor.execute("ALTER TABLE conversation ADD COLUMN context_data JSONB DEFAULT '{}'")
            print("  [OK] Added context_data column")
        else:
            print("  - context_data column already exists")

    # ========== Message table ==========
    if 'message' not in existing_tables:
        print("\nCreating message table...")
        cursor.execute("""
            CREATE TABLE message (
                id SERIAL PRIMARY KEY,
                conversation_id INTEGER NOT NULL REFERENCES conversation(id) ON DELETE CASCADE,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX idx_message_conversation_id ON message(conversation_id)")
        print("  [OK] Created message table")
    else:
        print("\n  - Message table already exists, skipping")

    # ========== MCP Tool Call table ==========
    if 'mcp_tool_call' not in existing_tables:
        print("\nCreating mcp_tool_call table...")
        cursor.execute("""
            CREATE TABLE mcp_tool_call (
                id SERIAL PRIMARY KEY,
                message_id INTEGER NOT NULL REFERENCES message(id) ON DELETE CASCADE,
                tool_name VARCHAR(100) NOT NULL,
                parameters JSONB DEFAULT '{}',
                result JSONB,
                status VARCHAR(20) DEFAULT 'pending' NOT NULL,
                error_message TEXT,
                execution_time_ms INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX idx_mcp_tool_call_message_id ON mcp_tool_call(message_id)")
        print("  [OK] Created mcp_tool_call table")
    else:
        print("\n  - MCP tool call table already exists, skipping")

    # Verify final structure
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    final_tables = [row[0] for row in cursor.fetchall()]
    print(f"\nFinal tables: {final_tables}")

    cursor.close()
    conn.close()
    print("\nMigration completed!")

except Exception as e:
    print(f"ERROR: {e}")
