import os
import psycopg2
from psycopg2.extras import execute_values


def get_connection_from_env():
        """Return a psycopg2 connection using DATABASE_URL or individual env vars.

        Prefer DATABASE_URL if present, otherwise read DB_NAME, DB_USER, DB_PASS, DB_HOST, DB_PORT.
        """
        database_url = os.environ.get('DATABASE_URL')
        if database_url:
                return psycopg2.connect(database_url)

        params = {
                'dbname': os.environ.get('DB_NAME', 'conversations_db'),
                'user': os.environ.get('DB_USER', 'postgres'),
                'password': os.environ.get('DB_PASS', 'your_password'),
                'host': os.environ.get('DB_HOST', 'localhost'),
                'port': os.environ.get('DB_PORT', '5432'),
        }
        return psycopg2.connect(**params)


def create_tables(conn):
        """Create required tables if they do not exist (users, conversations, messages)."""
        create_users = """
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT
        );
        """

        create_conversations = """
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id SERIAL PRIMARY KEY,
            user1_id INTEGER NOT NULL REFERENCES users(user_id),
            user2_id INTEGER NOT NULL REFERENCES users(user_id),
            started_at TIMESTAMPTZ DEFAULT now()
        );
        """

        create_messages = """
        CREATE TABLE IF NOT EXISTS messages (
            message_id SERIAL PRIMARY KEY,
            conversation_id INTEGER REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            sender_id INTEGER REFERENCES users(user_id),
            content TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        );
        """

        with conn.cursor() as cur:
                cur.execute(create_users)
                cur.execute(create_conversations)
                cur.execute(create_messages)
        conn.commit()



def get_or_create_user(cursor, username):
    """
    Gets the user_id for a username, creating the user if it doesn't exist.
    """
    cursor.execute("SELECT user_id FROM users WHERE username = %s;", (username,))
    user = cursor.fetchone()
    if user:
        return user[0]
    else:
        cursor.execute("INSERT INTO users (username) VALUES (%s) RETURNING user_id;", (username,))
        return cursor.fetchone()[0]


def create_user(cursor, username, password):
    """Create a new user with a plaintext password (no hashing).

    Returns the new user_id. Raises IntegrityError on duplicate username.
    """
    cursor.execute(
        "INSERT INTO users (username, password) VALUES (%s, %s) RETURNING user_id;",
        (username, password)
    )
    return cursor.fetchone()[0]


def get_user_by_username(cursor, username):
    """Return (user_id, username, password) or None if not found."""
    cursor.execute("SELECT user_id, username, password FROM users WHERE username = %s;", (username,))
    return cursor.fetchone()

def save_chat(cursor, user1_id, user2_id, chat_messages):
    """
    Saves a chat between two users to the database.
    Finds an existing conversation or creates a new one.
    `chat_messages` is a list of dicts, each with "sender_id" and "content" keys.
    """
    # 1. Find existing conversation or create a new one
    cursor.execute(
        "SELECT conversation_id FROM conversations WHERE (user1_id = %s AND user2_id = %s) OR (user1_id = %s AND user2_id = %s);",
        (user1_id, user2_id, user2_id, user1_id)
    )
    conversation = cursor.fetchone()
    if conversation:
        conversation_id = conversation[0]
    else:
        cursor.execute(
            "INSERT INTO conversations (user1_id, user2_id) VALUES (%s, %s) RETURNING conversation_id;",
            (user1_id, user2_id)
        )
        conversation_id = cursor.fetchone()[0]

    # 2. Prepare and save messages in a batch
    messages_to_insert = [
        (conversation_id, msg['sender_id'], msg['content'])
        for msg in chat_messages
    ]

    execute_values(
        cursor,
        "INSERT INTO messages (conversation_id, sender_id, content) VALUES %s;",
        messages_to_insert
    )

def save_llm_interaction(cursor, user_id, user_message, llm_response):
    """
    Saves a single interaction between a user and the LLM to the database.
    """
    assistant_id = get_or_create_user(cursor, "assistant")
    chat_messages = [
        {"sender_id": user_id, "content": user_message},
        {"sender_id": assistant_id, "content": llm_response},
    ]
    save_chat(cursor, user_id, assistant_id, chat_messages)

def get_conversation_history(cursor, user1_id, user2_id):
    """
    Retrieves the chat history between two users from the database.
    """
    # 1. Find the conversation
    cursor.execute(
        "SELECT conversation_id FROM conversations WHERE (user1_id = %s AND user2_id = %s) OR (user1_id = %s AND user2_id = %s);",
        (user1_id, user2_id, user2_id, user1_id)
    )
    conversation = cursor.fetchone()
    if not conversation:
        return [] # No conversation found

    conversation_id = conversation[0]

    # 2. Retrieve messages
    cursor.execute(
        "SELECT u.username, m.content FROM messages m JOIN users u ON m.sender_id = u.user_id WHERE m.conversation_id = %s ORDER BY m.created_at;",
        (conversation_id,)
    )
    return cursor.fetchall()
