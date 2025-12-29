import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

def create_tables():
    """
    Creates the users, conversations, and messages tables if they don't already exist.
    """
    commands = (
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password TEXT, 
            email TEXT UNIQUE,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id SERIAL PRIMARY KEY,
            user1_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
            user2_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
            title TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS messages (
            message_id SERIAL PRIMARY KEY,
            conversation_id INTEGER REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            sender_id INTEGER REFERENCES users(user_id) ON DELETE SET NULL,
            content TEXT,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        for command in commands:
            cur.execute(command)
        cur.close()
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()

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

def save_chat(cursor, user1_id, user2_id, chat_messages):
    """
    Saves a chat between two users to the database.
    Finds an existing conversation or creates a new one.
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
    messages_to_insert = []
    for sender, content in chat_messages:
        sender_id = user1_id if sender == "user" else user2_id
        messages_to_insert.append((conversation_id, sender_id, content))

    cursor.executemany(
        "INSERT INTO messages (conversation_id, sender_id, content) VALUES (%s, %s, %s);",
        messages_to_insert
    )

def save_transcript(transcript, user_username="user", assistant_username="assistant"):
    """
    Parses a transcript and saves it to the database.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get or create the specified users
        user_id = get_or_create_user(cur, user_username)
        assistant_id = get_or_create_user(cur, assistant_username)

        chat = [
            (user_username, transcript["input"]),
            (assistant_username, transcript["output"])
        ]
        save_chat(cur, user_id, assistant_id, chat)
        conn.commit()
        cur.close()
        print(f"Transcript for {user_username} saved to database successfully!")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error saving transcript to database: {error}")
    finally:
        if conn is not None:
            conn.close()

def get_conversation_history(user1_username, user2_username):
    """
    Retrieves the chat history between two users from the database.
    """
    history = []
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 1. Get user IDs
        cur.execute("SELECT user_id, username FROM users WHERE username IN (%s, %s);", (user1_username, user2_username))
        rows = cur.fetchall()
        user_map = {row[1]: row[0] for row in rows}
        user1_id = user_map.get(user1_username)
        user2_id = user_map.get(user2_username)

        if not all([user1_id, user2_id]):
            return [] # No history if one of the users doesn't exist

        # 2. Find the conversation
        cur.execute(
            "SELECT conversation_id FROM conversations WHERE (user1_id = %s AND user2_id = %s) OR (user1_id = %s AND user2_id = %s);",
            (user1_id, user2_id, user2_id, user1_id)
        )
        conversation = cur.fetchone()
        if not conversation:
            return [] # No conversation found

        conversation_id = conversation[0]

        # 3. Retrieve messages
        cur.execute(
            "SELECT u.username, m.content FROM messages m JOIN users u ON m.sender_id = u.user_id WHERE m.conversation_id = %s ORDER BY m.created_at;",
            (conversation_id,)
        )
        history = cur.fetchall()
        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error retrieving conversation history: {error}")
    finally:
        if conn is not None:
            conn.close()

    return history