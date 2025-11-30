import sqlite3

def create_tables():
    """
    Creates the users, user_profiles, conversations, and messages tables if they don't already exist.
    """
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    # Table for basic user identification
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE
        )
    ''')
    # New table for detailed user profiles, linked one-to-one with users
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id INTEGER PRIMARY KEY,
            full_name TEXT,
            email TEXT UNIQUE,
            bio TEXT,
            profile_picture_url TEXT,
            FOREIGN KEY (user_id) REFERENCES users (user_id)
        )
    ''')
    # Table for conversations between two users
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user1_id INTEGER,
            user2_id INTEGER,
            topics TEXT,
            FOREIGN KEY (user1_id) REFERENCES users (user_id),
            FOREIGN KEY (user2_id) REFERENCES users (user_id)
        )
    ''')
    # Table for individual messages within a conversation
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            sender_id INTEGER,
            content TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id),
            FOREIGN KEY (sender_id) REFERENCES users (user_id)
        )
    ''')
    conn.commit()
    conn.close()

def get_or_create_user(cursor, username):
    """
    Gets the user_id for a username, creating the user if it doesn't exist.
    """
    cursor.execute("SELECT user_id FROM users WHERE username = ?;", (username,))
    user = cursor.fetchone()
    if user:
        return user[0]
    else:
        cursor.execute("INSERT INTO users (username) VALUES (?);", (username,))
        return cursor.lastrowid

def save_chat(cursor, user1_id, user2_id, chat_messages):
    """
    Saves a chat between two users to the database.
    Finds an existing conversation or creates a new one.
    """
    # 1. Find existing conversation or create a new one
    cursor.execute(
        "SELECT conversation_id FROM conversations WHERE (user1_id = ? AND user2_id = ?) OR (user1_id = ? AND user2_id = ?);",
        (user1_id, user2_id, user2_id, user1_id)
    )
    conversation = cursor.fetchone()
    if conversation:
        conversation_id = conversation[0]
    else:
        cursor.execute(
            "INSERT INTO conversations (user1_id, user2_id) VALUES (?, ?);",
            (user1_id, user2_id)
        )
        conversation_id = cursor.lastrowid

    # 2. Prepare and save messages in a batch
    messages_to_insert = []
    for sender, content in chat_messages:
        sender_id = user1_id if sender == "user" else user2_id
        messages_to_insert.append((conversation_id, sender_id, content))

    cursor.executemany(
        "INSERT INTO messages (conversation_id, sender_id, content) VALUES (?, ?, ?);",
        messages_to_insert
    )

def save_transcript(transcript):
    """
    Parses a transcript and saves it to the database.
    """
    conn = None
    try:
        conn = sqlite3.connect('database.db')
        cur = conn.cursor()
        # Assuming "user" for the input and "assistant" for the output
        user_id = get_or_create_user(cur, "user")
        assistant_id = get_or_create_user(cur, "assistant")

        chat = [
            ("user", transcript["input"]),
            ("assistant", transcript["output"])
        ]
        save_chat(cur, user_id, assistant_id, chat)
        conn.commit()
        print("Transcript saved to database successfully!")

    except sqlite3.Error as error:
        print(f"Error saving transcript to database: {error}")
    finally:
        if conn:
            conn.close()

def get_conversation_history(user1_username, user2_username):
    """
    Retrieves the chat history between two users from the database.
    """
    history = []
    conn = None
    try:
        conn = sqlite3.connect('database.db')
        cur = conn.cursor()
        # 1. Get user IDs
        cur.execute("SELECT user_id, username FROM users WHERE username IN (?, ?);", (user1_username, user2_username))
        user_map = {username: user_id for username, user_id in cur.fetchall()}
        user1_id = user_map.get(user1_username)
        user2_id = user_map.get(user2_username)

        if not all([user1_id, user2_id]):
            return [] # No history if one of the users doesn't exist

        # 2. Find the conversation
        cur.execute(
            "SELECT conversation_id FROM conversations WHERE (user1_id = ? AND user2_id = ?) OR (user1_id = ? AND user2_id = ?);",
            (user1_id, user2_id, user2_id, user1_id)
        )
        conversation = cur.fetchone()
        if not conversation:
            return [] # No conversation found

        conversation_id = conversation[0]

        # 3. Retrieve messages
        cur.execute(
            "SELECT u.username, m.content FROM messages m JOIN users u ON m.sender_id = u.user_id WHERE m.conversation_id = ? ORDER BY m.created_at;",
            (conversation_id,)
        )
        history = cur.fetchall()

    except sqlite3.Error as error:
        print(f"Error retrieving conversation history: {error}")
    finally:
        if conn:
            conn.close()

    return history
