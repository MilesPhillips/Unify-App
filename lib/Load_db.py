# reload previous chats from database
+from lib.database_utils import save_transcript
# read the chat from our users transcript file and save to database

def retrieve_chat(user):
    """
    Retrieves the chat messages between two users from the database.
    """
    conn = None
    chat_messages = []
    try:
        conn = sqlite3.connect('database.db')
        cur = conn.cursor()

        # Get user IDs
        cur.execute("SELECT user_id FROM users WHERE username = ?;", (user,))
        user = cur.fetchone()

        user_id = user[0]

        # Get conversation ID
        # get conversation between user and assistant
        
        cur.execute("SELECT user_id FROM users WHERE username = ?;", ('assistant',))
        assistant = cur.fetchone()
        conversation = cur.fetchone()
        if not conversation:
            return chat_messages  # No conversation exists between the users

        conversation_id = conversation[0]

        # Retrieve messages
        cur.execute("""
            SELECT sender_id, content, created_at FROM messages 
            WHERE conversation_id = ? ORDER BY created_at ASC;
        """, (conversation_id,))
        rows = cur.fetchall()
        for row in rows:
            sender_id, content, created_at = row
            sender_username = user_id if sender_id == user1_id else user2_username
            chat_messages.append((sender_username, content, created_at))

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()
    return chat_messages

def main():
    user = retrieve_chat("user")

if __name__ == "retrieve_chat":
    main()