import psycopg2
from psycopg2.extras import execute_values

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
    """
    # 1. Create a conversation
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

    execute_values(
        cursor,
        "INSERT INTO messages (conversation_id, sender_id, content) VALUES %s;",
        messages_to_insert
    )

def save_transcript(transcript):
    """
    Parses a transcript and saves it to the database.
    """
    conn_params = {
        "dbname": "conversations_db",
        "user": "postgres",
        "password": "your_password",
        "host": "localhost",
        "port": "5432"
    }
    try:
        with psycopg2.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                # Assuming "user" for the input and "assistant" for the output
                user_id = get_or_create_user(cur, "user")
                assistant_id = get_or_create_user(cur, "assistant")

                chat = [
                    ("user", transcript["input"]),
                    ("assistant", transcript["output"])
                ]
                save_chat(cur, user_id, assistant_id, chat)
                print("Transcript saved to database successfully!")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error saving transcript to database: {error}")
