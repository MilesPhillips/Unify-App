import psycopg2
from lib.database_utils import get_connection_from_env, get_or_create_user, get_conversation_history

def retrieve_chat(username):
    """
    Retrieves the chat messages between a user and the assistant from the database.
    """
    conn = None
    try:
        conn = get_connection_from_env()
        with conn.cursor() as cur:
            user_id = get_or_create_user(cur, username)
            assistant_id = get_or_create_user(cur, "assistant")
            history = get_conversation_history(cur, user_id, assistant_id)
            for sender, message in history:
                print(f"{sender}: {message}")
            return history
    except psycopg2.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    retrieve_chat("user")