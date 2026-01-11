"""
Example: how to save chats using the project's `lib.database_utils`.

This script reads DB connection params from environment variables and
calls get_or_create_user + save_chat. It commits the transaction
so changes persist.

Note: The script does not create DB tables. Ensure your DB has the
expected tables: users, conversations, messages.
"""
import os
import psycopg2
from lib import database_utils as db


def get_db_params_from_env():
    return {
        "dbname": os.environ.get("DB_NAME", "conversations_db"),
        "user": os.environ.get("DB_USER", "postgres"),
        "password": os.environ.get("DB_PASS", "your_password"),
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": os.environ.get("DB_PORT", "5432"),
    }


def main():
    params = get_db_params_from_env()

    try:
        with psycopg2.connect(**params) as conn:
            with conn.cursor() as cur:
                # Create or get the two participants
                alice_id = db.get_or_create_user(cur, "alice")
                assistant_id = db.get_or_create_user(cur, "assistant")

                # Example chat between two identified parties.
                # The project helper expects chat_messages as an iterable of dicts.
                chat_messages = [
                    {"sender_id": alice_id, "content": "Hey, can you summarize the meeting notes?"},
                    {"sender_id": assistant_id, "content": "Sure — here are the highlights: ..."},
                    {"sender_id": alice_id, "content": "Thanks! Also, add action items."}
                ]

                # Save the chat. db.save_chat will create a conversation and insert messages.
                db.save_chat(cur, alice_id, assistant_id, chat_messages)

            # commit happens automatically when exiting the `with conn` block if no exception
        print("Chat saved (committed).")

    except Exception as e:
        print("Error saving chat:", e)


if __name__ == "__main__":
    main()
