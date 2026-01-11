import os
import uuid
import importlib.util

import pytest


MODULE_PATH = os.path.join(
    os.path.dirname(__file__),
    '..',
    'lib',
    'database_utils.py'
)

spec = importlib.util.spec_from_file_location("database_utils", MODULE_PATH)
database_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(database_utils)


@pytest.fixture
def postgres_connection():
    conn = database_utils.get_connection_from_env()
    database_utils.create_tables(conn)
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


def test_save_chat_and_read_history(postgres_connection):
    cursor = postgres_connection.cursor()
    try:
        username_a = f"test_user_{uuid.uuid4().hex[:8]}"
        username_b = f"test_user_{uuid.uuid4().hex[:8]}"

        user_a_id = database_utils.get_or_create_user(cursor, username_a)
        user_b_id = database_utils.get_or_create_user(cursor, username_b)

        chat_messages = [
            {"sender_id": user_a_id, "content": "hi there"},
            {"sender_id": user_b_id, "content": "hello"},
        ]

        database_utils.save_chat(cursor, user_a_id, user_b_id, chat_messages)
        history = database_utils.get_conversation_history(cursor, user_a_id, user_b_id)

        assert len(history) == 2
        assert history[0] == (username_a, "hi there")
        assert history[1] == (username_b, "hello")
    finally:
        cursor.close()
