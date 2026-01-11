import unittest
import uuid
import importlib.util
import os

# Import the database_utils module directly by file path to avoid importing
# the `lib` package (which can pull heavy optional deps like transformers).
spec = importlib.util.spec_from_file_location(
    "database_utils",
    os.path.join(os.path.dirname(__file__), '..', 'lib', 'database_utils.py')
)
db_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(db_utils)

class TestCreateUser(unittest.TestCase):
    def setUp(self):
        # Use connection helper from database_utils which reads env vars or defaults
        self.conn = db_utils.get_connection_from_env()
        # Ensure tables exist (idempotent)
        db_utils.create_tables(self.conn)
        self.cur = self.conn.cursor()

    def tearDown(self):
        # Clean up any test users we created during tests by pattern
        try:
            # Remove users with username starting with our test prefix
            self.cur.execute("DELETE FROM users WHERE username LIKE %s;", ("test_user_%",))
            self.conn.commit()
        finally:
            self.cur.close()
            self.conn.close()

    def test_create_and_get_user_with_password(self):
        username = f"test_user_{uuid.uuid4().hex[:8]}"
        password = "plain_text_test_password"

        # Create user using the helper that stores password
        user_id = db_utils.create_user(self.cur, username, password)
        # commit so data is visible
        self.conn.commit()

        # Retrieve the user and verify stored values
        row = db_utils.get_user_by_username(self.cur, username)
        self.assertIsNotNone(row, "User should exist in DB after creation")
        fetched_id, fetched_username, fetched_password = row
        self.assertEqual(fetched_username, username)
        self.assertEqual(fetched_password, password)
        self.assertEqual(fetched_id, user_id)

    def test_get_or_create_user_without_password(self):
        username = f"test_user_{uuid.uuid4().hex[:8]}"
        # Use get_or_create_user which creates username-only record
        user_id = db_utils.get_or_create_user(self.cur, username)
        self.conn.commit()

        row = db_utils.get_user_by_username(self.cur, username)
        self.assertIsNotNone(row)
        fetched_id, fetched_username, fetched_password = row
        self.assertEqual(fetched_username, username)
        # Password should be NULL/None for username-only creation
        self.assertTrue(fetched_password is None or fetched_password == "")
        self.assertEqual(fetched_id, user_id)


if __name__ == '__main__':
    unittest.main()
