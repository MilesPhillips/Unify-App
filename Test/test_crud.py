import unittest
import psycopg2

class TestUserCRUD(unittest.TestCase):
    def setUp(self):
        self.conn = psycopg2.connect(
            dbname="conversations_db",
            user="postgres",
            password="your_password",
            host="localhost",
            port="5432"
        )
        self.cur = self.conn.cursor()

    def tearDown(self):
        self.conn.rollback()
        self.cur.close()
        self.conn.close()

    def test_user_crud_operations(self):
        # Create a new user and get the ID
        self.cur.execute("INSERT INTO users (username, email) VALUES (%s, %s) RETURNING user_id;", ("testuser", "test@example.com"))
        user_id = self.cur.fetchone()[0]

        # Read the user and verify
        self.cur.execute("SELECT username, email FROM users WHERE user_id = %s;", (user_id,))
        user = self.cur.fetchone()
        self.assertEqual(user, ("testuser", "test@example.com"))

        # Update the user's email
        self.cur.execute("UPDATE users SET email = %s WHERE user_id = %s;", ("new.email@example.com", user_id))
        self.cur.execute("SELECT email FROM users WHERE user_id = %s;", (user_id,))
        updated_email = self.cur.fetchone()[0]
        self.assertEqual(updated_email, "new.email@example.com")

        # Delete the user
        self.cur.execute("DELETE FROM users WHERE user_id = %s;", (user_id,))
        self.cur.execute("SELECT * FROM users WHERE user_id = %s;", (user_id,))
        self.assertIsNone(self.cur.fetchone())

if __name__ == "__main__":
    unittest.main()
