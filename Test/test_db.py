import unittest
import psycopg2

class TestPostgresSetup(unittest.TestCase):
    def setUp(self):
        # Connect to the conversations_db database
        self.conn = psycopg2.connect(
            dbname="conversations_db",
            user="postgres",
            password="Red_Blue_Yellow42",
            host="localhost",
            port="5432"
        )
        self.cur = self.conn.cursor()

    def tearDown(self):
        self.cur.close()
        self.conn.close()

    def test_tables_exist(self):
        # Check that the tables exist
        for table in ["users", "conversations", "messages"]:
            self.cur.execute(f"""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table,))
            exists = self.cur.fetchone()[0]
            self.assertTrue(exists, f"Table '{table}' should exist.")

if __name__ == "__main__":
    unittest.main()