import psycopg2
from psycopg2 import sql
import os
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Connection to the default "postgres" database
conn = psycopg2.connect(
    dbname="postgres",
    user=DB_USER,
    password=DB_PASS,
    host=DB_HOST,
    port=DB_PORT
)
conn.autocommit = True
cur = conn.cursor()

# 1. Create a new database (optional, skip if already created)
# Check if database exists first
cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_NAME,))
if not cur.fetchone():
    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(DB_NAME)))
    print(f"Database {DB_NAME} created.")
else:
    print(f"Database {DB_NAME} already exists.")

cur.close()
conn.close()

# 2. Connect to the new database
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    host=DB_HOST,
    port=DB_PORT
)
cur = conn.cursor()


# 3. Create tables
schema = """
    CREATE TABLE users (
        user_id SERIAL PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE,
        created_at TIMESTAMPTZ DEFAULT now()
    );

    CREATE TABLE conversations (
        conversation_id SERIAL PRIMARY KEY,
        user1_id INT REFERENCES users(user_id) ON DELETE CASCADE,
        user2_id INT REFERENCES users(user_id) ON DELETE CASCADE,
        title TEXT,
        created_at TIMESTAMPTZ DEFAULT now()
    );

    CREATE TABLE messages (
        message_id SERIAL PRIMARY KEY,
        conversation_id INT REFERENCES conversations(conversation_id) ON DELETE CASCADE,
        sender_id INT REFERENCES users(user_id) ON DELETE SET NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMPTZ DEFAULT now()
    );"""


cur.execute(schema)
conn.commit()
cur.close()
conn.close()
print("✅ Database and tables created successfully!")