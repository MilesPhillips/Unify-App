import psycopg2
from psycopg2 import sql

# Connection to the default "postgres" database
conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    password="your_password",
    host="localhost",
    port="5432"
)
conn.autocommit = True
cur = conn.cursor()

# 1. Create a new database (optional, skip if already created)
cur.execute("CREATE DATABASE conversations_db;")
cur.close()
conn.close()

# 2. Connect to the new database
conn = psycopg2.connect(
    dbname="conversations_db",
    user="postgres",
    password="your_password",
    host="localhost",
    port="5432"
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