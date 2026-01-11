
import sys
import sys
from lib.database_utils import create_tables, get_connection_from_env

if __name__ == "__main__":
    conn = None
    try:
        conn = get_connection_from_env()
        create_tables(conn)
        print("Database tables created/updated successfully.")
    except Exception as e:
        print(f"Error creating database tables: {e}")
    finally:
        if conn:
            conn.close()
