
import sys
sys.path.append('./lib')
from database_utils import create_tables

if __name__ == "__main__":
    create_tables()
    print("Database tables created/updated successfully.")
