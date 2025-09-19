import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Get PostgreSQL database connection"""
    try:
        conn = psycopg2.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            database=os.getenv('DB_NAME', 'ai_chatbot'),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', 'password'),
            port=os.getenv('DB_PORT', '5432')
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise

def setup_database():
    """Setup database tables"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Read and execute SQL file
        with open('scripts/01_create_training_docs_table.sql', 'r') as f:
            sql_commands = f.read()
        
        cursor.execute(sql_commands)
        conn.commit()
        
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    setup_database()
