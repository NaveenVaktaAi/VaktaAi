import os
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'ai_chatbot'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password'),
            'port': os.getenv('DB_PORT', '5432')
        }
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def insert_training_doc(self, user_id: int, doc_name: str, original_filename: str, 
                           file_type: str, file_size: int, s3_url: str) -> int:
        """Insert new training document record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_doc (user_id, doc_name, original_filename, file_type, file_size, s3_url)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (user_id, doc_name, original_filename, file_type, file_size, s3_url))
            
            doc_id = cursor.fetchone()[0]
            conn.commit()
            return doc_id
    
    def update_processing_status(self, doc_id: int, status: str, chunk_count: int = None):
        """Update document processing status"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if chunk_count is not None:
                cursor.execute("""
                    UPDATE training_doc 
                    SET processing_status = %s, chunk_count = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (status, chunk_count, doc_id))
            else:
                cursor.execute("""
                    UPDATE training_doc 
                    SET processing_status = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (status, doc_id))
            conn.commit()
    
    def get_user_documents(self, user_id: int):
        """Get all documents for a user"""
        with self.get_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT * FROM training_doc 
                WHERE user_id = %s 
                ORDER BY upload_date DESC
            """, (user_id,))
            return cursor.fetchall()
    
    def insert_document_chunk(self, doc_id: int, chunk_index: int, chunk_text: str, 
                             chunk_metadata: dict, vector_id: str = None):
        """Insert document chunk"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO document_chunks (doc_id, chunk_index, chunk_text, chunk_metadata, vector_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """, (doc_id, chunk_index, chunk_text, chunk_metadata, vector_id))
            
            chunk_id = cursor.fetchone()[0]
            conn.commit()
            return chunk_id

# Global database manager instance
db_manager = DatabaseManager()
