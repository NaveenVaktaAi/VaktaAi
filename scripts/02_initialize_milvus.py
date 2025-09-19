import logging
from app.services.vector_store import vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_milvus():
    """Initialize Milvus vector database"""
    try:
        logger.info("Initializing Milvus vector database...")
        vector_store.initialize()
        logger.info("Milvus initialization completed successfully")
        
        # Get stats
        stats = vector_store.get_stats()
        logger.info(f"Vector store stats: {stats}")
        
    except Exception as e:
        logger.error(f"Milvus initialization failed: {e}")
        raise

if __name__ == "__main__":
    initialize_milvus()
