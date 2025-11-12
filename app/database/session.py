from pymongo import MongoClient
from app.config.settings import settings
import os


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "vakta_ai")

_mongo_client: MongoClient | None = None


def _get_client() -> MongoClient:
    """Get MongoDB client with optimized connection pool settings"""
    global _mongo_client
    if _mongo_client is None:
        # OPTIMIZATION: Configure connection pool for better concurrency
        # maxPoolSize: Maximum number of connections in pool (default: 100)
        # minPoolSize: Minimum number of connections in pool (default: 0)
        # maxIdleTimeMS: Close idle connections after 30 seconds
        # serverSelectionTimeoutMS: Fail fast if can't connect (5 seconds)
        _mongo_client = MongoClient(
            MONGO_URI,
            maxPoolSize=100,  # Allow up to 100 concurrent connections
            minPoolSize=10,   # Keep 10 connections ready
            maxIdleTimeMS=30000,  # Close idle connections after 30s
            serverSelectionTimeoutMS=5000,  # Fail fast if MongoDB unavailable
            connectTimeoutMS=5000,  # Connection timeout
            socketTimeoutMS=10000,  # Socket timeout for queries
        )
    return _mongo_client


def get_db():
    """
    Get database instance - non-blocking, uses connection pool
    FastAPI dependency automatically manages connection lifecycle
    """
    client = _get_client()
    db = client[MONGO_DB_NAME]
    try:
        yield db
    finally:
        # Connection is returned to pool automatically
        # No need to close - connection pooling handles it
        pass


