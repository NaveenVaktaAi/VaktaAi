from pymongo import MongoClient
from app.config.settings import settings
import os


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "vakta_ai")

_mongo_client: MongoClient | None = None


def _get_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(MONGO_URI)
    return _mongo_client


def get_db():
    client = _get_client()
    db = client[MONGO_DB_NAME]
    try:
        yield db
    finally:
        pass


