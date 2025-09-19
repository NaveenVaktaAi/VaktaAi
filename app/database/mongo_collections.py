from typing import Tuple, Dict, Any
from pymongo.database import Database
from bson import ObjectId


def get_collections(db: Database):
    documents = db["docSathi_ai_documents"]
    chunks = db["chunks"]

    # Ensure common indexes
    documents.create_index("user_id")
    documents.create_index("status")
    chunks.create_index("training_document_id")
    chunks.create_index("question_id")

    return documents, chunks


def create_document(db: Database, doc: Dict[str, Any]) -> str:
    documents, _ = get_collections(db)
    result = documents.insert_one(doc)
    return str(result.inserted_id)


def update_document_status(db: Database, document_id: str, fields: Dict[str, Any]) -> None:
    documents, _ = get_collections(db)
    documents.update_one({"_id": ObjectId(document_id)}, {"$set": fields})


def insert_chunk(db: Database, chunk_doc: Dict[str, Any]) -> str:
    _, chunks = get_collections(db)
    result = chunks.insert_one(chunk_doc)
    return str(result.inserted_id)


