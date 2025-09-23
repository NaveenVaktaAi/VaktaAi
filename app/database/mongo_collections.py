from typing import Tuple, Dict, Any
from pymongo.database import Database
from bson import ObjectId
from datetime import datetime


def get_collections(db: Database):
    documents = db["docSathi_ai_documents"]
    chunks = db["chunks"]
    chats = db["chats"]
    chat_messages = db["chat_messages"]

    # Ensure common indexes
    documents.create_index("user_id")
    documents.create_index("status")
    chunks.create_index("training_document_id")
    chunks.create_index("question_id")
    
    # Chat collection indexes
    chats.create_index("user_id")
    chats.create_index("document_id")
    chats.create_index("status")
    chats.create_index("created_at")
    
    # Chat messages collection indexes
    chat_messages.create_index("chat_id")
    chat_messages.create_index("is_bot")
    chat_messages.create_index("created_ts")

    return documents, chunks, chats, chat_messages


def create_document(db: Database, doc: Dict[str, Any]) -> str:
    documents, _, _, _ = get_collections(db)
    result = documents.insert_one(doc)
    return str(result.inserted_id)


def update_document_status(db: Database, document_id: str, fields: Dict[str, Any]) -> None:
    documents, _, _, _ = get_collections(db)
    documents.update_one({"_id": ObjectId(document_id)}, {"$set": fields})


def insert_chunk(db: Database, chunk_doc: Dict[str, Any]) -> str:
    _, chunks, _, _ = get_collections(db)
    result = chunks.insert_one(chunk_doc)
    return str(result.inserted_id)


# Chat CRUD operations
def create_chat(db: Database, chat_doc: Dict[str, Any]) -> str:
    """Create a new chat"""
    _, _, chats, _ = get_collections(db)
    result = chats.insert_one(chat_doc)
    return str(result.inserted_id)


def get_chat(db: Database, chat_id: str) -> Dict[str, Any]:
    """Get a chat by ID"""
    _, _, chats, _ = get_collections(db)
    return chats.find_one({"_id": ObjectId(chat_id)})


def get_user_chats(db: Database, user_id: int) -> list:
    """Get all chats for a user"""
    _, _, chats, _ = get_collections(db)
    return list(chats.find({"user_id": user_id}).sort("created_at", -1))


def update_chat(db: Database, chat_id: str, fields: Dict[str, Any]) -> None:
    """Update a chat"""
    _, _, chats, _ = get_collections(db)
    chats.update_one({"_id": ObjectId(chat_id)}, {"$set": fields})


def delete_chat(db: Database, chat_id: str) -> None:
    """Delete a chat and all its messages"""
    _, _, chats, chat_messages = get_collections(db)
    
    # Delete all messages for this chat
    chat_messages.delete_many({"chat_id": ObjectId(chat_id)})
    
    # Delete the chat
    chats.delete_one({"_id": ObjectId(chat_id)})


# Chat Messages CRUD operations
def create_chat_message(db: Database, message_doc: Dict[str, Any]) -> str:
    """Create a new chat message"""
    _, _, _, chat_messages = get_collections(db)
    result = chat_messages.insert_one(message_doc)
    return str(result.inserted_id)


def get_chat_messages(db: Database, chat_id: str, limit: int = 50, skip: int = 0) -> list:
    """Get messages for a chat with pagination"""
    _, _, _, chat_messages = get_collections(db)
    return list(chat_messages.find({"chat_id": ObjectId(chat_id)})
                .sort("created_ts", 1)
                .skip(skip)
                .limit(limit))


def get_chat_message(db: Database, message_id: str) -> Dict[str, Any]:
    """Get a specific chat message"""
    _, _, _, chat_messages = get_collections(db)
    return chat_messages.find_one({"_id": ObjectId(message_id)})


def update_chat_message(db: Database, message_id: str, fields: Dict[str, Any]) -> None:
    """Update a chat message"""
    _, _, _, chat_messages = get_collections(db)
    chat_messages.update_one({"_id": ObjectId(message_id)}, {"$set": fields})


def delete_chat_message(db: Database, message_id: str) -> None:
    """Delete a chat message"""
    _, _, _, chat_messages = get_collections(db)
    chat_messages.delete_one({"_id": ObjectId(message_id)})


def get_chat_message_count(db: Database, chat_id: str) -> int:
    """Get total message count for a chat"""
    _, _, _, chat_messages = get_collections(db)
    return chat_messages.count_documents({"chat_id": ObjectId(chat_id)})


