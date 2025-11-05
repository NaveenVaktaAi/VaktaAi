from typing import Dict, Any, Optional, List
from pymongo.database import Database
from bson import ObjectId
from datetime import datetime


def get_ai_tutor_collections(db: Database):
    """Get AI Tutor collections with proper indexing"""
    conversations = db["conversations"]
    
    # Conversations collection indexes
    conversations.create_index("user_id")
    conversations.create_index("status")
    conversations.create_index("created_at")
    conversations.create_index("updated_at")
    conversations.create_index([("user_id", 1), ("created_at", -1)])
    
    return conversations


# ===== CONVERSATION CRUD OPERATIONS =====

def create_conversation(db: Database, conversation_doc: Dict[str, Any]) -> str:
    """Create a new conversation"""
    conversations = get_ai_tutor_collections(db)
    # Initialize messages as empty list
    if "messages" not in conversation_doc:
        conversation_doc["messages"] = []
    result = conversations.insert_one(conversation_doc)
    return str(result.inserted_id)


def get_conversation(db: Database, conversation_id: str) -> Optional[Dict[str, Any]]:
    """Get a conversation by ID"""
    conversations = get_ai_tutor_collections(db)
    return conversations.find_one({"_id": ObjectId(conversation_id)})


def get_user_conversations(db: Database, user_id: int, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
    """Get all conversations for a user with pagination"""
    conversations = get_ai_tutor_collections(db)
    return list(conversations.find({"user_id": user_id})
                .sort("created_at", -1)
                .skip(skip)
                .limit(limit))


def get_user_conversations_count(db: Database, user_id: int) -> int:
    """Get total conversation count for a user"""
    conversations = get_ai_tutor_collections(db)
    return conversations.count_documents({"user_id": user_id})


def update_conversation(db: Database, conversation_id: str, fields: Dict[str, Any]) -> None:
    """Update a conversation"""
    conversations = get_ai_tutor_collections(db)
    fields["updated_at"] = datetime.utcnow()
    conversations.update_one({"_id": ObjectId(conversation_id)}, {"$set": fields})


def delete_conversation(db: Database, conversation_id: str) -> None:
    """Delete a conversation (messages are stored within conversation)"""
    conversations = get_ai_tutor_collections(db)
    conversations.delete_one({"_id": ObjectId(conversation_id)})


def get_conversations_by_status(db: Database, user_id: int, status: str, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
    """Get conversations by status"""
    conversations = get_ai_tutor_collections(db)
    return list(conversations.find({"user_id": user_id, "status": status})
                .sort("created_at", -1)
                .skip(skip)
                .limit(limit))


# ===== MESSAGE OPERATIONS (Now stored in conversation document) =====

def add_message_to_conversation(db: Database, conversation_id: str, message: Dict[str, Any]) -> None:
    """Add a message to conversation's messages array"""
    conversations = get_ai_tutor_collections(db)
    conversations.update_one(
        {"_id": ObjectId(conversation_id)},
        {
            "$push": {"messages": message},
            "$set": {"updated_at": datetime.utcnow()}
        }
    )


def get_conversation_messages(db: Database, conversation_id: str) -> List[Dict[str, Any]]:
    """Get all messages for a conversation"""
    conversations = get_ai_tutor_collections(db)
    conversation = conversations.find_one({"_id": ObjectId(conversation_id)})
    if conversation:
        return conversation.get("messages", [])
    return []


def save_messages_to_conversation(db: Database, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
    """Save/replace all messages for a conversation at once (for langgraph thread)"""
    conversations = get_ai_tutor_collections(db)
    conversations.update_one(
        {"_id": ObjectId(conversation_id)},
        {
            "$set": {
                "messages": messages,
                "updated_at": datetime.utcnow()
            }
        }
    )


def get_conversation_message_count(db: Database, conversation_id: str) -> int:
    """Get total message count for a conversation"""
    conversations = get_ai_tutor_collections(db)
    conversation = conversations.find_one({"_id": ObjectId(conversation_id)})
    if conversation:
        return len(conversation.get("messages", []))
    return 0


def get_conversations_with_message_counts(db: Database, user_id: int, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
    """Get conversations with message counts"""
    conversations = get_ai_tutor_collections(db)
    
    # Get conversations
    user_conversations = list(conversations.find({"user_id": user_id})
                              .sort("created_at", -1)
                              .skip(skip)
                              .limit(limit))
    
    # Add message count to each conversation
    for conversation in user_conversations:
        conversation["message_count"] = len(conversation.get("messages", []))
    
    return user_conversations


def get_user_messages_count(db: Database, user_id: int) -> int:
    """Get total message count for a user across all conversations"""
    conversations = get_ai_tutor_collections(db)
    
    # Get all conversations for the user
    user_conversations = list(conversations.find({"user_id": user_id}, {"messages": 1}))
    
    # Count all messages
    total_count = sum(len(conv.get("messages", [])) for conv in user_conversations)
    return total_count


def search_conversations(db: Database, user_id: int, search_query: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Search conversations by title"""
    conversations = get_ai_tutor_collections(db)
    return list(conversations.find({
        "user_id": user_id,
        "title": {"$regex": search_query, "$options": "i"}
    }).sort("created_at", -1).limit(limit))


def get_recent_active_conversations(db: Database, user_id: int, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
    """Get recently active conversations"""
    from datetime import timedelta
    conversations = get_ai_tutor_collections(db)
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    return list(conversations.find({
        "user_id": user_id,
        "updated_at": {"$gte": cutoff_date}
    }).sort("updated_at", -1).limit(limit))


# ===== BULK OPERATIONS =====

def bulk_delete_conversations(db: Database, conversation_ids: List[str]) -> int:
    """Bulk delete conversations (messages are stored within conversations)"""
    conversations = get_ai_tutor_collections(db)
    
    # Convert string IDs to ObjectId
    object_ids = [ObjectId(id) for id in conversation_ids]
    
    # Delete conversations (messages are part of conversation doc)
    result = conversations.delete_many({"_id": {"$in": object_ids}})
    return result.deleted_count


# ===== STATISTICS =====

def get_conversation_stats(db: Database, user_id: int) -> Dict[str, Any]:
    """Get conversation statistics for a user"""
    conversations = get_ai_tutor_collections(db)
    
    # Get all conversations for the user
    user_conversations = list(conversations.find({"user_id": user_id}, {"status": 1, "messages": 1}))
    
    # Count conversations by status
    active_count = sum(1 for conv in user_conversations if conv.get("status") == "active")
    archived_count = sum(1 for conv in user_conversations if conv.get("status") == "archived")
    
    # Count messages
    total_messages = 0
    bot_messages = 0
    user_messages = 0
    
    for conv in user_conversations:
        messages = conv.get("messages", [])
        total_messages += len(messages)
        bot_messages += sum(1 for msg in messages if msg.get("is_bot", False))
        user_messages += sum(1 for msg in messages if not msg.get("is_bot", False))
    
    return {
        "total_conversations": len(user_conversations),
        "active_conversations": active_count,
        "archived_conversations": archived_count,
        "total_messages": total_messages,
        "bot_messages": bot_messages,
        "user_messages": user_messages
    }

