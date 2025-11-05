"""
AI Tutor Collections Module

This module provides MongoDB collection operations for AI Tutor conversations.
Unlike the chat system, AI Tutor conversations are standalone and don't have
document references.

Collections:
- conversations: AI Tutor conversation sessions
- conversation_messages: Messages within conversations

Usage:
    from app.database.ai_tutor_collections import (
        create_conversation,
        create_conversation_message,
        get_user_conversations,
        get_conversation_messages
    )
"""

from .ai_tutor_collections import (
    # Collection getter
    get_ai_tutor_collections,
    
    # Conversation CRUD
    create_conversation,
    get_conversation,
    get_user_conversations,
    get_user_conversations_count,
    update_conversation,
    delete_conversation,
    get_conversations_by_status,
    get_conversations_with_message_counts,
    search_conversations,
    get_recent_active_conversations,
    
    # Message CRUD
    get_conversation_messages,
    get_conversation_message_count,
    get_user_messages_count,
    bulk_delete_conversations,
    
    # Statistics
    get_conversation_stats
)

__all__ = [
    # Collection getter
    "get_ai_tutor_collections",
    
    # Conversation CRUD
    "create_conversation",
    "get_conversation",
    "get_user_conversations",
    "get_user_conversations_count",
    "update_conversation",
    "delete_conversation",
    "get_conversations_by_status",
    "get_conversations_with_message_counts",
    "search_conversations",
    "get_recent_active_conversations",
    
    # Message CRUD
    
    
    # Statistics
    "get_conversation_stats"
]
