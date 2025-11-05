from typing import List, Dict, Any, Optional
from datetime import datetime
from bson import ObjectId
from pymongo.database import Database

from app.database.session import get_db
from app.database.ai_tutor_collections import (
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
    get_conversation_messages,
    save_messages_to_conversation,
    add_message_to_conversation,
    get_conversation_message_count,
    get_conversation_stats
)

from app.features.aiAvatar.schemas import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationWithMessages,
    ConversationListResponse,
    ConversationStatsResponse,
    Message
)


class AITutorRepository:
    """Repository for AI Tutor conversation operations"""
    
    def __init__(self):
        self.db = next(get_db())
    
    # ===== CONVERSATION OPERATIONS =====
    
    async def create_conversation(self, conversation_data: ConversationCreate, user_id: int) -> str:
        """Create a new conversation"""
        conversation_doc = {
            "user_id": user_id,
            "title": conversation_data.title or "",
            "status": conversation_data.status,
            "subject": conversation_data.subject,
            "tags": conversation_data.tags or [],
            "messages": conversation_data.messages or [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        return create_conversation(self.db, conversation_doc)
    
    async def get_conversation_by_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID"""
        return get_conversation(self.db, conversation_id)
    
    async def get_user_conversations(self, user_id: int, page: int = 1, limit: int = 20) -> ConversationListResponse:
        """Get all conversations for a user with pagination"""
        skip = (page - 1) * limit
        
        # Get total count
        total = get_user_conversations_count(self.db, user_id)
        
        # Get conversations
        conversations = get_user_conversations(self.db, user_id, limit, skip)
        
        # Convert to response format
        conversation_responses = []
        for conv in conversations:
            conversation_responses.append(ConversationResponse(
                _id=str(conv["_id"]),
                user_id=conv["user_id"],
                title=conv["title"],
                status=conv["status"],
                subject=conv.get("subject"),
                tags=conv.get("tags"),
                messages=conv.get("messages", []),
                created_at=conv["created_at"],
                updated_at=conv["updated_at"]
            ))
        
        return ConversationListResponse(
            conversations=conversation_responses,
            total=total,
            page=page,
            limit=limit
        )
    
    async def update_conversation(self, conversation_id: str, conversation_data: ConversationUpdate) -> bool:
        """Update a conversation"""
        update_fields = {}
        
        if conversation_data.title is not None:
            update_fields["title"] = conversation_data.title
        if conversation_data.status is not None:
            update_fields["status"] = conversation_data.status
        if conversation_data.subject is not None:
            update_fields["subject"] = conversation_data.subject
        if conversation_data.tags is not None:
            update_fields["tags"] = conversation_data.tags
        if conversation_data.messages is not None:
            update_fields["messages"] = conversation_data.messages
        
        if update_fields:
            update_conversation(self.db, conversation_id, update_fields)
            return True
        return False
    
    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation (messages are stored within conversation)"""
        try:
            delete_conversation(self.db, conversation_id)
            return True
        except Exception as e:
            print(f"Error deleting conversation {conversation_id}: {e}")
            return False
    
    async def get_conversations_by_status(self, user_id: int, status: str, page: int = 1, limit: int = 20) -> ConversationListResponse:
        """Get conversations by status"""
        skip = (page - 1) * limit
        conversations = get_conversations_by_status(self.db, user_id, status, limit, skip)
        
        # Count total for this status
        all_status_convs = get_conversations_by_status(self.db, user_id, status, limit=9999, skip=0)
        total = len(all_status_convs)
        
        # Convert to response format
        conversation_responses = []
        for conv in conversations:
            conversation_responses.append(ConversationResponse(
                _id=str(conv["_id"]),
                user_id=conv["user_id"],
                title=conv["title"],
                status=conv["status"],
                subject=conv.get("subject"),
                tags=conv.get("tags"),
                messages=conv.get("messages", []),
                created_at=conv["created_at"],
                updated_at=conv["updated_at"]
            ))
        
        return ConversationListResponse(
            conversations=conversation_responses,
            total=total,
            page=page,
            limit=limit
        )
    
    async def search_conversations(self, user_id: int, query: str, limit: int = 20) -> List[ConversationResponse]:
        """Search conversations by title"""
        conversations = search_conversations(self.db, user_id, query, limit)
        
        conversation_responses = []
        for conv in conversations:
            conversation_responses.append(ConversationResponse(
                _id=str(conv["_id"]),
                user_id=conv["user_id"],
                title=conv["title"],
                status=conv["status"],
                subject=conv.get("subject"),
                tags=conv.get("tags"),
                messages=conv.get("messages", []),
                created_at=conv["created_at"],
                updated_at=conv["updated_at"]
            ))
        
        return conversation_responses
    
    # ===== MESSAGE OPERATIONS (Now part of conversation) =====
    
    async def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a conversation"""
        return get_conversation_messages(self.db, conversation_id)
    
    async def get_last_messages(self, conversation_id: str, limit: int = 10) -> List[Message]:
        """Get last N messages for a conversation"""
        messages = get_conversation_messages(self.db, conversation_id)
        
        # Get last N messages
        last_messages = messages[-limit:] if len(messages) > limit else messages
        
        # Convert to Message objects
        message_responses = []
        for msg in last_messages:
            try:
                message_responses.append(Message(
                    message=msg.get("message", ""),
                    is_bot=msg.get("is_bot", False),
                    reaction=msg.get("reaction"),
                    token=msg.get("token"),
                    type=msg.get("type", "text"),
                    is_edited=msg.get("is_edited", False),
                    created_ts=msg.get("created_ts", datetime.utcnow()),
                    updated_ts=msg.get("updated_ts", datetime.utcnow())
                ))
            except Exception as e:
                print(f"Error converting message: {e}")
                continue
        
        return message_responses
    
    async def save_messages_to_conversation(self, conversation_id: str, messages: List[Dict[str, Any]]) -> None:
        """Save all messages to conversation at once (for langgraph thread)"""
        save_messages_to_conversation(self.db, conversation_id, messages)
    
    async def add_message_to_conversation(self, conversation_id: str, message: Dict[str, Any]) -> None:
        """Add a single message to conversation"""
        add_message_to_conversation(self.db, conversation_id, message)
        
        # Update conversation title if this is the first user message
        if not message.get("is_bot", False):
            await self._update_conversation_title_if_first_message(
                conversation_id,
                message.get("message", "")
            )
    
    async def _update_conversation_title_if_first_message(self, conversation_id: str, message: str):
        """Update conversation title if this is the first user message"""
        try:
            # Get conversation
            conv = get_conversation(self.db, conversation_id)
            if not conv:
                return
            
            messages = conv.get("messages", [])
            user_message_count = sum(1 for msg in messages if not msg.get("is_bot", False))
            
            # If this is the first user message, update conversation title
            if user_message_count == 1:
                # Check if title is empty
                if not conv.get("title") or conv.get("title") == "":
                    # Create title from first 50 characters
                    title = message[:50] + "..." if len(message) > 50 else message
                    update_conversation(self.db, conversation_id, {"title": title})
                    print(f"Updated conversation {conversation_id} title: {title}")
        
        except Exception as e:
            print(f"Error updating conversation title: {e}")
    
    async def get_conversation_with_messages(self, conversation_id: str) -> Optional[ConversationWithMessages]:
        """Get a conversation with its messages"""
        # Get conversation
        conv = await self.get_conversation_by_id(conversation_id)
        if not conv:
            return None
        
        # Get messages from conversation
        messages = conv.get("messages", [])
        
        # Create conversation response
        conversation_response = ConversationResponse(
            _id=str(conv["_id"]),
            user_id=conv["user_id"],
            title=conv["title"],
            status=conv["status"],
            subject=conv.get("subject"),
            tags=conv.get("tags"),
            messages=messages,
            created_at=conv["created_at"],
            updated_at=conv["updated_at"]
        )
        
        return ConversationWithMessages(
            conversation=conversation_response,
            total_messages=len(messages)
        )
    
    async def get_conversation_message_count(self, conversation_id: str) -> int:
        """Get total message count for a conversation"""
        return get_conversation_message_count(self.db, conversation_id)
    
    # ===== STATISTICS =====
    
    async def get_conversation_stats(self, user_id: int) -> ConversationStatsResponse:
        """Get conversation statistics for a user"""
        stats = get_conversation_stats(self.db, user_id)
        
        return ConversationStatsResponse(
            total_conversations=stats["total_conversations"],
            active_conversations=stats["active_conversations"],
            archived_conversations=stats["archived_conversations"],
            total_messages=stats["total_messages"],
            bot_messages=stats["bot_messages"],
            user_messages=stats["user_messages"]
        )
