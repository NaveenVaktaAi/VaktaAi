from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio
from bson import ObjectId
from app.database.session import get_db
from app.database.mongo_collections import (
    create_chat, get_chat, get_user_chats, update_chat, delete_chat,
    create_chat_message, get_chat_messages, get_chat_message, 
    update_chat_message, delete_chat_message, get_chat_message_count,
    get_collections
)
from app.features.chat.schemas import (
    ChatCreate, ChatUpdate, ChatResponse, ChatMessageCreate, 
    ChatMessageUpdate, ChatMessageResponse, ChatWithMessages, ChatListResponse
)
from app.features.chat.websocket_manager import WebSocketConnectionManager, ChatWebSocketResponse


class ChatRepository:
    """Repository for chat operations"""

    def __init__(self, websocket_manager: WebSocketConnectionManager = None):
        self.db = next(get_db())
        self.websocket_manager = websocket_manager

    async def create_chat(self, chat_data: ChatCreate) -> str:
        """Create a new chat"""
        # Ensure user_id is set (should be set by router from auth middleware)
        if not chat_data.user_id:
            raise ValueError("user_id is required and must be set from auth middleware")
        
        chat_doc = {
            "user_id": chat_data.user_id,
            "document_id": ObjectId(chat_data.document_id) if chat_data.document_id else None,
            "training_doc_id": ObjectId(chat_data.training_doc_id) if chat_data.training_doc_id else None,
            "title": chat_data.title,  # Use title from chat_data instead of empty string
            "status": chat_data.status,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        return create_chat(self.db, chat_doc)

    async def get_chat_by_id(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """Get a chat by ID"""
        return get_chat(self.db, chat_id)

    async def get_user_chats(self, user_id: str, page: int = 1, limit: int = 10) -> ChatListResponse:
        """Get all chats for a user with pagination"""
        skip = (page - 1) * limit
        chats = get_user_chats(self.db, user_id)
        
        # Apply pagination
        total = len(chats)
        paginated_chats = chats[skip:skip + limit]
        
        # Convert to response format
        chat_responses = []
        for chat in paginated_chats:
            chat_responses.append(ChatResponse(
                _id=str(chat["_id"]),
                user_id=str(chat["user_id"]),  # Convert to string
                document_id=str(chat["document_id"]) if chat.get("document_id") else None,
                title=chat["title"],
                status=chat["status"],
                created_at=chat["created_at"],
                updated_at=chat["updated_at"]
            ))
        
        return ChatListResponse(
            chats=chat_responses,
            total=total,
            page=page,
            limit=limit
        )

    async def update_chat(self, chat_id: str, chat_data: ChatUpdate) -> bool:
        """Update a chat"""
        update_fields = {}
        
        if chat_data.title is not None:
            update_fields["title"] = chat_data.title
        if chat_data.status is not None:
            update_fields["status"] = "active"
            
        if update_fields:
            update_fields["updated_at"] = datetime.now()
            update_chat(self.db, chat_id, update_fields)
            return True
        return False

    async def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat and all its messages"""
        try:
            delete_chat(self.db, chat_id)
            return True
        except Exception as e:
            print(f"Error deleting chat {chat_id}: {e}")
            return False

    async def create_chat_message(self, message_data: ChatMessageCreate) -> str:
        """Create a new chat message"""
        try:
            # ✅ OPTIMIZATION: Reduced duplicate check window from 5s to 2s and simplified query
            from datetime import timedelta
            recent_time = datetime.now() - timedelta(seconds=2)
            
            existing = list(self.db["chat_messages"].find({
                "chat_id": ObjectId(message_data.chat_id),
                "message": message_data.message,
                "is_bot": message_data.is_bot,
                "created_ts": {"$gte": recent_time}
            }).limit(1))
            
            if existing:
                return str(existing[0]["_id"])
            
            message_doc = {
                "chat_id": ObjectId(message_data.chat_id),
                "message": message_data.message,
                "is_bot": message_data.is_bot,
                "reaction": message_data.reaction,
                "token": message_data.token,
                "type": message_data.type,
                "is_edited": message_data.is_edited,
                "training_doc_id": ObjectId(message_data.training_doc_id) if message_data.training_doc_id else None,
                "citation": message_data.citation,  # Save citation source
                "created_ts": datetime.now(),
                "updated_ts": datetime.now()
            }
            message_id = create_chat_message(self.db, message_doc)
            
            # ✅ OPTIMIZATION: Update chat title in background (non-blocking)
            if not message_data.is_bot:
                # Run title update in background to not block message creation
                asyncio.create_task(self._update_chat_title_if_first_user_message(message_data.chat_id, message_data.message))
        except Exception as e:
            # If error occurs, try to find existing message
            existing = list(self.db["chat_messages"].find({
                "chat_id": ObjectId(message_data.chat_id),
                "message": message_data.message,
                "is_bot": message_data.is_bot
            }).sort("created_ts", -1).limit(1))
            
            if existing:
                return str(existing[0]["_id"])
            raise e
        
        # ✅ OPTIMIZATION: Send WebSocket notification in background (non-blocking)
        if self.websocket_manager and not message_data.is_bot:
            # Send notification in background to not block response
            asyncio.create_task(self.websocket_manager.send_to_chat(
                message_data.chat_id,
                {
                    "mt": "new_message",
                    "chatId": message_data.chat_id,
                    "messageId": message_id,
                    "message": message_data.message,
                    "isBot": message_data.is_bot,
                    "timestamp": datetime.now().isoformat(),
                }
            ))
        
        return message_id

    async def _update_chat_title_if_first_user_message(self, chat_id: str, message: str):
        """Update chat title and status to active if this is the first user message"""
        try:
            # Check if this is the first user message for this chat
            user_message_count = self.db["chat_messages"].count_documents({
                "chat_id": ObjectId(chat_id),
                "is_bot": False
            })
            
            print(f"[DEBUG] Chat {chat_id} has {user_message_count} user messages")
            
            # If this is the first user message (count = 1), update chat title
            if user_message_count == 1:
                # Create title from first 50 characters of the message
                title = message[:50] + "..." if len(message) > 50 else message
                
                # Update chat title and status to active
                update_chat(self.db, chat_id, {
                    "title": title,
                    "status": "active",
                    "updated_at": datetime.now()
                })
                print(f"[DEBUG] Updated chat {chat_id} - title: {title}, status: active")
                
        except Exception as e:
            print(f"[ERROR] Error updating chat title: {e}")
            import traceback
            traceback.print_exc()

    async def create_chat_message_with_websocket(
        self, 
        message_data: ChatMessageCreate,
        websocket_response: ChatWebSocketResponse = None
    ) -> str:
        """Create a new chat message with WebSocket streaming support"""
        # Create message in database
        message_doc = {
            "chat_id": ObjectId(message_data.chat_id),
            "message": message_data.message,
            "is_bot": message_data.is_bot,
            "reaction": message_data.reaction,
            "token": message_data.token,
            "type": message_data.type,
            "is_edited": message_data.is_edited,
            "training_doc_id": ObjectId(message_data.training_doc_id) if message_data.training_doc_id else None,
            "citation": message_data.citation,  # Save citation source
            "created_ts": datetime.now(),
            "updated_ts": datetime.now()
        }
        message_id = create_chat_message(self.db, message_doc)
        
        # Update chat title if this is the first user message
        if not message_data.is_bot:
            await self._update_chat_title_if_first_user_message(message_data.chat_id, message_data.message)
        
        # Send WebSocket notification
        if websocket_response:
            if message_data.is_bot:
                # Stream bot message
                await websocket_response.create_streaming_response(
                    message_data.message,
                    message_id,
                    message_data.type or "public",
                    True
                )
            else:
                # Send user message confirmation
                await websocket_response.send_user_message_confirmation(
                    message_data.message,
                    str(message_data.user_id) if hasattr(message_data, 'user_id') else "unknown",
                    message_data.token
                )
        
        return message_id

    async def get_chat_messages(self, chat_id: str, page: int = 1, limit: int = 50) -> List[Dict[str, Any]]:
        """Get messages for a chat with pagination"""
        skip = (page - 1) * limit
        messages = get_chat_messages(self.db, chat_id, limit, skip)
        
        # Convert to response format
        message_responses = []
        for message in messages:
            message_responses.append(ChatMessageResponse(
                _id=str(message["_id"]),
                chat_id=str(message["chat_id"]),
                message=message.get("message", ""),
                is_bot=message.get("is_bot", False),
                reaction=message.get("reaction"),
                token=message.get("token"),
                type=message.get("type", "text"),
                is_edited=message.get("is_edited", False),
                citation=message.get("citation"),  # Include citation in response
                created_ts=message.get("created_ts"),
                updated_ts=message.get("updated_ts")
            ))
        
        return message_responses

    async def get_last_chat_messages(self, chat_id: str, limit: int = 2, skip_latest: int = 1) -> List[Dict[str, Any]]:
        """Get last N messages for a chat, skipping the most recent ones
        
        Args:
            chat_id: Chat ID
            limit: Number of messages to fetch (default: 2)
            skip_latest: Number of latest messages to skip (default: 1 - skips current user message)
        """
        from bson import ObjectId
        
        # Get last N messages in descending order (newest first), skipping the latest
        _, _, _, chat_messages, _, _ = get_collections(self.db)
        messages = list(chat_messages.find({"chat_id": ObjectId(chat_id)})
                       .sort("created_ts", -1)  # Descending - newest first
                       .skip(skip_latest)  # Skip the latest message(s)
                       .limit(limit))
        
        # Reverse to get chronological order (oldest to newest)
        messages.reverse()
        
        # Convert to response format
        message_responses = []
        for message in messages:
            message_responses.append(ChatMessageResponse(
                _id=str(message["_id"]),
                chat_id=str(message["chat_id"]),
                message=message.get("message", ""),
                is_bot=message.get("is_bot", False),
                reaction=message.get("reaction"),
                token=message.get("token"),
                type=message.get("type", "text"),
                is_edited=message.get("is_edited", False),
                citation=message.get("citation"),  # Include citation in response
                created_ts=message.get("created_ts"),
                updated_ts=message.get("updated_ts")
            ))
        
        return message_responses

    async def get_chat_message_by_id(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chat message"""
        return get_chat_message(self.db, message_id)

    async def update_chat_message(self, message_id: str, message_data: ChatMessageUpdate) -> bool:
        """Update a chat message"""
        update_fields = {}
        
        if message_data.message is not None:
            update_fields["message"] = message_data.message
            update_fields["is_edited"] = True
        if message_data.reaction is not None:
            update_fields["reaction"] = message_data.reaction
        if message_data.is_edited is not None:
            update_fields["is_edited"] = message_data.is_edited
            
        if update_fields:
            update_fields["updated_ts"] = datetime.now()
            update_chat_message(self.db, message_id, update_fields)
            return True
        return False

    async def delete_chat_message(self, message_id: str) -> bool:
        """Delete a chat message"""
        try:
            delete_chat_message(self.db, message_id)
            return True
        except Exception as e:
            print(f"Error deleting message {message_id}: {e}")
            return False

    async def get_chat_with_messages(self, chat_id: str, page: int = 1, limit: int = 50) -> Optional[ChatWithMessages]:
        """Get a chat with its messages"""
        # Get chat
        chat = await self.get_chat_by_id(chat_id)
        if not chat:
            return None
        
        # Get messages
        messages = await self.get_chat_messages(chat_id, page, limit)
        
        # Get total message count
        total_messages = get_chat_message_count(self.db, chat_id)
        
        # Create chat response
        chat_response = ChatResponse(
            _id=str(chat["_id"]),
            user_id=str(chat["user_id"]),  # Convert to string
            document_id=str(chat["document_id"]) if chat.get("document_id") else None,
            title=chat["title"],
            status=chat["status"],
            created_at=chat["created_at"],
            updated_at=chat["updated_at"]
        )
        
        return ChatWithMessages(
            chat=chat_response,
            messages=messages,
            total_messages=total_messages
        )

    async def get_chat_message_count(self, chat_id: str) -> int:
        """Get total message count for a chat"""
        return get_chat_message_count(self.db, chat_id)

    async def add_reaction_to_message(self, message_id: str, reaction: str) -> bool:
        """Add or update reaction to a message"""
        try:
            update_fields = {
                "reaction": reaction,
                "updated_ts": datetime.now()
            }
            update_chat_message(self.db, message_id, update_fields)
            return True
        except Exception as e:
            print(f"Error adding reaction to message {message_id}: {e}")
            return False

    async def remove_reaction_from_message(self, message_id: str) -> bool:
        """Remove reaction from a message"""
        try:
            update_fields = {
                "reaction": None,
                "updated_ts": datetime.now()
            }
            update_chat_message(self.db, message_id, update_fields)
            return True
        except Exception as e:
            print(f"Error removing reaction from message {message_id}: {e}")
            return False
