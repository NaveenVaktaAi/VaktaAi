import asyncio
import json
import logging
import inspect
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import WebSocket, WebSocketDisconnect
from uuid import uuid4

logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """
    Manages WebSocket connections for the chat system.
    Similar to the bot system but adapted for MongoDB chat collections.
    """
    
    def __init__(self):
        # Dictionary to store active connections by chat_id
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Dictionary to store running tasks for each connection
        self.running_tasks: Dict[WebSocket, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, chat_id: str):
        """Accept a WebSocket connection and add it to the chat room"""
        await websocket.accept()
        if chat_id not in self.active_connections:
            self.active_connections[chat_id] = []
        self.active_connections[chat_id].append(websocket)
        logger.info(f"WebSocket connected for chat {chat_id}")

    def disconnect(self, websocket: WebSocket, chat_id: str):
        """Remove a WebSocket connection from the chat room"""
        if chat_id in self.active_connections:
            try:
                self.active_connections[chat_id].remove(websocket)
                # Clean up if no more connections for this chat
                if not self.active_connections[chat_id]:
                    del self.active_connections[chat_id]
            except ValueError:
                pass  # Connection not in list
        
        # Cancel any running tasks for this connection
        if websocket in self.running_tasks:
            task = self.running_tasks[websocket]
            if not task.done():
                task.cancel()
            del self.running_tasks[websocket]
        
        logger.info(f"WebSocket disconnected for chat {chat_id}")

    async def send_personal_message(self, websocket: WebSocket, message: dict):
        """Send a message to a specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            # Remove the connection if it's broken
            for chat_id, connections in self.active_connections.items():
                if websocket in connections:
                    self.disconnect(websocket, chat_id)
                    break

    async def send_to_chat(self, chat_id: str, message: dict):
        """Send a message to all connections in a chat room"""
        if chat_id in self.active_connections:
            disconnected_connections = []
            for websocket in self.active_connections[chat_id]:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    logger.error(f"Error sending to chat {chat_id}: {e}")
                    disconnected_connections.append(websocket)
            
            # Clean up disconnected connections
            for websocket in disconnected_connections:
                self.disconnect(websocket, chat_id)

    async def broadcast(self, message: dict):
        """Broadcast a message to all active connections"""
        for chat_id in list(self.active_connections.keys()):
            await self.send_to_chat(chat_id, message)

    def get_connection_count(self, chat_id: str) -> int:
        """Get the number of active connections for a chat"""
        return len(self.active_connections.get(chat_id, []))

    def is_chat_active(self, chat_id: str) -> bool:
        """Check if a chat has any active connections"""
        return chat_id in self.active_connections and len(self.active_connections[chat_id]) > 0


class ChatWebSocketResponse:
    """
    Handles WebSocket responses for chat messages.
    Adapted from the bot system for MongoDB chat collections.
    """

    def __init__(
        self,
        *,
        chat_id: str,
        user_id: str = None,
        websocket: WebSocket,
        connection_manager: WebSocketConnectionManager,
        user_message_token: str = None,
    ):
        self.chat_id = chat_id
        self.user_id = user_id
        self.connection_manager = connection_manager
        self.websocket = websocket
        self.user_message_token = user_message_token

    async def async_word_generator(self, text: str):
        """Async generator yielding words from the given text"""
        import re
        words = re.split(r"(\s+)", text)
        for word in words:
            yield word

    async def send_message_start(self, message_id: str):
        """Send message start signal"""
        await self.connection_manager.send_personal_message(
            self.websocket,
            {
                "mt": "chat_message_bot_partial",
                "chatId": self.chat_id,
                "start": message_id,
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def send_message_partial(self, message_id: str, partial_text: str):
        """Send partial message content"""
        await self.connection_manager.send_personal_message(
            self.websocket,
            {
                "mt": "chat_message_bot_partial",
                "chatId": self.chat_id,
                "uuid": message_id,
                "partial": partial_text,
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def send_message_end(self, message_id: str, citation_source: str = None):
        """Send message end signal with optional citation metadata"""
        message_data = {
            "mt": "chat_message_bot_partial",
            "chatId": self.chat_id,
            "stop": message_id,
            "timestamp": datetime.now().isoformat(),
        }
        # Add citation source to metadata if provided
        if citation_source:
            message_data["citation_source"] = citation_source
        await self.connection_manager.send_personal_message(
            self.websocket,
            message_data,
        )

    async def send_message_complete(
        self,
        message: str,
        is_bot: bool = True,
        message_type: str = "message_upload_confirm",
        doc_type: str = "public",
        token: str = None
    ):
        """Send complete message"""
        current_time = datetime.now().strftime("%H:%M")
        
        if not token:
            current_time_ms = int(datetime.now().timestamp() * 1000)
            token = f"{current_time_ms}_{self.chat_id}"

        await self.connection_manager.send_personal_message(
            self.websocket,
            {
                "mt": message_type,
                "chatId": self.chat_id,
                "message": message,
                "message_context": doc_type,
                "time": current_time,
                "isBot": is_bot,
                "token": token,
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def create_streaming_response(
        self,
        text,
        message_id: str = None,
        is_bot: bool = True,
        save_to_db: bool = True,
        citation_source: str = None  # ✅ Citation source: "document", "web_search", or "ai"
    ) -> str:
        """
        Streams text (string / coroutine / async generator / sync iterator) to the frontend,
        buffers smaller websocket frames, and returns the full produced text.
        
        Args:
            text: Text to stream (string, coroutine, async generator, or sync iterator)
            message_id: Unique identifier for this message
            is_bot: Whether this is a bot message
            save_to_db: If False, skips sending the complete message (useful when message is saved elsewhere)
        """
        if not message_id:
            message_id = str(uuid4())

        print(f"🎬 create_streaming_response: type={type(text).__name__}, message_id={message_id}")
        await self.send_message_start(message_id)

        # Defensive handling:
        # - If it's a coroutine, await it (prevent "was never awaited" warnings)
        # - If it's a string -> wrap into async word generator
        # - If it's an async generator / has __aiter__ -> use directly
        # - If it's a sync iterable -> wrap into async generator
        if inspect.iscoroutine(text):
            text = await text
            
        text_generator = None
        if isinstance(text, str):
            text_generator = self.async_word_generator(text)
        elif inspect.isasyncgen(text) or hasattr(text, "__aiter__"):
            text_generator = text
        else:
            print(f"⚠️ Unexpected text type: {type(text)}")
            return ""

        print(f"✅ Generator ready: {type(text_generator).__name__}")

        final_text = ""
        token_count = 0
        
        try:
            async for word in text_generator:
                if word:
                    token_count += 1
                    final_text += word
                    await self.send_message_partial(message_id, word)
                    # Small delay for smooth streaming (adjust as needed)
                    await asyncio.sleep(0.01)  # Reduced from 0.05 for faster streaming

            print(f"✅ Streaming completed: {token_count} tokens, {len(final_text)} chars")

        except Exception as e:
            print(f"🚨 Error during streaming: {e}")
            import traceback
            traceback.print_exc()
            # Send error indicator (pass citation_source if available)
            await self.send_message_end(message_id, citation_source)
            raise

        # Send end signal with citation source
        await self.send_message_end(message_id, citation_source)

        # Note: We DON'T send complete message here to avoid duplicate
        # The frontend should accumulate partial messages and display them
        # If you need to save to DB, do it in the calling function
        print(f"🏁 Streaming finished. Final text length: {len(final_text)} chars")

        return final_text

    async def send_error_message(self, error_message: str):
        """Send an error message to the client"""
        await self.connection_manager.send_personal_message(
            self.websocket,
            {
                "mt": "error",
                "chatId": self.chat_id,
                "message": error_message,
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def send_user_message_confirmation(
        self,
        message: str,
        user_id: str,
        token: str = None
    ):
        """Send confirmation that user message was received"""
        current_time = datetime.now().strftime("%H:%M")
        
        if not token:
            current_time_ms = int(datetime.now().timestamp() * 1000)
            token = f"{current_time_ms}_{self.chat_id}"

        await self.connection_manager.send_personal_message(
            self.websocket,
            {
                "mt": "message_upload_confirm",
                "chatId": self.chat_id,
                "message": message,
                "userId": user_id,
                "time": current_time,
                "isBot": False,
                "token": token,
                "timestamp": datetime.now().isoformat(),
            },
        )


# Global instance
websocket_manager = WebSocketConnectionManager()
