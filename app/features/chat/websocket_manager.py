import asyncio
import json
import logging
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

    async def send_message_partial(self, message_id: str, partial_text: str, doc_type: str = "public"):
        """Send partial message content"""
        await self.connection_manager.send_personal_message(
            self.websocket,
            {
                "mt": "chat_message_bot_partial",
                "chatId": self.chat_id,
                "uuid": message_id,
                "partial": partial_text,
                "message_context": doc_type,
                "timestamp": datetime.now().isoformat(),
            },
        )

    async def send_message_end(self, message_id: str):
        """Send message end signal"""
        await self.connection_manager.send_personal_message(
            self.websocket,
            {
                "mt": "chat_message_bot_partial",
                "chatId": self.chat_id,
                "stop": message_id,
                "timestamp": datetime.now().isoformat(),
            },
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
        text: str,
        message_id: str = None,
        doc_type: str = "public",
        is_bot: bool = True
    ) -> str:
        """Create a streaming response similar to the bot system"""
        if not message_id:
            message_id = str(uuid4())

        # Send start signal
        await self.send_message_start(message_id)

        # Stream the text word by word
        if isinstance(text, str):
            text_generator = self.async_word_generator(text)
        else:
            text_generator = text

        final_text = ""
        async for word in text_generator:
            if word:
                final_text += word
                await self.send_message_partial(message_id, word, doc_type)
                # Small delay to simulate streaming
                await asyncio.sleep(0.05)

        # Send end signal
        await self.send_message_end(message_id)

        # Send complete message
        await self.send_message_complete(final_text, is_bot, doc_type=doc_type)

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
