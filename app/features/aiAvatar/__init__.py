"""
AI Tutor (AI Avatar) Module

This module provides API endpoints and business logic for AI Tutor conversations.
Unlike the chat system, AI Tutor conversations are standalone and don't have
document references - they're pure AI tutoring sessions.

Messages are now stored as a list within the conversation document itself,
not in a separate collection. This allows for better integration with
langgraph and agentic flows where the entire thread is saved at once.

Components:
- router: FastAPI routes for AI Tutor endpoints with WebSocket support
- repository: Database operations and business logic
- schemas: Pydantic models for request/response validation
- bot_handler: AI tutor bot message handling without RAG (messages kept in memory)
- websocket_manager: WebSocket connection management
"""

from .router import router as ai_tutor_router
from .repository import AITutorRepository
from .schemas import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationWithMessages,
    ConversationListResponse,
    ConversationStatsResponse,
    Message
)
from .bot_handler import AITutorBotHandler, AITutorBotMessage
from .websocket_manager import ai_tutor_websocket_manager, AITutorWebSocketResponse

__all__ = [
    "ai_tutor_router",
    "AITutorRepository",
    # Bot handler
    "AITutorBotHandler",
    "AITutorBotMessage",
    # WebSocket
    "ai_tutor_websocket_manager",
    "AITutorWebSocketResponse",
    # Schemas
    "ConversationCreate",
    "ConversationUpdate",
    "ConversationResponse",
    "ConversationWithMessages",
    "ConversationListResponse",
    "ConversationStatsResponse",
    "Message"
]
