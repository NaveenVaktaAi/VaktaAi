from ast import Dict
from fastapi import APIRouter, HTTPException, Query, Path, WebSocket, WebSocketDisconnect, Depends, UploadFile, File, Form
from typing import List, Optional
import logging
import json
from datetime import datetime
import uuid
import os
from pathlib import Path as FilePath

from pymongo.database import Database

from app.database.session import get_db
from app.oauth import is_user_authorized
from app.features.aiAvatar.repository import AITutorRepository
from app.features.aiAvatar.schemas import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationWithMessages,
    ConversationListResponse,
    ConversationStatsResponse
)
from app.features.aiAvatar.websocket_manager import ai_tutor_websocket_manager, AITutorWebSocketResponse
from app.features.aiAvatar.bot_handler import AITutorBotMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai-tutor", tags=["AI Tutor"])

# Create static directories for file uploads if they don't exist
UPLOAD_DIR = FilePath("static/ai_tutor_images")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PDF_UPLOAD_DIR = FilePath("static/ai_tutor_pdfs")
PDF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ===== IMAGE UPLOAD ENDPOINT =====

@router.post("/upload-images")
async def upload_images(
    images: List[UploadFile] = File(...),
    conversation_id: str = Form(...)
):
    """
    Upload images for AI Tutor vision analysis
    Maximum 3 images allowed per upload
    """
    try:
        if len(images) > 3:
            raise HTTPException(status_code=400, detail="Maximum 3 images allowed")
        
        image_urls = []
        
        for image in images:
            # Validate file type
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"Invalid file type: {image.content_type}")
            
            # Generate unique filename
            file_extension = image.filename.split('.')[-1]
            unique_filename = f"{uuid.uuid4().hex[:12]}.{file_extension}"
            file_path = UPLOAD_DIR / unique_filename
            
            # Save file
            with open(file_path, "wb") as f:
                content = await image.read()
                f.write(content)
            
            # Generate URL (accessible via static files)
            image_url = f"/static/ai_tutor_images/{unique_filename}"
            image_urls.append(image_url)
            
            logger.info(f"Uploaded image: {unique_filename} for conversation: {conversation_id}")
        
        return {
            "success": True,
            "message": f"Uploaded {len(image_urls)} image(s) successfully",
            "image_urls": image_urls
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading images: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading images: {str(e)}")


# ===== PDF UPLOAD ENDPOINT =====

@router.post("/upload-pdfs")
async def upload_pdfs(
    pdfs: List[UploadFile] = File(...),
    conversation_id: str = Form(...)
):
    """
    Upload PDFs for AI Tutor document analysis
    Maximum 3 PDFs allowed per upload
    """
    try:
        if len(pdfs) > 3:
            raise HTTPException(status_code=400, detail="Maximum 3 PDFs allowed")
        
        pdf_urls = []
        
        for pdf in pdfs:
            # Validate file type
            if pdf.content_type != 'application/pdf':
                raise HTTPException(status_code=400, detail=f"Invalid file type: {pdf.content_type}. Only PDFs allowed.")
            
            # Generate unique filename
            unique_filename = f"{uuid.uuid4().hex[:12]}.pdf"
            file_path = PDF_UPLOAD_DIR / unique_filename
            
            # Save file
            with open(file_path, "wb") as f:
                content = await pdf.read()
                f.write(content)
            
            # Generate URL (accessible via static files)
            pdf_url = f"/static/ai_tutor_pdfs/{unique_filename}"
            pdf_urls.append(pdf_url)
            
            logger.info(f"Uploaded PDF: {unique_filename} for conversation: {conversation_id}")
        
        return {
            "success": True,
            "message": f"Uploaded {len(pdf_urls)} PDF(s) successfully",
            "pdf_urls": pdf_urls
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading PDFs: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading PDFs: {str(e)}")


# ===== CONVERSATION ENDPOINTS =====

@router.post("/conversation", response_model=dict)
async def create_conversation(conversation_data: ConversationCreate):
    """
    Create a new AI Tutor conversation
    
    Creates a new conversation session for AI tutoring.
    No document references needed - standalone conversation.
    Messages are stored as a list within the conversation document.
    """
    try:
        repo = AITutorRepository()
        conversation_id = await repo.create_conversation(conversation_data, 1)
        
        return {
            "success": True,
            "message": "Conversation created successfully",
            "data": {
                "conversation_id": conversation_id
            }
        }
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating conversation: {str(e)}")


@router.get("/conversations/user/{user_id}", response_model=ConversationListResponse)
async def get_user_conversations(
    user_id: int = Path(..., description="User ID"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Number of conversations per page"),
    status: str = Query(None, description="Filter by status (active/archived)")
):
    """Get all conversations for a user"""
    try:
        repo = AITutorRepository()
        
        if status:
            return await repo.get_conversations_by_status(user_id, status, page, limit)
        else:
            return await repo.get_user_conversations(user_id, page, limit)
    
    except Exception as e:
        logger.error(f"Error fetching user conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str = Path(..., description="Conversation ID")):
    """Get a specific conversation with its messages"""
    try:
        repo = AITutorRepository()
        conversation = await repo.get_conversation_by_id(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return ConversationResponse(
            _id=str(conversation["_id"]),
            user_id=conversation["user_id"],
            title=conversation["title"],
            status=conversation["status"],
            subject=conversation.get("subject"),
            tags=conversation.get("tags"),
            messages=conversation.get("messages", []),
            created_at=conversation["created_at"],
            updated_at=conversation["updated_at"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching conversation: {str(e)}")


@router.get("/conversations/{conversation_id}/full", response_model=ConversationWithMessages)
async def get_conversation_with_messages(
    conversation_id: str = Path(..., description="Conversation ID")
):
    """Get a conversation with its messages"""
    try:
        repo = AITutorRepository()
        
        conversation_with_messages = await repo.get_conversation_with_messages(conversation_id)
        
        if not conversation_with_messages:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conversation_with_messages
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation with messages: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching conversation with messages: {str(e)}")


@router.put("/conversations/{conversation_id}", response_model=dict)
async def update_conversation(
    conversation_id: str = Path(..., description="Conversation ID"),
    conversation_data: ConversationUpdate = None
):
    """Update a conversation (including messages)"""
    try:
        repo = AITutorRepository()
        
        # Check if conversation exists
        conversation = await repo.get_conversation_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Update conversation
        success = await repo.update_conversation(conversation_id, conversation_data)
        
        if success:
            return {
                "success": True,
                "message": "Conversation updated successfully",
                "data": {
                    "conversation_id": conversation_id
                }
            }
        else:
            return {
                "success": False,
                "message": "No fields to update",
                "data": {
                    "conversation_id": conversation_id
                }
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating conversation: {str(e)}")


@router.delete("/conversations/{conversation_id}", response_model=dict)
async def delete_conversation(conversation_id: str = Path(..., description="Conversation ID")):
    """Delete a conversation (messages are part of conversation document)"""
    try:
        repo = AITutorRepository()
        
        # Check if conversation exists
        conversation = await repo.get_conversation_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Delete conversation
        success = await repo.delete_conversation(conversation_id)
        
        if success:
            return {
                "success": True,
                "message": "Conversation deleted successfully",
                "data": {
                    "conversation_id": conversation_id
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete conversation")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")


# ===== STATISTICS ENDPOINT =====

@router.get("/users/{user_id}/stats", response_model=ConversationStatsResponse)
async def get_user_conversation_stats(user_id: int = Path(..., description="User ID")):
    """Get conversation statistics for a user"""
    try:
        repo = AITutorRepository()
        stats = await repo.get_conversation_stats(user_id)
        return stats
    
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")


# ===== SEARCH ENDPOINT =====

@router.get("/conversations/search/{user_id}", response_model=List[ConversationResponse])
async def search_conversations(
    user_id: int = Path(..., description="User ID"),
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results")
):
    """Search conversations by title"""
    try:
        repo = AITutorRepository()
        conversations = await repo.search_conversations(user_id, q, limit)
        return conversations
    
    except Exception as e:
        logger.error(f"Error searching conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching conversations: {str(e)}")


# ===== END CONVERSATION ENDPOINT =====

@router.put("/conversation/{conversation_id}/end", response_model=dict)
async def end_conversation_api(conversation_id: str = Path(..., description="Conversation ID")):
    """
    End an active conversation and save all in-memory messages to database.
    This endpoint is called when user clicks 'Done' button.
    """
    try:
        print(f"[AI_TUTOR] REST API: Ending conversation {conversation_id}")
        
        # Check if conversation exists in database
        repo = AITutorRepository()
        conversation = await repo.get_conversation_by_id(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Check if there's an active bot message handler for this conversation
        if conversation_id in bot_message_handlers:
            bot_message = bot_message_handlers[conversation_id]
            
            # Save all in-memory messages to database
            await bot_message.save_conversation_messages()
            
            # Remove handler from memory to clear in-memory messages
            del bot_message_handlers[conversation_id]
            
            print(f"[AI_TUTOR] REST API: Conversation {conversation_id} ended and messages saved")
            
            return {
                "success": True,
                "message": "Conversation ended and all messages saved to database",
                "data": {
                    "conversation_id": conversation_id,
                    "messages_saved": True,
                    "memory_cleared": True
                }
            }
        else:
            # No active handler means no in-memory messages
            print(f"[AI_TUTOR] REST API: No active handler found for conversation {conversation_id}")
            
            return {
                "success": True,
                "message": "Conversation ended (no active in-memory messages)",
                "data": {
                    "conversation_id": conversation_id,
                    "messages_saved": False,
                    "memory_cleared": True
                }
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AI_TUTOR] REST API: Error ending conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error ending conversation: {str(e)}")


# ===== WEBSOCKET ENDPOINT =====

@router.websocket("/conversation/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """
    WebSocket endpoint for real-time AI Tutor conversation.
    Pure AI tutoring without RAG/document search.
    Messages are kept in memory during session and saved when conversation ends.
    """
    try:
        print(f"[AI_TUTOR] WebSocket connected for conversation {conversation_id}")
        await connect_websocket(websocket, conversation_id)
        
        while True:
            data = await websocket.receive_text()
            print(f"[AI_TUTOR] Received data for conversation {conversation_id}")
            await process_received_data(data, conversation_id, websocket)

    except WebSocketDisconnect as e:
        print(f"[AI_TUTOR] WebSocket disconnected: {e}")
        await handle_websocket_disconnect(websocket, conversation_id)
    except Exception as e:
        print(f"[AI_TUTOR] WebSocket error: {e}")
        await handle_websocket_disconnect(websocket, conversation_id)


async def connect_websocket(websocket: WebSocket, conversation_id: str):
    """Connect WebSocket to conversation"""
    await ai_tutor_websocket_manager.connect(websocket, conversation_id)
    
    # Send connection confirmation
    await websocket.send_json({
        "mt": "connected",
        "conversationId": conversation_id,
        "message": "Connected to AI Tutor",
        "timestamp": datetime.utcnow().isoformat()
    })


async def process_received_data(data: str, conversation_id: str, websocket: WebSocket):
    """Process received WebSocket data"""
    try:
        message_data = json.loads(data)
        print(f"[AI_TUTOR] Processing message data: {message_data}")

        print(f"[AI_TUTOR] ------------ data: {data}")
        # Support both 'mt' (message type) and 'type' fields for compatibility
        message_type = message_data.get("mt") or message_data.get("type")
        
        # Map frontend message types to backend handlers
        if message_type == "message_upload" or message_type == "user_message":
            await handle_user_message(message_data, conversation_id, websocket)
        elif message_type == "typing":
            await handle_typing_indicator(message_data, conversation_id)
        elif message_type == "stop_typing":
            await handle_stop_typing(message_data, conversation_id)
        elif message_type == "end_conversation":
            await handle_end_conversation(message_data, conversation_id, websocket)
        else:
            print(f"[AI_TUTOR] Unknown message type: {message_type}")
            print(f"[AI_TUTOR] Available keys in message_data: {list(message_data.keys())}")
    
    except json.JSONDecodeError as e:
        print(f"[AI_TUTOR] Error parsing JSON: {e}")
        await websocket.send_json({
            "mt": "error",
            "error": "Invalid JSON format",
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"[AI_TUTOR] Error processing data: {e}")
        await websocket.send_json({
            "mt": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })


# Global storage for bot message handlers per conversation
bot_message_handlers = {}


async def handle_user_message(message_data: dict, conversation_id: str, websocket: WebSocket):
    """Handle user message from WebSocket"""
    try:
        print(f"[AI_TUTOR] Handling user message: {message_data}")
        user_message = message_data.get("message")
        user_id = str(message_data.get("userId", "unknown"))
        token = message_data.get("token")
        # Support both isAudio and is_audio for compatibility
        isAudio = message_data.get("isAudio", message_data.get("is_audio", True))
        # Extract images if present
        images = message_data.get("images", [])
        # Extract PDFs if present
        pdfs = message_data.get("pdfs", [])
        
        if not user_message and not images and not pdfs:
            print("[AI_TUTOR] No message content, images, or PDFs provided")
            return
        
        print(f"[AI_TUTOR] User message: {user_message}")
        print(f"[AI_TUTOR] Images: {len(images)} image(s)" if images else "[AI_TUTOR] No images")
        print(f"[AI_TUTOR] PDFs: {len(pdfs)} PDF(s)" if pdfs else "[AI_TUTOR] No PDFs")
        print(f"[AI_TUTOR] Audio mode: {'ENABLED (text + audio + summary)' if isAudio else 'DISABLED (text only)'}")
        
        # Get conversation subject and topic for context
        repo = AITutorRepository()
        conversation = await repo.get_conversation_by_id(conversation_id)
        selected_subject = conversation.get("subject", "") if conversation else ""
        selected_topic = conversation.get("topic", "") if conversation else ""  # ✅ Get topic
        print(f"[AI_TUTOR] 📚 Selected Subject: {selected_subject if selected_subject else 'Not specified'}")
        print(f"[AI_TUTOR] 🎯 Selected Topic: {selected_topic if selected_topic else 'Not specified'}")
        
        # Create WebSocket response handler
        websocket_response = AITutorWebSocketResponse(
            websocket,
            conversation_id,
            ai_tutor_websocket_manager
        )
        
        # Send user message confirmation
        await websocket_response.send_user_message_confirmation(
            user_message,
            user_id,
            token
        )
        
        # Initialize or reuse bot message handler for this conversation
        if conversation_id not in bot_message_handlers:
            bot_message_handlers[conversation_id] = AITutorBotMessage(
                conversation_id=conversation_id,
                user_id=user_id,
                websocket_response=websocket_response,
                timezone=message_data.get("timezone", "UTC"),
                language_code=message_data.get("languageCode", "en"),
                selected_subject=selected_subject,  # ✅ Pass subject context
                selected_topic=selected_topic  # ✅ Pass topic context
            )
        
        bot_message = bot_message_handlers[conversation_id]
        
        # Process and send bot response with audio flag, images, PDFs, and subject context
        await bot_message.send_bot_message(user_message, isAudio=isAudio, images=images, pdfs=pdfs)  # ✅ Pass isAudio flag, images, and PDFs
        
    except Exception as e:
        print(f"[AI_TUTOR] Error handling user message: {e}")
        await websocket.send_json({
            "mt": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })


async def handle_end_conversation(message_data: dict, conversation_id: str, websocket: WebSocket):
    """Handle end conversation - save all messages to database"""
    try:
        print(f"[AI_TUTOR] Ending conversation {conversation_id}, saving messages to DB...")
        
        # Get bot message handler for this conversation
        if conversation_id in bot_message_handlers:
            bot_message = bot_message_handlers[conversation_id]
            
            # Save all messages to database
            await bot_message.save_conversation_messages()
            
            # Remove handler from memory
            del bot_message_handlers[conversation_id]
            
            # Send confirmation
            await websocket.send_json({
                "mt": "conversation_ended",
                "conversationId": conversation_id,
                "message": "Conversation ended and messages saved",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            print(f"[AI_TUTOR] Conversation {conversation_id} ended successfully")
        else:
            print(f"[AI_TUTOR] No active handler found for conversation {conversation_id}")
            await websocket.send_json({
                "mt": "conversation_ended",
                "conversationId": conversation_id,
                "message": "Conversation ended (no messages to save)",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    except Exception as e:
        print(f"[AI_TUTOR] Error ending conversation: {e}")
        await websocket.send_json({
            "mt": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })


async def handle_typing_indicator(message_data: dict, conversation_id: str):
    """Handle typing indicator"""
    await ai_tutor_websocket_manager.send_to_conversation(
        conversation_id,
        {
            "mt": "typing_indicator",
            "conversationId": conversation_id,
            "userId": message_data.get("userId"),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


async def handle_stop_typing(message_data: dict, conversation_id: str):
    """Handle stop typing indicator"""
    await ai_tutor_websocket_manager.send_to_conversation(
        conversation_id,
        {
            "mt": "stop_typing",
            "conversationId": conversation_id,
            "userId": message_data.get("userId"),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


async def handle_websocket_disconnect(websocket: WebSocket, conversation_id: str):
    """Handle WebSocket disconnection"""
    # Save messages if there's an active handler
    if conversation_id in bot_message_handlers:
        try:
            bot_message = bot_message_handlers[conversation_id]
            await bot_message.save_conversation_messages()
            del bot_message_handlers[conversation_id]
            print(f"[AI_TUTOR] Messages saved on disconnect for conversation {conversation_id}")
        except Exception as e:
            print(f"[AI_TUTOR] Error saving messages on disconnect: {e}")
    
    ai_tutor_websocket_manager.disconnect(websocket, conversation_id)
    print(f"[AI_TUTOR] WebSocket disconnected from conversation {conversation_id}")
