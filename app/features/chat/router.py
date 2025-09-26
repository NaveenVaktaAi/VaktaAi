from fastapi import APIRouter, HTTPException, Query, Path, WebSocket, WebSocketDisconnect
from typing import List, Optional
import json
import asyncio
from datetime import datetime

from app.features.chat.repository import ChatRepository
from app.features.chat.schemas import (
    ChatCreate, ChatUpdate, ChatResponse, ChatMessageCreate, 
    ChatMessageUpdate, ChatMessageResponse, ChatWithMessages, ChatListResponse
)
from app.features.chat.websocket_manager import websocket_manager, ChatWebSocketResponse
from app.features.chat.bot_handler import MongoDBBotMessage

router = APIRouter(prefix="/chat", tags=["chat"])

# Store running tasks for WebSocket connections
running_tasks = {}


@router.post("/", response_model=dict)
async def create_chat(chat_data: ChatCreate):
    """Create a new chat"""
    try:
        chat_repo = ChatRepository()
        chat_id = await chat_repo.create_chat(chat_data)
        return {
            "success": True,
            "message": "Chat created successfully",
            "data": {
                "chat_id": chat_id
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating chat: {str(e)}")

@router.get("/user/{user_id}", response_model=ChatListResponse)
async def get_user_chats(
    user_id: int = Path(..., description="User ID"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Number of chats per page")
):
    """Get all chats for a user"""
    try:
        chat_repo = ChatRepository()
        return await chat_repo.get_user_chats(user_id, page, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching user chats: {str(e)}")


@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(chat_id: str = Path(..., description="Chat ID")):
    """Get a specific chat"""
    try:
        chat_repo = ChatRepository()
        chat = await chat_repo.get_chat_by_id(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        return ChatResponse(
            _id=str(chat["_id"]),
            user_id=chat["user_id"],
            document_id=str(chat["document_id"]) if chat.get("document_id") else None,
            title=chat["title"],
            status=chat["status"],
            created_at=chat["created_at"],
            updated_at=chat["updated_at"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chat: {str(e)}")


@router.put("/{chat_id}", response_model=dict)
async def update_chat(
    chat_id: str = Path(..., description="Chat ID"),
    chat_data: ChatUpdate = None
):
    """Update a chat"""
    try:
        chat_repo = ChatRepository()
        
        # Check if chat exists
        chat = await chat_repo.get_chat_by_id(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        success = await chat_repo.update_chat(chat_id, chat_data)
        if success:
            return {
                "success": True,
                "message": "Chat updated successfully"
            }
        else:
            return {
                "success": False,
                "message": "No changes made"
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating chat: {str(e)}")


@router.delete("/{chat_id}", response_model=dict)
async def delete_chat(chat_id: str = Path(..., description="Chat ID")):
    """Delete a chat and all its messages"""
    try:
        chat_repo = ChatRepository()
        
        # Check if chat exists
        chat = await chat_repo.get_chat_by_id(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        success = await chat_repo.delete_chat(chat_id)
        if success:
            return {
                "success": True,
                "message": "Chat deleted successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Error deleting chat")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting chat: {str(e)}")


@router.get("/{chat_id}/messages", response_model=List[ChatMessageResponse])
async def get_chat_messages(
    chat_id: str = Path(..., description="Chat ID"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(50, ge=1, le=100, description="Number of messages per page")
):
    """Get messages for a chat"""
    try:
        chat_repo = ChatRepository()
        
        # Check if chat exists
        chat = await chat_repo.get_chat_by_id(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        messages = await chat_repo.get_chat_messages(chat_id, page, limit)
        return messages
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching messages: {str(e)}")


@router.post("/{chat_id}/messages", response_model=dict)
async def create_chat_message(
    chat_id: str = Path(..., description="Chat ID"),
    message_data: ChatMessageCreate = None
):
    """Create a new message in a chat"""
    try:
        chat_repo = ChatRepository()
        
        # Check if chat exists
        chat = await chat_repo.get_chat_by_id(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Set chat_id in message data
        message_data.chat_id = chat_id
        
        message_id = await chat_repo.create_chat_message(message_data)
        return {
            "success": True,
            "message": "Message created successfully",
            "message_id": message_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating message: {str(e)}")


@router.get("/{chat_id}/messages/{message_id}", response_model=ChatMessageResponse)
async def get_chat_message(
    chat_id: str = Path(..., description="Chat ID"),
    message_id: str = Path(..., description="Message ID")
):
    """Get a specific chat message"""
    try:
        chat_repo = ChatRepository()
        
        # Check if chat exists
        chat = await chat_repo.get_chat_by_id(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        message = await chat_repo.get_chat_message_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        return ChatMessageResponse(
            _id=str(message["_id"]),
            chat_id=str(message["chat_id"]),
            message=message["message"],
            is_bot=message["is_bot"],
            reaction=message.get("reaction"),
            token=message.get("token"),
            type=message.get("type", "text"),
            is_edited=message.get("is_edited", False),
            created_ts=message["created_ts"],
            updated_ts=message["updated_ts"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching message: {str(e)}")


@router.put("/{chat_id}/messages/{message_id}", response_model=dict)
async def update_chat_message(
    chat_id: str = Path(..., description="Chat ID"),
    message_id: str = Path(..., description="Message ID"),
    message_data: ChatMessageUpdate = None
):
    """Update a chat message"""
    try:
        chat_repo = ChatRepository()
        
        # Check if chat exists
        chat = await chat_repo.get_chat_by_id(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Check if message exists
        message = await chat_repo.get_chat_message_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        success = await chat_repo.update_chat_message(message_id, message_data)
        if success:
            return {
                "success": True,
                "message": "Message updated successfully"
            }
        else:
            return {
                "success": False,
                "message": "No changes made"
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating message: {str(e)}")


@router.delete("/{chat_id}/messages/{message_id}", response_model=dict)
async def delete_chat_message(
    chat_id: str = Path(..., description="Chat ID"),
    message_id: str = Path(..., description="Message ID")
):
    """Delete a chat message"""
    try:
        chat_repo = ChatRepository()
        
        # Check if chat exists
        chat = await chat_repo.get_chat_by_id(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Check if message exists
        message = await chat_repo.get_chat_message_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        success = await chat_repo.delete_chat_message(message_id)
        if success:
            return {
                "success": True,
                "message": "Message deleted successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Error deleting message")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting message: {str(e)}")


@router.post("/{chat_id}/messages/{message_id}/reaction", response_model=dict)
async def add_reaction_to_message(
    chat_id: str = Path(..., description="Chat ID"),
    message_id: str = Path(..., description="Message ID"),
    reaction: str = Query(..., description="Reaction emoji")
):
    """Add reaction to a message"""
    try:
        chat_repo = ChatRepository()
        
        # Check if chat exists
        chat = await chat_repo.get_chat_by_id(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Check if message exists
        message = await chat_repo.get_chat_message_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        success = await chat_repo.add_reaction_to_message(message_id, reaction)
        if success:
            return {
                "success": True,
                "message": "Reaction added successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Error adding reaction")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding reaction: {str(e)}")


@router.delete("/{chat_id}/messages/{message_id}/reaction", response_model=dict)
async def remove_reaction_from_message(
    chat_id: str = Path(..., description="Chat ID"),
    message_id: str = Path(..., description="Message ID")
):
    """Remove reaction from a message"""
    try:
        chat_repo = ChatRepository()
        
        # Check if chat exists
        chat = await chat_repo.get_chat_by_id(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Check if message exists
        message = await chat_repo.get_chat_message_by_id(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        
        success = await chat_repo.remove_reaction_from_message(message_id)
        if success:
            return {
                "success": True,
                "message": "Reaction removed successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Error removing reaction")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing reaction: {str(e)}")


@router.get("/{chat_id}/full", response_model=ChatWithMessages)
async def get_chat_with_messages(
    chat_id: str = Path(..., description="Chat ID"),
    page: int = Query(1, ge=1, description="Page number for messages"),
    limit: int = Query(50, ge=1, le=100, description="Number of messages per page")
):
    """Get a chat with its messages"""
    try:
        chat_repo = ChatRepository()
        result = await chat_repo.get_chat_with_messages(chat_id, page, limit)
        if not result:
            raise HTTPException(status_code=404, detail="Chat not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching chat with messages: {str(e)}")


# WebSocket endpoints
@router.websocket("/ws/{chat_id}")
async def websocket_endpoint(websocket: WebSocket, chat_id: str):
    """
    WebSocket endpoint for real-time chat communication.
    Adapted from the bot system to work with MongoDB chat collections.
    """
    try:
        print(f"WebSocket connected-------------- for chat {chat_id}")
        await connect_websocket(websocket, chat_id)
        current_time = datetime.now().strftime("%H:%M")
        is_error = False

        while True:
            data = await websocket.receive_text()
            await process_received_data(data, chat_id, current_time, is_error, websocket)

    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected: {e}")
        await handle_websocket_disconnect(websocket, chat_id, current_time)
    except Exception as e:
        print(f"WebSocket error: {e}")
        await handle_websocket_disconnect(websocket, chat_id, current_time)


async def connect_websocket(websocket: WebSocket, chat_id: str):
    """Connect WebSocket to chat room"""
    return await websocket_manager.connect(websocket, chat_id)


async def process_received_data(data: str, chat_id: str, current_time: str, is_error: bool, websocket: WebSocket):
    """Process received WebSocket data"""
    try:
        received_data = json.loads(data)

        print("received_data----==================data received------", received_data)
        message_type = received_data.get("mt", "")
        message = received_data.get("message", "")
        user_id = received_data.get("userId", "")
        timezone = received_data.get("timezone", "UTC")
        language_code = received_data.get("selectedLanguage", "en")
        
        # Validate chat_id
        try:
            # Check if chat exists in MongoDB
            chat_repo = ChatRepository()
            chat = await chat_repo.get_chat_by_id(chat_id)
            if not chat:
                await handle_invalid_chat_id(websocket, chat_id)
                return
        except Exception as e:
            print(f"Error validating chat_id: {e}")
            await handle_invalid_chat_id(websocket, chat_id)
            return

        if message_type == "message_upload" and not is_error:
            # Handle message upload
            task = asyncio.create_task(
                handle_message_upload(received_data, chat_id, current_time, websocket)
            )
            running_tasks[websocket] = task

        elif message_type == "stop":
            # Handle stop generation
            if websocket in running_tasks:
                task = running_tasks[websocket]
                task.cancel()
                del running_tasks[websocket]
            
            # Send stop confirmation
            await websocket_manager.send_personal_message(
                websocket,
                {
                    "mt": "stopped_generation",
                    "isBot": True,
                    "message": message,
                    "chatId": chat_id,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        await websocket_manager.send_personal_message(
            websocket,
            {
                "mt": "error",
                "message": "Invalid JSON format",
                "chatId": chat_id,
            }
        )
    except Exception as e:
        print(f"Error processing received data: {e}")
        await websocket_manager.send_personal_message(
            websocket,
            {
                "mt": "error",
                "message": "Error processing message",
                "chatId": chat_id,
            }
        )


async def handle_message_upload(received_data: dict, chat_id: str, current_time: str, websocket: WebSocket):
    """Handle message upload and generate bot response"""
    try:
        message = received_data.get("message", "")
        user_id = received_data.get("userId", "")
        timezone = received_data.get("timezone", "UTC")
        language_code = received_data.get("selectedLanguage", "en")
        document_id = received_data.get("documentId", "")
        # Generate token for user message
        current_time_ms = int(datetime.now().timestamp() * 1000)
        user_message_token = f"{current_time_ms}_{chat_id}"
        
        # Send user message confirmation
        await websocket_manager.send_personal_message(
            websocket,
            {
                "mt": "message_upload_confirm",
                "chatId": chat_id,
                "message": message,
                "userId": user_id,
                "time": current_time,
                "isBot": False,
                "token": user_message_token,
                "timestamp": datetime.now().isoformat(),
            }
        )
        
        # Create WebSocket response handler
        websocket_response = ChatWebSocketResponse(
            chat_id=chat_id,
            user_id=user_id,
            websocket=websocket,
            connection_manager=websocket_manager,
            user_message_token=user_message_token
        )
        
        # Create bot message handler
        bot_message = MongoDBBotMessage(
            chat_id=chat_id,
            user_id=user_id,
            document_id=document_id,
            websocket_response=websocket_response,
            timezone=timezone,
            language_code=language_code
        )
        
        # Generate bot response
        success = await bot_message.send_bot_message(message)
        
        if not success:
            # Send error message
            error_message = "I'm having trouble right now. You can try again."
            await websocket_response.send_message_complete(
                error_message,
                is_bot=True,
                message_type="message_upload_confirm",
                token=f"{int(datetime.now().timestamp() * 1000)}_{chat_id}"
            )
            
    except asyncio.CancelledError:
        print(f"Task for chat {chat_id} was cancelled.")
        await websocket_manager.send_personal_message(
            websocket,
            {
                "mt": "generation_stopped",
                "message": "Bot response generation was stopped.",
                "chatId": chat_id,
            }
        )
    except Exception as e:
        print(f"Error in handle_message_upload: {e}")
        await websocket_manager.send_personal_message(
            websocket,
            {
                "mt": "error",
                "message": "Error generating bot response",
                "chatId": chat_id,
            }
        )


async def handle_invalid_chat_id(websocket: WebSocket, chat_id: str):
    """Handle invalid chat ID"""
    error_message = f"Invalid chat ID: {chat_id}"
    await websocket_manager.send_personal_message(
        websocket,
        {
            "mt": "chat_access_denied",
            "message": error_message,
            "chatId": chat_id,
        }
    )


async def handle_websocket_disconnect(websocket: WebSocket, chat_id: str, current_time: str):
    """Handle WebSocket disconnection"""
    websocket_manager.disconnect(websocket, chat_id)
    
    # Cancel any running tasks
    if websocket in running_tasks:
        task = running_tasks[websocket]
        if not task.done():
            task.cancel()
        del running_tasks[websocket]
    
    print(f"WebSocket disconnected for chat {chat_id}")


@router.get("/ws/{chat_id}/status")
async def get_websocket_status(chat_id: str):
    """Get WebSocket connection status for a chat"""
    connection_count = websocket_manager.get_connection_count(chat_id)
    is_active = websocket_manager.is_chat_active(chat_id)
    
    return {
        "chat_id": chat_id,
        "is_active": is_active,
        "connection_count": connection_count,
        "timestamp": datetime.now().isoformat()
    }
