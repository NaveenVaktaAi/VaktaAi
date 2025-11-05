from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from bson import ObjectId


class ChatCreate(BaseModel):
    """Schema for creating a new chat"""
    user_id: int = Field(..., description="ID of the user creating the chat")
    document_id: Optional[str] = Field(None, description="ID of the document this chat is related to")
    training_doc_id: Optional[str] = Field(None, description="ID of the training document this chat is related to")
    title: str = Field(..., description="Title of the chat")
    status: str = Field(default="active", description="Status of the chat (active/inactive)")


class ChatUpdate(BaseModel):
    """Schema for updating a chat"""
    title: Optional[str] = Field(None, description="Title of the chat")
    status: Optional[str] = Field(None, description="Status of the chat (active/inactive)")
    training_doc_id: Optional[str] = Field(None, description="ID of the training document this chat is related to")


class ChatResponse(BaseModel):
    """Schema for chat response"""
    id: str = Field(..., alias="_id")
    user_id: int
    document_id: Optional[str]
    title: str
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }


class ChatMessageCreate(BaseModel):
    """Schema for creating a new chat message"""
    chat_id: str = Field(..., description="ID of the chat this message belongs to")
    message: str = Field(..., description="The message content")
    is_bot: bool = Field(default=False, description="Whether this message is from bot or user")
    reaction: Optional[str] = Field(None, description="Reaction to the message")
    token: Optional[str] = Field(None, description="Token for the message")
    type: Optional[str] = Field(default="text", description="Type of message (text, image, etc.)")
    is_edited: bool = Field(default=False, description="Whether the message has been edited")
    training_doc_id: Optional[str] = Field(None, description="ID of the training document this message is related to")


class ChatMessageUpdate(BaseModel):
    """Schema for updating a chat message"""
    message: Optional[str] = Field(None, description="The message content")
    reaction: Optional[str] = Field(None, description="Reaction to the message")
    is_edited: Optional[bool] = Field(None, description="Whether the message has been edited")
    training_doc_id: Optional[str] = Field(None, description="ID of the training document this message is related to")


class ChatMessageResponse(BaseModel):
    """Schema for chat message response"""
    id: str = Field(..., alias="_id")
    chat_id: str
    message: str
    is_bot: Optional[bool] = False
    reaction: Optional[str]
    token: Optional[str]
    type: Optional[str]
    is_edited: Optional[bool] = False
    created_ts: Optional[datetime]
    updated_ts: Optional[datetime]

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }


class ChatWithMessages(BaseModel):
    """Schema for chat with its messages"""
    chat: ChatResponse
    messages: List[ChatMessageResponse]
    total_messages: int


class ChatListResponse(BaseModel):
    """Schema for list of chats"""
    chats: List[ChatResponse]
    total: int
    page: int
    limit: int
