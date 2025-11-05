from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId


class Message(BaseModel):
    """Schema for a single message within a conversation"""
    message: str = Field(..., description="The message content")
    is_bot: bool = Field(default=False, description="Whether this message is from bot or user")
    reaction: Optional[str] = Field(None, description="Reaction to the message")
    token: Optional[str] = Field(None, description="Token for the message")
    type: Optional[str] = Field(default="text", description="Type of message (text, image, etc.)")
    is_edited: bool = Field(default=False, description="Whether the message has been edited")
    created_ts: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when message was created")
    updated_ts: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when message was last updated")


class ConversationCreate(BaseModel):
    """Schema for creating a new conversation"""
    user_id: int = Field(..., description="ID of the user creating the conversation")
    title: str = Field(default="", description="Title of the conversation")
    status: str = Field(default="active", description="Status of the conversation (active/archived)")
    subject: Optional[str] = Field(None, description="Subject of conversation (e.g., Physics, Math)")
    topic: Optional[str] = Field(None, description="Specific topic within subject (e.g., Newton's Laws, Calculus)")
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")
    messages: Optional[List[Dict[str, Any]]] = Field(default=[], description="List of messages in the conversation")


class ConversationUpdate(BaseModel):
    """Schema for updating a conversation"""
    title: Optional[str] = Field(None, description="Title of the conversation")
    status: Optional[str] = Field(None, description="Status of the conversation (active/archived)")
    subject: Optional[str] = Field(None, description="Subject of conversation")
    topic: Optional[str] = Field(None, description="Specific topic within subject")
    tags: Optional[List[str]] = Field(None, description="Tags for categorization")
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="List of messages to replace")


class ConversationResponse(BaseModel):
    """Schema for conversation response"""
    id: str = Field(..., alias="_id")
    user_id: int
    title: str
    status: str
    subject: Optional[str] = None
    topic: Optional[str] = None
    tags: Optional[List[str]] = None
    messages: Optional[List[Dict[str, Any]]] = Field(default=[], description="List of messages in the conversation")
    created_at: datetime
    updated_at: datetime
    
    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat()
        }


class ConversationWithMessages(BaseModel):
    """Schema for conversation with its messages"""
    conversation: ConversationResponse
    total_messages: int


class ConversationListResponse(BaseModel):
    """Schema for list of conversations"""
    conversations: List[ConversationResponse]
    total: int
    page: int
    limit: int


class ConversationStatsResponse(BaseModel):
    """Schema for conversation statistics"""
    total_conversations: int
    active_conversations: int
    archived_conversations: int
    total_messages: int
    bot_messages: int
    user_messages: int
