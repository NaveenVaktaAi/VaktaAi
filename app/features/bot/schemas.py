from datetime import datetime

from pydantic import BaseModel


class MessageUploadData(BaseModel):
    mt: str
    isBot: bool
    message: str
    token: str | None
    timezone: str
    selectedLanguage:str
    userId:str


class MessageData(BaseModel):
    time: str
    chatId: str
    message: str
    userId:str
    isBot: bool
    token: str
    mt: str

class MessageResponse(BaseModel):
    id: int
    created: datetime
    text: str
    file: str
    isBot: bool


class CreateChatId(BaseModel):
    time: str
    clientId: str
    message: str
    isBot: bool
    mt: str


class GenerateQuesFromChatHistory(BaseModel):
    botMessage: str
    userMessage: str
    currentUserMessage:str
    messageId: str
    chat_messages_id: int
    prevBotMessage:str

class DeleteSchema(BaseModel):
    chat_id: int