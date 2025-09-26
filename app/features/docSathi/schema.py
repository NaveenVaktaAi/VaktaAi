from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime

class PreSignedUrl(BaseModel):
    fileFormat: str


class FileData(BaseModel):
    signedUrl: str
    fileNameTime: str


class UploadDocuments(BaseModel):
    file_data: Optional["FileData"] = Field(None, alias="FileData")
    type: Optional[str] = None
    website_url: Optional[str] = Field(None, alias="WebsiteUrl")   # single URL instead of array
    youtube_url: Optional[str] = Field(None, alias="YoutubeUrl")   # single URL instead of array
    document_format: Optional[str] = Field(None, alias="documentFormat")



class AgentAIDocument(BaseModel):
    user_id: Optional[int]
    organization_id: int
    name: Optional[str] = None
    url: Optional[str] = None
    status: Optional[str] = None
    document_format: Optional[str] = None  # pdf, word
    type: Optional[str] = None
    summary: Optional[str] = None
    created_ts: datetime = datetime.now()
    updated_ts: datetime = datetime.now()


class Chunk(BaseModel):
    detail: str
    keywords: str
    meta_summary: str
    chunk: str
    organization_id: int
    question_id: Optional[int] = None
    training_document_id: Optional[int] = None
    created_ts: datetime = datetime.now()
    updated_ts: datetime = datetime.now()



class documentsIds(BaseModel):
     document_ids: List[str]  