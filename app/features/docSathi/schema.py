from typing import List, Optional
from pydantic import BaseModel, EmailStr
from datetime import datetime

class PreSignedUrl(BaseModel):
    fileFormat: str



class FileData(BaseModel):
    signedUrl: str
    fileNameTime: str


class WebsiteUrls(BaseModel):
    url: str


class YoutubeUrls(BaseModel):
    url: str


class UploadDocuments(BaseModel):
    FileData: Optional[List[FileData]] = None
    type: Optional[str] = None
    WebsiteUrls: Optional[List[WebsiteUrls]] = None 
    YoutubeUrls: Optional[List[YoutubeUrls]] = None
    documentFormat: Optional[str] = None




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
