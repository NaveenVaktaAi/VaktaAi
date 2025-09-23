# Doc-Chat RAG System - Pydantic Models
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum

class SourceType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPT = "ppt"
    URL = "url"
    YOUTUBE = "youtube"
    AUDIO = "audio"
    VIDEO = "video"

class DocumentStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Request Models
class IngestRequest(BaseModel):
    tenant_id: str
    source_type: SourceType
    url: Optional[str] = None
    upload_id: Optional[str] = None
    language_prefs: List[str] = ["en"]
    fallback_stt: bool = False

class ChatRequest(BaseModel):
    tenant_id: str
    doc_ids: List[str]
    query: str
    top_k: int = 6
    return_sources: bool = True

class StudyRequest(BaseModel):
    tenant_id: str
    doc_ids: List[str]
    sections: List[str] = ["summary", "notes", "quiz", "glossary"]

# Response Models
class DocumentSegment(BaseModel):
    tenant_id: str
    doc_id: str
    source_type: SourceType
    source_url: str
    text: str
    page: Optional[int] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    language: str = "en"
    hash: str
    metadata: Dict[str, Any] = {}

class SourceInfo(BaseModel):
    doc_id: str
    source_type: SourceType
    text: str
    page: Optional[int] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    similarity: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo] = []

class StudyResponse(BaseModel):
    doc_ids: List[str]
    sections: Dict[str, str]
    generated_at: datetime

class DocumentInfo(BaseModel):
    id: str
    name: str
    type: SourceType
    status: DocumentStatus
    uploaded_at: datetime
    size: Optional[int] = None
    segments_count: Optional[int] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]
    error: Optional[str] = None

class MetricsResponse(BaseModel):
    timestamp: datetime
    tenant_stats: List[Dict[str, Any]]
    redis_memory_used: str
    redis_connected_clients: int
