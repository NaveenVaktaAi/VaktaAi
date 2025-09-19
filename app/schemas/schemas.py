from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from bson import ObjectId


# Helper for ObjectId (MongoDB uses it for _id)
class PyObjectId(ObjectId):

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)


# ----------------------------
# AgentAIDocument Schema
# ----------------------------
class DocSathiAIDocumentBase(BaseModel):
    user_id: Optional[int]
    name: Optional[str] = None
    url: Optional[str] = None
    status: Optional[str] = None
    document_format: Optional[str] = None   # pdf, word, etc.
    type: Optional[str] = None
    summary: Optional[str] = None
    created_ts: datetime = Field(default_factory=datetime.now)
    updated_ts: datetime = Field(default_factory=datetime.now)


class DocSathiAIDocumentCreate(DocSathiAIDocumentBase):
    pass


class DocSathiAIDocumentOut(DocSathiAIDocumentBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    class Config:
        json_encoders = {ObjectId: str}
        allow_population_by_field_name = True


# ----------------------------
# Chunk Schema
# ----------------------------
class ChunkBase(BaseModel):
    detail: str
    keywords: str
    meta_summary: str
    chunk: str
    question_id: Optional[int] = None
    training_document_id: int
    created_ts: datetime = Field(default_factory=datetime.now)
    updated_ts: datetime = Field(default_factory=datetime.now)


class ChunkCreate(ChunkBase):
    pass


class ChunkOut(ChunkBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")

    class Config:
        json_encoders = {ObjectId: str}
        allow_population_by_field_name = True
