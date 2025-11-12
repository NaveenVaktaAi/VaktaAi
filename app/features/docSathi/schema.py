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
    url: Optional[str] = Field(None, alias="Url")  # ✅ Single URL field - backend will auto-detect type
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



class documentsId(BaseModel):
     document_id: str

class DocumentSummary(BaseModel):
    summary: str
    key_points: List[str]
    title: str

class DocumentSummaryResponse(BaseModel):
    success: bool
    message: str
    data: Optional[DocumentSummary] = None

class DocumentNotes(BaseModel):
    notes: List[str]
    title: str

class DocumentNotesResponse(BaseModel):
    success: bool
    message: str
    data: Optional[DocumentNotes] = None

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    correct_answer: int
    explanation: str

class DocumentQuiz(BaseModel):
    questions: List[QuizQuestion]
    title: str

class DocumentQuizResponse(BaseModel):
    success: bool
    message: str
    data: Optional[DocumentQuiz] = None

class GenerateQuizRequest(BaseModel):
    quiz_name: str
    document_id: str
    user_id: str
    level: str  # easy | medium | hard
    number_of_questions: int

class StudentQuiz(BaseModel):
    quiz_name: str
    related_doc_id: str
    created_by: str
    level: str
    no_of_questions: int
    is_submitted: bool = False
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class QuestionAnswer(BaseModel):
    quiz_id: str
    question_type: str  # mcq | true_false
    question_text: str
    options: List[str]
    correct_answer: str
    student_answer: Optional[str] = None
    AI_explanation: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class GenerateQuizResponse(BaseModel):
    success: bool
    message: str
    data: Optional[StudentQuiz] = None

class QuizQuestionResponse(BaseModel):
    question_id: str
    question_type: str
    question_text: str
    options: List[str]
    correct_answer: str
    student_answer: Optional[str] = None
    AI_explanation: str

class QuizResponse(BaseModel):
    quiz_id: str
    quiz_name: str
    level: str
    no_of_questions: int
    is_submitted: bool
    created_at: datetime
    updated_at: datetime
    questions: List[QuizQuestionResponse]

class GetDocumentQuizzesResponse(BaseModel):
    success: bool
    message: str
    data: List[QuizResponse]

class SubmitQuizRequest(BaseModel):
    answers: List[dict]  # [{"question_id": "string", "selected_answer": "string"}]

class QuizSubmissionResult(BaseModel):
    quiz_id: str
    score: int
    total_questions: int
    correct_answers: int
    wrong_answers: int
    percentage: float
    submitted_at: datetime
    questions: List[QuizQuestionResponse]

class SubmitQuizResponse(BaseModel):
    success: bool
    message: str
    data: Optional[QuizSubmissionResult] = None

class DocumentTextResponse(BaseModel):
    success: bool
    message: str
    data: Optional[dict] = None  # {"content": str, "type": str}

class DocumentChat(BaseModel):
    chat_id: str
    document_id: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    last_message: Optional[str] = None
    message_count: Optional[int] = None
    title: Optional[str] = None

class DocumentChatsResponse(BaseModel):
    success: bool
    message: str
    data: Optional[List[DocumentChat]] = None