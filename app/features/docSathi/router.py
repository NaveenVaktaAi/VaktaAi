

from fastapi import APIRouter, Depends, Query

from app.features.auth.router import get_current_user
from app.common.schemas import ResponseModal
from app.request import Request
from app.features.docSathi.schema import PreSignedUrl , UploadDocuments, DocumentSummaryResponse, DocumentNotesResponse, DocumentQuizResponse, GenerateQuizRequest, GenerateQuizResponse, GetDocumentQuizzesResponse, SubmitQuizRequest, SubmitQuizResponse, DocumentTextResponse, DocumentChatsResponse, documentsId
from app.features.docSathi.repository import check_doc_status, generate_presigned_url, get_documents_by_user_id, read_and_train_private_file, generate_document_summary, generate_document_notes, generate_student_quiz, get_document_quizzes, submit_quiz, get_document_text, get_document_chats


router = APIRouter(tags=["DocSathi"], prefix="/docSathi")

# Generate pre signed url
@router.post("/pre-signed-url")
async def create_signed_url(
    request: Request,
    data: PreSignedUrl,
    current_user: dict = Depends(get_current_user),
):
    print("-----------------------------------")
    return await generate_presigned_url(data.fileFormat)



# Upload documents
@router.post("/upload-document")
async def upload_documents(
    request: Request,
    data: UploadDocuments,
    current_user: dict = Depends(get_current_user),
):
    print("-----------------------------------")
    print("current_user object:", current_user)  # Debug: print full user object
    # get_current_user returns user object with _id field (MongoDB ObjectId converted to string)
    user_id = current_user.get("_id") or current_user.get("id")
    print(user_id, "user_id-----------------------------------  ")
    if not user_id:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found in token"
        )
    return await read_and_train_private_file(data, str(user_id))


# get all the uploaded documents
@router.get("/get-all-documents/{user_id}")
async def get_all_documents(
    request: Request,
    user_id: str,
    current_user: dict = Depends(get_current_user),
):
    print("---------------cdsvdsvs--------------------")
    # Use authenticated user's ID instead of path parameter for security
    # get_current_user returns user object with _id field (MongoDB ObjectId converted to string)
    authenticated_user_id = current_user.get("_id") or current_user.get("id")
    if not authenticated_user_id:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found in token"
        )
    return await get_documents_by_user_id(str(authenticated_user_id))

# Generate document summary
@router.post("/documents/{document_id}/summary")
async def generate_document_summary_endpoint(
    document_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Generate summary for a document based on its chunks"""
    return await generate_document_summary(document_id)

# Generate document notes
@router.post("/documents/{document_id}/notes")
async def generate_document_notes_endpoint(
    document_id: str,
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    """Generate study notes for a document based on its chunks"""
    return await generate_document_notes(document_id)



# Generate student quiz (new comprehensive quiz generation)
@router.post("/generate-quiz")
async def generate_student_quiz_endpoint(
    request: GenerateQuizRequest,
    current_user: dict = Depends(get_current_user),
):
    """Generate a complete student quiz with questions and save to database"""
    return await generate_student_quiz(request)

# Get document quizzes
@router.get("/documents/{document_id}/quizzes")
async def get_document_quizzes_endpoint(
    document_id: str,
    created_by: str = Query(None, description="Filter by creator user ID"),
    request: Request = None,
    current_user: dict = Depends(get_current_user),
):
    """Get all quizzes for a document, optionally filtered by creator"""
    return await get_document_quizzes(document_id, created_by)

# Submit quiz
@router.post("/quizzes/{quiz_id}/submit")
async def submit_quiz_endpoint(
    quiz_id: str,
    request: SubmitQuizRequest,
    current_user: dict = Depends(get_current_user),
):
    """Submit quiz answers and calculate score"""
    return await submit_quiz(quiz_id, request.answers)

# Get document text content
@router.get("/documents/{document_id}/text")
async def get_document_text_endpoint(
    document_id: str,
    request: Request = None,
    current_user: dict = Depends(get_current_user),
):
    """Get combined text content from document chunks"""
    return await get_document_text(document_id)

# Get document chats
@router.get("/documents/{document_id}/chats")
async def get_document_chats_endpoint(
    document_id: str,
    limit: int = Query(20, description="Number of chats to return (max 50)"),
    offset: int = Query(0, description="Number of chats to skip"),
    request: Request = None,
    current_user: dict = Depends(get_current_user),
):
    """Get all chats for a document with pagination"""
    return await get_document_chats(document_id, limit, offset)

# check the training document status
@router.post("/check-document-status")
async def check_document_status(
    request: Request,
    data: documentsId,
    current_user: dict = Depends(get_current_user),
):
    return await check_doc_status(data.document_id)
