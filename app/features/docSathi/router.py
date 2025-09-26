
from app.database import get_db
from fastapi import APIRouter, Depends, Query

from app.common.schemas import ResponseModal
from app.request import Request
from app.features.docSathi.schema import PreSignedUrl , UploadDocuments, documentsIds
from app.features.docSathi.repository import check_doc_status, generate_presigned_url, get_documents_by_user_id, read_and_train_private_file


router = APIRouter(tags=["DocSathi"], prefix="/docSathi")

# Generate pre signed url
@router.post("/pre-signed-url")
async def create_signed_url(
    request: Request,
    data: PreSignedUrl,
):
    print("-----------------------------------")
    return await generate_presigned_url(data.fileFormat)



# Upload documents
@router.post("/upload-document")
async def upload_documents(
    request: Request,
    data: UploadDocuments,
):
    print("-----------------------------------")
   
    return await read_and_train_private_file(data)


# get all the uploaded documents
@router.get("/get-all-documents/{user_id}")
async def get_all_documents(
    request: Request,
    user_id: str,
):
    print("---------------cdsvdsvs--------------------")
    return await get_documents_by_user_id(user_id)

# check the training document status
@router.post("/check-document-status")
async def check_document_status(
    request: Request,
    data: documentsIds,

):
    return await check_doc_status(data.document_ids)
