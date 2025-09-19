
from app.database import get_db
from fastapi import APIRouter, Depends, Query

from app.common.schemas import ResponseModal
from app.request import Request
from app.features.docSathi.schema import PreSignedUrl , UploadDocuments
from app.features.docSathi.repository import generate_presigned_url, read_and_train_private_file


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


