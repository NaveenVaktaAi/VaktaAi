# from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
# from fastapi.responses import JSONResponse
# from typing import List, Optional, Tuple
# import logging
# from app.services.s3_service import s3_service
# from app.services.file_processor import file_processor
# from app.database.db_manager import db_manager

# logger = logging.getLogger(__name__)
# router = APIRouter(prefix="/api/upload", tags=["upload"])

# @router.post("/document")
# async def upload_document(
#     file: UploadFile = File(...),
#     user_id: int = 1  # TODO: Get from authentication
# ):
#     """Upload document and store metadata"""
#     try:
#         # Read file content
#         file_content = await file.read()
#         file_size = len(file_content)
        
#         # Validate file
#         is_valid, message = file_processor.validate_file(file.filename, file_size)
#         if not is_valid:
#             raise HTTPException(status_code=400, detail=message)
        
#         # Upload to S3
#         success, s3_url = s3_service.upload_file(
#             file_content=file_content,
#             original_filename=file.filename,
#             content_type=file.content_type
#         )
        
#         if not success:
#             raise HTTPException(status_code=500, detail="Failed to upload file to storage")
        
#         # Get file metadata
#         metadata = file_processor.get_file_metadata(
#             filename=file.filename,
#             file_size=file_size,
#             content_type=file.content_type
#         )
        
#         # Store in database
#         doc_id = db_manager.insert_training_doc(
#             user_id=user_id,
#             doc_name=file.filename,
#             original_filename=file.filename,
#             file_type=metadata['file_type'],
#             file_size=file_size,
#             s3_url=s3_url
#         )
        
#         logger.info(f"Document uploaded successfully: {doc_id}")
        
#         return JSONResponse(
#             status_code=200,
#             content={
#                 "success": True,
#                 "message": "Document uploaded successfully",
#                 "doc_id": doc_id,
#                 "s3_url": s3_url,
#                 "metadata": metadata
#             }
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Upload failed: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# @router.get("/documents")
# async def get_user_documents(user_id: int = 1):
#     """Get all documents for a user"""
#     try:
#         documents = db_manager.get_user_documents(user_id)
        
#         return JSONResponse(
#             status_code=200,
#             content={
#                 "success": True,
#                 "documents": [dict(doc) for doc in documents]
#             }
#         )
        
#     except Exception as e:
#         logger.error(f"Failed to get documents: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# @router.delete("/document/{doc_id}")
# async def delete_document(doc_id: int, user_id: int = 1):
#     """Delete document and its S3 file"""
#     try:
#         # Get document info
#         documents = db_manager.get_user_documents(user_id)
#         document = next((doc for doc in documents if doc['id'] == doc_id), None)
        
#         if not document:
#             raise HTTPException(status_code=404, detail="Document not found")
        
#         # Delete from S3
#         s3_service.delete_file(document['s3_url'])
        
#         # Delete from database (CASCADE will handle chunks)
#         with db_manager.get_connection() as conn:
#             cursor = conn.cursor()
#             cursor.execute("DELETE FROM training_doc WHERE id = %s AND user_id = %s", (doc_id, user_id))
#             conn.commit()
        
#         return JSONResponse(
#             status_code=200,
#             content={
#                 "success": True,
#                 "message": "Document deleted successfully"
#             }
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Delete failed: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# @router.post("/process/{doc_id}")
# async def process_document(doc_id: int, user_id: int = 1):
#     """Trigger document processing for chunking and embedding"""
#     try:
#         # Get document info
#         documents = db_manager.get_user_documents(user_id)
#         document = next((doc for doc in documents if doc['id'] == doc_id), None)
        
#         if not document:
#             raise HTTPException(status_code=404, detail="Document not found")
        
#         # Update status to processing
#         db_manager.update_processing_status(doc_id, "processing")
        
#         return JSONResponse(
#             status_code=200,
#             content={
#                 "success": True,
#                 "message": "Document processing started",
#                 "doc_id": doc_id
#             }
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Processing failed: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")
