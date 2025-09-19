from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging
from app.services.document_processor import document_processor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/processing", tags=["processing"])

@router.post("/document/{doc_id}")
async def process_document(doc_id: int, background_tasks: BackgroundTasks):
    """Start document processing in background"""
    try:
        # Add to background processing
        background_tasks.add_task(document_processor.process_document, doc_id)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Document processing started",
                "doc_id": doc_id
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/status/{doc_id}")
async def get_processing_status(doc_id: int):
    """Get processing status for a document"""
    try:
        status = document_processor.get_processing_status(doc_id)
        
        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "status": status
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get processing status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/stats")
async def get_processing_stats():
    """Get overall processing statistics"""
    try:
        # TODO: Implement processing statistics
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "stats": {
                    "total_documents": 0,
                    "processed_documents": 0,
                    "failed_documents": 0,
                    "total_chunks": 0
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get processing stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
