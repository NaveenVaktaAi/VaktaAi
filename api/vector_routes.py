from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/vector", tags=["vector"])

@router.post("/store/{doc_id}")
async def store_document_vectors(doc_id: int):
    """Store document chunks in vector database"""
    try:
        result = vector_store.store_document_chunks(doc_id)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": result
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to store vectors: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/search")
async def search_vectors(
    query: str,
    top_k: int = 5,
    doc_filter: Optional[List[int]] = None
):
    """Search for relevant chunks using vector similarity"""
    try:
        results = vector_store.search_relevant_chunks(
            query=query,
            top_k=top_k,
            doc_filter=doc_filter
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }
        )
        
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/document/{doc_id}")
async def delete_document_vectors(doc_id: int):
    """Delete all vectors for a document"""
    try:
        success = vector_store.delete_document_vectors(doc_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to delete vectors")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Vectors deleted for document {doc_id}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete vectors: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/stats")
async def get_vector_stats():
    """Get vector store statistics"""
    try:
        stats = vector_store.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "stats": stats
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get vector stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
