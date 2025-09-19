from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional
import logging
from app.services.parent_agent import parent_agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag", tags=["rag"])

@router.post("/query")
async def process_rag_query(
    query: str,
    client_id: str,
    conversation_history: Optional[List[Dict]] = None
):
    """Process query through Agentic RAG system"""
    try:
        response = parent_agent.process_user_query(
            client_id=client_id,
            query=query,
            conversation_history=conversation_history or []
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "result": response
            }
        )
        
    except Exception as e:
        logger.error(f"RAG query processing failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/stats/{client_id}")
async def get_client_stats(client_id: str):
    """Get client interaction statistics"""
    try:
        stats = parent_agent.get_client_stats(client_id)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "stats": stats
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get client stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/patterns/{client_id}")
async def get_interaction_patterns(client_id: str):
    """Get interaction patterns for a client"""
    try:
        patterns = parent_agent.get_interaction_patterns(client_id)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "patterns": patterns
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get interaction patterns: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
