from typing import List, Optional, Dict
from fastapi import APIRouter, HTTPException, Query, Path, WebSocket, WebSocketDisconnect, Depends, UploadFile, File, Form
import logging
import json
from datetime import datetime
import uuid
import os
from pathlib import Path as FilePath

from pymongo.database import Database

from app.database.session import get_db
from app.oauth import is_user_authorized
from app.features.aiAvatar.repository import AITutorRepository
from app.features.aiAvatar.schemas import (
    ConversationCreate,
    ConversationUpdate,
    ConversationResponse,
    ConversationWithMessages,
    ConversationListResponse,
    ConversationStatsResponse
)
from app.features.aiAvatar.websocket_manager import ai_tutor_websocket_manager, AITutorWebSocketResponse
from app.features.aiAvatar.bot_handler import AITutorBotMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai-tutor", tags=["AI Tutor"])

# Create static directories for file uploads if they don't exist
UPLOAD_DIR = FilePath("static/ai_tutor_images")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PDF_UPLOAD_DIR = FilePath("static/ai_tutor_pdfs")
PDF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ===== IMAGE UPLOAD ENDPOINT =====

@router.post("/upload-images")
async def upload_images(
    images: List[UploadFile] = File(...),
    conversation_id: str = Form(...)
):
    """
    Upload images for AI Tutor vision analysis
    Maximum 3 images allowed per upload
    """
    try:
        if len(images) > 3:
            raise HTTPException(status_code=400, detail="Maximum 3 images allowed")
        
        image_urls = []
        
        for image in images:
            # Validate file type
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"Invalid file type: {image.content_type}")
            
            # Generate unique filename
            file_extension = image.filename.split('.')[-1]
            unique_filename = f"{uuid.uuid4().hex[:12]}.{file_extension}"
            file_path = UPLOAD_DIR / unique_filename
            
            # Save file
            with open(file_path, "wb") as f:
                content = await image.read()
                f.write(content)
            
            # Generate URL (accessible via static files)
            image_url = f"/static/ai_tutor_images/{unique_filename}"
            image_urls.append(image_url)
            
            logger.info(f"Uploaded image: {unique_filename} for conversation: {conversation_id}")
        
        return {
            "success": True,
            "message": f"Uploaded {len(image_urls)} image(s) successfully",
            "image_urls": image_urls
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading images: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading images: {str(e)}")


# ===== PDF UPLOAD ENDPOINT =====

@router.post("/upload-pdfs")
async def upload_pdfs(
    pdfs: List[UploadFile] = File(...),
    conversation_id: str = Form(...)
):
    """
    Upload PDFs for AI Tutor document analysis
    Maximum 3 PDFs allowed per upload
    """
    try:
        if len(pdfs) > 3:
            raise HTTPException(status_code=400, detail="Maximum 3 PDFs allowed")
        
        pdf_urls = []
        
        for pdf in pdfs:
            # Validate file type
            if pdf.content_type != 'application/pdf':
                raise HTTPException(status_code=400, detail=f"Invalid file type: {pdf.content_type}. Only PDFs allowed.")
            
            # Generate unique filename
            unique_filename = f"{uuid.uuid4().hex[:12]}.pdf"
            file_path = PDF_UPLOAD_DIR / unique_filename
            
            # Save file
            with open(file_path, "wb") as f:
                content = await pdf.read()
                f.write(content)
            
            # Generate URL (accessible via static files)
            pdf_url = f"/static/ai_tutor_pdfs/{unique_filename}"
            pdf_urls.append(pdf_url)
            
            logger.info(f"Uploaded PDF: {unique_filename} for conversation: {conversation_id}")
        
        return {
            "success": True,
            "message": f"Uploaded {len(pdf_urls)} PDF(s) successfully",
            "pdf_urls": pdf_urls
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading PDFs: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading PDFs: {str(e)}")


# ===== CONVERSATION ENDPOINTS =====

@router.post("/conversation", response_model=dict)
async def create_conversation(conversation_data: ConversationCreate):
    """
    Create a new AI Tutor conversation
    
    Creates a new conversation session for AI tutoring.
    No document references needed - standalone conversation.
    Messages are stored as a list within the conversation document.
    """
    try:
        repo = AITutorRepository()
        conversation_id = await repo.create_conversation(conversation_data, 1)
        
        return {
            "success": True,
            "message": "Conversation created successfully",
            "data": {
                "conversation_id": conversation_id
            }
        }
    except Exception as e:
        logger.error(f"Error creating conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating conversation: {str(e)}")


@router.get("/conversations/user/{user_id}", response_model=ConversationListResponse)
async def get_user_conversations(
    user_id: int = Path(..., description="User ID"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Number of conversations per page"),
    status: str = Query(None, description="Filter by status (active/archived)")
):
    """Get all conversations for a user"""
    try:
        repo = AITutorRepository()
        
        if status:
            return await repo.get_conversations_by_status(user_id, status, page, limit)
        else:
            return await repo.get_user_conversations(user_id, page, limit)
    
    except Exception as e:
        logger.error(f"Error fetching user conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str = Path(..., description="Conversation ID")):
    """Get a specific conversation with its messages"""
    try:
        repo = AITutorRepository()
        conversation = await repo.get_conversation_by_id(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return ConversationResponse(
            _id=str(conversation["_id"]),
            user_id=conversation["user_id"],
            title=conversation["title"],
            status=conversation["status"],
            exam_type=conversation.get("exam_type"),
            exam_name=conversation.get("exam_name"),
            subject=conversation.get("subject"),
            topic=conversation.get("topic"),
            tags=conversation.get("tags", []),
            messages=conversation.get("messages", []),
            explain_concept=conversation.get("explain_concept"),
            practice_problem=conversation.get("practice_problem"),
            study_guide=conversation.get("study_guide"),
            key_points=conversation.get("key_points"),
            created_at=conversation["created_at"],
            updated_at=conversation["updated_at"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching conversation: {str(e)}")


@router.get("/conversations/{conversation_id}/full", response_model=ConversationWithMessages)
async def get_conversation_with_messages(
    conversation_id: str = Path(..., description="Conversation ID")
):
    """Get a conversation with its messages"""
    try:
        repo = AITutorRepository()
        
        conversation_with_messages = await repo.get_conversation_with_messages(conversation_id)
        
        if not conversation_with_messages:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return conversation_with_messages
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching conversation with messages: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching conversation with messages: {str(e)}")


@router.put("/conversations/{conversation_id}", response_model=dict)
async def update_conversation(
    conversation_id: str = Path(..., description="Conversation ID"),
    conversation_data: ConversationUpdate = None
):
    """Update a conversation (including messages)"""
    try:
        repo = AITutorRepository()
        
        # Check if conversation exists
        conversation = await repo.get_conversation_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Update conversation
        success = await repo.update_conversation(conversation_id, conversation_data)
        
        if success:
            return {
                "success": True,
                "message": "Conversation updated successfully",
                "data": {
                    "conversation_id": conversation_id
                }
            }
        else:
            return {
                "success": False,
                "message": "No fields to update",
                "data": {
                    "conversation_id": conversation_id
                }
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating conversation: {str(e)}")


@router.delete("/conversations/{conversation_id}", response_model=dict)
async def delete_conversation(conversation_id: str = Path(..., description="Conversation ID")):
    """Delete a conversation (messages are part of conversation document)"""
    try:
        repo = AITutorRepository()
        
        # Check if conversation exists
        conversation = await repo.get_conversation_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Delete conversation
        success = await repo.delete_conversation(conversation_id)
        
        if success:
            return {
                "success": True,
                "message": "Conversation deleted successfully",
                "data": {
                    "conversation_id": conversation_id
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete conversation")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")


# ===== STATISTICS ENDPOINT =====

@router.get("/users/{user_id}/stats", response_model=ConversationStatsResponse)
async def get_user_conversation_stats(user_id: int = Path(..., description="User ID")):
    """Get conversation statistics for a user"""
    try:
        repo = AITutorRepository()
        stats = await repo.get_conversation_stats(user_id)
        return stats
    
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {str(e)}")


# ===== EXAM-TYPE CONVERSATIONS ENDPOINT =====

@router.get("/conversations/exam/{user_id}/{exam_type}", response_model=Dict[str, List[ConversationResponse]])
async def get_exam_conversations_grouped_by_subject(
    user_id: int = Path(..., description="User ID"),
    exam_type: str = Path(..., description="Exam type (IIT JEE, NEET, etc.)")
):
    """Get conversations by exam_type grouped by subject"""
    try:
        repo = AITutorRepository()
        grouped_conversations = await repo.get_conversations_by_exam_type_grouped_by_subject(user_id, exam_type)
        return grouped_conversations
    
    except Exception as e:
        logger.error(f"Error fetching exam conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching exam conversations: {str(e)}")


# ===== SEARCH ENDPOINT =====

@router.get("/conversations/search/{user_id}", response_model=List[ConversationResponse])
async def search_conversations(
    user_id: int = Path(..., description="User ID"),
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results")
):
    """Search conversations by title"""
    try:
        repo = AITutorRepository()
        conversations = await repo.search_conversations(user_id, q, limit)
        return conversations
    
    except Exception as e:
        logger.error(f"Error searching conversations: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching conversations: {str(e)}")


# ===== QUICK ACTION ENDPOINTS =====
# Practice Problem, Study Guide, Key Points

@router.post("/practice-problem")
async def practice_problem(
    conversation_id: str = Form(..., description="Conversation ID"),
    subject: Optional[str] = Form(None, description="Subject (e.g., Physics, Math)"),
    topic: Optional[str] = Form(None, description="Topic (e.g., Newton's Laws, Calculus)")
):
    """
    Generate a practice problem using web search and OpenAI.
    Saves to conversation as a special message type.
    """
    try:
        import asyncio
        from langchain_community.tools.tavily_search import TavilySearchResults
        from app.features.chat.utils.response import ResponseCreator
        from datetime import datetime
        from app.database.ai_tutor_collections import add_message_to_conversation
        
        logger.info(f"[PRACTICE PROBLEM] Starting for conversation {conversation_id}, Subject: {subject}, Topic: {topic}")
        
        if not subject and not topic:
            raise HTTPException(status_code=400, detail="Either subject or topic must be provided")
        
        repo = AITutorRepository()
        conversation = await repo.get_conversation_by_id(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        search_query = f"{subject} {topic} practice problems examples".strip() if subject and topic else (topic or subject)
        logger.info(f"[PRACTICE PROBLEM] Search query: {search_query}")
        
        # Perform web search
        web_results = []
        try:
            TAVILY_API_KEY = "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"
            tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
            loop = asyncio.get_event_loop()
            web_results = await loop.run_in_executor(None, tool.invoke, search_query)
            
            if web_results:
                logger.info(f"[PRACTICE PROBLEM] ✅ Web search completed: {len(web_results)} results")
            else:
                logger.warning("[PRACTICE PROBLEM] ⚠️ No web search results found")
        except Exception as e:
            logger.error(f"[PRACTICE PROBLEM] ❌ Web search failed: {e}")
            web_results = []
        
        # Generate practice problem
        try:
            web_context = ""
            if web_results:
                top_results = web_results[:3]
                web_context = "\n\n".join([
                    f"**{result.get('title', 'Source')}**:\n{result.get('content', '')}"
                    for result in top_results
                ])
            
            prompt = [
                {
                    "role": "system",
                    "content": f"""You are an expert {subject or 'teacher'} creating practice problems for students.

Your task is to create a well-structured practice problem that:
- Is appropriate for the student's level
- Tests understanding of key concepts
- Includes clear instructions
- Has a step-by-step solution
- Uses real-world applications when possible

Format your response with markdown:
- Use headings for Problem Statement and Solution
- Use code blocks for formulas and calculations
- Use bullet points for steps
- Make it engaging and educational"""
                },
                {
                    "role": "user",
                    "content": f"""Create a practice problem for:

**Subject:** {subject or 'General'}
**Topic:** {topic or 'General'}

### Web Search Context (for accuracy):
{web_context if web_context else "No additional web context available (using your knowledge)"}

Provide a complete practice problem with:
1. Problem Statement (clear and well-defined)
2. Step-by-step Solution (detailed explanation)
3. Key Takeaways (what the student should learn)

Your Practice Problem:"""
                }
            ]
            
            logger.info("[PRACTICE PROBLEM] 🤖 Generating practice problem with OpenAI...")
            problem = await ResponseCreator().gpt_response_without_stream(prompt)
            
            if not problem or not problem.strip():
                raise Exception("Empty problem generated")
            
            logger.info(f"[PRACTICE PROBLEM] ✅ Problem generated: {len(problem)} characters")
            
            # Update conversation with practice_problem field (directly in database)
            # NOTE: We do NOT add this as a chat message - it should only appear in the quick action modal
            from app.database.ai_tutor_collections import update_conversation
            update_fields = {
                "practice_problem": {
                    "subject": subject,
                    "topic": topic,
                    "problem": problem,
                    "created_at": datetime.utcnow()
                }
            }
            update_conversation(repo.db, conversation_id, update_fields)
            
            logger.info(f"[PRACTICE PROBLEM] ✅ Problem saved to conversation {conversation_id}")
            
            return {
                "success": True,
                "message": "Practice problem generated successfully",
                "data": {
                    "problem": problem,
                    "subject": subject,
                    "topic": topic,
                    "conversation_id": conversation_id,
                    "web_results_used": len(web_results) > 0
                }
            }
            
        except Exception as e:
            logger.error(f"[PRACTICE PROBLEM] ❌ Error generating problem: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating practice problem: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PRACTICE PROBLEM] ❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating practice problem: {str(e)}")


@router.post("/study-guide")
async def study_guide(
    conversation_id: str = Form(..., description="Conversation ID"),
    subject: Optional[str] = Form(None, description="Subject (e.g., Physics, Math)"),
    topic: Optional[str] = Form(None, description="Topic (e.g., Newton's Laws, Calculus)")
):
    """
    Generate a comprehensive study guide using web search and OpenAI.
    Saves to conversation as a special message type.
    """
    try:
        import asyncio
        from langchain_community.tools.tavily_search import TavilySearchResults
        from app.features.chat.utils.response import ResponseCreator
        from datetime import datetime
        from app.database.ai_tutor_collections import add_message_to_conversation
        
        logger.info(f"[STUDY GUIDE] Starting for conversation {conversation_id}, Subject: {subject}, Topic: {topic}")
        
        if not subject and not topic:
            raise HTTPException(status_code=400, detail="Either subject or topic must be provided")
        
        repo = AITutorRepository()
        conversation = await repo.get_conversation_by_id(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        search_query = f"{subject} {topic} study guide summary".strip() if subject and topic else (topic or subject)
        logger.info(f"[STUDY GUIDE] Search query: {search_query}")
        
        # Perform web search
        web_results = []
        try:
            TAVILY_API_KEY = "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"
            tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
            loop = asyncio.get_event_loop()
            web_results = await loop.run_in_executor(None, tool.invoke, search_query)
            
            if web_results:
                logger.info(f"[STUDY GUIDE] ✅ Web search completed: {len(web_results)} results")
            else:
                logger.warning("[STUDY GUIDE] ⚠️ No web search results found")
        except Exception as e:
            logger.error(f"[STUDY GUIDE] ❌ Web search failed: {e}")
            web_results = []
        
        # Generate study guide
        try:
            web_context = ""
            if web_results:
                top_results = web_results[:3]
                web_context = "\n\n".join([
                    f"**{result.get('title', 'Source')}**:\n{result.get('content', '')}"
                    for result in top_results
                ])
            
            prompt = [
                {
                    "role": "system",
                    "content": f"""You are an expert {subject or 'teacher'} creating comprehensive study guides for students.

Your task is to create a well-organized study guide that:
- Covers all key concepts systematically
- Includes definitions, formulas, and important facts
- Has clear sections and subsections
- Includes examples and applications
- Provides learning tips and mnemonics
- Is easy to review and memorize

Format your response with markdown:
- Use clear headings (##, ###) for sections
- Use bullet points for lists
- Use **bold** for important terms
- Use code blocks for formulas
- Make it comprehensive but organized"""
                },
                {
                    "role": "user",
                    "content": f"""Create a comprehensive study guide for:

**Subject:** {subject or 'General'}
**Topic:** {topic or 'General'}

### Web Search Context (for accuracy):
{web_context if web_context else "No additional web context available (using your knowledge)"}

Provide a complete study guide with:
1. Overview (what this topic covers)
2. Key Concepts (definitions and explanations)
3. Important Formulas/Equations (if applicable)
4. Examples and Applications
5. Study Tips and Mnemonics
6. Summary (key takeaways)

Your Study Guide:"""
                }
            ]
            
            logger.info("[STUDY GUIDE] 🤖 Generating study guide with OpenAI...")
            guide = await ResponseCreator().gpt_response_without_stream(prompt)
            
            if not guide or not guide.strip():
                raise Exception("Empty study guide generated")
            
            logger.info(f"[STUDY GUIDE] ✅ Study guide generated: {len(guide)} characters")
            
            # Update conversation with study_guide field (directly in database)
            # NOTE: We do NOT add this as a chat message - it should only appear in the quick action modal
            from app.database.ai_tutor_collections import update_conversation
            update_fields = {
                "study_guide": {
                    "subject": subject,
                    "topic": topic,
                    "guide": guide,
                    "created_at": datetime.utcnow()
                }
            }
            update_conversation(repo.db, conversation_id, update_fields)
            
            logger.info(f"[STUDY GUIDE] ✅ Study guide saved to conversation {conversation_id}")
            
            return {
                "success": True,
                "message": "Study guide generated successfully",
                "data": {
                    "guide": guide,
                    "subject": subject,
                    "topic": topic,
                    "conversation_id": conversation_id,
                    "web_results_used": len(web_results) > 0
                }
            }
            
        except Exception as e:
            logger.error(f"[STUDY GUIDE] ❌ Error generating study guide: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating study guide: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[STUDY GUIDE] ❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating study guide: {str(e)}")


@router.post("/key-points")
async def key_points(
    conversation_id: str = Form(..., description="Conversation ID"),
    subject: Optional[str] = Form(None, description="Subject (e.g., Physics, Math)"),
    topic: Optional[str] = Form(None, description="Topic (e.g., Newton's Laws, Calculus)")
):
    """
    Generate key points summary using web search and OpenAI.
    Saves to conversation as a special message type.
    """
    try:
        import asyncio
        from langchain_community.tools.tavily_search import TavilySearchResults
        from app.features.chat.utils.response import ResponseCreator
        from datetime import datetime
        from app.database.ai_tutor_collections import add_message_to_conversation
        
        logger.info(f"[KEY POINTS] Starting for conversation {conversation_id}, Subject: {subject}, Topic: {topic}")
        
        if not subject and not topic:
            raise HTTPException(status_code=400, detail="Either subject or topic must be provided")
        
        repo = AITutorRepository()
        conversation = await repo.get_conversation_by_id(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        search_query = f"{subject} {topic} key points summary".strip() if subject and topic else (topic or subject)
        logger.info(f"[KEY POINTS] Search query: {search_query}")
        
        # Perform web search
        web_results = []
        try:
            TAVILY_API_KEY = "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"
            tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
            loop = asyncio.get_event_loop()
            web_results = await loop.run_in_executor(None, tool.invoke, search_query)
            
            if web_results:
                logger.info(f"[KEY POINTS] ✅ Web search completed: {len(web_results)} results")
            else:
                logger.warning("[KEY POINTS] ⚠️ No web search results found")
        except Exception as e:
            logger.error(f"[KEY POINTS] ❌ Web search failed: {e}")
            web_results = []
        
        # Generate key points
        try:
            web_context = ""
            if web_results:
                top_results = web_results[:3]
                web_context = "\n\n".join([
                    f"**{result.get('title', 'Source')}**:\n{result.get('content', '')}"
                    for result in top_results
                ])
            
            prompt = [
                {
                    "role": "system",
                    "content": f"""You are an expert {subject or 'teacher'} summarizing key points for students.

Your task is to create a concise, memorable list of key points that:
- Highlights the most important concepts
- Is easy to remember and review
- Uses clear, simple language
- Covers essential facts, formulas, and principles
- Is organized logically
- Can be used for quick revision

Format your response with markdown:
- Use clear headings for organization
- Use numbered or bulleted lists
- Use **bold** for key terms
- Use code blocks for formulas
- Keep it concise but comprehensive"""
                },
                {
                    "role": "user",
                    "content": f"""Create key points summary for:

**Subject:** {subject or 'General'}
**Topic:** {topic or 'General'}

### Web Search Context (for accuracy):
{web_context if web_context else "No additional web context available (using your knowledge)"}

Provide a well-organized list of key points that students should remember:
1. Main Concepts (3-5 most important ideas)
2. Key Definitions (essential terms)
3. Important Formulas/Equations (if applicable)
4. Practical Applications (real-world uses)
5. Quick Tips (memory aids or shortcuts)

Your Key Points:"""
                }
            ]
            
            logger.info("[KEY POINTS] 🤖 Generating key points with OpenAI...")
            key_points_text = await ResponseCreator().gpt_response_without_stream(prompt)
            
            if not key_points_text or not key_points_text.strip():
                raise Exception("Empty key points generated")
            
            logger.info(f"[KEY POINTS] ✅ Key points generated: {len(key_points_text)} characters")
            
            # Update conversation with key_points field (directly in database)
            # NOTE: We do NOT add this as a chat message - it should only appear in the quick action modal
            from app.database.ai_tutor_collections import update_conversation
            update_fields = {
                "key_points": {
                    "subject": subject,
                    "topic": topic,
                    "key_points": key_points_text,
                    "created_at": datetime.utcnow()
                }
            }
            update_conversation(repo.db, conversation_id, update_fields)
            
            logger.info(f"[KEY POINTS] ✅ Key points saved to conversation {conversation_id}")
            
            return {
                "success": True,
                "message": "Key points generated successfully",
                "data": {
                    "key_points": key_points_text,
                    "subject": subject,
                    "topic": topic,
                    "conversation_id": conversation_id,
                    "web_results_used": len(web_results) > 0
                }
            }
            
        except Exception as e:
            logger.error(f"[KEY POINTS] ❌ Error generating key points: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating key points: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[KEY POINTS] ❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating key points: {str(e)}")


# ===== EXPLAIN CONCEPT ENDPOINT =====

@router.post("/explain-concept")
async def explain_concept(
    conversation_id: str = Form(..., description="Conversation ID"),
    subject: Optional[str] = Form(None, description="Subject (e.g., Physics, Math)"),
    topic: Optional[str] = Form(None, description="Topic (e.g., Newton's Laws, Calculus)")
):
    """
    Explain a concept using web search and OpenAI.
    Saves explanation to conversation as a special message type.
    """
    try:
        import asyncio
        from langchain_community.tools.tavily_search import TavilySearchResults
        from app.features.chat.utils.response import ResponseCreator
        
        logger.info(f"[EXPLAIN CONCEPT] Starting for conversation {conversation_id}, Subject: {subject}, Topic: {topic}")
        
        # Validate inputs
        if not subject and not topic:
            raise HTTPException(status_code=400, detail="Either subject or topic must be provided")
        
        # Get conversation
        repo = AITutorRepository()
        conversation = await repo.get_conversation_by_id(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        user_id = conversation.get("user_id")
        
        # Construct search query
        search_query = f"{subject} {topic}".strip() if subject and topic else (topic or subject)
        logger.info(f"[EXPLAIN CONCEPT] Search query: {search_query}")
        
        # Perform web search
        web_results = []
        try:
            TAVILY_API_KEY = "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"
            tool = TavilySearchResults(tavily_api_key=TAVILY_API_KEY)
            
            loop = asyncio.get_event_loop()
            web_results = await loop.run_in_executor(None, tool.invoke, search_query)
            
            if web_results:
                logger.info(f"[EXPLAIN CONCEPT] ✅ Web search completed: {len(web_results)} results")
                logger.info(f"[EXPLAIN CONCEPT] First result: {web_results[0].get('title', 'No title')}")
            else:
                logger.warning("[EXPLAIN CONCEPT] ⚠️ No web search results found")
        except Exception as e:
            logger.error(f"[EXPLAIN CONCEPT] ❌ Web search failed: {e}")
            web_results = []
        
        # Generate explanation using OpenAI
        try:
            web_context = ""
            if web_results:
                # Combine top 3 results
                top_results = web_results[:3]
                web_context = "\n\n".join([
                    f"**{result.get('title', 'Source')}**:\n{result.get('content', '')}"
                    for result in top_results
                ])
            
            # Create comprehensive prompt for explanation (as messages array format)
            explanation_prompt = [
                {
                    "role": "system",
                    "content": f"""You are an expert {subject or 'teacher'} explaining concepts to students in a clear, engaging, and comprehensive way.

Your task is to provide a well-structured explanation that covers:
- **Introduction**: What is this concept?
- **Key Points**: Main ideas and principles
- **Examples**: Real-world examples or analogies
- **Why it matters**: Practical applications or importance
- **Summary**: Key takeaways

Use simple language that students can understand, break down complex ideas into digestible parts, and make it engaging and educational.
Keep it comprehensive but not overwhelming (aim for 300-500 words).
Use markdown formatting for better readability."""
                },
                {
                    "role": "user",
                    "content": f"""Please explain the following concept:

{'**Subject:** ' + subject if subject else ''}
{'**Topic:** ' + topic if topic else ''}

### Web Search Context (for accuracy):
{web_context if web_context else "No additional web context available (using your knowledge)"}

Provide a clear, well-structured explanation with markdown formatting:
- Use headings (##, ###) for sections
- Use bullet points for lists
- Use **bold** for important terms
- Use code blocks for formulas or technical terms

Your Explanation:"""
                }
            ]
            
            logger.info("[EXPLAIN CONCEPT] 🤖 Generating explanation with OpenAI...")
            explanation = await ResponseCreator().gpt_response_without_stream(explanation_prompt)
            
            if not explanation or not explanation.strip():
                raise Exception("Empty explanation generated")
            
            logger.info(f"[EXPLAIN CONCEPT] ✅ Explanation generated: {len(explanation)} characters")
            
            # Update conversation with explain_concept field (directly in database)
            # NOTE: We do NOT add this as a chat message - it should only appear in the quick action modal
            from app.database.ai_tutor_collections import update_conversation
            update_fields = {
                "explain_concept": {
                    "subject": subject,
                    "topic": topic,
                    "explanation": explanation,
                    "created_at": datetime.utcnow()
                }
            }
            update_conversation(repo.db, conversation_id, update_fields)
            
            logger.info(f"[EXPLAIN CONCEPT] ✅ Explanation saved to conversation {conversation_id}")
            
            return {
                "success": True,
                "message": "Concept explained successfully",
                "data": {
                    "explanation": explanation,
                    "subject": subject,
                    "topic": topic,
                    "conversation_id": conversation_id,
                    "web_results_used": len(web_results) > 0
                }
            }
            
        except Exception as e:
            logger.error(f"[EXPLAIN CONCEPT] ❌ Error generating explanation: {e}")
            raise HTTPException(status_code=500, detail=f"Error generating explanation: {str(e)}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[EXPLAIN CONCEPT] ❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error explaining concept: {str(e)}")


# ===== END CONVERSATION ENDPOINT =====

@router.put("/conversation/{conversation_id}/end", response_model=dict)
async def end_conversation_api(conversation_id: str = Path(..., description="Conversation ID")):
    """
    End an active conversation and save all in-memory messages to database.
    This endpoint is called when user clicks 'Done' button.
    """
    try:
        print(f"[AI_TUTOR] REST API: Ending conversation {conversation_id}")
        
        # Check if conversation exists in database
        repo = AITutorRepository()
        conversation = await repo.get_conversation_by_id(conversation_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Check if there's an active bot message handler for this conversation
        if conversation_id in bot_message_handlers:
            bot_message = bot_message_handlers[conversation_id]
            
            # Save all in-memory messages to database
            await bot_message.save_conversation_messages()
            
            # Remove handler from memory to clear in-memory messages
            del bot_message_handlers[conversation_id]
            
            print(f"[AI_TUTOR] REST API: Conversation {conversation_id} ended and messages saved")
            
            return {
                "success": True,
                "message": "Conversation ended and all messages saved to database",
                "data": {
                    "conversation_id": conversation_id,
                    "messages_saved": True,
                    "memory_cleared": True
                }
            }
        else:
            # No active handler means no in-memory messages
            print(f"[AI_TUTOR] REST API: No active handler found for conversation {conversation_id}")
            
            return {
                "success": True,
                "message": "Conversation ended (no active in-memory messages)",
                "data": {
                    "conversation_id": conversation_id,
                    "messages_saved": False,
                    "memory_cleared": True
                }
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[AI_TUTOR] REST API: Error ending conversation: {e}")
        raise HTTPException(status_code=500, detail=f"Error ending conversation: {str(e)}")


# ===== WEBSOCKET ENDPOINT =====

@router.websocket("/conversation/ws/{conversation_id}")
async def websocket_endpoint(websocket: WebSocket, conversation_id: str):
    """
    WebSocket endpoint for real-time AI Tutor conversation.
    Pure AI tutoring without RAG/document search.
    Messages are kept in memory during session and saved when conversation ends.
    """
    try:
        print(f"[AI_TUTOR] WebSocket connected for conversation {conversation_id}")
        await connect_websocket(websocket, conversation_id)
        
        while True:
            data = await websocket.receive_text()
            print(f"[AI_TUTOR] Received data for conversation {conversation_id}")
            await process_received_data(data, conversation_id, websocket)

    except WebSocketDisconnect as e:
        print(f"[AI_TUTOR] WebSocket disconnected: {e}")
        await handle_websocket_disconnect(websocket, conversation_id)
    except Exception as e:
        print(f"[AI_TUTOR] WebSocket error: {e}")
        await handle_websocket_disconnect(websocket, conversation_id)


async def connect_websocket(websocket: WebSocket, conversation_id: str):
    """Connect WebSocket to conversation"""
    await ai_tutor_websocket_manager.connect(websocket, conversation_id)
    
    # Send connection confirmation
    await websocket.send_json({
        "mt": "connected",
        "conversationId": conversation_id,
        "message": "Connected to AI Tutor",
        "timestamp": datetime.utcnow().isoformat()
    })


async def process_received_data(data: str, conversation_id: str, websocket: WebSocket):
    """Process received WebSocket data"""
    try:
        message_data = json.loads(data)
        print(f"[AI_TUTOR] Processing message data: {message_data}")

        print(f"[AI_TUTOR] ------------ data: {data}")
        # Support both 'mt' (message type) and 'type' fields for compatibility
        message_type = message_data.get("mt") or message_data.get("type")
        
        # Map frontend message types to backend handlers
        if message_type == "message_upload" or message_type == "user_message":
            await handle_user_message(message_data, conversation_id, websocket)
        elif message_type == "typing":
            await handle_typing_indicator(message_data, conversation_id)
        elif message_type == "stop_typing":
            await handle_stop_typing(message_data, conversation_id)
        elif message_type == "end_conversation":
            await handle_end_conversation(message_data, conversation_id, websocket)
        else:
            print(f"[AI_TUTOR] Unknown message type: {message_type}")
            print(f"[AI_TUTOR] Available keys in message_data: {list(message_data.keys())}")
    
    except json.JSONDecodeError as e:
        print(f"[AI_TUTOR] Error parsing JSON: {e}")
        await websocket.send_json({
            "mt": "error",
            "error": "Invalid JSON format",
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        print(f"[AI_TUTOR] Error processing data: {e}")
        await websocket.send_json({
            "mt": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })


# Global storage for bot message handlers per conversation
bot_message_handlers = {}


async def handle_user_message(message_data: dict, conversation_id: str, websocket: WebSocket):
    """Handle user message from WebSocket"""
    try:
        print(f"[AI_TUTOR] Handling user message: {message_data}")
        user_message = message_data.get("message")
        user_id = str(message_data.get("userId", "unknown"))
        token = message_data.get("token")
        # Support both isAudio and is_audio for compatibility
        isAudio = message_data.get("isAudio", message_data.get("is_audio", True))
        # Extract images if present
        images = message_data.get("images", [])
        # Extract PDFs if present
        pdfs = message_data.get("pdfs", [])
        
        if not user_message and not images and not pdfs:
            print("[AI_TUTOR] No message content, images, or PDFs provided")
            return
        
        print(f"[AI_TUTOR] User message: {user_message}")
        print(f"[AI_TUTOR] Images: {len(images)} image(s)" if images else "[AI_TUTOR] No images")
        print(f"[AI_TUTOR] PDFs: {len(pdfs)} PDF(s)" if pdfs else "[AI_TUTOR] No PDFs")
        print(f"[AI_TUTOR] Audio mode: {'ENABLED (text + audio + summary)' if isAudio else 'DISABLED (text only)'}")
        
        # Get conversation subject and topic for context
        repo = AITutorRepository()
        conversation = await repo.get_conversation_by_id(conversation_id)
        selected_subject = conversation.get("subject", "") if conversation else ""
        selected_topic = conversation.get("topic", "") if conversation else ""
        exam_type = conversation.get("exam_type", "") if conversation else ""
        print(f"[AI_TUTOR] 📚 Selected Subject: {selected_subject if selected_subject else 'Not specified'}")
        print(f"[AI_TUTOR] 🎯 Selected Topic: {selected_topic if selected_topic else 'Not specified'}")
        print(f"[AI_TUTOR] 🎓 Exam Type: {exam_type if exam_type else 'Not specified'}")
        
        # Create WebSocket response handler
        websocket_response = AITutorWebSocketResponse(
            websocket,
            conversation_id,
            ai_tutor_websocket_manager
        )
        
        # Send user message confirmation
        await websocket_response.send_user_message_confirmation(
            user_message,
            user_id,
            token
        )
        
        # Initialize or reuse bot message handler for this conversation
        if conversation_id not in bot_message_handlers:
            bot_message_handlers[conversation_id] = AITutorBotMessage(
                conversation_id=conversation_id,
                user_id=user_id,
                websocket_response=websocket_response,
                timezone=message_data.get("timezone", "UTC"),
                language_code=message_data.get("languageCode", "en"),
                selected_subject=selected_subject,
                selected_topic=selected_topic,
                exam_type=exam_type
            )
        
        bot_message = bot_message_handlers[conversation_id]
        
        # Process and send bot response with audio flag, images, PDFs, and subject context
        await bot_message.send_bot_message(user_message, isAudio=isAudio, images=images, pdfs=pdfs)  # ✅ Pass isAudio flag, images, and PDFs
        
    except Exception as e:
        print(f"[AI_TUTOR] Error handling user message: {e}")
        await websocket.send_json({
            "mt": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })


async def handle_end_conversation(message_data: dict, conversation_id: str, websocket: WebSocket):
    """Handle end conversation - save all messages to database"""
    try:
        print(f"[AI_TUTOR] Ending conversation {conversation_id}, saving messages to DB...")
        
        # Get bot message handler for this conversation
        if conversation_id in bot_message_handlers:
            bot_message = bot_message_handlers[conversation_id]
            
            # Save all messages to database
            await bot_message.save_conversation_messages()
            
            # Remove handler from memory
            del bot_message_handlers[conversation_id]
            
            # Send confirmation
            await websocket.send_json({
                "mt": "conversation_ended",
                "conversationId": conversation_id,
                "message": "Conversation ended and messages saved",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            print(f"[AI_TUTOR] Conversation {conversation_id} ended successfully")
        else:
            print(f"[AI_TUTOR] No active handler found for conversation {conversation_id}")
            await websocket.send_json({
                "mt": "conversation_ended",
                "conversationId": conversation_id,
                "message": "Conversation ended (no messages to save)",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    except Exception as e:
        print(f"[AI_TUTOR] Error ending conversation: {e}")
        await websocket.send_json({
            "mt": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })


async def handle_typing_indicator(message_data: dict, conversation_id: str):
    """Handle typing indicator"""
    await ai_tutor_websocket_manager.send_to_conversation(
        conversation_id,
        {
            "mt": "typing_indicator",
            "conversationId": conversation_id,
            "userId": message_data.get("userId"),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


async def handle_stop_typing(message_data: dict, conversation_id: str):
    """Handle stop typing indicator"""
    await ai_tutor_websocket_manager.send_to_conversation(
        conversation_id,
        {
            "mt": "stop_typing",
            "conversationId": conversation_id,
            "userId": message_data.get("userId"),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


async def handle_websocket_disconnect(websocket: WebSocket, conversation_id: str):
    """Handle WebSocket disconnection"""
    # Save messages if there's an active handler
    if conversation_id in bot_message_handlers:
        try:
            bot_message = bot_message_handlers[conversation_id]
            await bot_message.save_conversation_messages()
            del bot_message_handlers[conversation_id]
            print(f"[AI_TUTOR] Messages saved on disconnect for conversation {conversation_id}")
        except Exception as e:
            print(f"[AI_TUTOR] Error saving messages on disconnect: {e}")
    
    ai_tutor_websocket_manager.disconnect(websocket, conversation_id)
    print(f"[AI_TUTOR] WebSocket disconnected from conversation {conversation_id}")
