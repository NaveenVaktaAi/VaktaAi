from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class ParentAgent:
    """
    Simplified parent agent for VaktaAi backend.
    Handles user queries and provides responses with RAG capabilities.
    """
    
    def __init__(self):
        self.interaction_history = {}  # In-memory storage for session tracking
    
    def track_interaction(self, client_id: str, query: str, response: Dict[str, Any]):
        """Track user interactions and retrieval patterns"""
        try:
            if client_id not in self.interaction_history:
                self.interaction_history[client_id] = []
            
            interaction = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "response_summary": {
                    "source_documents": response.get("source_documents", []),
                    "context_chunks_used": response.get("context_chunks_used", 0),
                    "retrieval_performed": response.get("retrieval_performed", False),
                    "processing_time": response.get("processing_time", 0)
                }
            }
            
            self.interaction_history[client_id].append(interaction)
            
            # Keep only last 50 interactions per client
            if len(self.interaction_history[client_id]) > 50:
                self.interaction_history[client_id] = self.interaction_history[client_id][-50:]
            
            logger.info(f"Tracked interaction for client {client_id}")
            
        except Exception as e:
            logger.error(f"Failed to track interaction: {e}")
    
    def get_interaction_patterns(self, client_id: str) -> Dict[str, Any]:
        """Analyze interaction patterns for a client"""
        try:
            if client_id not in self.interaction_history:
                return {"message": "No interaction history found"}
            
            interactions = self.interaction_history[client_id]
            
            # Calculate statistics
            total_queries = len(interactions)
            total_context_chunks = sum(
                i["response_summary"]["context_chunks_used"] 
                for i in interactions
            )
            retrieval_success_rate = sum(
                1 for i in interactions 
                if i["response_summary"]["retrieval_performed"]
            ) / total_queries if total_queries > 0 else 0
            
            avg_processing_time = sum(
                i["response_summary"]["processing_time"] 
                for i in interactions
            ) / total_queries if total_queries > 0 else 0
            
            return {
                "total_queries": total_queries,
                "total_context_chunks_used": total_context_chunks,
                "retrieval_success_rate": round(retrieval_success_rate, 2),
                "avg_processing_time": round(avg_processing_time, 3),
                "recent_queries": [i["query"] for i in interactions[-5:]]
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze interaction patterns: {e}")
            return {"error": str(e)}
    
    def process_user_query(
        self, 
        client_id: str, 
        query: str, 
        conversation_history: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process user query and return response with metadata.
        This is a simplified version that works with the existing chat system.
        """
        try:
            start_time = datetime.now()
            
            # For now, we'll create a simple response structure
            # In a full implementation, this would integrate with RAG and document search
            response_content = self._generate_response(query, conversation_history)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create response with metadata
            response = {
                "response": response_content,
                "source_documents": [],  # Would be populated by RAG system
                "context_chunks_used": 0,  # Would be populated by RAG system
                "retrieval_performed": False,  # Would be true if RAG was used
                "processing_time": processing_time,
                "suggestion": self._generate_suggestion(query)
            }
            
            # Track the interaction
            self.track_interaction(client_id, query, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing user query: {e}")
            return {
                "response": "I'm sorry, I encountered an error while processing your request. Please try again.",
                "source_documents": [],
                "context_chunks_used": 0,
                "retrieval_performed": False,
                "processing_time": 0,
                "suggestion": None,
                "error": str(e)
            }
    
    def _generate_response(self, query: str, conversation_history: List[Dict] = None) -> str:
        """
        Generate a response based on the user query.
        This is a placeholder that would be replaced with actual RAG/LLM integration.
        """
        # Simple response logic - in production this would use actual RAG
        query_lower = query.lower()
        
        if "hello" in query_lower or "hi" in query_lower:
            return "Hello! I'm your AI assistant. I can help you with questions about your documents and provide information on various topics."
        elif "document" in query_lower or "upload" in query_lower:
            return "I can help you search through your uploaded documents. Please upload documents through the Documents tab to enable document-based responses."
        elif "help" in query_lower:
            return "I'm here to help! You can ask me questions about your documents, request information on various topics, or get assistance with general queries."
        elif "thank" in query_lower:
            return "You're welcome! Feel free to ask if you need any more assistance."
        else:
            return f"I understand you're asking about: '{query}'. I'm currently operating in a simplified mode. For full RAG capabilities with document search, please ensure documents are uploaded and the system is properly configured."
    
    def _generate_suggestion(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Generate follow-up suggestions based on the query.
        """
        query_lower = query.lower()
        
        if "document" in query_lower:
            return {
                "suggest": True,
                "reason": "Document-related query",
                "suggestion": "Would you like to upload a document to get more specific answers?"
            }
        elif "help" in query_lower:
            return {
                "suggest": True,
                "reason": "Help request",
                "suggestion": "Try asking about specific topics or uploading documents for document-based assistance."
            }
        
        return None

# Global parent agent instance
parent_agent = ParentAgent()
