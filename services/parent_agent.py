from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from app.services.rag_agent import rag_agent
from app.database.db_manager import db_manager

logger = logging.getLogger(__name__)

class ParentAgent:
    def __init__(self):
        self.rag_agent = rag_agent
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
            total_interactions = len(interactions)
            retrieval_count = sum(1 for i in interactions if i["response_summary"]["retrieval_performed"])
            avg_processing_time = sum(i["response_summary"]["processing_time"] for i in interactions) / total_interactions
            
            # Most used documents
            doc_usage = {}
            for interaction in interactions:
                for doc in interaction["response_summary"]["source_documents"]:
                    doc_usage[doc] = doc_usage.get(doc, 0) + 1
            
            most_used_docs = sorted(doc_usage.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "client_id": client_id,
                "total_interactions": total_interactions,
                "retrieval_rate": retrieval_count / total_interactions if total_interactions > 0 else 0,
                "avg_processing_time": avg_processing_time,
                "most_used_documents": most_used_docs,
                "recent_queries": [i["query"] for i in interactions[-5:]]
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze interaction patterns: {e}")
            return {"error": str(e)}
    
    def should_suggest_documents(self, client_id: str, query: str) -> Dict[str, Any]:
        """Determine if we should suggest document upload based on query patterns"""
        try:
            if client_id not in self.interaction_history:
                return {"suggest": False, "reason": "No history"}
            
            interactions = self.interaction_history[client_id]
            recent_interactions = interactions[-5:]  # Last 5 interactions
            
            # Check if recent queries had low retrieval success
            low_retrieval_count = sum(1 for i in recent_interactions 
                                    if i["response_summary"]["context_chunks_used"] == 0)
            
            if low_retrieval_count >= 3:
                return {
                    "suggest": True,
                    "reason": "Multiple recent queries found no relevant documents",
                    "suggestion": "Consider uploading documents related to your questions for better answers"
                }
            
            # Check for repeated similar queries
            recent_queries = [i["query"].lower() for i in recent_interactions]
            query_lower = query.lower()
            
            similar_count = sum(1 for q in recent_queries if self._queries_similar(q, query_lower))
            
            if similar_count >= 2:
                return {
                    "suggest": True,
                    "reason": "Repeated similar queries detected",
                    "suggestion": "Upload specific documents about this topic for more detailed answers"
                }
            
            return {"suggest": False, "reason": "No suggestion needed"}
            
        except Exception as e:
            logger.error(f"Failed to determine document suggestion: {e}")
            return {"suggest": False, "error": str(e)}
    
    def _queries_similar(self, query1: str, query2: str) -> bool:
        """Simple similarity check between queries"""
        words1 = set(query1.split())
        words2 = set(query2.split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity > 0.3  # 30% word overlap threshold
    
    def process_user_query(self, client_id: str, query: str, 
                          conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Main entry point for processing user queries with tracking"""
        try:
            # Process query through RAG agent
            rag_response = self.rag_agent.process_query(
                query=query,
                conversation_history=conversation_history,
                user_id=1  # TODO: Get actual user_id
            )
            
            # Track the interaction
            self.track_interaction(client_id, query, rag_response)
            
            # Check if we should suggest document upload
            suggestion = self.should_suggest_documents(client_id, query)
            
            # Combine response with tracking info
            response = {
                "response": rag_response["response"],
                "source_documents": rag_response.get("source_documents", []),
                "context_chunks_used": rag_response.get("context_chunks_used", 0),
                "processing_time": rag_response.get("processing_time", 0),
                "retrieval_performed": rag_response.get("retrieval_performed", False),
                "suggestion": suggestion if suggestion["suggest"] else None,
                "timestamp": datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Parent agent query processing failed: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your question.",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a client"""
        try:
            patterns = self.get_interaction_patterns(client_id)
            
            # Get user documents from database
            user_docs = db_manager.get_user_documents(1)  # TODO: Get actual user_id
            
            return {
                "interaction_patterns": patterns,
                "uploaded_documents": len(user_docs),
                "document_list": [{"id": doc["id"], "name": doc["doc_name"], 
                                 "status": doc["processing_status"]} for doc in user_docs]
            }
            
        except Exception as e:
            logger.error(f"Failed to get client stats: {e}")
            return {"error": str(e)}

# Global parent agent instance
parent_agent = ParentAgent()
