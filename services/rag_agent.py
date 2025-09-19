from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.services.query_rewriter import query_rewriter
from app.services.vector_store import vector_store
from app.database.db_manager import db_manager

logger = logging.getLogger(__name__)

class RAGAgent:
    def __init__(self):
        self.llm = OpenAI(temperature=0.3)
        
        # RAG response generation prompt
        self.rag_prompt = PromptTemplate(
            input_variables=["query", "context", "conversation_history"],
            template="""
You are an AI assistant that answers questions based on provided context from documents.

Conversation History:
{conversation_history}

User Question: {query}

Relevant Context from Documents:
{context}

Instructions:
1. Answer the question using ONLY the information provided in the context
2. If the context doesn't contain enough information, say so clearly
3. Cite which document(s) your answer comes from
4. Be concise but comprehensive
5. If there are conflicting information in different documents, mention this
6. Do not make up information not present in the context

Answer:"""
        )
        
        self.rag_chain = LLMChain(
            llm=self.llm,
            prompt=self.rag_prompt
        )
        
        # Query analysis prompt
        self.analysis_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
Analyze this user query and determine:
1. Is this a factual question that would benefit from document retrieval?
2. What type of information is the user looking for?
3. How specific or general is the query?

Query: {query}

Analysis:
- Needs retrieval: (Yes/No)
- Information type: 
- Specificity level: (High/Medium/Low)
- Key topics:
"""
        )
        
        self.analysis_chain = LLMChain(
            llm=self.llm,
            prompt=self.analysis_prompt
        )
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine retrieval strategy"""
        try:
            analysis = self.analysis_chain.run(query=query)
            
            # Parse analysis (simplified)
            needs_retrieval = "yes" in analysis.lower()
            
            return {
                "needs_retrieval": needs_retrieval,
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            return {
                "needs_retrieval": True,  # Default to retrieval
                "analysis": "Analysis failed",
                "timestamp": datetime.now().isoformat()
            }
    
    def retrieve_context(self, query: str, conversation_history: List[Dict] = None, 
                        top_k: int = 5) -> Tuple[List[Dict], Dict[str, Any]]:
        """Retrieve relevant context using multi-step approach"""
        try:
            retrieval_info = {
                "original_query": query,
                "steps": [],
                "total_chunks": 0
            }
            
            # Step 1: Rewrite query
            rewritten_query = query_rewriter.rewrite_query(query, conversation_history)
            retrieval_info["rewritten_query"] = rewritten_query
            retrieval_info["steps"].append("Query rewritten")
            
            # Step 2: Expand query for multiple search angles
            query_variations = query_rewriter.expand_query(rewritten_query)
            retrieval_info["query_variations"] = query_variations
            retrieval_info["steps"].append(f"Generated {len(query_variations)} query variations")
            
            # Step 3: Search with each variation
            all_results = []
            for i, variation in enumerate(query_variations):
                results = vector_store.search_relevant_chunks(
                    query=variation,
                    top_k=max(2, top_k // len(query_variations))
                )
                
                # Add variation info to results
                for result in results:
                    result["query_variation"] = i
                    result["search_query"] = variation
                
                all_results.extend(results)
            
            # Step 4: Deduplicate and rank results
            unique_results = self._deduplicate_results(all_results)
            ranked_results = self._rank_results(unique_results, query)[:top_k]
            
            retrieval_info["total_chunks"] = len(ranked_results)
            retrieval_info["steps"].append(f"Retrieved and ranked {len(ranked_results)} unique chunks")
            
            logger.info(f"Retrieved {len(ranked_results)} relevant chunks for query")
            return ranked_results, retrieval_info
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return [], {"error": str(e)}
    
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks based on chunk_id"""
        seen_chunks = set()
        unique_results = []
        
        for result in results:
            chunk_id = result.get("chunk_id")
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_results(self, results: List[Dict], original_query: str) -> List[Dict]:
        """Rank results by relevance score and other factors"""
        try:
            # Sort by score (higher is better for cosine similarity)
            ranked = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
            
            # Add ranking information
            for i, result in enumerate(ranked):
                result["rank"] = i + 1
            
            return ranked
            
        except Exception as e:
            logger.error(f"Result ranking failed: {e}")
            return results
    
    def generate_response(self, query: str, context_chunks: List[Dict], 
                         conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Generate RAG response using retrieved context"""
        try:
            # Format context
            context_text = ""
            source_docs = set()
            
            for i, chunk in enumerate(context_chunks):
                doc_name = chunk.get("doc_name", "Unknown Document")
                text = chunk.get("text", "")
                score = chunk.get("score", 0)
                
                context_text += f"\n[Document: {doc_name}, Relevance: {score:.3f}]\n{text}\n"
                source_docs.add(doc_name)
            
            # Format conversation history
            history_text = ""
            if conversation_history:
                recent_history = conversation_history[-3:]  # Last 3 messages
                for msg in recent_history:
                    role = msg.get('sender', 'user')
                    content = msg.get('content', '')
                    history_text += f"{role}: {content}\n"
            
            # Generate response
            response = self.rag_chain.run(
                query=query,
                context=context_text,
                conversation_history=history_text
            )
            
            return {
                "response": response.strip(),
                "source_documents": list(source_docs),
                "context_chunks_used": len(context_chunks),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "response": "I apologize, but I encountered an error while generating a response.",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def process_query(self, query: str, conversation_history: List[Dict] = None, 
                     user_id: int = 1) -> Dict[str, Any]:
        """Main RAG processing pipeline"""
        try:
            start_time = datetime.now()
            
            # Step 1: Analyze query
            analysis = self.analyze_query(query)
            
            # Step 2: Retrieve context if needed
            if analysis["needs_retrieval"]:
                context_chunks, retrieval_info = self.retrieve_context(
                    query=query,
                    conversation_history=conversation_history,
                    top_k=5
                )
            else:
                context_chunks = []
                retrieval_info = {"message": "No retrieval needed"}
            
            # Step 3: Generate response
            if context_chunks:
                response_info = self.generate_response(
                    query=query,
                    context_chunks=context_chunks,
                    conversation_history=conversation_history
                )
            else:
                # Fallback to general response
                response_info = {
                    "response": "I don't have specific information about that in my knowledge base. Could you provide more context or ask about something else?",
                    "source_documents": [],
                    "context_chunks_used": 0
                }
            
            # Step 4: Log interaction
            processing_time = (datetime.now() - start_time).total_seconds()
            
            interaction_log = {
                "user_id": user_id,
                "query": query,
                "analysis": analysis,
                "retrieval_info": retrieval_info,
                "response_info": response_info,
                "processing_time_seconds": processing_time,
                "timestamp": start_time.isoformat()
            }
            
            # TODO: Store interaction log in database
            
            return {
                "response": response_info["response"],
                "source_documents": response_info.get("source_documents", []),
                "context_chunks_used": response_info.get("context_chunks_used", 0),
                "processing_time": processing_time,
                "retrieval_performed": analysis["needs_retrieval"]
            }
            
        except Exception as e:
            logger.error(f"RAG query processing failed: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your question.",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Global RAG agent instance
rag_agent = RAGAgent()
