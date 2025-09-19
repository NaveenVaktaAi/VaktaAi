from typing import List, Dict, Any, Optional
import logging
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re

logger = logging.getLogger(__name__)

class QueryRewriter:
    def __init__(self):
        self.llm = OpenAI(temperature=0.1)
        
        # Query rewriting prompt template
        self.rewrite_prompt = PromptTemplate(
            input_variables=["original_query", "conversation_history"],
            template="""
You are an expert query rewriter for a RAG (Retrieval-Augmented Generation) system.
Your task is to rewrite user queries to improve document retrieval accuracy and reduce hallucination.

Conversation History:
{conversation_history}

Original Query: {original_query}

Please rewrite this query to:
1. Make it more specific and searchable
2. Add relevant context from conversation history if needed
3. Break down complex questions into focused search terms
4. Remove ambiguous language
5. Ensure it targets factual information

Rewritten Query:"""
        )
        
        self.rewrite_chain = LLMChain(
            llm=self.llm,
            prompt=self.rewrite_prompt
        )
        
        # Query expansion prompt for multiple search angles
        self.expand_prompt = PromptTemplate(
            input_variables=["query"],
            template="""
Given this search query: {query}

Generate 2-3 alternative phrasings or related queries that would help find relevant information:

1. 
2. 
3. 

Keep each alternative concise and focused on the same topic but with different keywords or perspectives.
"""
        )
        
        self.expand_chain = LLMChain(
            llm=self.llm,
            prompt=self.expand_prompt
        )
    
    def rewrite_query(self, original_query: str, conversation_history: List[Dict] = None) -> str:
        """Rewrite query for better retrieval"""
        try:
            # Format conversation history
            history_text = ""
            if conversation_history:
                recent_history = conversation_history[-5:]  # Last 5 messages
                for msg in recent_history:
                    role = msg.get('sender', 'user')
                    content = msg.get('content', '')
                    history_text += f"{role}: {content}\n"
            
            # Rewrite the query
            rewritten = self.rewrite_chain.run(
                original_query=original_query,
                conversation_history=history_text
            )
            
            # Clean up the response
            rewritten = rewritten.strip()
            
            logger.info(f"Query rewritten: '{original_query}' -> '{rewritten}'")
            return rewritten
            
        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            return original_query
    
    def expand_query(self, query: str) -> List[str]:
        """Generate alternative query phrasings"""
        try:
            expanded = self.expand_chain.run(query=query)
            
            # Parse the numbered list
            alternatives = []
            lines = expanded.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if re.match(r'^\d+\.', line):
                    # Remove number and clean
                    alt_query = re.sub(r'^\d+\.\s*', '', line).strip()
                    if alt_query and alt_query != query:
                        alternatives.append(alt_query)
            
            # Include original query
            all_queries = [query] + alternatives
            
            logger.info(f"Expanded query into {len(all_queries)} variations")
            return all_queries
            
        except Exception as e:
            logger.error(f"Query expansion failed: {e}")
            return [query]
    
    def extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for filtering"""
        try:
            # Simple keyword extraction
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how', 'who'}
            
            # Extract words
            words = re.findall(r'\b\w+\b', query.lower())
            key_terms = [word for word in words if word not in stop_words and len(word) > 2]
            
            return key_terms[:10]  # Limit to top 10 terms
            
        except Exception as e:
            logger.error(f"Key term extraction failed: {e}")
            return []

# Global query rewriter instance
query_rewriter = QueryRewriter()
