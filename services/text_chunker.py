from typing import List, Dict, Any
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config.settings import settings
import logging

logger = logging.getLogger(__name__)

class TextChunker:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        
        # Initialize LangChain text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-$$$$]', '', text)
        
        # Remove empty lines
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def create_chunks(self, text: str, doc_id: int, doc_name: str) -> List[Dict[str, Any]]:
        """Create chunks from text with metadata"""
        try:
            # Clean text first
            cleaned_text = self.clean_text(text)
            
            if not cleaned_text:
                logger.warning(f"No text content found for document {doc_id}")
                return []
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(cleaned_text)
            
            # Create chunk objects with metadata
            chunk_objects = []
            for i, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) < 50:  # Skip very short chunks
                    continue
                
                chunk_metadata = {
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "chunk_index": i,
                    "chunk_length": len(chunk_text),
                    "chunk_words": len(chunk_text.split()),
                    "source": "document"
                }
                
                chunk_objects.append({
                    "text": chunk_text.strip(),
                    "metadata": chunk_metadata,
                    "index": i
                })
            
            logger.info(f"Created {len(chunk_objects)} chunks for document {doc_id}")
            return chunk_objects
            
        except Exception as e:
            logger.error(f"Chunking failed for document {doc_id}: {e}")
            return []
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        if not chunks:
            return {"total_chunks": 0, "avg_length": 0, "total_words": 0}
        
        total_length = sum(len(chunk["text"]) for chunk in chunks)
        total_words = sum(len(chunk["text"].split()) for chunk in chunks)
        
        return {
            "total_chunks": len(chunks),
            "avg_length": total_length // len(chunks),
            "total_words": total_words,
            "min_length": min(len(chunk["text"]) for chunk in chunks),
            "max_length": max(len(chunk["text"]) for chunk in chunks)
        }

# Global text chunker instance
text_chunker = TextChunker()
