from typing import List, Dict, Any, Optional
import logging
from app.services.milvus_service import milvus_service
from app.services.embedding_service import embedding_service
from app.database.db_manager import db_manager

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.milvus = milvus_service
        self.embedding_service = embedding_service
    
    def initialize(self):
        """Initialize vector store"""
        try:
            self.milvus.initialize()
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            logger.warning("Vector store will operate without Milvus capabilities")
            # Don't raise the exception, allow the app to continue
    
    def store_document_chunks(self, doc_id: int) -> Dict[str, Any]:
        """Store document chunks in vector database"""
        try:
            # Check if Milvus is available
            if not self.milvus.is_available():
                return {"error": "Milvus is not available"}
            # Get chunks from PostgreSQL
            with db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, chunk_index, chunk_text, chunk_metadata 
                    FROM document_chunks 
                    WHERE doc_id = %s 
                    ORDER BY chunk_index
                """, (doc_id,))
                
                chunks = cursor.fetchall()
            
            if not chunks:
                return {"error": "No chunks found for document"}
            
            # Get document info
            documents = db_manager.get_user_documents(1)  # TODO: Get actual user_id
            document = next((doc for doc in documents if doc['id'] == doc_id), None)
            
            if not document:
                return {"error": "Document not found"}
            
            # Prepare chunks for Milvus
            chunks_data = []
            for chunk in chunks:
                chunk_id, chunk_index, chunk_text, chunk_metadata = chunk
                
                # Get embedding from metadata or generate new one
                embedding = chunk_metadata.get('embedding')
                if not embedding:
                    embedding = self.embedding_service.generate_single_embedding(chunk_text)
                    if not embedding:
                        continue
                
                chunk_data = {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                    "doc_name": document['doc_name'],
                    "metadata": chunk_metadata,
                    "embedding": embedding
                }
                chunks_data.append(chunk_data)
            
            # Insert into Milvus
            inserted_ids = self.milvus.insert_chunks(chunks_data)
            
            # Update chunk records with Milvus IDs
            for i, milvus_id in enumerate(inserted_ids):
                chunk_id = chunks_data[i]["chunk_id"]
                with db_manager.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE document_chunks 
                        SET vector_id = %s 
                        WHERE id = %s
                    """, (str(milvus_id), chunk_id))
                    conn.commit()
            
            return {
                "success": True,
                "doc_id": doc_id,
                "chunks_stored": len(inserted_ids),
                "milvus_ids": inserted_ids
            }
            
        except Exception as e:
            logger.error(f"Failed to store chunks in vector database: {e}")
            return {"error": str(e)}
    
    def search_relevant_chunks(self, query: str, top_k: int = 5, 
                             doc_filter: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Search for relevant chunks based on query"""
        try:
            # Check if Milvus is available
            if not self.milvus.is_available():
                logger.warning("Milvus is not available, returning empty results")
                return []
            # Generate query embedding
            query_embedding = self.embedding_service.generate_single_embedding(query)
            if not query_embedding:
                return []
            
            # Search in Milvus
            results = self.milvus.search_similar_chunks(
                query_embedding=query_embedding,
                top_k=top_k,
                doc_filter=doc_filter
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_document_vectors(self, doc_id: int) -> bool:
        """Delete all vectors for a document"""
        try:
            return self.milvus.delete_document_chunks(doc_id)
        except Exception as e:
            logger.error(f"Failed to delete document vectors: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            milvus_stats = self.milvus.get_collection_stats()
            embedding_info = self.embedding_service.get_model_info()
            
            return {
                "vector_database": milvus_stats,
                "embedding_model": embedding_info
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

# Global vector store instance
vector_store = VectorStore()
