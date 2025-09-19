import asyncio
from typing import List, Dict, Any, Optional
import logging
from app.services.s3_service import s3_service
from app.services.file_processor import file_processor
from app.services.text_chunker import text_chunker
from app.services.embedding_service import embedding_service
from app.database.db_manager import db_manager

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.processing_queue = asyncio.Queue()
        self.is_processing = False
    
    async def process_document(self, doc_id: int) -> Dict[str, Any]:
        """Process a document: extract text, chunk, embed, and store"""
        try:
            # Update status to processing
            db_manager.update_processing_status(doc_id, "processing")
            
            # Get document info from database
            documents = db_manager.get_user_documents(1)  # TODO: Get actual user_id
            document = next((doc for doc in documents if doc['id'] == doc_id), None)
            
            if not document:
                raise Exception(f"Document {doc_id} not found")
            
            logger.info(f"Processing document {doc_id}: {document['doc_name']}")
            
            # Step 1: Download file from S3
            file_content = s3_service.get_file_content(document['s3_url'])
            if not file_content:
                raise Exception("Failed to download file from S3")
            
            # Step 2: Extract text from file
            extracted_text = file_processor.extract_text(file_content, document['original_filename'])
            if not extracted_text:
                raise Exception("Failed to extract text from file")
            
            logger.info(f"Extracted {len(extracted_text)} characters from document")
            
            # Step 3: Create chunks
            chunks = text_chunker.create_chunks(
                text=extracted_text,
                doc_id=doc_id,
                doc_name=document['doc_name']
            )
            
            if not chunks:
                raise Exception("Failed to create chunks from text")
            
            # Step 4: Generate embeddings
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = embedding_service.generate_embeddings(chunk_texts)
            
            if len(embeddings) != len(chunks):
                raise Exception("Embedding count mismatch with chunk count")
            
            # Step 5: Store chunks in database
            stored_chunks = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Add embedding to metadata
                chunk["metadata"]["embedding"] = embedding
                
                # Store in database
                chunk_id = db_manager.insert_document_chunk(
                    doc_id=doc_id,
                    chunk_index=chunk["index"],
                    chunk_text=chunk["text"],
                    chunk_metadata=chunk["metadata"]
                )
                
                stored_chunks.append({
                    "chunk_id": chunk_id,
                    "index": chunk["index"],
                    "text_length": len(chunk["text"])
                })
            
            # Step 6: Update document status
            db_manager.update_processing_status(doc_id, "completed", len(chunks))
            
            # Get chunk statistics
            chunk_stats = text_chunker.get_chunk_stats(chunks)
            
            result = {
                "doc_id": doc_id,
                "status": "completed",
                "chunks_created": len(chunks),
                "chunks_stored": len(stored_chunks),
                "text_length": len(extracted_text),
                "chunk_stats": chunk_stats,
                "processing_time": "calculated_later"  # TODO: Add timing
            }
            
            logger.info(f"Successfully processed document {doc_id}: {len(chunks)} chunks created")
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed for {doc_id}: {e}")
            
            # Update status to failed
            db_manager.update_processing_status(doc_id, "failed")
            
            return {
                "doc_id": doc_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def process_document_async(self, doc_id: int):
        """Add document to processing queue"""
        await self.processing_queue.put(doc_id)
        
        if not self.is_processing:
            asyncio.create_task(self._process_queue())
    
    async def _process_queue(self):
        """Process documents in queue"""
        self.is_processing = True
        
        try:
            while not self.processing_queue.empty():
                doc_id = await self.processing_queue.get()
                await self.process_document(doc_id)
                self.processing_queue.task_done()
        finally:
            self.is_processing = False
    
    def get_processing_status(self, doc_id: int) -> Dict[str, Any]:
        """Get processing status for a document"""
        try:
            documents = db_manager.get_user_documents(1)  # TODO: Get actual user_id
            document = next((doc for doc in documents if doc['id'] == doc_id), None)
            
            if not document:
                return {"error": "Document not found"}
            
            return {
                "doc_id": doc_id,
                "status": document['processing_status'],
                "chunk_count": document['chunk_count'],
                "upload_date": document['upload_date'].isoformat() if document['upload_date'] else None,
                "updated_at": document['updated_at'].isoformat() if document['updated_at'] else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get processing status: {e}")
            return {"error": str(e)}

# Global document processor instance
document_processor = DocumentProcessor()
