import datetime
import json
from requests import Session
from sqlalchemy import update

from app.schemas.milvus.collection.milvus_collections import chunk_msmarcos_collection
from app.utils.transformers.models import msmarco_model,multi_qa_mpnet_model
from pymilvus import Collection
from pymilvus import SearchResult
from pymilvus import MilvusException, SearchResult
from asgiref.sync import sync_to_async


__all__ = (
    "questions_msmarco_distilbert_base_tas_b",
    "keywords_msmarco_distilbert_base_tas_b",
)
 



async def insert_chunk_to_milvus(
    db: Session, chunk: str, mongo_document_id: str, mongo_chunk_id: str
):
    try:
        # Check if collection is available
        if chunk_msmarcos_collection is None:
            print("Milvus collection not available, skipping vector insertion")
            return
            
        chunk = chunk.lower().replace("\n", " ")
        msmarco_embeddings = multi_qa_mpnet_model.encode([chunk], normalize_embeddings=True)
        
        print("msmarco_embeddings>>>>>>>>>>>>>>>>>>>>>>>>>>>>",len(msmarco_embeddings[0]))
        chunk_msmarcos_collection.insert(
            [
                [mongo_chunk_id],  # mongo_chunk_id field
                msmarco_embeddings.tolist(),  # vector field
                [mongo_document_id]  # mongo_document_id field
            ],
            _async=False,
        )
        
        # No need to commit for MongoDB - it auto-commits
        # db.commit() is not needed for MongoDB sessions
        
        print("Document trained successfully>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        
        
    except Exception as e:
        print(f"Error in inserting chunk data: {e}")
        # Don't re-raise the exception to prevent stopping the training process





