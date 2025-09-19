from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
from app.config.settings import settings

logger = logging.getLogger(__name__)

class MilvusService:
    def __init__(self):
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self.embedding_dim = settings.EMBEDDING_DIMENSION
        self.collection = None
        self.is_connected = False
    
    def connect(self):
        """Connect to Milvus server"""
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            self.is_connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self.is_connected = False
            raise
    
    def disconnect(self):
        """Disconnect from Milvus server"""
        try:
            connections.disconnect("default")
            self.is_connected = False
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to disconnect from Milvus: {e}")
    
    def create_collection(self):
        """Create collection with schema for document chunks"""
        try:
            if not self.is_connected:
                self.connect()
            
            # Check if collection already exists
            if utility.has_collection(self.collection_name):
                logger.info(f"Collection {self.collection_name} already exists")
                self.collection = Collection(self.collection_name)
                return
            
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="doc_id", dtype=DataType.INT64),
                FieldSchema(name="chunk_id", dtype=DataType.INT64),
                FieldSchema(name="chunk_index", dtype=DataType.INT64),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Document chunks with embeddings for RAG"
            )
            
            # Create collection
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )
            
            logger.info(f"Created collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise
    
    def create_index(self):
        """Create IVFFLAT index for efficient similarity search"""
        try:
            if not self.collection:
                raise Exception("Collection not initialized")
            
            # Check if index already exists
            indexes = self.collection.indexes
            if indexes:
                logger.info("Index already exists")
                return
            
            # Create IVFFLAT index
            index_params = {
                "metric_type": "COSINE",  # Use cosine similarity
                "index_type": "IVF_FLAT",
                "params": {
                    "nlist": 1024  # Number of cluster units
                }
            }
            
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            logger.info("Created IVFFLAT index on embedding field")
            
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            raise
    
    def load_collection(self):
        """Load collection into memory for search"""
        try:
            if not self.collection:
                raise Exception("Collection not initialized")
            
            self.collection.load()
            logger.info(f"Loaded collection {self.collection_name} into memory")
            
        except Exception as e:
            logger.error(f"Failed to load collection: {e}")
            raise
    
    def insert_chunks(self, chunks_data: List[Dict[str, Any]]) -> List[int]:
        """Insert document chunks into Milvus"""
        try:
            if not self.collection:
                raise Exception("Collection not initialized")
            
            if not chunks_data:
                return []
            
            # Prepare data for insertion
            data = {
                "doc_id": [chunk["doc_id"] for chunk in chunks_data],
                "chunk_id": [chunk["chunk_id"] for chunk in chunks_data],
                "chunk_index": [chunk["chunk_index"] for chunk in chunks_data],
                "text": [chunk["text"][:2000] for chunk in chunks_data],  # Truncate if needed
                "doc_name": [chunk["doc_name"][:255] for chunk in chunks_data],
                "metadata": [json.dumps(chunk["metadata"])[:1000] for chunk in chunks_data],
                "embedding": [chunk["embedding"] for chunk in chunks_data]
            }
            
            # Insert data
            insert_result = self.collection.insert(data)
            
            # Get inserted IDs
            inserted_ids = insert_result.primary_keys
            
            logger.info(f"Inserted {len(chunks_data)} chunks into Milvus")
            return inserted_ids
            
        except Exception as e:
            logger.error(f"Failed to insert chunks: {e}")
            raise
    
    def search_similar_chunks(self, query_embedding: List[float], top_k: int = 5, 
                            doc_filter: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity"""
        try:
            if not self.collection:
                raise Exception("Collection not initialized")
            
            # Search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {
                    "nprobe": 16  # Number of clusters to search
                }
            }
            
            # Build filter expression if doc_filter is provided
            filter_expr = None
            if doc_filter:
                doc_ids_str = ",".join(map(str, doc_filter))
                filter_expr = f"doc_id in [{doc_ids_str}]"
            
            # Perform search
            search_results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=["doc_id", "chunk_id", "chunk_index", "text", "doc_name", "metadata"]
            )
            
            # Process results
            results = []
            for hits in search_results:
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "score": float(hit.score),
                        "doc_id": hit.entity.get("doc_id"),
                        "chunk_id": hit.entity.get("chunk_id"),
                        "chunk_index": hit.entity.get("chunk_index"),
                        "text": hit.entity.get("text"),
                        "doc_name": hit.entity.get("doc_name"),
                        "metadata": json.loads(hit.entity.get("metadata", "{}"))
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete_document_chunks(self, doc_id: int) -> bool:
        """Delete all chunks for a specific document"""
        try:
            if not self.collection:
                raise Exception("Collection not initialized")
            
            # Delete chunks by doc_id
            delete_expr = f"doc_id == {doc_id}"
            self.collection.delete(delete_expr)
            
            logger.info(f"Deleted chunks for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete chunks: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            if not self.collection:
                return {"error": "Collection not initialized"}
            
            stats = self.collection.num_entities
            
            return {
                "collection_name": self.collection_name,
                "total_entities": stats,
                "embedding_dimension": self.embedding_dim,
                "index_type": "IVF_FLAT"
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
    
    def is_available(self) -> bool:
        """Check if Milvus service is available"""
        try:
            if not self.is_connected:
                self.connect()
            return True
        except Exception:
            return False
    
    def initialize(self):
        """Initialize Milvus service - connect, create collection, index, and load"""
        try:
            self.connect()
            self.create_collection()
            self.create_index()
            self.load_collection()
            logger.info("Milvus service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Milvus service: {e}")
            self.is_connected = False
            raise

# Global Milvus service instance
milvus_service = MilvusService()
