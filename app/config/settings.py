import os
from typing import Optional

class Settings:
    # Database settings
    DB_HOST: str = os.getenv('DB_HOST', 'localhost')
    DB_NAME: str = os.getenv('DB_NAME', 'ai_chatbot')
    DB_USER: str = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD: str = os.getenv('DB_PASSWORD', 'password')
    DB_PORT: str = os.getenv('DB_PORT', '5432')
    
    # S3 settings
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION: str = os.getenv('AWS_REGION', 'us-east-1')
    S3_BUCKET_NAME: str = os.getenv('S3_BUCKET_NAME', 'xr-technolab-vakta-ai')
    
    # Milvus settings
    MILVUS_HOST: str = os.getenv('MILVUS_HOST', 'localhost')
    MILVUS_PORT: str = os.getenv('MILVUS_PORT', '19530')
    MILVUS_COLLECTION_NAME: str = os.getenv('MILVUS_COLLECTION_NAME', 'document_chunks')
    
    # Embedding settings
    EMBEDDING_MODEL: str = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    EMBEDDING_DIMENSION: int = int(os.getenv('EMBEDDING_DIMENSION', '384'))
    
    # Chunking settings
    CHUNK_SIZE: int = int(os.getenv('CHUNK_SIZE', '800'))
    CHUNK_OVERLAP: int = int(os.getenv('CHUNK_OVERLAP', '100'))
    
    # File upload settings
    MAX_FILE_SIZE: int = int(os.getenv('MAX_FILE_SIZE', '50000000'))  # 50MB
    ALLOWED_EXTENSIONS: list = ['pdf', 'doc', 'docx', 'txt', 'ppt', 'pptx']

settings = Settings()
