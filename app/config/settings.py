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
    
    # Authentication settings
    JWT_SECRET_KEY: str = os.getenv('JWT_SECRET_KEY', 'your-secret-key-change-in-production')
    JWT_ALGORITHM: str = os.getenv('JWT_ALGORITHM', 'HS256')
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', '1440'))  # 24 hours
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv('REFRESH_TOKEN_EXPIRE_DAYS', '30'))
    
    # OTP settings
    OTP_LENGTH: int = int(os.getenv('OTP_LENGTH', '6'))
    OTP_EXPIRE_MINUTES: int = int(os.getenv('OTP_EXPIRE_MINUTES', '5'))
    
    # Email settings
    SENDER_EMAIL: str = os.getenv('SENDER_EMAIL', 'naveen.sharma@vaktaai.com')
    EMAIL_VERIFICATION_EXPIRE_MINUTES: int = int(os.getenv('EMAIL_VERIFICATION_EXPIRE_MINUTES', '10'))
    
    # SMS settings (placeholder for future SMS service integration)
    SMS_SERVICE_ENABLED: bool = os.getenv('SMS_SERVICE_ENABLED', 'false').lower() == 'true'
    
    # MongoDB settings
    MONGO_URI: str = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
    MONGO_DB_NAME: str = os.getenv('MONGO_DB_NAME', 'vakta_ai')
    
    # AI Service API Keys
    GROQ_API_KEY: Optional[str] = os.getenv('GROQ_API_KEY')
    GROQ_MODEL: str = os.getenv('GROQ_MODEL', 'llama-3.1-70b-versatile')
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

settings = Settings()
