# import os

# from dotenv import dotenv_values


# def env_variables():
#     current_directory = os.path.dirname(os.path.abspath(__file__))
#     mode = os.getenv("MODE", "local")

#     if mode == "production":
#         env_file = os.path.join(current_directory, "../.env.production")
#     elif mode == "development":
#         env_file = os.path.join(current_directory, "../.env.development")
#     elif mode == "uat":
#         env_file = os.path.join(current_directory, "../.env.uat")
#     else:
#         env_file = os.path.join(current_directory, "../.env.local")
#     # if mode == "production" else "../.env.development"
#     env = dotenv_values(env_file)

#     return env
import os
from typing import Optional


class AI_Settings:
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    STT_METHOD: str = "groq_whisper"
    TTS_METHOD: str = "aws_polly"  # Changed from edge_tts to aws_polly
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    KOKORO_URL: str = "http://localhost:8001"
    WEB_CONCURRENCY: int = 10
    # AWS Polly settings
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")


class Config:
    env_file = ".env"

ai_settings = AI_Settings()