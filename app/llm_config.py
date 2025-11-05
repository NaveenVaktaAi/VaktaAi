import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = "fsdfdsfdsfdsf"
    MODEL_NAME: str = "gpt-3.5-turbo"
    MAX_TOKENS: int = 5000
    TEMPERATURE: float = 0.7
    
    # Validation
    def __post_init__(self):
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")

settings = Settings()
