import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "500"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Validation
    def __post_init__(self):
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")

settings = Settings()
