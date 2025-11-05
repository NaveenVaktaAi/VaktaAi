import openai

from app.aws.secretKey import get_secret_keys
from dotenv import load_dotenv

load_dotenv()
import os

def start_openai():
    client = openai.AsyncOpenAI(
        api_key= os.getenv("OPENAI_API_KEY"),
    )
    return client
