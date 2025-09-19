import openai

from app.aws.secretKey import get_secret_keys

keys = get_secret_keys()


def start_openai():
    client = openai.AsyncOpenAI(
        api_key=keys.get("OPENAI_API_KEY"),
    )
    return client
