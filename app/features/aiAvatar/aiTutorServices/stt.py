import aiohttp
from typing import Optional


class GroqSTT:
    def __init__(self, api_key: str):
        self.api_key = api_key
        # NOTE: verify the correct endpoint in groq docs
        self.endpoint = "https://api.groq.com/v1/audio/transcriptions"


    async def transcribe_bytes(self, audio_bytes: bytes, language: str = "hi") -> str:
        """Transcribe an audio blob. For real-time, prefer streaming-compatible provider.
        """
        form = aiohttp.FormData()
        form.add_field('file', audio_bytes, filename='audio.wav', content_type='audio/wav')
        form.add_field('model', 'whisper-large-v3')
        form.add_field('language', language)


        headers = {"Authorization": f"Bearer {self.api_key}"}


        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, data=form, headers=headers, timeout=60) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"STT request failed: {resp.status} - {text}")
                result = await resp.json()
                return result.get('text', '')