
import asyncio
import base64
from typing import AsyncGenerator
import edge_tts

class EdgeTTS:
    def __init__(self, voice: str = "hi-IN-MadhurNeural"):
        self.voice = voice


    async def synthesize_streaming(self, text: str) -> AsyncGenerator[bytes, None]:
    # edge_tts.Communicate.stream yields dicts with 'type' and 'data' (base64 for audio)
        communicate = edge_tts.Communicate(text, self.voice)
        async for chunk in communicate.stream():
        # chunk is like {'type': 'audio', 'data': b'...'} or {'type': 'event', ...}
            if chunk.get("type") == "audio":
                data = chunk.get("data")
                # data might already be bytes; if base64 string decode it
                if isinstance(data, str):
                    yield base64.b64decode(data)
                else:
                    yield data
            await asyncio.sleep(0) # cooperative


# Kokoro example
import aiohttp
class KokoroTTS:
    def __init__(self, server_url: str):
        self.server_url = server_url


    async def synthesize_streaming(self, text: str) -> AsyncGenerator[bytes, None]:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{self.server_url}/synthesize", json={"text": text}) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Kokoro synth failed: {resp.status}")
                async for chunk in resp.content.iter_chunked(4096):
                    yield chunk