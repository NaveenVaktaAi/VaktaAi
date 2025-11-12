import asyncio
from fastapi import WebSocket

from app.avatar_config import ai_settings
from app.features.aiAvatar.aiTutorServices.agentRunner import run_agent_streaming
from app.features.aiAvatar.aiTutorServices.stt import GroqSTT
from app.features.aiAvatar.aiTutorServices.tts import EdgeTTS, KokoroTTS
from app.features.aiAvatar.audio_compression import audio_compression


stt_client = GroqSTT(ai_settings.GROQ_API_KEY) if ai_settings.GROQ_API_KEY else None
tts_client = EdgeTTS() if ai_settings.TTS_METHOD == "edge_tts" else KokoroTTS(ai_settings.KOKORO_URL)

# Enable compression for WebSocket streaming (can be disabled for lower latency)
ENABLE_WS_COMPRESSION = True  # Set to False if client doesn't support decompression


async def handle_student_ws(websocket: WebSocket):
    await websocket.accept()
    audio_buffer = bytearray()
    current_tts_task = None


    async def speak_text(text_chunk: str):
        # stream tts and send bytes to websocket
        async for audio in tts_client.synthesize_streaming(text_chunk):
            # 🚀 Compress audio before sending (50-60% bandwidth reduction)
            if ENABLE_WS_COMPRESSION and audio_compression.should_compress(len(audio)):
                try:
                    compression_result = await audio_compression.compress(audio)
                    # Send compressed audio with metadata
                    await websocket.send_json({
                        "type": "audio_chunk_compressed",
                        "compressed": True,
                        "size": compression_result['compressed_size'],
                        "original_size": compression_result['original_size']
                    })
                    await websocket.send_bytes(compression_result['compressed'])
                except Exception as e:
                    # Fallback to uncompressed if compression fails
                    print(f"[WS] ⚠️ Compression failed, sending uncompressed: {e}")
                    await websocket.send_bytes(audio)
            else:
                # Send uncompressed audio
                await websocket.send_bytes(audio)


    async def on_token(token_text: str):
        # Forward text chunk to client for UI and also initiate TTS
        await websocket.send_json({"type": "text_chunk", "text": token_text})
        # we synthesize in background but wait if you want ordered audio
        await speak_text(token_text)


    try:
        while True:
            # For clarity use receive_text / receive_bytes explicitly
            msg = await websocket.receive_text()
            if msg == "audio_complete":
                # If you receive raw bytes via separate messages, adjust accordingly
                if audio_buffer:
                    transcription = await stt_client.transcribe_bytes(bytes(audio_buffer), language="hi")
                    audio_buffer.clear()
                    await websocket.send_json({"type": "transcription", "text": transcription})

                    # Run agent streaming and pipe responses
                    await run_agent_streaming(transcription, on_token)

                    await websocket.send_json({"type": "response_complete"})

            elif msg == "interrupt":
                # Cancel currently playing TTS if any
                if current_tts_task and not current_tts_task.done():
                    current_tts_task.cancel()
                    await websocket.send_json({"type": "interrupted"})


    except Exception as exc:
        try:
            await websocket.close()
        except Exception:
            pass
        raise