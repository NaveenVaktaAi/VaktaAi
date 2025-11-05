from typing import Dict
from fastapi import WebSocket
from datetime import datetime
import logging
import asyncio

logger = logging.getLogger(__name__)


class AITutorWebSocketManager:
    """
    WebSocket connection manager for AI Tutor conversations.
    Similar to chat websocket manager but for AI tutoring.
    """
    
    def __init__(self):
        # Store active connections: {conversation_id: [websocket1, websocket2, ...]}
        self.active_connections: Dict[str, list[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, conversation_id: str):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        if conversation_id not in self.active_connections:
            self.active_connections[conversation_id] = []
        
        self.active_connections[conversation_id].append(websocket)
        logger.info(f"WebSocket connected to conversation {conversation_id}. Total connections: {len(self.active_connections[conversation_id])}")
    
    def disconnect(self, websocket: WebSocket, conversation_id: str):
        """Remove a WebSocket connection"""
        if conversation_id in self.active_connections:
            if websocket in self.active_connections[conversation_id]:
                self.active_connections[conversation_id].remove(websocket)
                logger.info(f"WebSocket disconnected from conversation {conversation_id}. Remaining: {len(self.active_connections[conversation_id])}")
            
            # Clean up empty conversation entries
            if not self.active_connections[conversation_id]:
                del self.active_connections[conversation_id]
    
    async def send_to_conversation(self, conversation_id: str, message: dict):
        """Send a message to all connections in a conversation"""
        if conversation_id in self.active_connections:
            dead_connections = []
            
            for connection in self.active_connections[conversation_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending to WebSocket: {e}")
                    dead_connections.append(connection)
            
            # Remove dead connections
            for dead_conn in dead_connections:
                self.disconnect(dead_conn, conversation_id)
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all active connections"""
        for conversation_id in list(self.active_connections.keys()):
            await self.send_to_conversation(conversation_id, message)
    
    def get_connection_count(self, conversation_id: str) -> int:
        """Get number of active connections for a conversation"""
        return len(self.active_connections.get(conversation_id, []))
    
    def get_total_connections(self) -> int:
        """Get total number of active connections"""
        return sum(len(connections) for connections in self.active_connections.values())


# Global WebSocket manager instance
ai_tutor_websocket_manager = AITutorWebSocketManager()


class AITutorWebSocketResponse:
    """
    WebSocket response handler for AI Tutor.
    Handles streaming responses and message formatting.
    """
    
    def __init__(self, websocket: WebSocket, conversation_id: str, connection_manager: AITutorWebSocketManager = None):
        self.websocket = websocket
        self.conversation_id = conversation_id
        self.connection_manager = connection_manager or ai_tutor_websocket_manager
    
    async def create_streaming_response(
        self,
        response_generator,
        message_id: str,
        is_final: bool = False
    ) -> str:
        """
        Create a streaming response from an async generator.
        Returns the complete text after streaming.
        """
        try:
            full_text = ""
            chunk_count = 0
            
            # Start streaming
            await self.send_stream_start(message_id)
            
            # Stream chunks with minimal delay for faster start (like DocSathi)
            async for chunk in response_generator:
                if chunk:
                    full_text += chunk
                    chunk_count += 1
                    await self.send_stream_chunk(message_id, chunk)
                    # ✅ Minimal delay for smooth streaming (10ms like DocSathi, faster than before)
                    await asyncio.sleep(0.01)
            
            # End streaming
            await self.send_stream_end(message_id, full_text, is_final)
            
            print(f"✅ Streamed {chunk_count} chunks, total length: {len(full_text)}")
            return full_text
            
        except Exception as e:
            logger.error(f"Error in create_streaming_response: {e}")
            await self.send_error(str(e))
            return ""
    
    async def send_stream_start(self, message_id: str):
        """Send stream start signal"""
        await self.connection_manager.send_to_conversation(
            self.conversation_id,
            {
                "mt": "stream_start",
                "conversationId": self.conversation_id,
                "messageId": message_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def send_stream_chunk(self, message_id: str, chunk: str):
        """Send a stream chunk"""
        await self.connection_manager.send_to_conversation(
            self.conversation_id,
            {
                "mt": "stream_chunk",
                "conversationId": self.conversation_id,
                "messageId": message_id,
                "chunk": chunk,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def send_stream_end(self, message_id: str, full_text: str, is_final: bool = False):
        """Send stream end signal"""
        await self.connection_manager.send_to_conversation(
            self.conversation_id,
            {
                "mt": "stream_end",
                "conversationId": self.conversation_id,
                "messageId": message_id,
                "fullText": full_text,
                "isFinal": is_final,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def send_user_message_confirmation(self, message: str, user_id: str, token: str = None):
        """Send confirmation that user message was received"""
        await self.connection_manager.send_to_conversation(
            self.conversation_id,
            {
                "mt": "user_message_received",
                "conversationId": self.conversation_id,
                "message": message,
                "userId": user_id,
                "token": token,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def send_error(self, error_message: str):
        """Send error message"""
        await self.connection_manager.send_to_conversation(
            self.conversation_id,
            {
                "mt": "error",
                "conversationId": self.conversation_id,
                "error": error_message,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def send_typing_indicator(self):
        """Send typing indicator"""
        await self.connection_manager.send_to_conversation(
            self.conversation_id,
            {
                "mt": "typing_indicator",
                "conversationId": self.conversation_id,
                "isBot": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def stop_typing_indicator(self):
        """Stop typing indicator"""
        await self.connection_manager.send_to_conversation(
            self.conversation_id,
            {
                "mt": "stop_typing_indicator",
                "conversationId": self.conversation_id,
                "isBot": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def send_thinking_indicator(
        self, 
        message: str = "Thinking...", 
        status: str = "thinking",
        language_code: str = "en",
        enable_audio: bool = False,
        display_message: str = None  # Optional separate message for UI display
    ):
        """
        Send thinking/searching indicator to frontend.
        Shows user that LLM is processing their question.
        Optionally generates audio for the thinking message (like a real teacher talking).
        
        Args:
            message: Audio message (teacher's voice)
            display_message: Text message for UI (if different from audio)
            enable_audio: Whether to generate audio
        """
        # Use display_message for UI, or fallback to audio message
        ui_message = display_message if display_message else message
        
        # Send text indicator (always sent for UI)
        await self.connection_manager.send_to_conversation(
            self.conversation_id,
            {
                "mt": "thinking_indicator",
                "conversationId": self.conversation_id,
                "isBot": True,
                "message": ui_message,  # Text for UI display
                "status": status,  # "thinking", "searching", "analyzing", "generating"
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        print(f"[THINKING] 💭 Sent thinking indicator: {ui_message}")
        
        # If audio enabled, generate and send audio for thinking message
        if enable_audio:
            try:
                from app.features.aiAvatar.tts_service import tts_service
                
                print(f"[THINKING] 🎤 Generating audio for thinking message...")
                
                # Generate short audio for thinking message
                audio_data = await tts_service.generate_audio_chunk(
                    text=message,
                    language_code=language_code,
                    is_first_chunk=True,
                    context="greeting"  # Use greeting context for short messages
                )
                
                if audio_data:
                    # Send as special thinking audio chunk
                    await self.send_audio_chunk(
                        audio_data=audio_data,
                        chunk_index=0,
                        total_chunks=1,
                        message_id=f"thinking_{status}"
                    )
                    print(f"[THINKING] 🎤 ✅ Thinking audio sent")
                    
            except Exception as e:
                print(f"[THINKING] ⚠️ Failed to generate thinking audio: {e}")
    
    async def send_audio_chunk(
        self,
        audio_data: dict,
        chunk_index: int,
        total_chunks: int,
        message_id: str = None
    ):
        """
        Send an audio chunk to the frontend with phonemes for Unity lip-sync.
        Audio data should contain: audio_url, text, sampling_rate, language_code, duration, phonemes
        """
        try:
            # Prepare payload with phonemes for Unity avatar
            payload = {
                "mt": "audio_chunk",
                "conversationId": self.conversation_id,
                "messageId": message_id,
                "chunkIndex": chunk_index,
                "totalChunks": total_chunks,
                "audioUrl": audio_data.get("audio_url"),  # Audio file URL instead of base64
                "audioFile": audio_data.get("audio_file"),  # Filename for reference
                "phonemeJsonUrl": audio_data.get("phoneme_json_url"),  # Phoneme JSON file URL
                "text": audio_data.get("text"),
                "samplingRate": audio_data.get("sampling_rate"),
                "languageCode": audio_data.get("language_code", "en"),
                "duration": audio_data.get("duration"),
                "timestamp": datetime.utcnow().isoformat(),
                # Phoneme data for Unity avatar lip-sync
                "phonemes": audio_data.get("phonemes", []),  # Keep inline phonemes for backward compatibility
                "phonemeCount": audio_data.get("phoneme_count", 0)
            }
            
            await self.connection_manager.send_to_conversation(
                self.conversation_id,
                payload
            )
            
            phoneme_count = audio_data.get("phoneme_count", 0)
            phonemes_data = audio_data.get("phonemes", [])
            logger.info(f"[AUDIO] Sent audio chunk {chunk_index + 1}/{total_chunks} with {phoneme_count} phonemes for conversation {self.conversation_id}")
            if phonemes_data:
                phoneme_details = [p.get('phoneme', '?') for p in phonemes_data[:3]]
                logger.info(f"[AUDIO] 🎭 Phonemes sample: {phoneme_details}")
                logger.info(f"[AUDIO] 🎭 Full phoneme data available for Unity lip-sync")
            else:
                logger.warning(f"[AUDIO] 🎭 No phonemes in chunk {chunk_index + 1}")
        except Exception as e:
            logger.error(f"[AUDIO] Error sending audio chunk: {e}")
    
    async def send_audio_generation_start(self, message_id: str = None):
        """Notify frontend that audio generation has started"""
        await self.connection_manager.send_to_conversation(
            self.conversation_id,
            {
                "mt": "audio_generation_start",
                "conversationId": self.conversation_id,
                "messageId": message_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def send_audio_generation_complete(self, message_id: str = None, total_chunks: int = 0):
        """Notify frontend that audio generation is complete"""
        await self.connection_manager.send_to_conversation(
            self.conversation_id,
            {
                "mt": "audio_generation_complete",
                "conversationId": self.conversation_id,
                "messageId": message_id,
                "totalChunks": total_chunks,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def create_streaming_response_with_audio(
        self,
        response_generator,
        message_id: str,
        is_final: bool = False,
        language_code: str = "en",
        enable_tts: bool = True
    ) -> str:
        """
        Create a streaming response with REAL-TIME parallel audio generation.
        Audio generates WHILE text is streaming, not after!
        """
        try:
            full_text = ""
            chunk_count = 0
            sentence_buffer = ""
            audio_chunk_counter = 0
            
            # Start streaming
            await self.send_stream_start(message_id)
            
            # Start audio generation notification
            if enable_tts:
                await self.send_audio_generation_start(message_id)
            
            # Stream text chunks and generate audio in PARALLEL
            async for chunk in response_generator:
                if chunk:
                    full_text += chunk
                    sentence_buffer += chunk
                    chunk_count += 1
                    await self.send_stream_chunk(message_id, chunk)
                    
                    # Check if we have complete sentences in buffer
                    if enable_tts:
                        complete_sentences = self._extract_complete_sentences(sentence_buffer)
                        
                        # If we have 2+ complete sentences, generate audio NOW
                        if len(complete_sentences) >= 2:
                            sentences_text = " ".join(complete_sentences)
                            print(f"[AUDIO] 🎤 Generating audio for: {sentences_text[:100]}...")
                            
                            # Generate audio in parallel (don't wait)
                            asyncio.create_task(
                                self._generate_single_audio_chunk(
                                    sentences_text,
                                    message_id,
                                    language_code,
                                    audio_chunk_counter
                                )
                            )
                            audio_chunk_counter += 1
                            
                            # Remove processed sentences from buffer
                            for sentence in complete_sentences:
                                sentence_buffer = sentence_buffer.replace(sentence, "", 1)
                            sentence_buffer = sentence_buffer.strip()
                    
                    # Small delay for smooth streaming
                    await asyncio.sleep(0.02)  # 20ms delay
            
            # End text streaming
            await self.send_stream_end(message_id, full_text, is_final)
            
            print(f"✅ Text streaming complete: {chunk_count} chunks, total length: {len(full_text)}")
            
            # Generate audio for remaining buffer
            if enable_tts and sentence_buffer.strip():
                print(f"[AUDIO] 🎤 Generating audio for remaining text...")
                asyncio.create_task(
                    self._generate_single_audio_chunk(
                        sentence_buffer,
                        message_id,
                        language_code,
                        audio_chunk_counter
                    )
                )
            
            return full_text
            
        except Exception as e:
            logger.error(f"Error in create_streaming_response_with_audio: {e}")
            await self.send_error(str(e))
            return ""
    
    def _extract_complete_sentences(self, text: str) -> list:
        """Extract complete sentences from buffer for real-time audio generation"""
        import re
        # Match sentences ending with . ! ? । ॥
        pattern = r'[^.!?।॥]*[.!?।॥]+'
        matches = re.findall(pattern, text)
        return [s.strip() for s in matches if s.strip()]
    
    async def _generate_single_audio_chunk(
        self,
        text: str,
        message_id: str,
        language_code: str,
        chunk_index: int
    ):
        """
        Generate audio for a single chunk of text in real-time.
        This runs in parallel with text streaming.
        """
        try:
            from app.features.aiAvatar.tts_service import tts_service
            
            print(f"[AUDIO] 🎤 Chunk {chunk_index}: Generating audio for '{text[:80]}...'")
            
            # Generate audio chunk
            audio_data = await tts_service.generate_audio_chunk(
                text,
                language_code=language_code,
                is_first_chunk=(chunk_index == 0),
                context="explanation"
            )
            
            if audio_data:
                # Send audio chunk to frontend
                await self.send_audio_chunk(
                    audio_data,
                    chunk_index,
                    chunk_index + 1,  # We don't know total yet in real-time
                    message_id
                )
                print(f"[AUDIO] ✅ Chunk {chunk_index} sent successfully")
            else:
                print(f"[AUDIO] ⚠️ Chunk {chunk_index} - No audio generated")
                
        except Exception as e:
            logger.error(f"[AUDIO] Error generating audio chunk {chunk_index}: {e}")
    
    async def _generate_intelligent_audio(self, text: str, message_id: str, language_code: str):
        """
        Generate audio using intelligent sentence buffering approach.
        Better flow and natural teaching style.
        """
        try:
            from app.features.aiAvatar.tts_service import tts_service
            
            print(f"[AUDIO] Starting intelligent audio generation for message {message_id}")
            await self.send_audio_generation_start(message_id)
            
            # Callback to send each audio chunk
            async def audio_chunk_callback(audio_data, chunk_index, total_chunks):
                await self.send_audio_chunk(
                    audio_data,
                    chunk_index,
                    total_chunks,
                    message_id
                )
                print(f"[AUDIO] ✅ Sent chunk {chunk_index+1}/{total_chunks}")
            
            # Generate audio with intelligent chunking
            await tts_service.generate_audio_for_text_stream(
                text,
                audio_chunk_callback,
                language_code=language_code
            )
            
            print(f"[AUDIO] 🎉 Intelligent audio generation complete for message {message_id}")
            
        except Exception as e:
            logger.error(f"[AUDIO] Error in intelligent audio generation: {e}")
            print(f"[AUDIO] Failed to generate audio: {e}")
    
    async def generate_and_send_audio_chunks(
        self,
        text: str,
        message_id: str,
        language_code: str = "en"
    ):
        """
        Generate audio from teacher-style summary and send in chunks.
        Used for parallel chain approach where summary is pre-generated.
        """
        try:
            from app.features.aiAvatar.tts_service import tts_service
            
            print(f"[AUDIO] 🎤 Generating audio chunks from summary: {text[:100]}...")
            
            # Split summary into sentences for chunking
            sentences = self._extract_complete_sentences(text)
            
            if not sentences:
                sentences = [text]
            
            total_chunks = len(sentences)
            print(f"[AUDIO] 📊 Splitting into {total_chunks} audio chunks")
            
            # Generate audio for each sentence chunk
            for chunk_idx, sentence in enumerate(sentences):
                if not sentence.strip():
                    continue
                
                print(f"[AUDIO] 🎤 Chunk {chunk_idx}: Generating audio for '{sentence[:50]}...'")
                
                # Generate audio with AWS Polly
                audio_data = await tts_service.generate_audio_chunk(
                    text=sentence,
                    language_code=language_code,
                    is_first_chunk=(chunk_idx == 0),
                    context="explanation"
                )
                
                if audio_data:
                    # Send audio chunk to frontend
                    await self.send_audio_chunk(
                        audio_data=audio_data,
                        chunk_index=chunk_idx,
                        total_chunks=total_chunks,
                        message_id=message_id
                    )
                    print(f"[AUDIO] ✅ Sent chunk {chunk_idx + 1}/{total_chunks}")
                else:
                    print(f"[AUDIO] ⚠️ Chunk {chunk_idx} - No audio generated")
            
            print(f"[AUDIO] 🎉 Summary audio generation complete: {total_chunks} chunks sent")
            
        except Exception as e:
            logger.error(f"[AUDIO] Error generating audio chunks: {e}")
            print(f"[AUDIO] Failed to generate audio chunks: {e}")

