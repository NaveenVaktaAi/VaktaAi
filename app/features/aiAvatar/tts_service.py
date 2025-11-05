import asyncio
import base64
import logging
from typing import Optional, List, Dict
import re
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import os
import json
import uuid
from pathlib import Path
from app.features.chat.utils.response import ResponseCreator

logger = logging.getLogger(__name__)

# Audio files directory
AUDIO_FILES_DIR = Path("static/audio_files")
AUDIO_FILES_DIR.mkdir(parents=True, exist_ok=True)

# AWS Polly Viseme to Unity Blendshape mapping
POLLY_TO_UNITY_VISEME_MAP = {
    # Silence
    'sil': 'sil',
    
    # Consonants
    'p': 'B_M_P',      # p, b, m (lips closed)
    'f': 'F_V',        # f, v (teeth on lower lip)
    'T': 'TH',         # th (tongue between teeth)
    't': 'T_L_D_N',    # t, d, n (tongue behind teeth)
    'S': 'Ch_J',       # sh, ch, j (wide lips)
    's': 'S_Z',        # s, z (narrow lips)
    'k': 'K_G_H_NG',   # k, g, ng (back of throat)
    'r': 'R',          # r (lips slightly rounded)
    
    # Vowels
    'a': 'Ah',         # ah (mouth wide open)
    '@': 'Er',         # er, schwa (neutral position)
    'e': 'EE',         # ee (smile, lips stretched)
    'E': 'AE',         # ae (mouth open, lips stretched)
    'i': 'IH',         # ih (slight smile)
    'o': 'Oh',         # oh (lips rounded)
    'u': 'W_OO',       # oo, w (lips very rounded)
}

# Initialize AWS Polly client
try:
    polly = boto3.client(
        'polly',
        region_name=os.getenv('AWS_REGION', 'us-east-1'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    POLLY_AVAILABLE = True
    logger.info("[TTS] ✅ AWS Polly initialized successfully")
except Exception as e:
    POLLY_AVAILABLE = False
    polly = None
    logger.error(f"[TTS] ❌ AWS Polly initialization failed: {e}")


class ImprovedTTSService:
    """
    AWS Polly TTS Service with Automatic Phoneme Generation
    - Uses AWS Polly Neural voices for high-quality speech
    - Automatic phoneme extraction for Unity lip-sync
    - Multi-language support (English, Hindi, and more)
    """
    
    def __init__(self):
        # AWS Polly voice mapping - Female Teacher Voices
        self.voice_map = {
            # English - Polite Female Teacher Voice
            "en": {"voice": "Ruth", "lang_code": "en-US", "engine": "neural"},  # Female, clear, friendly teacher
            
            # Hindi/Indian - Polite Female Teacher Voice
            "hi": {"voice": "Kajal", "lang_code": "hi-IN", "engine": "neural"},  # Female, clear, patient Hindi teacher
            "hinglish": {"voice": "Kajal", "lang_code": "hi-IN", "engine": "neural"},  # Patient Hinglish teacher
            
            # Other Indian languages - using Kajal (patient teacher style)
            "mr": {"voice": "Kajal", "lang_code": "hi-IN", "engine": "neural"},
            "gu": {"voice": "Kajal", "lang_code": "hi-IN", "engine": "neural"},
            "ta": {"voice": "Kajal", "lang_code": "hi-IN", "engine": "neural"},
            "te": {"voice": "Kajal", "lang_code": "hi-IN", "engine": "neural"},
            "bn": {"voice": "Kajal", "lang_code": "hi-IN", "engine": "neural"},
        }
        
        # Buffer management
        self.sentence_buffer = []
        self.buffer_threshold = 3
        self.min_words = 20
        
        logger.info("[TTS] 🎙️ AWS Polly TTS initialized with automatic phoneme support")
    
    async def generate_audio_chunk(
        self,
        text: str,
        language_code: str = "en",
        is_first_chunk: bool = False,
        context: str = "explanation"
    ) -> Optional[dict]:
        """
        Generate audio with AWS Polly and extract phonemes for Unity lip-sync
        """
        if not text or not text.strip():
            return None
        
        try:
            # Clean text
            processed_text = self._clean_text_for_speech(text)
            
            if not processed_text or len(processed_text.strip()) < 2:
                return None
            
            # Get voice configuration
            lang_code = language_code.lower()
            if "hinglish" in lang_code:
                lang_code = "hinglish"
            
            voice_config = self.voice_map.get(lang_code, self.voice_map.get("en"))
            voice_id = voice_config["voice"]
            lang_code_polly = voice_config["lang_code"]
            engine = voice_config["engine"]
            
            logger.info(f"[TTS] 🎤 AWS Polly Voice: {voice_id} | Language: {lang_code_polly} | Engine: {engine}")
            
            # Check AWS Polly availability
            if not POLLY_AVAILABLE:
                logger.error("[TTS] ❌ AWS Polly not available")
                return None
            
            # Generate audio and phonemes with AWS Polly
            loop = asyncio.get_event_loop()
            
            # Step 1: Synthesize speech audio
            logger.info(f"[TTS] 🎵 Synthesizing audio for: {processed_text[:50]}...")
            audio_response = await loop.run_in_executor(
                None,
                lambda: polly.synthesize_speech(
                    Text=processed_text,
                    VoiceId=voice_id,
                    OutputFormat='mp3',
                    LanguageCode=lang_code_polly,
                    Engine=engine
                )
            )
            
            # Get audio stream
            if "AudioStream" not in audio_response:
                logger.error("[TTS] ❌ No audio stream in Polly response")
                return None
            
            audio_bytes = audio_response["AudioStream"].read()
            if not audio_bytes:
                logger.error("[TTS] ❌ Empty audio stream")
                return None
            
            logger.info(f"[TTS] ✅ Audio generated: {len(audio_bytes)} bytes")
            
            # Save audio to file instead of base64
            audio_filename = f"audio_{uuid.uuid4().hex[:12]}.mp3"
            audio_filepath = AUDIO_FILES_DIR / audio_filename
            
            # Write audio bytes to file
            with open(audio_filepath, 'wb') as f:
                f.write(audio_bytes)
            
            # Create URL path for frontend
            # Get backend URL from environment or use default
            import os
            backend_url = os.getenv("BACKEND_URL", "http://localhost:5000")
            
            # Always use full backend URL (port 5000) for audio files
            # Frontend will be on port 3001, but audio files are served from backend (5000)
            audio_url = f"{backend_url}/static/audio_files/{audio_filename}"
            logger.info(f"[TTS] 💾 Audio saved to: {audio_filepath}")
            logger.info(f"[TTS] 🔗 Audio URL (backend): {audio_url}")
            
            # Step 2: Get viseme (phoneme-like) metadata using speech marks
            logger.info(f"[TTS] 🎭 Requesting viseme metadata from AWS Polly...")
            phoneme_response = await loop.run_in_executor(
                None,
                lambda: polly.synthesize_speech(
                    Text=processed_text,
                    VoiceId=voice_id,
                    OutputFormat='json',
                    LanguageCode=lang_code_polly,
                    SpeechMarkTypes=['viseme'],  # Use viseme instead of phoneme
                    Engine=engine
                )
            )
            
            # Parse viseme (phoneme) metadata
            phonemes = []
            phoneme_json_url = None
            
            if "AudioStream" in phoneme_response:
                phoneme_data = phoneme_response["AudioStream"].read().decode('utf-8')
                phonemes = self._parse_polly_phonemes(phoneme_data)
                logger.info(f"[TTS] 🎭 ✅ AWS Polly returned {len(phonemes)} visemes (phonemes)")
                
                # Save phonemes as JSON file
                if phonemes:
                    phoneme_filename = f"phonemes_{uuid.uuid4().hex[:12]}.json"
                    phoneme_filepath = AUDIO_FILES_DIR / phoneme_filename
                    
                    # Save phonemes to JSON file
                    with open(phoneme_filepath, 'w', encoding='utf-8') as f:
                        json.dump(phonemes, f, indent=2, ensure_ascii=False)
                    
                    phoneme_json_url = f"/static/audio_files/{phoneme_filename}"
                    logger.info(f"[TTS] 💾 Phonemes saved to: {phoneme_filepath}")
                    logger.info(f"[TTS] 🔗 Phoneme JSON URL: {phoneme_json_url}")
            else:
                logger.warning("[TTS] ⚠️ No viseme metadata in Polly response")
            
            # Prepare result with audio file URL and phoneme JSON URL
            result = {
                "audio_url": audio_url,  # File URL instead of base64
                "audio_file": audio_filename,  # Filename for reference
                "phoneme_json_url": phoneme_json_url,  # Phoneme JSON file URL
                "text": processed_text,
                "sampling_rate": 24000,
                "language_code": language_code,
                "voice_used": voice_id,
                "duration": self._estimate_duration(processed_text),
                "format": "mp3",
                "context": context,
                "phonemes": phonemes,  # Keep inline phonemes for backward compatibility
                "phoneme_count": len(phonemes) if phonemes else 0
            }
            
            logger.info(f"[TTS] 🎭 AWS Polly audio with {len(phonemes)} visemes (phonemes) ready for frontend")
            if phonemes:
                sample_phonemes = [p['phoneme'] for p in phonemes[:5]]
                logger.info(f"[TTS] 🎭 Viseme/Phoneme sample: {sample_phonemes}")
            
            return result
            
        except (BotoCoreError, ClientError) as error:
            logger.error(f"[TTS] ❌ AWS Polly error: {error}")
            return None
        except Exception as e:
            logger.error(f"[TTS] ❌ Error: {e}")
            return None
    
    async def generate_audio_for_text_stream(
        self,
        text: str,
        callback,
        language_code: str = "en"
    ):
        """
        Generate audio for text stream with teaching-style explanation
        """
        try:
            # Convert text to teaching-style explanation
            logger.info(f"[TTS] 🧠 Converting text to teaching-style explanation...")
            teaching_version = await self._convert_to_teaching_explanation(text, language_code)
            
            if not teaching_version or not teaching_version.strip():
                logger.warning("[TTS] Failed to create teaching version, using original")
                teaching_version = text
            
            logger.info(f"[TTS] 🎓 Teaching version created: {teaching_version[:150]}...")
            
            # Split into sentences for chunking
            sentences = self._extract_complete_sentences(teaching_version)
            
            if not sentences:
                # Fallback: treat whole text as one chunk
                audio_data = await self.generate_audio_chunk(
                    teaching_version,
                    language_code,
                    is_first_chunk=True,
                    context="explanation"
                )
                if audio_data:
                    await callback(audio_data, 0, 1)
                return
            
            logger.info(f"[TTS] Processing {len(sentences)} sentences from teaching version")
            
            # Group sentences into chunks (3 sentences per chunk)
            chunks = []
            current_chunk = []
            
            for sentence in sentences:
                current_chunk.append(sentence)
                if len(current_chunk) >= 3:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            total_chunks = len(chunks)
            logger.info(f"[TTS] Total chunks to process: {total_chunks}")
            
            # Generate audio for each chunk
            for i, chunk_text in enumerate(chunks):
                is_first = (i == 0)
                audio_data = await self.generate_audio_chunk(
                    chunk_text,
                    language_code,
                    is_first_chunk=is_first,
                    context="explanation"
                )
                
                if audio_data:
                    await callback(audio_data, i, total_chunks)
                    logger.info(f"[TTS] ✅ Sent chunk {i + 1}/{total_chunks}")
            
            logger.info(f"[TTS] ✅ Completed all {total_chunks} chunks")
                    
        except Exception as e:
            logger.error(f"[TTS] Error in text stream generation: {e}")
    
    def _parse_polly_phonemes(self, phoneme_data: str) -> List[Dict]:
        """
        Parse AWS Polly viseme speech marks into Unity-compatible phoneme format.
        
        Polly returns JSON lines like:
        {"time":0,"type":"viseme","value":"p"}
        {"time":50,"type":"viseme","value":"t"}
        
        Visemes are mouth shapes that correspond to phonemes.
        """
        phonemes = []
        
        try:
            lines = phoneme_data.strip().split('\n')
            
            for idx, line in enumerate(lines):
                if not line.strip():
                    continue
                    
                try:
                    mark = json.loads(line)
                    if mark.get('type') == 'viseme':
                        polly_viseme = mark.get('value', '')
                        unity_blendshape = POLLY_TO_UNITY_VISEME_MAP.get(polly_viseme, polly_viseme)
                        
                        phoneme_obj = {
                            'phoneme': polly_viseme,           # AWS Polly viseme (original)
                            'unity_blendshape': unity_blendshape,  # Unity blendshape name
                            'start_time': mark.get('time', 0) / 1000.0,  # Convert ms to seconds
                            'end_time': (mark.get('time', 0) + 50) / 1000.0,  # Estimate end time
                            'duration': 0.05,  # Default 50ms duration
                            'index': idx
                        }
                        phonemes.append(phoneme_obj)
                except json.JSONDecodeError as e:
                    logger.warning(f"[PHONEMES] Failed to parse line: {line[:50]}")
                    continue
            
            # Calculate proper durations based on next phoneme start time
            for i in range(len(phonemes) - 1):
                phonemes[i]['end_time'] = phonemes[i + 1]['start_time']
                phonemes[i]['duration'] = phonemes[i]['end_time'] - phonemes[i]['start_time']
            
            logger.info(f"[PHONEMES] ✅ Parsed {len(phonemes)} AWS Polly visemes (phonemes)")
            return phonemes
            
        except Exception as e:
            logger.error(f"[PHONEMES] ❌ Error parsing Polly visemes: {e}")
            return []
    
    async def _convert_to_teaching_explanation(self, text: str, language_code: str = "en") -> str:
        """Convert text to polite teacher-style explanation"""
        try:
            response_creator = ResponseCreator()
            
            # Create polite teaching prompt
            teaching_prompt = f"""
            You are a polite, patient, and friendly female teacher explaining concepts to students.
            
            Convert this text into a warm, encouraging teaching explanation:
            
            Text: {text}
            
            Guidelines:
            1. Use a gentle, patient tone like a caring teacher
            2. Add polite phrases like "Let me help you understand", "Let's explore", "I'm happy to explain"
            3. Use encouraging words like "Great question!", "That's interesting!", "You're doing well!"
            4. Break down complex ideas into simple, easy-to-understand parts
            5. Be conversational and friendly, like talking to a curious student
            6. Show empathy and understanding if it's a doubt or question
            7. Keep it clear but add warmth and encouragement
            
            Make it sound like a supportive teacher who genuinely wants to help students learn.
            """
            
            # Get teaching version
            teaching_text = await response_creator.get_text_response(
                teaching_prompt,
                language_code=language_code
            )
            
            return teaching_text if teaching_text else text
                
        except Exception as e:
            logger.error(f"[TTS] Error converting to teaching style: {e}")
            return text
    
    def _extract_complete_sentences(self, text: str) -> List[str]:
        """Extract complete sentences from text"""
        # Split on sentence endings
        sentences = re.split(r'([.!?]+)', text)
        
        # Reconstruct sentences with their punctuation
        complete_sentences = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i + 1].strip() in ['.', '!', '?', '...']:
                complete_sentences.append(sentences[i].strip() + sentences[i + 1])
                i += 2
            elif sentences[i].strip():
                complete_sentences.append(sentences[i].strip())
                i += 1
            else:
                i += 1
        
        return [s for s in complete_sentences if s and len(s.strip()) > 2]
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for speech synthesis"""
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
        
        # Remove markdown
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'__([^_]+)__', r'\1', text)
        text = re.sub(r'_([^_]+)_', r'\1', text)
        
        # Remove lists
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Remove URLs and emails
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special chars
        text = re.sub(r'[*#`_~>\[\](){}|<>]', '', text)
        
        # Clean whitespace
        text = re.sub(r'\n\n+', '. ', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Clean up punctuation spacing
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        return text.strip()
    
    def _estimate_duration(self, text: str) -> float:
        """Estimate audio duration"""
        words = len(text.split())
        return (words / 150) * 60  # ~150 words per minute
    

# Global TTS service instance
tts_service = ImprovedTTSService()
