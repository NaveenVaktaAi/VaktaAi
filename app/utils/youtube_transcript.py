#!/usr/bin/env python3
"""
Robust YouTube transcript extraction with Caption-first, ASR-second fallback
"""
import time
import re
import os
import tempfile
import subprocess
import requests
import logging
from typing import List, Optional, Tuple, Dict, Any
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

logger = logging.getLogger(__name__)

# Configuration
LANGS = ["en", "en-IN", "hi", "hi-IN"]
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
MAX_RETRIES = 5
CACHE_TTL = 1209600  # 14 days

def _delays(max_tries: int = MAX_RETRIES):
    """Generate exponential backoff delays"""
    for i in range(max_tries):
        yield min(2**i, 30)

def _clean(text: str) -> str:
    """Clean transcript text by removing junk HTML and tracking data"""
    if not text:
        return ""
    
    # Remove common junk patterns
    bad_patterns = [
        "trackingParams", "\\u0026", "ytcfg", "WIZ_global_data",
        "window.yt", "ytInitialData", "ytInitialPlayerResponse",
        "clickTrackingParams", "commandMetadata", "webCommandMetadata",
        "browseEndpoint", "watchEndpoint", "innertubeCommand"
    ]
    
    if any(pattern in text for pattern in bad_patterns):
        return ""
    
    # Clean up HTML entities and extra whitespace
    text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    
    return text

def _normalize(items) -> List[str]:
    """Normalize transcript items to clean text segments"""
    out = []
    for item in items:
        # Handle both dict and FetchedTranscriptSnippet objects
        if hasattr(item, 'text'):
            text = _clean(item.text)
        elif isinstance(item, dict):
            text = _clean(item.get("text", ""))
        else:
            continue
            
        if len(text) > 1:
            out.append(text)
    return out

def try_captions(video_id: str) -> Optional[List[str]]:
    """Try to get official YouTube captions"""
    logger.info(f"🔍 Trying official captions for video: {video_id}")
    print("+++try_captions+++++++++++++++++++++videoid+++++++++++++++++++++++++++++",video_id)
    for delay in _delays():
        try:
            # Create an instance of YouTubeTranscriptApi and call fetch method
            api = YouTubeTranscriptApi()
            
            # Try different language combinations
            language_combinations = [
                ['hi', 'en'],  # Hindi first, then English
                ['en', 'hi'],  # English first, then Hindi
                ['hi'],        # Only Hindi
                ['en'],        # Only English
            ]
            
            for languages in language_combinations:
                try:
                    transcript = api.fetch(video_id, languages=languages, preserve_formatting=True)
                    segments = _normalize(transcript)
                    if segments:
                        logger.info(f"✅ Successfully extracted {len(segments)} caption segments using languages: {languages}")
                        return segments
                except (TranscriptsDisabled, NoTranscriptFound) as e:
                    logger.debug(f"❌ No transcript found for languages {languages}: {e}")
                    continue
            
            return None
        except (VideoUnavailable) as e:
            logger.warning(f"❌ Video unavailable: {e}")
            return None
        except Exception as e:
            logger.warning(f"❌ Captions failed, retrying in {delay}s: {e}")
            time.sleep(delay)
    
    return None

def try_timedtext(video_id: str) -> Optional[List[str]]:
    """Try to get transcript via YouTube's timedtext API"""
    logger.info(f"🔍 Trying timedtext API for video: {video_id}")
    
    endpoints = [
        f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=srv3",
        f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=ttml",
        f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=vtt",
        f"https://www.youtube.com/api/timedtext?v={video_id}&lang=hi&fmt=srv3",
    ]
    
    for url in endpoints:
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            if response.status_code == 200 and response.text:
                # Check if it's not HTML junk
                if "<html" in response.text.lower() or "trackingParams" in response.text:
                    continue
                
                # Parse different formats
                lines = re.split(r"[\r\n]+", response.text)
                lines = [_clean(re.sub(r"<[^>]+>", " ", line)) for line in lines]
                lines = [line for line in lines if len(line) > 1]
                
                if len(lines) > 3:
                    logger.info(f"✅ Successfully extracted {len(lines)} timedtext segments")
                    return lines
        except Exception as e:
            logger.warning(f"❌ Timedtext endpoint failed: {e}")
            continue
    
    return None

def _yt_dlp_audio(video_url: str, cookies_path: Optional[str] = None) -> str:
    """Download audio using yt-dlp"""
    logger.info(f"🎵 Downloading audio for: {video_url}")
    
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".m4a").name
    cmd = [
        "yt-dlp", 
        "-f", "bestaudio/best", 
        "-x", "--audio-format", "m4a", 
        "-o", out, 
        video_url,
        "--quiet"
    ]
    
    if cookies_path and os.path.exists(cookies_path):
        cmd += ["--cookies", cookies_path]
    
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        logger.info(f"✅ Audio downloaded: {out}")
        return out
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ yt-dlp failed: {e.stderr.decode()}")
        raise
    except subprocess.TimeoutExpired:
        logger.error("❌ yt-dlp timeout")
        raise

def try_asr(audio_path: str, model_size: str = "medium") -> List[str]:
    """Try ASR using faster-whisper"""
    logger.info(f"🎤 Running ASR on audio: {audio_path}")
    
    try:
        from faster_whisper import WhisperModel
        
        model = WhisperModel(model_size, device="cpu")  # Use CPU for compatibility
        segments, _ = model.transcribe(audio_path, language="en", task="transcribe")
        
        out = []
        for segment in segments:
            text = _clean(segment.text or "")
            if len(text) > 1:
                out.append(text)
        
        logger.info(f"✅ ASR extracted {len(out)} segments")
        return out
        
    except ImportError:
        logger.warning("❌ faster-whisper not available, trying openai-whisper")
        try:
            import whisper
            
            model = whisper.load_model(model_size)
            result = model.transcribe(audio_path, language="en")
            
            out = []
            for segment in result["segments"]:
                text = _clean(segment["text"])
                if len(text) > 1:
                    out.append(text)
            
            logger.info(f"✅ Whisper ASR extracted {len(out)} segments")
            return out
            
        except ImportError:
            logger.error("❌ Neither faster-whisper nor openai-whisper available")
            return []
    except Exception as e:
        logger.error(f"❌ ASR failed: {e}")
        return []

async def get_transcript(video_url: str, video_id: str, cookies_path: Optional[str] = None) -> Tuple[str, List[str]]:
    """
    Get transcript using Caption-first, ASR-second fallback strategy
    
    Returns:
        Tuple of (source, segments) where source ∈ {"captions", "timedtext", "asr", "metadata"}
    """
    logger.info(f"🎯 Starting transcript extraction for video: {video_id}")
    
    print("<insert>get_transcript+++++++++++++++++++++videoid+++++++++++++++++++++++++++++",video_id)
    # Method 1: Try official captions first (most reliable)
    segments = try_captions(video_id)
    if segments:
        return "captions", segments
    
    # Method 2: Try timedtext API
    segments = try_timedtext(video_id)
    if segments:
        return "timedtext", segments
    
    # Method 3: ASR fallback
    try:
        audio_path = _yt_dlp_audio(video_url, cookies_path)
        try:
            segments = try_asr(audio_path)
            if segments:
                return "asr", segments
        finally:
            # Clean up audio file
            try:
                os.remove(audio_path)
            except:
                pass
    except Exception as e:
        logger.warning(f"❌ ASR fallback failed: {e}")
    
    # Method 4: Return empty metadata
    logger.warning(f"❌ All transcript methods failed for {video_id}")
    return "metadata", []

async def get_transcript_with_cache(video_url: str, video_id: str, cache_client=None, cookies_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get transcript with Redis caching
    
    Returns:
        Dict with source, segments, cached, and metadata
    """
    # Check cache first
    if cache_client:
        for method in ["captions", "timedtext", "asr"]:
            cache_key = f"youtube_transcript:{video_id}:{method}"
            cached = cache_client.get(cache_key)
            if cached:
                try:
                    segments = eval(cached) if isinstance(cached, str) else cached
                    logger.info(f"✅ Found cached transcript: {method}")
                    return segments
                except:
                    continue
    print("+++get_transcript_with_cache+++++++++++++++++++++videoid+++++++++++++++++++++++++++++",video_id)
    # Get fresh transcript
    source, segments = await get_transcript(video_url, video_id, cookies_path)
    
    # Cache successful results
    if cache_client and source in ["captions", "timedtext", "asr"] and segments:
        cache_key = f"youtube_transcript:{video_id}:{source}"
        cache_client.setex(cache_key, CACHE_TTL, str(segments))
        logger.info(f"✅ Cached transcript: {source}")
    
    return segments
