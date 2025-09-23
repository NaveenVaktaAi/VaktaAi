"""
Ultra-Powerful YouTube Transcript Extractor
Uses cutting-edge methods to extract actual speech content
"""

import asyncio
import logging
import requests
import tempfile
import os
import re
import json
import time
from typing import Optional, Dict, List, Tuple
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
import yt_dlp

logger = logging.getLogger(__name__)

class UltraPowerfulTranscriptExtractor:
    """Ultra-powerful transcript extractor using cutting-edge methods"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Referer': 'https://www.youtube.com/',
            'Origin': 'https://www.youtube.com',
        })
    
    async def extract_transcript(self, video_id: str) -> Optional[str]:
        """Extract transcript using ultra-powerful methods"""
        logger.info(f"🚀 Starting ULTRA-POWERFUL transcript extraction for: {video_id}")
        
        # Method 1: Advanced yt-dlp with Custom Extractors
        transcript = await self._extract_with_advanced_ytdlp(video_id)
        if transcript and self._is_actual_speech(transcript):
            return transcript
        
        # Method 2: YouTube Innertube API (2024 Method)
        transcript = await self._extract_with_innertube_api(video_id)
        if transcript and self._is_actual_speech(transcript):
            return transcript
        
        # Method 3: Direct Caption API Calls
        transcript = await self._extract_with_caption_api(video_id)
        if transcript and self._is_actual_speech(transcript):
            return transcript
        
        # Method 4: Professional Web Scraping
        transcript = await self._extract_with_professional_scraping(video_id)
        if transcript and self._is_actual_speech(transcript):
            return transcript
        
        # Method 5: Third-Party Service Integration
        transcript = await self._extract_with_third_party_services(video_id)
        if transcript and self._is_actual_speech(transcript):
            return transcript
        
        logger.warning(f"❌ All ULTRA-POWERFUL methods failed for video: {video_id}")
        return None
    
    async def _extract_with_advanced_ytdlp(self, video_id: str) -> Optional[str]:
        """Extract using advanced yt-dlp with custom configurations"""
        try:
            logger.info(f"🔍 Method 1: Advanced yt-dlp extraction")
            
            # Advanced yt-dlp configuration
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en', 'hi', 'en-US', 'en-GB', 'hi-IN', 'en-IN', 'en-AU', 'en-CA'],
                'skip_download': True,
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'writethumbnail': False,
                'writeinfojson': False,
                'ignoreerrors': True,
                'extractor_retries': 5,
                'fragment_retries': 5,
                'retries': 5,
                'socket_timeout': 60,
                'http_chunk_size': 10485760,
                'concurrent_fragment_downloads': 8,
                'extractors': [
                    'youtube:transcript',
                    'youtube:captions',
                    'youtube:timedtext',
                    'youtube:player',
                    'youtube:webpage',
                ],
                'extractor_args': {
                    'youtube': {
                        'skip': ['dash', 'hls'],
                        'player_skip': ['webpage'],
                    }
                }
            }
            
            with tempfile.TemporaryDirectory() as temp_dir:
                ydl_opts['outtmpl'] = os.path.join(temp_dir, '%(id)s.%(ext)s')
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    
                    # Extract with retries
                    for attempt in range(3):
                        try:
                            info = ydl.extract_info(video_url, download=False)
                            break
                        except Exception as e:
                            if attempt == 2:
                                raise e
                            await asyncio.sleep(2)
                    
                    logger.info(f"📊 Video: {info.get('title', 'Unknown')}")
                    
                    # Try manual subtitles first (highest quality)
                    subtitles = info.get('subtitles', {})
                    for lang in ['en', 'hi', 'en-US', 'en-GB', 'hi-IN', 'en-IN', 'en-AU', 'en-CA']:
                        if lang in subtitles:
                            logger.info(f"✅ Found manual subtitles for {lang}")
                            for subtitle_info in subtitles[lang]:
                                try:
                                    subtitle_url = subtitle_info['url']
                                    response = self.session.get(subtitle_url, timeout=30)
                                    if response.status_code == 200:
                                        transcript = self._parse_subtitle_content(response.text)
                                        if transcript and self._is_actual_speech(transcript):
                                            logger.info(f"🎤 Manual transcript: {len(transcript)} chars")
                                            return transcript
                                except Exception as e:
                                    logger.warning(f"❌ Subtitle {lang} failed: {str(e)}")
                                    continue
                    
                    # Try automatic captions
                    automatic_captions = info.get('automatic_captions', {})
                    for lang in ['en', 'hi', 'en-US', 'en-GB', 'hi-IN', 'en-IN', 'en-AU', 'en-CA']:
                        if lang in automatic_captions:
                            logger.info(f"✅ Found auto-captions for {lang}")
                            for caption_info in automatic_captions[lang]:
                                try:
                                    caption_url = caption_info['url']
                                    response = self.session.get(caption_url, timeout=30)
                                    if response.status_code == 200:
                                        transcript = self._parse_subtitle_content(response.text)
                                        if transcript and self._is_actual_speech(transcript):
                                            logger.info(f"🎤 Auto-caption transcript: {len(transcript)} chars")
                                            return transcript
                                except Exception as e:
                                    logger.warning(f"❌ Caption {lang} failed: {str(e)}")
                                    continue
                    
                    # Try video description as fallback
                    description = info.get('description', '')
                    if description and len(description) > 200:
                        clean_desc = self._clean_transcript_text(description)
                        if clean_desc and self._is_actual_speech(clean_desc):
                            logger.info(f"📝 Description transcript: {len(clean_desc)} chars")
                            return clean_desc
                    
        except Exception as e:
            logger.warning(f"❌ Advanced yt-dlp extraction failed: {str(e)}")
        
        return None
    
    async def _extract_with_innertube_api(self, video_id: str) -> Optional[str]:
        """Extract using YouTube Innertube API (2024 method)"""
        try:
            logger.info(f"🔍 Method 2: YouTube Innertube API")
            
            # Step 1: Get video page to extract API key
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            response = self.session.get(video_url, timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch video page: {response.status_code}")
                return None
            
            # Extract INNERTUBE_API_KEY
            api_key_match = re.search(r'"INNERTUBE_API_KEY":"([^"]+)"', response.text)
            if not api_key_match:
                logger.warning("INNERTUBE_API_KEY not found")
                return None
            
            api_key = api_key_match.group(1)
            logger.info(f"✅ Found API key: {api_key[:20]}...")
            
            # Step 2: Get video info using Innertube API
            innertube_url = "https://www.youtube.com/youtubei/v1/player"
            
            payload = {
                "context": {
                    "client": {
                        "clientName": "WEB",
                        "clientVersion": "2.20231201.00.00"
                    }
                },
                "videoId": video_id
            }
            
            headers = {
                'Content-Type': 'application/json',
                'X-YouTube-Client-Name': '1',
                'X-YouTube-Client-Version': '2.20231201.00.00',
            }
            
            response = self.session.post(
                f"{innertube_url}?key={api_key}",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract captions from player response
                captions = data.get('captions', {})
                if captions:
                    logger.info("✅ Found captions in Innertube API response")
                    
                    # Try to get caption tracks
                    caption_tracks = captions.get('playerCaptionsTracklistRenderer', {}).get('captionTracks', [])
                    
                    for track in caption_tracks:
                        try:
                            track_url = track.get('baseUrl')
                            if track_url:
                                caption_response = self.session.get(track_url, timeout=30)
                                if caption_response.status_code == 200:
                                    transcript = self._parse_subtitle_content(caption_response.text)
                                    if transcript and self._is_actual_speech(transcript):
                                        logger.info(f"🎤 Innertube transcript: {len(transcript)} chars")
                                        return transcript
                        except Exception as e:
                            logger.warning(f"❌ Caption track failed: {str(e)}")
                            continue
            
        except Exception as e:
            logger.warning(f"❌ Innertube API extraction failed: {str(e)}")
        
        return None
    
    async def _extract_with_caption_api(self, video_id: str) -> Optional[str]:
        """Extract using direct caption API calls"""
        try:
            logger.info(f"🔍 Method 3: Direct Caption API calls")
            
            # Multiple caption API endpoints
            caption_endpoints = [
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=srv3",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=hi&fmt=srv3",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=ttml",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=hi&fmt=ttml",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=vtt",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=hi&fmt=vtt",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=srv1",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=hi&fmt=srv1",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=srv2",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=hi&fmt=srv2",
            ]
            
            for endpoint in caption_endpoints:
                try:
                    logger.info(f"🌐 Trying caption API: {endpoint}")
                    response = self.session.get(endpoint, timeout=15)
                    
                    if response.status_code == 200:
                        transcript = self._parse_subtitle_content(response.text)
                        if transcript and self._is_actual_speech(transcript):
                            logger.info(f"🎤 Caption API transcript: {len(transcript)} chars")
                            return transcript
                            
                except Exception as e:
                    logger.warning(f"❌ Caption API {endpoint} failed: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.warning(f"❌ Caption API extraction failed: {str(e)}")
        
        return None
    
    async def _extract_with_professional_scraping(self, video_id: str) -> Optional[str]:
        """Extract using professional web scraping"""
        try:
            logger.info(f"🔍 Method 4: Professional web scraping")
            
            # Multiple scraping sources
            scraping_sources = [
                f"https://youtubetotranscript.com/transcript?v={video_id}",
                f"https://www.youtube.com/watch?v={video_id}",
                f"https://youtube.com/watch?v={video_id}",
                f"https://m.youtube.com/watch?v={video_id}",
            ]
            
            for source_url in scraping_sources:
                try:
                    logger.info(f"🌐 Scraping: {source_url}")
                    response = self.session.get(source_url, timeout=15)
                    
                    if response.status_code == 200:
                        if 'youtubetotranscript.com' in source_url:
                            # Parse transcript website
                            transcript = self._extract_from_transcript_website(response.text)
                            if transcript and self._is_actual_speech(transcript):
                                logger.info(f"🎤 Transcript website: {len(transcript)} chars")
                                return transcript
                        else:
                            # Parse YouTube page
                            transcript = self._extract_from_youtube_page(response.text)
                            if transcript and self._is_actual_speech(transcript):
                                logger.info(f"🎤 YouTube page: {len(transcript)} chars")
                                return transcript
                                
                except Exception as e:
                    logger.warning(f"❌ Scraping {source_url} failed: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.warning(f"❌ Professional scraping extraction failed: {str(e)}")
        
        return None
    
    async def _extract_with_third_party_services(self, video_id: str) -> Optional[str]:
        """Extract using third-party services"""
        try:
            logger.info(f"🔍 Method 5: Third-party services")
            
            # Try multiple third-party services
            third_party_sources = [
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=ttml",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=hi&fmt=ttml",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=vtt",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=hi&fmt=vtt",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=srv3",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=hi&fmt=srv3",
            ]
            
            for source_url in third_party_sources:
                try:
                    logger.info(f"🌐 Third-party: {source_url}")
                    response = self.session.get(source_url, timeout=15)
                    
                    if response.status_code == 200:
                        transcript = self._parse_subtitle_content(response.text)
                        if transcript and self._is_actual_speech(transcript):
                            logger.info(f"🎤 Third-party transcript: {len(transcript)} chars")
                            return transcript
                            
                except Exception as e:
                    logger.warning(f"❌ Third-party {source_url} failed: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.warning(f"❌ Third-party extraction failed: {str(e)}")
        
        return None
    
    def _is_actual_speech(self, text: str) -> bool:
        """Check if text contains actual speech content (not HTML/CSS)"""
        if not text or len(text) < 50:
            return False
        
        # Check for HTML/CSS indicators
        html_css_indicators = [
            'body{padding', 'margin:0', 'overflow-y:scroll', 'display:none',
            'webkit-', 'moz-', 'ms-', 'o-', 'grecaptcha-badge',
            'animationActivationEntityKey', 'trackingParams', 'youtubei/v1/',
            'window.ytcsi', 'ytcsi.tick', 'serializedShareEntity'
        ]
        
        text_lower = text.lower()
        for indicator in html_css_indicators:
            if indicator in text_lower:
                return False
        
        # Check for speech indicators
        speech_indicators = [
            'hello', 'hi', 'good', 'welcome', 'today', 'we', 'will', 'learn',
            'हाय', 'हेलो', 'गुड', 'वेलकम', 'आज', 'हम', 'सीखेंगे', 'करेंगे',
            'students', 'teacher', 'class', 'lecture', 'topic', 'chapter',
            'स्टूडेंट', 'टीचर', 'क्लास', 'लेक्चर', 'टॉपिक', 'चैप्टर',
            'explains', 'discusses', 'covers', 'teaches', 'shows', 'demonstrates'
        ]
        
        speech_count = sum(1 for indicator in speech_indicators if indicator in text_lower)
        
        # Must have at least 2 speech indicators and no HTML/CSS
        return speech_count >= 2
    
    def _parse_subtitle_content(self, content: str) -> str:
        """Parse subtitle content from various formats"""
        try:
            # First try JSON caption parsing (YouTube's new format)
            if '{' in content and 'wireMagic' in content:
                try:
                    from .json_caption_parser import JSONCaptionParser
                    parser = JSONCaptionParser()
                    json_result = parser.extract_clean_transcript(content)
                    if json_result and len(json_result) > 50:
                        logger.info(f"✅ JSON caption parsed: {len(json_result)} chars")
                        return json_result
                except Exception as e:
                    logger.warning(f"❌ JSON caption parsing failed: {str(e)}")
            
            # Try XML format (YouTube timedtext)
            if '<transcript>' in content or '<text' in content:
                try:
                    root = ET.fromstring(content)
                    texts = []
                    for text_elem in root.findall('.//text'):
                        text = text_elem.text
                        if text:
                            texts.append(text.strip())
                    return ' '.join(texts)
                except:
                    pass
            
            # Try SRT format
            if '-->' in content:
                srt_pattern = r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n(.+?)(?=\n\d+\n|\n\n|$)'
                matches = re.findall(srt_pattern, content, re.DOTALL)
                if matches:
                    return ' '.join([match.strip() for match in matches])
            
            # Try VTT format
            if 'WEBVTT' in content:
                vtt_pattern = r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\n(.+?)(?=\n\d{2}:\d{2}:\d{2}\.\d{3}|\n\n|$)'
                matches = re.findall(vtt_pattern, content, re.DOTALL)
                if matches:
                    return ' '.join([match.strip() for match in matches])
            
            # Try TTML format
            if '<tt' in content or '<p' in content:
                try:
                    root = ET.fromstring(content)
                    texts = []
                    for p_elem in root.findall('.//p'):
                        text = p_elem.text
                        if text:
                            texts.append(text.strip())
                    return ' '.join(texts)
                except:
                    pass
            
            # Fallback: extract all text content
            lines = content.split('\n')
            text_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.isdigit() and '-->' not in line and '<' not in line:
                    text_lines.append(line)
            
            return ' '.join(text_lines)
            
        except Exception as e:
            logger.warning(f"Error parsing subtitle content: {str(e)}")
            return content
    
    def _extract_from_transcript_website(self, html_content: str) -> str:
        """Extract transcript from transcript website"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text()
            
            # Look for speech content
            lines = text_content.split('\n')
            speech_lines = []
            
            for line in lines:
                line = line.strip()
                if (len(line) > 50 and 
                    (self._has_speech_content(line) or 
                     any(char in line for char in ['ह', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ']))):
                    speech_lines.append(line)
            
            return '\n'.join(speech_lines) if speech_lines else ''
            
        except Exception as e:
            logger.warning(f"Error extracting from transcript website: {str(e)}")
            return ''
    
    def _extract_from_youtube_page(self, html_content: str) -> str:
        """Extract transcript from YouTube page"""
        try:
            # Look for transcript data in script tags
            soup = BeautifulSoup(html_content, 'html.parser')
            scripts = soup.find_all('script')
            
            for script in scripts:
                if script.string and 'transcript' in script.string.lower():
                    # Try to extract transcript data
                    content = script.string
                    if 'captions' in content or 'subtitles' in content:
                        # This is where we'd parse the transcript data
                        # For now, return empty as this is complex
                        pass
            
            return ''
            
        except Exception as e:
            logger.warning(f"Error extracting from YouTube page: {str(e)}")
            return ''
    
    def _has_speech_content(self, text: str) -> bool:
        """Check if text contains actual speech content"""
        speech_indicators = [
            'hello', 'hi', 'good', 'welcome', 'today', 'we', 'will', 'learn',
            'हाय', 'हेलो', 'गुड', 'वेलकम', 'आज', 'हम', 'सीखेंगे', 'करेंगे',
            'students', 'teacher', 'class', 'lecture', 'topic', 'chapter',
            'स्टूडेंट', 'टीचर', 'क्लास', 'लेक्चर', 'टॉपिक', 'चैप्टर'
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in speech_indicators)
    
    def _clean_transcript_text(self, text: str) -> str:
        """Clean and format transcript text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove timestamps
        text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)
        
        # Clean up
        text = text.strip()
        
        return text
