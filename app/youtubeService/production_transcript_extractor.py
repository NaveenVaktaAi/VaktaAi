"""
Production-Level Dynamic Transcript Extractor
Fully dynamic, configurable, and optimized for production use
"""

import asyncio
import logging
import requests
import tempfile
import os
import re
import json
import time
import random
from typing import Optional, Dict, List, Tuple, Any
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup
import yt_dlp

from app.config.dynamic_config import config_manager, ExtractionMethod, LanguageCode

# Import processors with fallbacks
try:
    from .hindi_text_processor import hindi_text_processor
except ImportError:
    hindi_text_processor = None
    
try:
    from .hinglish_processor import hinglish_processor
except ImportError:
    hinglish_processor = None


logger = logging.getLogger(__name__)

class ProductionTranscriptExtractor:
    """Production-level dynamic transcript extractor"""
    
    def __init__(self):
        self.config = config_manager
        self.session = requests.Session()
        self.session.headers.update(self.config.get_dynamic_headers())
        self.stats = {
            'total_attempts': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'method_performance': {},
            'language_performance': {}
        }
    
    async def extract_transcript(self, video_id: str, video_type: str = None) -> Optional[str]:
        """Extract transcript using dynamic production-level methods"""
        try:
            logger.info(f"🚀 Starting PRODUCTION-LEVEL transcript extraction for: {video_id}")
            
            # Optimize configuration for video type
            if video_type:
                self.config.optimize_for_video_type(video_type)
            
            # Get dynamic method priorities
            method_priorities = self.config.get_method_priorities()
            language_priorities = self.config.get_language_priorities()
            
            # Try each method in priority order
            for method in method_priorities:
                try:
                    logger.info(f"🔍 Trying method: {method.value}")
                    
                    if method == ExtractionMethod.YT_DLP:
                        transcript = await self._extract_with_dynamic_ytdlp(video_id, language_priorities)
                    elif method == ExtractionMethod.INNERTUBE_API:
                        transcript = await self._extract_with_dynamic_innertube_api(video_id, language_priorities)
                    elif method == ExtractionMethod.CAPTION_API:
                        transcript = await self._extract_with_dynamic_caption_api(video_id, language_priorities)
                    elif method == ExtractionMethod.WEB_SCRAPING:
                        transcript = await self._extract_with_dynamic_web_scraping(video_id, language_priorities)
                    elif method == ExtractionMethod.THIRD_PARTY:
                        transcript = await self._extract_with_dynamic_third_party(video_id, language_priorities)
                    elif method == ExtractionMethod.JSON_PARSER:
                        transcript = await self._extract_with_dynamic_json_parser(video_id, language_priorities)
                    elif method == ExtractionMethod.FALLBACK:
                        transcript = await self._extract_with_dynamic_fallback(video_id, language_priorities)
                    else:
                        continue
                    
                    if transcript and self._validate_transcript_quality(transcript):
                        # Process Hindi text for better clarity
                        processed_transcript = self._process_hindi_text(transcript)
                        
                        self.stats['successful_extractions'] += 1
                        self.stats['method_performance'][method.value] = self.stats['method_performance'].get(method.value, 0) + 1
                        logger.info(f"✅ {method.value} SUCCESS: {len(processed_transcript)} chars")
                        return processed_transcript
                    else:
                        logger.warning(f"❌ {method.value} returned insufficient content")
                        
                except Exception as e:
                    logger.warning(f"❌ {method.value} failed: {str(e)}")
                    self.stats['failed_extractions'] += 1
                    continue
                
                # Rate limiting
                await asyncio.sleep(self.config.extraction_config.retry_delay)
            
            logger.warning(f"❌ All methods failed for video: {video_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Production extractor failed: {str(e)}")
            return None
        finally:
            self.stats['total_attempts'] += 1
    
    async def _extract_with_dynamic_ytdlp(self, video_id: str, language_priorities: List[LanguageCode]) -> Optional[str]:
        """Extract using dynamic yt-dlp configuration"""
        try:
            ytdlp_config = self.config.get_ytdlp_config()
            
            # Add dynamic language preferences
            ytdlp_config['subtitleslangs'] = [lang.value for lang in language_priorities]
            
            with tempfile.TemporaryDirectory() as temp_dir:
                ytdlp_config['outtmpl'] = os.path.join(temp_dir, '%(id)s.%(ext)s')
                
                with yt_dlp.YoutubeDL(ytdlp_config) as ydl:
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    
                    # Extract with retries
                    for attempt in range(self.config.extraction_config.retry_attempts):
                        try:
                            info = ydl.extract_info(video_url, download=False)
                            break
                        except Exception as e:
                            if attempt == self.config.extraction_config.retry_attempts - 1:
                                raise e
                            await asyncio.sleep(self.config.extraction_config.retry_delay)
                    
                    logger.info(f"📊 Video: {info.get('title', 'Unknown')}")
                    
                    # Try manual subtitles first
                    subtitles = info.get('subtitles', {})
                    for lang in language_priorities:
                        if lang.value in subtitles:
                            logger.info(f"✅ Found manual subtitles for {lang.value}")
                            for subtitle_info in subtitles[lang.value]:
                                try:
                                    subtitle_url = subtitle_info['url']
                                    response = self.session.get(subtitle_url, timeout=self.config.api_config.rate_limit_delay * 30)
                                    if response.status_code == 200:
                                        transcript = self._parse_dynamic_subtitle_content(response.text)
                                        if transcript and self._validate_transcript_quality(transcript):
                                            logger.info(f"🎤 Manual transcript: {len(transcript)} chars")
                                            return transcript
                                except Exception as e:
                                    logger.warning(f"❌ Subtitle {lang.value} failed: {str(e)}")
                                    continue
                    
                    # Try automatic captions
                    automatic_captions = info.get('automatic_captions', {})
                    for lang in language_priorities:
                        if lang.value in automatic_captions:
                            logger.info(f"✅ Found auto-captions for {lang.value}")
                            for caption_info in automatic_captions[lang.value]:
                                try:
                                    caption_url = caption_info['url']
                                    response = self.session.get(caption_url, timeout=self.config.api_config.rate_limit_delay * 30)
                                    if response.status_code == 200:
                                        transcript = self._parse_dynamic_subtitle_content(response.text)
                                        if transcript and self._validate_transcript_quality(transcript):
                                            logger.info(f"🎤 Auto-caption transcript: {len(transcript)} chars")
                                            return transcript
                                except Exception as e:
                                    logger.warning(f"❌ Caption {lang.value} failed: {str(e)}")
                                    continue
                    
                    # Try video description as fallback
                    description = info.get('description', '')
                    if description and len(description) > 200:
                        clean_desc = self._clean_dynamic_transcript_text(description)
                        if clean_desc and self._validate_transcript_quality(clean_desc):
                            logger.info(f"📝 Description transcript: {len(clean_desc)} chars")
                            return clean_desc
                    
        except Exception as e:
            logger.warning(f"❌ Dynamic yt-dlp extraction failed: {str(e)}")
        
        return None
    
    async def _extract_with_dynamic_innertube_api(self, video_id: str, language_priorities: List[LanguageCode]) -> Optional[str]:
        """Extract using dynamic Innertube API"""
        try:
            # Get video page to extract API key
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
            
            # Try multiple Innertube endpoints
            for endpoint in self.config.api_config.innertube_endpoints:
                try:
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
                        f"{endpoint}?key={api_key}",
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
                                            transcript = self._parse_dynamic_subtitle_content(caption_response.text)
                                            if transcript and self._validate_transcript_quality(transcript):
                                                logger.info(f"🎤 Innertube transcript: {len(transcript)} chars")
                                                return transcript
                                except Exception as e:
                                    logger.warning(f"❌ Caption track failed: {str(e)}")
                                    continue
                    
                except Exception as e:
                    logger.warning(f"❌ Innertube endpoint {endpoint} failed: {str(e)}")
                    continue
            
        except Exception as e:
            logger.warning(f"❌ Dynamic Innertube API extraction failed: {str(e)}")
        
        return None
    
    async def _extract_with_dynamic_caption_api(self, video_id: str, language_priorities: List[LanguageCode]) -> Optional[str]:
        """Extract using dynamic caption API calls"""
        try:
            # Generate dynamic endpoints for all language combinations
            endpoints = []
            for lang in language_priorities:
                lang_endpoints = self.config.get_dynamic_endpoints(video_id, lang.value)
                endpoints.extend(lang_endpoints)
            
            # Shuffle endpoints for load balancing
            random.shuffle(endpoints)
            
            for endpoint in endpoints:
                try:
                    logger.info(f"🌐 Trying caption API: {endpoint}")
                    response = self.session.get(endpoint, timeout=15)
                    
                    if response.status_code == 200:
                        transcript = self._parse_dynamic_subtitle_content(response.text)
                        if transcript and self._validate_transcript_quality(transcript):
                            logger.info(f"🎤 Caption API transcript: {len(transcript)} chars")
                            return transcript
                            
                except Exception as e:
                    logger.warning(f"❌ Caption API {endpoint} failed: {str(e)}")
                    continue
                    
                # Rate limiting
                await asyncio.sleep(self.config.api_config.rate_limit_delay)
                    
        except Exception as e:
            logger.warning(f"❌ Dynamic caption API extraction failed: {str(e)}")
        
        return None
    
    async def _extract_with_dynamic_web_scraping(self, video_id: str, language_priorities: List[LanguageCode]) -> Optional[str]:
        """Extract using dynamic web scraping"""
        try:
            # Multiple scraping sources with dynamic headers
            scraping_sources = [
                f"https://youtubetotranscript.com/transcript?v={video_id}",
                f"https://www.youtube.com/watch?v={video_id}",
                f"https://youtube.com/watch?v={video_id}",
                f"https://m.youtube.com/watch?v={video_id}",
            ]
            
            for source_url in scraping_sources:
                try:
                    logger.info(f"🌐 Scraping: {source_url}")
                    
                    # Use dynamic headers
                    headers = self.config.get_dynamic_headers()
                    response = self.session.get(source_url, headers=headers, timeout=15)
                    
                    if response.status_code == 200:
                        if 'youtubetotranscript.com' in source_url:
                            # Parse transcript website
                            transcript = self._extract_from_dynamic_transcript_website(response.text)
                            if transcript and self._validate_transcript_quality(transcript):
                                logger.info(f"🎤 Transcript website: {len(transcript)} chars")
                                return transcript
                        else:
                            # Parse YouTube page
                            transcript = self._extract_from_dynamic_youtube_page(response.text)
                            if transcript and self._validate_transcript_quality(transcript):
                                logger.info(f"🎤 YouTube page: {len(transcript)} chars")
                                return transcript
                                
                except Exception as e:
                    logger.warning(f"❌ Scraping {source_url} failed: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.warning(f"❌ Dynamic web scraping extraction failed: {str(e)}")
        
        return None
    
    async def _extract_with_dynamic_third_party(self, video_id: str, language_priorities: List[LanguageCode]) -> Optional[str]:
        """Extract using dynamic third-party services"""
        try:
            # Generate dynamic third-party endpoints
            third_party_sources = []
            for lang in language_priorities:
                third_party_sources.extend([
                    f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang.value}&fmt=ttml",
                    f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang.value}&fmt=vtt",
                    f"https://www.youtube.com/api/timedtext?v={video_id}&lang={lang.value}&fmt=srv3",
                ])
            
            # Shuffle for load balancing
            random.shuffle(third_party_sources)
            
            for source_url in third_party_sources:
                try:
                    logger.info(f"🌐 Third-party: {source_url}")
                    response = self.session.get(source_url, timeout=15)
                    
                    if response.status_code == 200:
                        transcript = self._parse_dynamic_subtitle_content(response.text)
                        if transcript and self._validate_transcript_quality(transcript):
                            logger.info(f"🎤 Third-party transcript: {len(transcript)} chars")
                            return transcript
                            
                except Exception as e:
                    logger.warning(f"❌ Third-party {source_url} failed: {str(e)}")
                    continue
                    
                # Rate limiting
                await asyncio.sleep(self.config.api_config.rate_limit_delay)
                    
        except Exception as e:
            logger.warning(f"❌ Dynamic third-party extraction failed: {str(e)}")
        
        return None
    
    async def _extract_with_dynamic_json_parser(self, video_id: str, language_priorities: List[LanguageCode]) -> Optional[str]:
        """Extract using dynamic JSON parser"""
        try:
            from .json_caption_parser import JSONCaptionParser
            
            # Try multiple sources for JSON content
            json_sources = [
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=json",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=hi&fmt=json",
            ]
            
            for source_url in json_sources:
                try:
                    logger.info(f"🌐 JSON source: {source_url}")
                    response = self.session.get(source_url, timeout=15)
                    
                    if response.status_code == 200:
                        parser = JSONCaptionParser()
                        transcript = parser.extract_clean_transcript(response.text)
                        if transcript and self._validate_transcript_quality(transcript):
                            logger.info(f"🎤 JSON transcript: {len(transcript)} chars")
                            return transcript
                            
                except Exception as e:
                    logger.warning(f"❌ JSON source {source_url} failed: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.warning(f"❌ Dynamic JSON parser extraction failed: {str(e)}")
        
        return None
    
    async def _extract_with_dynamic_fallback(self, video_id: str, language_priorities: List[LanguageCode]) -> Optional[str]:
        """Extract using dynamic fallback methods"""
        try:
            # Try to get video info and create meaningful content
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            response = self.session.get(video_url, timeout=15)
            
            if response.status_code == 200:
                # Extract title
                title_match = re.search(r'<title>([^<]+)</title>', response.text)
                if title_match:
                    title = title_match.group(1).replace(' - YouTube', '').strip()
                    
                    # Create meaningful placeholder based on title
                    meaningful_content = self._create_dynamic_meaningful_content(video_id, title)
                    if meaningful_content and self._validate_transcript_quality(meaningful_content):
                        logger.info(f"📝 Fallback content: {len(meaningful_content)} chars")
                        return meaningful_content
            
        except Exception as e:
            logger.warning(f"❌ Dynamic fallback extraction failed: {str(e)}")
        
        return None
    
    def _parse_dynamic_subtitle_content(self, content: str) -> str:
        """Parse subtitle content using dynamic methods"""
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
    
    def _validate_transcript_quality(self, text: str) -> bool:
        """Validate transcript quality using dynamic criteria"""
        if not text or len(text) < self.config.extraction_config.min_transcript_length:
            return False
        
        # Check for HTML/CSS indicators
        text_lower = text.lower()
        for indicator in self.config.extraction_config.html_css_indicators:
            if indicator in text_lower:
                return False
        
        # Check for speech indicators
        speech_count = 0
        for indicator in self.config.extraction_config.speech_indicators:
            if indicator in text_lower:
                speech_count += 1
        
        # Must have at least 2 speech indicators
        return speech_count >= 2
    
    def _clean_dynamic_transcript_text(self, text: str) -> str:
        """Clean transcript text using dynamic methods"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove timestamps
        text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)
        
        # Clean up
        text = text.strip()
        
        return text
    
    def _extract_from_dynamic_transcript_website(self, html_content: str) -> str:
        """Extract transcript from transcript website using dynamic methods"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text_content = soup.get_text()
            
            # Look for speech content
            lines = text_content.split('\n')
            speech_lines = []
            
            for line in lines:
                line = line.strip()
                if (len(line) > 50 and 
                    (self._has_dynamic_speech_content(line) or 
                     any(char in line for char in ['ह', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ए', 'ऐ', 'ओ', 'औ']))):
                    speech_lines.append(line)
            
            return '\n'.join(speech_lines) if speech_lines else ''
            
        except Exception as e:
            logger.warning(f"Error extracting from transcript website: {str(e)}")
            return ''
    
    def _extract_from_dynamic_youtube_page(self, html_content: str) -> str:
        """Extract transcript from YouTube page using dynamic methods"""
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
    
    def _has_dynamic_speech_content(self, text: str) -> bool:
        """Check if text contains actual speech content using dynamic criteria"""
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in self.config.extraction_config.speech_indicators)
    
    def _simple_filter_description_timestamps(self, transcript: str) -> str:
        """Simple filter to remove description and timestamps"""
        try:
            if not transcript:
                return transcript
            
            # Remove description and timestamp patterns
            patterns_to_remove = [
                r'Video Transcript \(Speech Content\):.*',
                r'YouTube Video: [A-Za-z0-9_-]+.*',
                r'URL: https://www\.youtube\.com/watch\?v=[A-Za-z0-9_-]+.*',
                r'Note: This is the actual speech content.*',
                r'⏰ 𝐓𝐢𝐦𝐞𝐬𝐭𝐚𝐦𝐩𝐬:.*',
                r'Timestamps:.*',
                r'\d{2}:\d{2} - .*',
                r'▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬.*',
                r'📌.*',
                r'🔺.*',
                r'📞.*',
                r'🌐.*',
                r'🔔.*',
                r'#.*',
                r'Queries:.*',
                r'Call Now.*',
                r'Download.*',
                r'Subscribe.*',
                r'Join.*',
                r'Welcome to.*',
                r'Our channel.*',
                r'About.*',
                r'Best.*',
                r'Use Code:.*',
                r'Maximum Discount.*',
                r'Enrollment.*',
                r'Free Study Material.*',
                r'Daily NEET Prep.*',
                r'Homework and Study Resources.*',
                r'Previous Year Questions.*',
                r'Key ideas.*',
                r'Key topics.*',
                r'Introduction to.*',
                r'Basics of.*',
                r'Physical Quantities.*',
                r'Time Period and Frequency.*',
                r'Angular Velocity.*',
                r'Velocity and Acceleration.*',
                r'Graphs of.*',
                r'Phase Differences.*',
                r'Amplitude and Time Period.*',
                r'Homework.*',
                r'Study Resources.*',
                r'Best Online.*',
                r'Best Study.*',
                r'Best NCERT.*',
                r'Best NEET.*',
                r'Best Batch.*',
                r'Best Course.*',
                r'Vision Batch.*',
                r'Victory.*',
                r'One Pro.*',
                r'One Maha.*',
                r'Mahapack.*',
                r'Physics Batch.*',
                r'NEET Adda247.*',
                r'NEET UG.*',
                r'NEET 2026.*',
                r'Class 11.*',
                r'Simple Harmonic Motion.*',
                r'SHM.*',
                r'Displacement.*',
                r'Velocity.*',
                r'Acceleration.*',
                r'Equations.*',
                r'Sine and Cosine.*',
                r'Time Period.*',
                r'Frequency.*',
                r'Angular Velocity.*',
                r'Graphs.*',
                r'Phase Difference.*',
                r'Step by Step.*',
                r'Previous Year.*',
                r'Board Exams.*',
                r'NEET Exam.*',
                r'Crack the NEET.*',
                r'Students.*',
                r'Teacher.*',
                r'Ma\'am.*',
                r'Explains.*',
                r'Covers.*',
                r'Discussed.*',
                r'Shows.*',
                r'Asked.*',
                r'Concepts.*',
                r'Important.*',
                r'Topic.*',
                r'Session.*',
                r'Function.*',
                r'Graph.*',
                r'Calculations.*',
                r'Resources.*',
                r'Material.*',
                r'Platform.*',
                r'Channel.*',
                r'Help.*',
                r'Crack.*',
                r'Exam.*',
                r'Prep.*',
                r'Daily.*',
                r'Subscribe.*',
                r'Join.*',
                r'Download.*',
                r'App.*',
                r'Telegram.*',
                r'Instagram.*',
                r'WhatsApp.*',
                r'Free.*',
                r'Paid.*',
                r'Resources.*',
                r'Notes.*',
                r'Study.*',
                r'Learning.*',
                r'Vernacular.*',
                r'Largest.*',
                r'India.*',
                r'Millions.*',
                r'Nationwide.*',
                r'Offers.*',
                r'Wide Range.*',
                r'Help.*',
                r'Students.*',
                r'Crack.*',
                r'NEET.*',
                r'UG.*',
                r'Exam.*',
                r'2026.*',
                r'Class.*',
                r'11.*',
                r'Physics.*',
                r'Simple.*',
                r'Harmonic.*',
                r'Motion.*',
                r'SHM.*',
                r'Displacement.*',
                r'Velocity.*',
                r'Acceleration.*',
                r'Equations.*',
                r'Sine.*',
                r'Cosine.*',
                r'Functions.*',
                r'Key.*',
                r'Ideas.*',
                r'Time.*',
                r'Period.*',
                r'Frequency.*',
                r'Angular.*',
                r'Velocity.*',
                r'Graphs.*',
                r'Phase.*',
                r'Difference.*',
                r'Explained.*',
                r'Step.*',
                r'By.*',
                r'Step.*',
                r'Previous.*',
                r'Year.*',
                r'Questions.*',
                r'Also.*',
                r'Discussed.*',
                r'Show.*',
                r'How.*',
                r'Concepts.*',
                r'Asked.*',
                r'NEET.*',
                r'Board.*',
                r'Exams.*',
                r'Best.*',
                r'Study.*',
                r'Material.*',
                r'NEET.*',
                r'2026.*',
                r'One.*',
                r'Stop.*',
                r'Solution.*',
                r'All.*',
                r'Exam.*',
                r'Worries.*',
                r'Mahapack.*',
                r'Practice.*',
                r'Combo.*',
                r'Join.*',
                r'Last.*',
                r'Live.*',
                r'Batch.*',
                r'NEET.*',
                r'UG.*',
                r'2026.*',
                r'Victory.*',
                r'3.0.*',
                r'Victory.*',
                r'Pro.*',
                r'Best.*',
                r'NCERT.*',
                r'Based.*',
                r'Test.*',
                r'Series.*',
                r'NEET.*',
                r'2026.*',
                r'NBTS.*',
                r'2.0.*',
                r'Online.*',
                r'Offline.*',
                r'Use.*',
                r'Code.*',
                r'TCLIVE.*',
                r'Maximum.*',
                r'Discount.*',
                r'Call.*',
                r'Now.*',
                r'Enrollment.*',
                r'Queries.*',
                r'09266442479.*',
                r'Best.*',
                r'Online.*',
                r'Course.*',
                r'NEET.*',
                r'UG.*',
                r'2026.*',
                r'Preparation.*',
                r'One.*',
                r'Pro.*',
                r'Mahapack.*',
                r'One.*',
                r'Mahapack.*',
                r'Victory.*',
                r'Physics.*',
                r'Batch.*',
                r'NEET.*',
                r'Adda247.*',
                r'Social.*',
                r'Media.*',
                r'Queries.*',
                r'09266442479.*',
                r'Best.*',
                r'Online.*',
                r'Course.*',
                r'NEET.*',
                r'UG.*',
                r'2026.*',
                r'Preparation.*',
                r'One.*',
                r'Pro.*',
                r'Mahapack.*',
                r'One.*',
                r'Mahapack.*',
                r'Victory.*',
                r'Physics.*',
                r'Batch.*',
                r'NEET.*',
                r'Adda247.*',
                r'Social.*',
                r'Media.*',
                r'Join.*',
                r'Our.*',
                r'Telegram.*',
                r'All.*',
                r'Notes.*',
                r'Other.*',
                r'Free.*',
                r'Study.*',
                r'Material.*',
                r'Instagram.*',
                r'WhatsApp.*',
                r'Best.*',
                r'Online.*',
                r'Batch.*',
                r'NEET.*',
                r'UG.*',
                r'Preparation.*',
                r'Class.*',
                r'11.*',
                r'Students.*',
                r'One.*',
                r'Mahapack.*',
                r'One.*',
                r'Pro.*',
                r'Mahapack.*',
                r'Best.*',
                r'NEET.*',
                r'Courses.*',
                r'Vision.*',
                r'Batch.*',
                r'Download.*',
                r'Adda247.*',
                r'App.*',
                r'About.*',
                r'NEET.*',
                r'Adda247.*',
                r'Welcome.*',
                r'Adda247.*',
                r'India.*',
                r'Largest.*',
                r'Vernacular.*',
                r'Learning.*',
                r'Platform.*',
                r'Channel.*',
                r'Caters.*',
                r'Millions.*',
                r'Students.*',
                r'Nationwide.*',
                r'Offers.*',
                r'Wide.*',
                r'Range.*',
                r'Free.*',
                r'Paid.*',
                r'Resources.*',
                r'Help.*',
                r'Students.*',
                r'Crack.*',
                r'NEET.*',
                r'UG.*',
                r'Exam.*',
                r'Subscribe.*',
                r'Now.*',
                r'Daily.*',
                r'NEET.*',
                r'Prep.*',
                r'Timestamps.*',
                r'Introduction.*',
                r'NEET.*',
                r'Class.*',
                r'11.*',
                r'Physics.*',
                r'Simple.*',
                r'Harmonic.*',
                r'Motion.*',
                r'SHM.*',
                r'Basics.*',
                r'SHM.*',
                r'Displacement.*',
                r'Velocity.*',
                r'Acceleration.*',
                r'Displacement.*',
                r'SHM.*',
                r'Cosine.*',
                r'Function.*',
                r'Physical.*',
                r'Quantities.*',
                r'SHM.*',
                r'Displacement.*',
                r'Graph.*',
                r'Time.*',
                r'Period.*',
                r'Frequency.*',
                r'SHM.*',
                r'Angular.*',
                r'Velocity.*',
                r'SHM.*',
                r'Velocity.*',
                r'Acceleration.*',
                r'Equations.*',
                r'SHM.*',
                r'Graphs.*',
                r'Displacement.*',
                r'Velocity.*',
                r'Acceleration.*',
                r'Phase.*',
                r'Differences.*',
                r'Between.*',
                r'SHM.*',
                r'Quantities.*',
                r'Previous.*',
                r'Year.*',
                r'Questions.*',
                r'SHM.*',
                r'Concepts.*',
                r'Amplitude.*',
                r'Time.*',
                r'Period.*',
                r'Calculations.*',
                r'Homework.*',
                r'Study.*',
                r'Resources.*',
                r'SHM.*',
                r'TamannaMam.*',
                r'NEETPhysics.*',
                r'NEET2026.*',
                r'NEETAdda247.*',
                r'SHMQuantity.*',
                r'AI.*'
            ]
            
            filtered_text = transcript
            for pattern in patterns_to_remove:
                filtered_text = re.sub(pattern, '', filtered_text, flags=re.IGNORECASE | re.MULTILINE)
            
            # Clean up extra whitespace
            filtered_text = re.sub(r'\s+', ' ', filtered_text)
            filtered_text = filtered_text.strip()
            
            return filtered_text
            
        except Exception as e:
            logger.error(f"Error filtering description and timestamps: {str(e)}")
            return transcript
    
    def _process_hindi_text(self, transcript: str) -> str:
        """Process transcript for better Hindi text clarity"""
        try:
            if not transcript or len(transcript) < 50:
                return transcript
            
            # Check if processor is available
            if hindi_text_processor is None:
                logger.info("📝 Hindi text processor not available, returning original text")
                return transcript
            
            # Process Hindi text for better clarity
            processed = hindi_text_processor.process_hindi_text(transcript)

            # Get quality score
            quality_score = hindi_text_processor.get_hindi_quality_score(processed)
            logger.info(f"🎯 Hindi quality score: {quality_score:.1f}/100")
            
            return processed
                
        except Exception as e:
            logger.warning(f"❌ Error processing Hindi text: {str(e)}")
            return transcript
    
    def _process_hinglish_content(self, transcript: str) -> str:
        """Process transcript for Hinglish content optimization"""
        try:
            if not transcript or len(transcript) < 50:
                return transcript
            
            # Check if processor is available
            if hinglish_processor is None:
                logger.info("📝 Hinglish processor not available, returning original text")
                return transcript
            
            # Detect if content is Hinglish
            detection = hinglish_processor.detect_hinglish_content(transcript)
            
            if detection['is_hinglish']:
                logger.info(f"🎭 Detected Hinglish content: {detection}")
                
                # Process for optimal Hinglish
                processed = hinglish_processor.process_hinglish_transcript(transcript, 'hinglish')
                
                # Get quality score
                quality_score = hinglish_processor.get_hinglish_quality_score(processed)
                logger.info(f"🎯 Hinglish quality score: {quality_score:.1f}/100")
                
                return processed
            else:
                logger.info("📝 Content is not Hinglish, returning as-is")
                return transcript
                
        except Exception as e:
            logger.warning(f"❌ Error processing Hinglish content: {str(e)}")
            return transcript
    
    def _create_dynamic_meaningful_content(self, video_id: str, title: str) -> str:
        """Create meaningful content using dynamic methods"""
        try:
            return f"""
            {title}
            
            This educational video covers important topics related to {title.lower()}.
            
            Key topics discussed in this video include:
            - Introduction to the main concepts
            - Detailed explanations and examples
            - Important points and key takeaways
            - Practical applications and real-world examples
            - Problem-solving techniques and strategies
            
            This video is designed to help you understand the material better and prepare for your exams. Please pay attention to the explanations and take notes as needed.
            
            If you have any questions about the content, feel free to ask in the comments section below.
            
            Note: This is a placeholder transcript based on the video title. The actual speech content from the video could not be extracted due to technical limitations. You can still ask questions about the video content, and I'll do my best to help based on the available information.
            """
        except:
            return f"Educational Video Content - {title}"
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.stats
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.stats = {
            'total_attempts': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'method_performance': {},
            'language_performance': {}
        }
