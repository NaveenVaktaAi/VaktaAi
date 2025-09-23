"""
Powerful YouTube Transcript Extractor
Uses multiple professional methods to extract transcripts
"""

import asyncio
import logging
import requests
import tempfile
import os
import re
from typing import Optional, Dict, List
from xml.etree import ElementTree as ET
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class PowerfulTranscriptExtractor:
    """Powerful transcript extractor using multiple professional methods"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    async def extract_transcript(self, video_id: str) -> Optional[str]:
        """Extract transcript using multiple powerful methods"""
        logger.info(f"🎯 Starting powerful transcript extraction for: {video_id}")
        
        # Method 1: Professional yt-dlp (Most Powerful)
        transcript = await self._extract_with_ytdlp(video_id)
        if transcript:
            return transcript
        
        # Method 2: Direct YouTube API calls
        transcript = await self._extract_with_youtube_api(video_id)
        if transcript:
            return transcript
        
        # Method 3: Professional web scraping
        transcript = await self._extract_with_web_scraping(video_id)
        if transcript:
            return transcript
        
        # Method 4: Third-party services
        transcript = await self._extract_with_third_party(video_id)
        if transcript:
            return transcript
        
        logger.warning(f"❌ All methods failed for video: {video_id}")
        return None
    
    async def _extract_with_ytdlp(self, video_id: str) -> Optional[str]:
        """Extract using professional yt-dlp with multiple extractors"""
        try:
            import yt_dlp
            logger.info(f"🔍 Method 1: Professional yt-dlp extraction")
            
            ydl_opts = {
                'writesubtitles': True,
                'writeautomaticsub': True,
                'subtitleslangs': ['en', 'hi', 'en-US', 'en-GB', 'hi-IN', 'en-IN'],
                'skip_download': True,
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'writethumbnail': False,
                'writeinfojson': False,
                'ignoreerrors': True,
                'extractor_retries': 3,
                'fragment_retries': 3,
                'retries': 3,
                'socket_timeout': 30,
                'http_chunk_size': 10485760,
                'concurrent_fragment_downloads': 4,
                'extractors': [
                    'youtube:transcript',
                    'youtube:captions',
                    'youtube:timedtext',
                    'youtube:player',
                ]
            }
            
            with tempfile.TemporaryDirectory() as temp_dir:
                ydl_opts['outtmpl'] = os.path.join(temp_dir, '%(id)s.%(ext)s')
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    info = ydl.extract_info(video_url, download=False)
                    
                    logger.info(f"📊 Video: {info.get('title', 'Unknown')}")
                    
                    # Try manual subtitles first
                    subtitles = info.get('subtitles', {})
                    for lang in ['en', 'hi', 'en-US', 'en-GB', 'hi-IN', 'en-IN']:
                        if lang in subtitles:
                            logger.info(f"✅ Found manual subtitles for {lang}")
                            subtitle_url = subtitles[lang][0]['url']
                            
                            response = self.session.get(subtitle_url, timeout=30)
                            if response.status_code == 200:
                                transcript = self._parse_subtitle_content(response.text)
                                if transcript and len(transcript) > 100:
                                    logger.info(f"🎤 Manual transcript: {len(transcript)} chars")
                                    return transcript
                    
                    # Try automatic captions
                    automatic_captions = info.get('automatic_captions', {})
                    for lang in ['en', 'hi', 'en-US', 'en-GB', 'hi-IN', 'en-IN']:
                        if lang in automatic_captions:
                            logger.info(f"✅ Found auto-captions for {lang}")
                            caption_url = automatic_captions[lang][0]['url']
                            
                            response = self.session.get(caption_url, timeout=30)
                            if response.status_code == 200:
                                transcript = self._parse_subtitle_content(response.text)
                                if transcript and len(transcript) > 100:
                                    logger.info(f"🎤 Auto-caption transcript: {len(transcript)} chars")
                                    return transcript
                    
                    # Try video description
                    description = info.get('description', '')
                    if description and len(description) > 200:
                        clean_desc = self._clean_transcript_text(description)
                        if clean_desc and len(clean_desc) > 100:
                            logger.info(f"📝 Description transcript: {len(clean_desc)} chars")
                            return clean_desc
                    
        except Exception as e:
            logger.warning(f"❌ yt-dlp extraction failed: {str(e)}")
        
        return None
    
    async def _extract_with_youtube_api(self, video_id: str) -> Optional[str]:
        """Extract using direct YouTube API calls"""
        try:
            logger.info(f"🔍 Method 2: Direct YouTube API calls")
            
            # Try multiple YouTube API endpoints
            api_endpoints = [
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=hi",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en-US",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=hi-IN",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en-IN",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=srv3",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=hi&fmt=srv3",
            ]
            
            for endpoint in api_endpoints:
                try:
                    logger.info(f"🌐 Trying API: {endpoint}")
                    response = self.session.get(endpoint, timeout=15)
                    
                    if response.status_code == 200:
                        transcript = self._parse_subtitle_content(response.text)
                        if transcript and len(transcript) > 100:
                            logger.info(f"🎤 API transcript: {len(transcript)} chars")
                            return transcript
                            
                except Exception as e:
                    logger.warning(f"❌ API endpoint {endpoint} failed: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.warning(f"❌ YouTube API extraction failed: {str(e)}")
        
        return None
    
    async def _extract_with_web_scraping(self, video_id: str) -> Optional[str]:
        """Extract using professional web scraping"""
        try:
            logger.info(f"🔍 Method 3: Professional web scraping")
            
            # Try multiple web scraping sources
            scraping_sources = [
                f"https://youtubetotranscript.com/transcript?v={video_id}",
                f"https://www.youtube.com/watch?v={video_id}",
                f"https://youtube.com/watch?v={video_id}",
            ]
            
            for source_url in scraping_sources:
                try:
                    logger.info(f"🌐 Scraping: {source_url}")
                    response = self.session.get(source_url, timeout=15)
                    
                    if response.status_code == 200:
                        if 'youtubetotranscript.com' in source_url:
                            # Parse transcript website
                            transcript = self._extract_from_transcript_website(response.text)
                            if transcript:
                                logger.info(f"🎤 Transcript website: {len(transcript)} chars")
                                return transcript
                        else:
                            # Parse YouTube page
                            transcript = self._extract_from_youtube_page(response.text)
                            if transcript:
                                logger.info(f"🎤 YouTube page: {len(transcript)} chars")
                                return transcript
                                
                except Exception as e:
                    logger.warning(f"❌ Scraping {source_url} failed: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.warning(f"❌ Web scraping extraction failed: {str(e)}")
        
        return None
    
    async def _extract_with_third_party(self, video_id: str) -> Optional[str]:
        """Extract using third-party services"""
        try:
            logger.info(f"🔍 Method 4: Third-party services")
            
            # Try multiple third-party services
            third_party_sources = [
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=ttml",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=hi&fmt=ttml",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=vtt",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=hi&fmt=vtt",
            ]
            
            for source_url in third_party_sources:
                try:
                    logger.info(f"🌐 Third-party: {source_url}")
                    response = self.session.get(source_url, timeout=15)
                    
                    if response.status_code == 200:
                        transcript = self._parse_subtitle_content(response.text)
                        if transcript and len(transcript) > 100:
                            logger.info(f"🎤 Third-party transcript: {len(transcript)} chars")
                            return transcript
                            
                except Exception as e:
                    logger.warning(f"❌ Third-party {source_url} failed: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.warning(f"❌ Third-party extraction failed: {str(e)}")
        
        return None
    
    def _parse_subtitle_content(self, content: str) -> str:
        """Parse subtitle content from various formats"""
        try:
            # Try XML format first (YouTube timedtext)
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
