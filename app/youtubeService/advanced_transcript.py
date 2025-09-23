#!/usr/bin/env python3
"""
Advanced YouTube transcript extraction using multiple working methods
"""
import requests
import logging
import json
import re
import time
from typing import Optional, List, Dict, Any
from bs4 import BeautifulSoup
import urllib.parse

logger = logging.getLogger(__name__)

class AdvancedTranscriptExtractor:
    """Advanced transcript extractor using multiple working methods"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    async def get_transcript(self, video_id: str) -> Optional[str]:
        """Get actual speech transcript using advanced methods"""
        logger.info(f"🎯 Advanced transcript extraction for video: {video_id}")
        
        # Method 1: Try YouTube Transcript API (most reliable)
        transcript = await self._try_youtube_api(video_id)
        if transcript:
            return transcript
        
        # Method 2: Try web scraping approach
        transcript = await self._try_web_scraping(video_id)
        if transcript:
            return transcript
        
        # Method 3: Try alternative transcript services
        transcript = await self._try_alternative_services(video_id)
        if transcript:
            return transcript
        
        # Method 4: Try direct video page parsing
        transcript = await self._try_direct_parsing(video_id)
        if transcript:
            return transcript
        
        logger.warning(f"❌ All advanced methods failed for {video_id}")
        return None
    
    async def _try_youtube_api(self, video_id: str) -> Optional[str]:
        """Try YouTube Transcript API with better error handling"""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            # Try to get transcript list
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            logger.info(f"📋 Available transcripts: {[t.language_code for t in transcript_list]}")
            
            # Try preferred languages first
            for transcript_info in transcript_list:
                if transcript_info.language_code in ['en', 'hi', 'en-US', 'en-GB', 'hi-IN']:
                    try:
                        transcript = transcript_info.fetch()
                        transcript_text = ' '.join([item['text'] for item in transcript])
                        logger.info(f"✅ YouTube API success: {transcript_info.language_code}")
                        logger.info(f"📝 Preview: {transcript_text[:200]}...")
                        return transcript_text
                    except Exception as e:
                        logger.warning(f"❌ Failed to fetch {transcript_info.language_code}: {e}")
                        continue
            
            # Try any available transcript
            if transcript_list:
                try:
                    transcript = transcript_list[0].fetch()
                    transcript_text = ' '.join([item['text'] for item in transcript])
                    logger.info(f"✅ YouTube API success: {transcript_list[0].language_code}")
                    logger.info(f"📝 Preview: {transcript_text[:200]}...")
                    return transcript_text
                except Exception as e:
                    logger.warning(f"❌ Failed to fetch first available transcript: {e}")
                    
        except Exception as e:
            logger.warning(f"❌ YouTube API failed: {e}")
        
        return None
    
    async def _try_web_scraping(self, video_id: str) -> Optional[str]:
        """Try web scraping approach"""
        try:
            logger.info(f"🔍 Trying web scraping for {video_id}")
            
            # Try different YouTube URLs
            urls = [
                f"https://www.youtube.com/watch?v={video_id}",
                f"https://youtu.be/{video_id}",
                f"https://m.youtube.com/watch?v={video_id}"
            ]
            
            for url in urls:
                try:
                    response = self.session.get(url, timeout=15)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Look for transcript data in various places
                        transcript_text = self._extract_from_html(soup, video_id)
                        if transcript_text:
                            logger.info(f"✅ Web scraping success!")
                            logger.info(f"📝 Preview: {transcript_text[:200]}...")
                            return transcript_text
                            
                except Exception as e:
                    logger.warning(f"❌ Web scraping failed for {url}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"❌ Web scraping approach failed: {e}")
        
        return None
    
    async def _try_alternative_services(self, video_id: str) -> Optional[str]:
        """Try alternative transcript services"""
        try:
            logger.info(f"🔍 Trying alternative services for {video_id}")
            
            # Try youtubetotranscript.com approach
            transcript = await self._try_youtube_to_transcript(video_id)
            if transcript:
                return transcript
            
            # Try other services
            services = [
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=srv3",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=ttml",
                f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en&fmt=vtt",
            ]
            
            for service_url in services:
                try:
                    response = self.session.get(service_url, timeout=10)
                    if response.status_code == 200 and response.text:
                        # Check if it's not HTML junk
                        if "<html" not in response.text.lower() and "trackingParams" not in response.text:
                            transcript_text = self._parse_transcript_content(response.text)
                            if transcript_text:
                                logger.info(f"✅ Alternative service success!")
                                logger.info(f"📝 Preview: {transcript_text[:200]}...")
                                return transcript_text
                except Exception as e:
                    logger.warning(f"❌ Alternative service failed: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"❌ Alternative services failed: {e}")
        
        return None
    
    async def _try_direct_parsing(self, video_id: str) -> Optional[str]:
        """Try direct video page parsing"""
        try:
            logger.info(f"🔍 Trying direct parsing for {video_id}")
            
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            response = self.session.get(video_url, timeout=15)
            
            if response.status_code == 200:
                # Look for transcript data in the page content
                page_content = response.text
                
                # Try to find transcript data in various formats
                patterns = [
                    r'"captions":\s*({[^}]+})',
                    r'"transcript":\s*({[^}]+})',
                    r'"subtitles":\s*({[^}]+})',
                    r'"automaticCaptions":\s*({[^}]+})',
                    r'ytInitialData.*?captions.*?({.*?})',
                    r'ytInitialPlayerResponse.*?captions.*?({.*?})'
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, page_content, re.DOTALL)
                    if matches:
                        for match in matches:
                            try:
                                # Try to parse as JSON
                                data = json.loads(match)
                                transcript_text = self._extract_from_json_data(data)
                                if transcript_text:
                                    logger.info(f"✅ Direct parsing success!")
                                    logger.info(f"📝 Preview: {transcript_text[:200]}...")
                                    return transcript_text
                            except:
                                # Try to extract text directly
                                transcript_text = self._extract_text_from_match(match)
                                if transcript_text:
                                    logger.info(f"✅ Direct parsing success!")
                                    logger.info(f"📝 Preview: {transcript_text[:200]}...")
                                    return transcript_text
                                    
        except Exception as e:
            logger.warning(f"❌ Direct parsing failed: {e}")
        
        return None
    
    async def _try_youtube_to_transcript(self, video_id: str) -> Optional[str]:
        """Try youtubetotranscript.com approach"""
        try:
            logger.info(f"🔍 Trying youtubetotranscript.com for {video_id}")
            
            url = f"https://youtubetotranscript.com/transcript?v={video_id}"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Look for transcript content
                transcript_selectors = [
                    '.transcript-content',
                    '.transcript-text',
                    '.transcript',
                    '#transcript',
                    '.content',
                    '.text'
                ]
                
                for selector in transcript_selectors:
                    elements = soup.select(selector)
                    for element in elements:
                        text = element.get_text(strip=True)
                        if text and len(text) > 100:  # Meaningful content
                            # Clean up the text
                            clean_text = self._clean_transcript_text(text)
                            if clean_text:
                                logger.info(f"✅ youtubetotranscript.com success!")
                                logger.info(f"📝 Preview: {clean_text[:200]}...")
                                return clean_text
                                
        except Exception as e:
            logger.warning(f"❌ youtubetotranscript.com failed: {e}")
        
        return None
    
    def _extract_from_html(self, soup: BeautifulSoup, video_id: str) -> Optional[str]:
        """Extract transcript from HTML content"""
        try:
            # Look for various transcript indicators
            transcript_indicators = [
                'transcript', 'caption', 'subtitle', 'speech', 'text'
            ]
            
            for indicator in transcript_indicators:
                # Look for elements containing transcript data
                elements = soup.find_all(text=re.compile(indicator, re.I))
                for element in elements:
                    parent = element.parent
                    if parent:
                        text = parent.get_text(strip=True)
                        if text and len(text) > 100:
                            clean_text = self._clean_transcript_text(text)
                            if clean_text:
                                return clean_text
                                
        except Exception as e:
            logger.warning(f"❌ HTML extraction failed: {e}")
        
        return None
    
    def _extract_from_json_data(self, data: Dict) -> Optional[str]:
        """Extract transcript from JSON data"""
        try:
            # Navigate through the JSON structure to find transcript
            if isinstance(data, dict):
                # Look for common transcript keys
                transcript_keys = ['captions', 'transcript', 'subtitles', 'automaticCaptions']
                
                for key in transcript_keys:
                    if key in data:
                        transcript_data = data[key]
                        if isinstance(transcript_data, dict):
                            # Look for caption tracks
                            if 'captionTracks' in transcript_data:
                                tracks = transcript_data['captionTracks']
                                if isinstance(tracks, list) and tracks:
                                    # Get the first track
                                    track = tracks[0]
                                    if 'baseUrl' in track:
                                        # Download the caption
                                        response = self.session.get(track['baseUrl'], timeout=10)
                                        if response.status_code == 200:
                                            return self._parse_transcript_content(response.text)
                                            
        except Exception as e:
            logger.warning(f"❌ JSON extraction failed: {e}")
        
        return None
    
    def _extract_text_from_match(self, match: str) -> Optional[str]:
        """Extract text from regex match"""
        try:
            # Try to extract text content from the match
            # Remove HTML tags
            clean_text = re.sub(r'<[^>]+>', ' ', match)
            # Remove extra whitespace
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            if len(clean_text) > 100:
                return clean_text
                
        except Exception as e:
            logger.warning(f"❌ Text extraction failed: {e}")
        
        return None
    
    def _parse_transcript_content(self, content: str) -> Optional[str]:
        """Parse transcript content in various formats"""
        try:
            lines = []
            
            # Try to parse as JSON first
            try:
                data = json.loads(content)
                if 'events' in data:
                    for event in data['events']:
                        if 'segs' in event:
                            for seg in event['segs']:
                                if 'utf8' in seg:
                                    text = seg['utf8'].strip()
                                    if text and len(text) > 2:
                                        lines.append(text)
                if lines:
                    return ' '.join(lines)
            except:
                pass
            
            # Try to parse as VTT
            if 'WEBVTT' in content:
                vtt_lines = content.split('\n')
                for line in vtt_lines:
                    line = line.strip()
                    if (line and 
                        not line.startswith('WEBVTT') and 
                        not line.startswith('NOTE') and 
                        not '-->' in line and 
                        not line.isdigit() and
                        len(line) > 3):
                        clean_line = re.sub(r'<[^>]+>', '', line)
                        if clean_line:
                            lines.append(clean_line)
                if lines:
                    return ' '.join(lines)
            
            # Try to parse as XML/TTML
            if '<tt' in content or '<transcript' in content:
                # Extract text from XML tags
                text_pattern = r'<[^>]*>([^<]+)</[^>]*>'
                matches = re.findall(text_pattern, content)
                for match in matches:
                    text = match.strip()
                    if text and len(text) > 3:
                        lines.append(text)
                if lines:
                    return ' '.join(lines)
            
            return None
            
        except Exception as e:
            logger.warning(f"❌ Content parsing failed: {e}")
            return None
    
    def _clean_transcript_text(self, text: str) -> str:
        """Clean transcript text"""
        if not text:
            return ""
        
        # Remove common junk patterns
        junk_patterns = [
            "trackingParams", "\\u0026", "ytcfg", "WIZ_global_data",
            "window.yt", "ytInitialData", "ytInitialPlayerResponse",
            "clickTrackingParams", "commandMetadata", "webCommandMetadata",
            "browseEndpoint", "watchEndpoint", "innertubeCommand",
            "animationActivationEntityKey", "youtubei/v1/"
        ]
        
        for pattern in junk_patterns:
            if pattern in text:
                return ""
        
        # Clean up HTML entities and extra whitespace
        text = re.sub(r"&[a-zA-Z0-9#]+;", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        
        # Check if it's meaningful content
        if len(text) < 50 or any(word in text.lower() for word in ['error', 'not found', 'unavailable']):
            return ""
        
        return text
