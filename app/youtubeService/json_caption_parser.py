"""
JSON Caption Parser for YouTube Transcripts
Parses YouTube's JSON caption format to extract clean speech text
"""

import json
import re
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class JSONCaptionParser:
    """Parser for YouTube's JSON caption format"""
    
    def __init__(self):
        self.speech_indicators = [
            'hello', 'hi', 'good', 'welcome', 'today', 'we', 'will', 'learn',
            'हाय', 'हेलो', 'गुड', 'वेलकम', 'आज', 'हम', 'सीखेंगे', 'करेंगे',
            'students', 'teacher', 'class', 'lecture', 'topic', 'chapter',
            'स्टूडेंट', 'टीचर', 'क्लास', 'लेक्चर', 'टॉपिक', 'चैप्टर',
            'hey', 'everyone', 'back', 'exciting', 'video', 'going', 'see',
            'models', 'locally', 'using', 'docker', 'supports', 'containerization',
            'create', 'containers', 'images', 'right', 'inside', 'cli'
        ]
    
    def parse_json_caption(self, json_content: str) -> Optional[str]:
        """Parse YouTube JSON caption format to extract clean speech text"""
        try:
            logger.info("🔍 Parsing JSON caption format")
            
            # Try to parse as JSON
            try:
                data = json.loads(json_content)
            except json.JSONDecodeError:
                # If not valid JSON, try to extract JSON from text
                json_match = re.search(r'\{.*\}', json_content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    logger.warning("❌ No valid JSON found in content")
                    return None
            
            # Extract speech text from events
            speech_text = self._extract_speech_from_events(data)
            
            if speech_text and len(speech_text) > 50:
                logger.info(f"✅ Extracted speech text: {len(speech_text)} chars")
                return speech_text
            else:
                logger.warning("❌ No speech text found in JSON")
                return None
                
        except Exception as e:
            logger.warning(f"❌ JSON caption parsing failed: {str(e)}")
            return None
    
    def _extract_speech_from_events(self, data: Dict) -> str:
        """Extract speech text from events array"""
        try:
            events = data.get('events', [])
            if not events:
                logger.warning("❌ No events found in JSON data")
                return ""
            
            speech_segments = []
            
            for event in events:
                if 'segs' in event:
                    segments = event['segs']
                    for seg in segments:
                        if 'utf8' in seg:
                            text = seg['utf8']
                            # Clean up the text
                            text = self._clean_text(text)
                            if text:  # Accept all non-empty text
                                speech_segments.append(text)
            
            # Join segments and clean up
            full_text = ' '.join(speech_segments)
            full_text = self._clean_final_text(full_text)
            
            # Check if we have meaningful content
            if full_text and len(full_text) > 10:
                logger.info(f"✅ Extracted {len(full_text)} chars from events")
                return full_text
            else:
                logger.warning("❌ No meaningful content extracted from events")
                return ""
            
        except Exception as e:
            logger.warning(f"❌ Error extracting speech from events: {str(e)}")
            return ""
    
    def _clean_text(self, text: str) -> str:
        """Clean individual text segments"""
        if not text:
            return ""
        
        # Remove newlines and extra spaces
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that are not speech
        text = re.sub(r'[^\w\s.,!?;:\'"-]', '', text)
        
        return text.strip()
    
    def _is_speech_content(self, text: str) -> bool:
        """Check if text contains actual speech content"""
        if not text or len(text) < 1:
            return False
        
        text_lower = text.lower().strip()
        
        # Skip empty or very short text
        if len(text_lower) < 1:
            return False
        
        # Check for speech indicators
        for indicator in self.speech_indicators:
            if indicator in text_lower:
                return True
        
        # Check for common words
        common_words = ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must', 'shall']
        if any(word in text_lower for word in common_words):
            return True
        
        # Check for punctuation (indicates speech)
        if any(char in text for char in ['.', ',', '!', '?', ';', ':']):
            return True
        
        # Check for capital letters (indicates proper nouns/speech)
        if any(char.isupper() for char in text):
            return True
        
        # Check if it's a meaningful word (not just numbers or symbols)
        if text_lower.isalpha() and len(text_lower) > 1:
            return True
        
        # Check for words with vowels (likely speech)
        vowels = 'aeiou'
        if any(vowel in text_lower for vowel in vowels) and len(text_lower) > 2:
            return True
        
        return False
    
    def _clean_final_text(self, text: str) -> str:
        """Clean the final extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove duplicate words/phrases
        words = text.split()
        cleaned_words = []
        prev_word = ""
        
        for word in words:
            if word != prev_word:
                cleaned_words.append(word)
            prev_word = word
        
        # Join and clean up
        text = ' '.join(cleaned_words)
        
        # Remove common artifacts
        text = re.sub(r'\b(acAsrConf|tOffsetMs|wWinId|aAppend|dDurationMs|tStartMs)\b', '', text)
        text = re.sub(r'\b\d+\b', '', text)  # Remove standalone numbers
        text = re.sub(r'\s+', ' ', text)  # Clean up spaces
        
        return text.strip()
    
    def extract_clean_transcript(self, raw_content: str) -> Optional[str]:
        """Main method to extract clean transcript from raw content"""
        try:
            logger.info("🚀 Starting clean transcript extraction")
            
            # First try JSON parsing
            if '{' in raw_content and '}' in raw_content:
                json_result = self.parse_json_caption(raw_content)
                if json_result:
                    return json_result
            
            # Fallback: try to extract text from the content
            text_result = self._extract_text_fallback(raw_content)
            if text_result:
                return text_result
            
            logger.warning("❌ No clean transcript could be extracted")
            return None
            
        except Exception as e:
            logger.warning(f"❌ Clean transcript extraction failed: {str(e)}")
            return None
    
    def _extract_text_fallback(self, content: str) -> Optional[str]:
        """Fallback method to extract text from content"""
        try:
            # Look for quoted text that might be speech
            quoted_texts = re.findall(r'"utf8":\s*"([^"]+)"', content)
            
            if quoted_texts:
                # Filter for speech content
                speech_texts = []
                for text in quoted_texts:
                    if self._is_speech_content(text):
                        speech_texts.append(text)
                
                if speech_texts:
                    full_text = ' '.join(speech_texts)
                    return self._clean_final_text(full_text)
            
            return None
            
        except Exception as e:
            logger.warning(f"❌ Fallback text extraction failed: {str(e)}")
            return None
