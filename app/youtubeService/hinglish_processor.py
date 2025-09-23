"""
Hinglish Transcript Processor
Handles mixed Hindi-English content processing and optimization
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class HinglishPattern:
    """Hinglish pattern for detection and processing"""
    pattern: str
    replacement: str
    description: str

class HinglishProcessor:
    """Process and optimize Hinglish (Hindi-English mixed) content"""
    
    def __init__(self):
        self.hinglish_patterns = self._initialize_hinglish_patterns()
        self.hindi_indicators = self._initialize_hindi_indicators()
        self.english_indicators = self._initialize_english_indicators()
        self.mixed_indicators = self._initialize_mixed_indicators()
    
    def _initialize_hinglish_patterns(self) -> List[HinglishPattern]:
        """Initialize common Hinglish patterns"""
        return [
            # Common Hinglish phrases
            HinglishPattern(r'\baaj\s+hum\b', 'today we', 'aaj hum -> today we'),
            HinglishPattern(r'\btoday\s+we\b', 'aaj hum', 'today we -> aaj hum'),
            HinglishPattern(r'\byeh\s+topic\b', 'this topic', 'yeh topic -> this topic'),
            HinglishPattern(r'\bthis\s+topic\b', 'yeh topic', 'this topic -> yeh topic'),
            HinglishPattern(r'\bphysics\s+mein\b', 'in physics', 'physics mein -> in physics'),
            HinglishPattern(r'\bin\s+physics\b', 'physics mein', 'in physics -> physics mein'),
            HinglishPattern(r'\bmaths\s+ka\b', 'of maths', 'maths ka -> of maths'),
            HinglishPattern(r'\bof\s+maths\b', 'maths ka', 'of maths -> maths ka'),
            HinglishPattern(r'\bvideo\s+mein\b', 'in this video', 'video mein -> in this video'),
            HinglishPattern(r'\bin\s+this\s+video\b', 'video mein', 'in this video -> video mein'),
            HinglishPattern(r'\bdekhte\s+hain\b', 'let\'s see', 'dekhte hain -> let\'s see'),
            HinglishPattern(r'\blet\'s\s+see\b', 'dekhte hain', 'let\'s see -> dekhte hain'),
            HinglishPattern(r'\bsamjha\s+deta\s+hun\b', 'let me explain', 'samjha deta hun -> let me explain'),
            HinglishPattern(r'\blet\s+me\s+explain\b', 'samjha deta hun', 'let me explain -> samjha deta hun'),
            
            # Common words
            HinglishPattern(r'\bpahle\b', 'first', 'pahle -> first'),
            HinglishPattern(r'\bfirst\b', 'pahle', 'first -> pahle'),
            HinglishPattern(r'\bphir\b', 'then', 'phir -> then'),
            HinglishPattern(r'\bthen\b', 'phir', 'then -> phir'),
            HinglishPattern(r'\bab\b', 'now', 'ab -> now'),
            HinglishPattern(r'\bnow\b', 'ab', 'now -> ab'),
            HinglishPattern(r'\bkyunki\b', 'because', 'kyunki -> because'),
            HinglishPattern(r'\bbecause\b', 'kyunki', 'because -> kyunki'),
            HinglishPattern(r'\bjab\b', 'when', 'jab -> when'),
            HinglishPattern(r'\bwhen\b', 'jab', 'when -> jab'),
            HinglishPattern(r'\bagar\b', 'if', 'agar -> if'),
            HinglishPattern(r'\bif\b', 'agar', 'if -> agar'),
            HinglishPattern(r'\btoh\b', 'so', 'toh -> so'),
            HinglishPattern(r'\bso\b', 'toh', 'so -> toh'),
            HinglishPattern(r'\blekin\b', 'but', 'lekin -> but'),
            HinglishPattern(r'\bbut\b', 'lekin', 'but -> lekin'),
            
            # Pronouns
            HinglishPattern(r'\bmain\b', 'I', 'main -> I'),
            HinglishPattern(r'\bI\b', 'main', 'I -> main'),
            HinglishPattern(r'\btum\b', 'you', 'tum -> you'),
            HinglishPattern(r'\byou\b', 'tum', 'you -> tum'),
            HinglishPattern(r'\bwoh\b', 'that', 'woh -> that'),
            HinglishPattern(r'\bthat\b', 'woh', 'that -> woh'),
            HinglishPattern(r'\byeh\b', 'this', 'yeh -> this'),
            HinglishPattern(r'\bthis\b', 'yeh', 'this -> yeh'),
            
            # Question words
            HinglishPattern(r'\bkya\b', 'what', 'kya -> what'),
            HinglishPattern(r'\bwhat\b', 'kya', 'what -> kya'),
            HinglishPattern(r'\bkaise\b', 'how', 'kaise -> how'),
            HinglishPattern(r'\bhow\b', 'kaise', 'how -> kaise'),
            HinglishPattern(r'\bkahan\b', 'where', 'kahan -> where'),
            HinglishPattern(r'\bwhere\b', 'kahan', 'where -> kahan'),
            HinglishPattern(r'\bkab\b', 'when', 'kab -> when'),
            HinglishPattern(r'\bwhen\b', 'kab', 'when -> kab'),
            HinglishPattern(r'\bkyun\b', 'why', 'kyun -> why'),
            HinglishPattern(r'\bwhy\b', 'kyun', 'why -> kyun'),
            HinglishPattern(r'\bkaun\b', 'who', 'kaun -> who'),
            HinglishPattern(r'\bwho\b', 'kaun', 'who -> kaun'),
        ]
    
    def _initialize_hindi_indicators(self) -> List[str]:
        """Initialize Hindi language indicators"""
        return [
            'हाय', 'हेलो', 'गुड', 'वेलकम', 'आज', 'हम', 'सीखेंगे', 'करेंगे',
            'स्टूडेंट', 'टीचर', 'क्लास', 'लेक्चर', 'टॉपिक', 'चैप्टर',
            'भौतिकी', 'गणित', 'विज्ञान', 'अध्ययन', 'पढ़ाई', 'शिक्षा',
            'मुख्य', 'महत्वपूर्ण', 'समझना', 'समझाना', 'बताना', 'दिखाना',
            'पहले', 'फिर', 'अब', 'क्योंकि', 'जब', 'अगर', 'तो', 'लेकिन',
            'मैं', 'तुम', 'वो', 'यह', 'क्या', 'कैसे', 'कहाँ', 'कब', 'क्यों', 'कौन'
        ]
    
    def _initialize_english_indicators(self) -> List[str]:
        """Initialize English language indicators"""
        return [
            'hello', 'hi', 'good', 'welcome', 'today', 'we', 'will', 'learn',
            'students', 'teacher', 'class', 'lecture', 'topic', 'chapter',
            'physics', 'maths', 'science', 'study', 'education', 'learning',
            'main', 'important', 'understand', 'explain', 'tell', 'show',
            'first', 'then', 'now', 'because', 'when', 'if', 'so', 'but',
            'I', 'you', 'that', 'this', 'what', 'how', 'where', 'when', 'why', 'who'
        ]
    
    def _initialize_mixed_indicators(self) -> List[str]:
        """Initialize mixed language indicators"""
        return [
            'hello guys', 'hi everyone', 'good morning', 'welcome back',
            'aaj hum', 'today we', 'let me explain', 'samjha deta hun',
            'students ko', 'for students', 'yeh topic', 'this topic',
            'physics mein', 'in physics', 'maths ka', 'of maths',
            'video mein', 'in this video', 'dekhte hain', 'let\'s see',
            'pahle', 'first', 'phir', 'then', 'ab', 'now', 'kyunki', 'because',
            'jab', 'when', 'agar', 'if', 'toh', 'so', 'lekin', 'but',
            'main', 'I', 'tum', 'you', 'woh', 'that', 'yeh', 'this',
            'kya', 'what', 'kaise', 'how', 'kahan', 'where', 'kab', 'when',
            'kyun', 'why', 'kaun', 'who', 'kisne', 'who did', 'kisko', 'whom'
        ]
    
    def detect_hinglish_content(self, text: str) -> Dict[str, any]:
        """Detect if content is Hinglish and analyze language mix"""
        try:
            text_lower = text.lower()
            
            # Count indicators
            hindi_count = sum(1 for indicator in self.hindi_indicators if indicator in text_lower)
            english_count = sum(1 for indicator in self.english_indicators if indicator in text_lower)
            mixed_count = sum(1 for indicator in self.mixed_indicators if indicator in text_lower)
            
            # Calculate percentages
            total_indicators = hindi_count + english_count + mixed_count
            if total_indicators == 0:
                return {
                    'is_hinglish': False,
                    'hindi_percentage': 0,
                    'english_percentage': 0,
                    'mixed_percentage': 0,
                    'confidence': 0
                }
            
            hindi_percentage = (hindi_count / total_indicators) * 100
            english_percentage = (english_count / total_indicators) * 100
            mixed_percentage = (mixed_count / total_indicators) * 100
            
            # Determine if it's Hinglish
            is_hinglish = (
                (hindi_percentage > 20 and english_percentage > 20) or  # Both languages present
                mixed_percentage > 30 or  # High mixed content
                (hindi_percentage > 10 and english_percentage > 10 and mixed_percentage > 10)  # All three present
            )
            
            confidence = min(100, (hindi_percentage + english_percentage + mixed_percentage) / 3)
            
            return {
                'is_hinglish': is_hinglish,
                'hindi_percentage': round(hindi_percentage, 2),
                'english_percentage': round(english_percentage, 2),
                'mixed_percentage': round(mixed_percentage, 2),
                'confidence': round(confidence, 2),
                'hindi_count': hindi_count,
                'english_count': english_count,
                'mixed_count': mixed_count
            }
            
        except Exception as e:
            logger.error(f"Error detecting Hinglish content: {str(e)}")
            return {
                'is_hinglish': False,
                'hindi_percentage': 0,
                'english_percentage': 0,
                'mixed_percentage': 0,
                'confidence': 0
            }
    
    def process_hinglish_transcript(self, text: str, target_language: str = 'hinglish') -> str:
        """Process transcript to optimize for Hinglish content"""
        try:
            if not text or len(text) < 10:
                return text
            
            # Detect current language mix
            detection = self.detect_hinglish_content(text)
            
            if not detection['is_hinglish']:
                logger.info("Content is not Hinglish, returning as-is")
                return text
            
            logger.info(f"Detected Hinglish content: {detection}")
            
            # Process based on target language
            if target_language.lower() == 'hinglish':
                return self._optimize_for_hinglish(text, detection)
            elif target_language.lower() == 'hindi':
                return self._convert_to_hindi(text)
            elif target_language.lower() == 'english':
                return self._convert_to_english(text)
            else:
                return text
                
        except Exception as e:
            logger.error(f"Error processing Hinglish transcript: {str(e)}")
            return text
    
    def _optimize_for_hinglish(self, text: str, detection: Dict) -> str:
        """Optimize text for Hinglish content"""
        try:
            # If already well-mixed, return as-is
            if detection['mixed_percentage'] > 40:
                return text
            
            # If too much Hindi, add some English
            if detection['hindi_percentage'] > 70:
                return self._add_english_mix(text)
            
            # If too much English, add some Hindi
            if detection['english_percentage'] > 70:
                return self._add_hindi_mix(text)
            
            # Apply Hinglish patterns
            processed_text = text
            for pattern in self.hinglish_patterns:
                processed_text = re.sub(pattern.pattern, pattern.replacement, processed_text, flags=re.IGNORECASE)
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Error optimizing for Hinglish: {str(e)}")
            return text
    
    def _add_english_mix(self, text: str) -> str:
        """Add English words to Hindi-heavy text"""
        try:
            # Common Hindi to English replacements for better mixing
            replacements = {
                'हम': 'hum',
                'मैं': 'main',
                'तुम': 'tum',
                'यह': 'yeh',
                'वो': 'woh',
                'क्या': 'kya',
                'कैसे': 'kaise',
                'कहाँ': 'kahan',
                'कब': 'kab',
                'क्यों': 'kyun',
                'कौन': 'kaun',
                'पहले': 'pahle',
                'फिर': 'phir',
                'अब': 'ab',
                'क्योंकि': 'kyunki',
                'जब': 'jab',
                'अगर': 'agar',
                'तो': 'toh',
                'लेकिन': 'lekin'
            }
            
            processed_text = text
            for hindi, hinglish in replacements.items():
                processed_text = processed_text.replace(hindi, hinglish)
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Error adding English mix: {str(e)}")
            return text
    
    def _add_hindi_mix(self, text: str) -> str:
        """Add Hindi words to English-heavy text"""
        try:
            # Common English to Hindi replacements for better mixing
            replacements = {
                'we': 'hum',
                'I': 'main',
                'you': 'tum',
                'this': 'yeh',
                'that': 'woh',
                'what': 'kya',
                'how': 'kaise',
                'where': 'kahan',
                'when': 'kab',
                'why': 'kyun',
                'who': 'kaun',
                'first': 'pahle',
                'then': 'phir',
                'now': 'ab',
                'because': 'kyunki',
                'when': 'jab',
                'if': 'agar',
                'so': 'toh',
                'but': 'lekin'
            }
            
            processed_text = text
            for english, hinglish in replacements.items():
                processed_text = re.sub(r'\b' + english + r'\b', hinglish, processed_text, flags=re.IGNORECASE)
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Error adding Hindi mix: {str(e)}")
            return text
    
    def _convert_to_hindi(self, text: str) -> str:
        """Convert text to Hindi"""
        try:
            # This would require a proper translation service
            # For now, just return as-is
            logger.info("Hindi conversion not implemented yet")
            return text
        except Exception as e:
            logger.error(f"Error converting to Hindi: {str(e)}")
            return text
    
    def _convert_to_english(self, text: str) -> str:
        """Convert text to English"""
        try:
            # This would require a proper translation service
            # For now, just return as-is
            logger.info("English conversion not implemented yet")
            return text
        except Exception as e:
            logger.error(f"Error converting to English: {str(e)}")
            return text
    
    def _clean_speech_content(self, text: str) -> str:
        """Clean text to extract only actual speech content"""
        try:
            if not text:
                return text
            
            # Remove common metadata patterns
            patterns_to_remove = [
                r'Video Transcript \(Speech Content\):',
                r'YouTube Video: [A-Za-z0-9_-]+',
                r'URL: https://www\.youtube\.com/watch\?v=[A-Za-z0-9_-]+',
                r'Note: This is the actual speech content.*',
                r'Video Information:.*',
                r'Video Content Summary:.*',
                r'▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬.*',
                r'📌.*',
                r'🔺.*',
                r'📞.*',
                r'🌐.*',
                r'🔔.*',
                r'⏰.*',
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
                r'Timestamps:.*',
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
                r'Students.*',
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
            
            cleaned_text = text
            for pattern in patterns_to_remove:
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE | re.MULTILINE)
            
            # Remove extra whitespace and newlines
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
            cleaned_text = cleaned_text.strip()
            
            # If cleaned text is too short, return original
            if len(cleaned_text) < 50:
                return text
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error cleaning speech content: {str(e)}")
            return text
    
    def get_hinglish_quality_score(self, text: str) -> float:
        """Get quality score for Hinglish content (0-100)"""
        try:
            detection = self.detect_hinglish_content(text)
            
            if not detection['is_hinglish']:
                return 0.0
            
            # Calculate quality score based on language balance
            balance_score = 100 - abs(detection['hindi_percentage'] - detection['english_percentage'])
            mixed_score = detection['mixed_percentage'] * 2
            confidence_score = detection['confidence']
            
            # Weighted average
            quality_score = (balance_score * 0.4) + (mixed_score * 0.3) + (confidence_score * 0.3)
            
            return min(100, max(0, quality_score))
            
        except Exception as e:
            logger.error(f"Error calculating Hinglish quality score: {str(e)}")
            return 0.0

# Global instance
hinglish_processor = HinglishProcessor()
