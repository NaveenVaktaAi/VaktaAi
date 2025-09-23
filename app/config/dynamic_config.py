"""
Dynamic Configuration System for Production-Level YouTube Transcript Extraction
All methods, functions, and parameters are fully dynamic and configurable
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class ExtractionMethod(Enum):
    """Available extraction methods"""
    YT_DLP = "yt_dlp"
    INNERTUBE_API = "innertube_api"
    CAPTION_API = "caption_api"
    WEB_SCRAPING = "web_scraping"
    THIRD_PARTY = "third_party"
    JSON_PARSER = "json_parser"
    FALLBACK = "fallback"

class LanguageCode(Enum):
    """Supported language codes"""
    EN = "en"
    HI = "hi"
    EN_US = "en-US"
    EN_GB = "en-GB"
    HI_IN = "hi-IN"
    EN_IN = "en-IN"
    EN_AU = "en-AU"
    EN_CA = "en-CA"
    HINGLISH = "hinglish"
    MIXED = "mixed"

@dataclass
class ExtractionConfig:
    """Dynamic extraction configuration"""
    # Method priorities (order matters)
    method_priorities: List[ExtractionMethod] = None
    
    # Language preferences
    language_priorities: List[LanguageCode] = None
    
    # Quality thresholds
    min_transcript_length: int = 100
    max_processing_time: int = 60
    retry_attempts: int = 3
    retry_delay: float = 2.0
    
    # Content validation
    speech_indicators: List[str] = None
    html_css_indicators: List[str] = None
    
    # Performance settings
    concurrent_requests: int = 4
    request_timeout: int = 30
    chunk_size: int = 10485760
    
    # Fallback settings
    enable_fallback: bool = True
    fallback_placeholder: bool = True
    
    def __post_init__(self):
        if self.method_priorities is None:
            self.method_priorities = [
                ExtractionMethod.YT_DLP,
                ExtractionMethod.INNERTUBE_API,
                ExtractionMethod.CAPTION_API,
                ExtractionMethod.WEB_SCRAPING,
                ExtractionMethod.THIRD_PARTY,
                ExtractionMethod.JSON_PARSER,
                ExtractionMethod.FALLBACK
            ]
        
        if self.language_priorities is None:
            self.language_priorities = [
                LanguageCode.HINGLISH,
                LanguageCode.MIXED,
                LanguageCode.HI,
                LanguageCode.EN,
                LanguageCode.HI_IN,
                LanguageCode.EN_IN,
                LanguageCode.EN_US,
                LanguageCode.EN_GB,
                LanguageCode.EN_AU,
                LanguageCode.EN_CA
            ]
        
        if self.speech_indicators is None:
            self.speech_indicators = [
                # English indicators
                'hello', 'hi', 'good', 'welcome', 'today', 'we', 'will', 'learn',
                'students', 'teacher', 'class', 'lecture', 'topic', 'chapter',
                'hey', 'everyone', 'back', 'exciting', 'video', 'going', 'see',
                'models', 'locally', 'using', 'docker', 'supports', 'containerization',
                'create', 'containers', 'images', 'right', 'inside', 'cli',
                
                # Hindi indicators
                'हाय', 'हेलो', 'गुड', 'वेलकम', 'आज', 'हम', 'सीखेंगे', 'करेंगे',
                'स्टूडेंट', 'टीचर', 'क्लास', 'लेक्चर', 'टॉपिक', 'चैप्टर',
                'भौतिकी', 'गणित', 'विज्ञान', 'अध्ययन', 'पढ़ाई', 'शिक्षा',
                
                # Hinglish indicators (mixed Hindi-English)
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
        
        if self.html_css_indicators is None:
            self.html_css_indicators = [
                'body{padding', 'margin:0', 'overflow-y:scroll', 'display:none',
                'webkit-', 'moz-', 'ms-', 'o-', 'grecaptcha-badge',
                'animationActivationEntityKey', 'trackingParams', 'youtubei/v1/',
                'window.ytcsi', 'ytcsi.tick', 'serializedShareEntity'
            ]

@dataclass
class YTDlpConfig:
    """Dynamic yt-dlp configuration"""
    writesubtitles: bool = True
    writeautomaticsub: bool = True
    skip_download: bool = True
    quiet: bool = True
    no_warnings: bool = True
    extract_flat: bool = False
    writethumbnail: bool = False
    writeinfojson: bool = False
    ignoreerrors: bool = True
    extractor_retries: int = 5
    fragment_retries: int = 5
    retries: int = 5
    socket_timeout: int = 60
    http_chunk_size: int = 10485760
    concurrent_fragment_downloads: int = 8
    
    # Dynamic extractors
    extractors: List[str] = None
    
    def __post_init__(self):
        if self.extractors is None:
            self.extractors = [
                'youtube:transcript',
                'youtube:captions',
                'youtube:timedtext',
                'youtube:player',
                'youtube:webpage',
            ]

@dataclass
class APIConfig:
    """Dynamic API configuration"""
    # YouTube API endpoints
    timedtext_endpoints: List[str] = None
    innertube_endpoints: List[str] = None
    
    # Headers
    user_agents: List[str] = None
    accept_languages: List[str] = None
    
    # Rate limiting
    rate_limit_delay: float = 1.0
    max_requests_per_minute: int = 60
    
    def __post_init__(self):
        if self.timedtext_endpoints is None:
            self.timedtext_endpoints = [
                "https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}&fmt=srv3",
                "https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}&fmt=ttml",
                "https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}&fmt=vtt",
                "https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}&fmt=srv1",
                "https://www.youtube.com/api/timedtext?v={video_id}&lang={lang}&fmt=srv2",
            ]
        
        if self.innertube_endpoints is None:
            self.innertube_endpoints = [
                "https://www.youtube.com/youtubei/v1/player",
                "https://www.youtube.com/youtubei/v1/next",
                "https://www.youtube.com/youtubei/v1/browse",
            ]
        
        if self.user_agents is None:
            self.user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/121.0',
            ]
        
        if self.accept_languages is None:
            self.accept_languages = [
                'en-US,en;q=0.9',
                'hi-IN,hi;q=0.9,en;q=0.8',
                'en-GB,en;q=0.9',
                'en-AU,en;q=0.9',
                'en-CA,en;q=0.9',
            ]

class DynamicConfigManager:
    """Dynamic configuration manager for production-level optimization"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or "dynamic_config.json"
        self.extraction_config = ExtractionConfig()
        self.ytdlp_config = YTDlpConfig()
        self.api_config = APIConfig()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file or create default"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    self._apply_config(config_data)
                logger.info(f"✅ Loaded configuration from {self.config_file}")
            else:
                # Use defaults without creating file
                logger.info("✅ Using default configuration")
        except Exception as e:
            logger.warning(f"❌ Failed to load config: {str(e)}, using defaults")
            # Initialize with safe defaults
            self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize with safe default configuration"""
        self.extraction_config = ExtractionConfig(
            method_priorities=[ExtractionMethod.YT_DLP, ExtractionMethod.CAPTION_API],
            language_priorities=[LanguageCode.EN, LanguageCode.HI]
        )
        self.ytdlp_config = YTDlpConfig()
        self.api_config = APIConfig()
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            config_data = {
                'extraction_config': asdict(self.extraction_config),
                'ytdlp_config': asdict(self.ytdlp_config),
                'api_config': asdict(self.api_config)
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Saved configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save config: {str(e)}")
    
    def _apply_config(self, config_data: Dict[str, Any]):
        """Apply configuration data to objects"""
        try:
            if 'extraction_config' in config_data:
                extraction_data = config_data['extraction_config']
                self.extraction_config = ExtractionConfig(**extraction_data)
            
            if 'ytdlp_config' in config_data:
                ytdlp_data = config_data['ytdlp_config']
                self.ytdlp_config = YTDlpConfig(**ytdlp_data)
            
            if 'api_config' in config_data:
                api_data = config_data['api_config']
                self.api_config = APIConfig(**api_data)
                
        except Exception as e:
            logger.warning(f"❌ Failed to apply config: {str(e)}")
    
    def update_config(self, section: str, **kwargs):
        """Update configuration dynamically"""
        try:
            if section == 'extraction':
                for key, value in kwargs.items():
                    if hasattr(self.extraction_config, key):
                        setattr(self.extraction_config, key, value)
            elif section == 'ytdlp':
                for key, value in kwargs.items():
                    if hasattr(self.ytdlp_config, key):
                        setattr(self.ytdlp_config, key, value)
            elif section == 'api':
                for key, value in kwargs.items():
                    if hasattr(self.api_config, key):
                        setattr(self.api_config, key, value)
            
            self.save_config()
            logger.info(f"✅ Updated {section} configuration")
        except Exception as e:
            logger.error(f"❌ Failed to update config: {str(e)}")
    
    def get_method_priorities(self) -> List[ExtractionMethod]:
        """Get current method priorities"""
        return self.extraction_config.method_priorities
    
    def get_language_priorities(self) -> List[LanguageCode]:
        """Get current language priorities"""
        return self.extraction_config.language_priorities
    
    def get_ytdlp_config(self) -> Dict[str, Any]:
        """Get yt-dlp configuration as dict"""
        return asdict(self.ytdlp_config)
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration as dict"""
        return asdict(self.api_config)
    
    def optimize_for_video_type(self, video_type: str):
        """Optimize configuration for specific video type"""
        try:
            if video_type.lower() in ['educational', 'lecture', 'tutorial']:
                self.update_config('extraction', 
                    min_transcript_length=200,
                    max_processing_time=90,
                    retry_attempts=5
                )
                logger.info("✅ Optimized for educational content")
            
            elif video_type.lower() in ['music', 'song', 'audio']:
                self.update_config('extraction',
                    min_transcript_length=50,
                    max_processing_time=30,
                    retry_attempts=2
                )
                logger.info("✅ Optimized for music content")
            
            elif video_type.lower() in ['short', 'shorts', 'quick']:
                self.update_config('extraction',
                    min_transcript_length=30,
                    max_processing_time=15,
                    retry_attempts=1
                )
                logger.info("✅ Optimized for short content")
            
            elif video_type.lower() in ['hindi', 'indian']:
                # Prioritize Hindi languages
                hindi_priorities = [
                    LanguageCode.HI, LanguageCode.HI_IN, LanguageCode.EN_IN,
                    LanguageCode.EN, LanguageCode.EN_US, LanguageCode.EN_GB
                ]
                self.update_config('extraction', language_priorities=hindi_priorities)
                logger.info("✅ Optimized for Hindi content")
            
            elif video_type.lower() in ['hinglish', 'mixed', 'indian_english']:
                # Prioritize Hinglish and mixed content
                hinglish_priorities = [
                    LanguageCode.HINGLISH, LanguageCode.MIXED, LanguageCode.HI_IN,
                    LanguageCode.EN_IN, LanguageCode.HI, LanguageCode.EN,
                    LanguageCode.EN_US, LanguageCode.EN_GB
                ]
                self.update_config('extraction', language_priorities=hinglish_priorities)
                logger.info("✅ Optimized for Hinglish content")
            
            else:
                logger.info("✅ Using default configuration")
                
        except Exception as e:
            logger.error(f"❌ Failed to optimize for {video_type}: {str(e)}")
    
    def get_dynamic_headers(self) -> Dict[str, str]:
        """Get dynamic headers for requests"""
        import random
        
        return {
            'User-Agent': random.choice(self.api_config.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': random.choice(self.api_config.accept_languages),
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Referer': 'https://www.youtube.com/',
            'Origin': 'https://www.youtube.com',
        }
    
    def get_dynamic_endpoints(self, video_id: str, language: str) -> List[str]:
        """Get dynamic endpoints for API calls"""
        endpoints = []
        
        for endpoint_template in self.api_config.timedtext_endpoints:
            try:
                endpoint = endpoint_template.format(video_id=video_id, lang=language)
                endpoints.append(endpoint)
            except:
                continue
        
        return endpoints

# Global configuration manager instance
config_manager = DynamicConfigManager()
