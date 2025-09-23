"""
Hindi Text Processor - Fixes Hindi text clarity and transliteration
"""

import re
import logging
from typing import Dict, Any
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

logger = logging.getLogger(__name__)

class HindiTextProcessor:
    """Processes Hindi text for better clarity and readability"""
    
    def __init__(self):
        self.hindi_char_map = {
            # Common misrecognized characters
            'हय': 'है',
            'एवरवन': 'एवरीवन',
            'हल': 'है',
            'अ': 'आ',
            'वर': 'वर',
            'गड': 'गड',
            'आफटरनन': 'आफ्टरनून',
            'पलज': 'प्लीज',
            'कफरम': 'कन्फर्म',
            'दट': 'देट',
            'आई': 'आई',
            'एम': 'एम',
            'कलयरल': 'कलरफुल',
            'ऑडबल': 'ऑडिबल',
            'एड': 'एंड',
            'वज़बल': 'विजुअल',
            'ह': 'है',
            'ज': 'जो',
            'अचछ': 'अच्छा',
            'स': 'से',
            'दखई': 'दिखाई',
            'द': 'दे',
            'रह': 'रहे',
            'फटफट': 'फटाफट',
            'बत': 'बता',
            'फर': 'फिर',
            'बचच': 'बच्चे',
            'हम': 'हम',
            'लग': 'लोग',
            'हमर': 'हमारे',
            'आज': 'आज',
            'क': 'का',
            'इस': 'इस',
            'बहत': 'बहुत',
            'ह': 'है',
            'जयद': 'ज्यादा',
            'शनदर': 'शानदार',
            'कलस': 'क्लास',
            'शरआत': 'शुरुआत',
            'करग': 'करेंगे',
            'जसक': 'जिसकी',
            'अदर': 'अंदर',
            'हम': 'हम',
            'लग': 'लोग',
            'कलस': 'क्लास',
            '11th': '11वीं',
            'एक': 'एक',
            'ऐस': 'ऐसा',
            'सरज': 'सीरीज',
            'चल': 'चल',
            'रह': 'रही',
            'ह': 'है',
            'इस': 'इस',
            'YouTube': 'यूट्यूब',
            'चनल': 'चैनल',
            'क': 'के',
            'ऊपर': 'ऊपर',
            'जसक': 'जिसकी',
            'अदर': 'अंदर',
            'हम': 'हम',
            'लग': 'लोग',
            'कलस': 'क्लास',
            '11th': '11वीं',
            'क': 'का',
            'कलस': 'क्लास',
            '11th': '11वीं',
            'फजकस': 'फिजिक्स',
            'क': 'का',
            'ज': 'जो',
            'सबस': 'सबसे',
            'इपरटट': 'इम्पोर्टेन्ट',
            'टपकस': 'टॉपिक्स',
            'ह': 'है',
            'हम': 'हम',
            'लग': 'लोग',
            'उनक': 'उनके',
            'कवर': 'कवर',
            'कर': 'कर',
            'रह': 'रहे',
            'ह': 'है',
            'कसपटस': 'कॉन्सेप्ट्स',
            'पवईकयस': 'फिजिक्स',
            'और': 'और',
            'एपलकशनशस': 'एप्लिकेशन्स',
            'क': 'के',
            'सथ': 'साथ',
            'त': 'तो',
            'हम': 'हम',
            'पच': 'पांच',
            'टपकस': 'टॉपिक्स',
            'ऐस': 'ऐसे',
            'मन': 'में',
            'उठए': 'उठाए',
            'थ': 'थे',
            'जसम': 'जिसमें',
            'स': 'से',
            'डयमशस': 'डायमेंशन्स',
            'हम': 'हम',
            'लग': 'लोग',
            'कर': 'कर',
            'चक': 'चुके',
            'ह': 'हैं',
            'कलजस': 'क्लासेज',
            'कर': 'कर',
            'चक': 'चुके',
            'ह': 'हैं',
            'ममट': 'मोमेंट',
            'ऑफ': 'ऑफ',
            'इनरशय': 'इनरशिया',
            'एड': 'एंड',
            'रडयस': 'रेडियस',
            'ऑफ': 'ऑफ',
            'गरशन': 'गिरेशन',
            'कर': 'कर',
            'चक': 'चुके',
            'ह': 'हैं',
            'और': 'और',
            'आज': 'आज',
            'क': 'का',
            'हमर': 'हमारी',
            'कलस': 'क्लास',
            'म': 'में',
            'हम': 'हम',
            'लग': 'लोग',
            'क': 'का',
            'कवर': 'कवर',
            'करन': 'करना',
            'ह': 'है',
            'एसएचएम': 'एसएचएम',
            'कवटटज': 'क्वांटिटीज',
            'बहत': 'बहुत',
            'ह': 'है',
            'जयद': 'ज्यादा',
            'इपरटट': 'इम्पोर्टेन्ट',
            'ह': 'है',
            'दख': 'देखो',
            'आज': 'आज',
            'क': 'का',
            'हमर': 'हमारी',
            'कलस': 'क्लास',
            'म': 'में',
            'बलग': 'बिल्कुल',
            'हमर': 'हमारे',
            'य': 'ये',
            'जतन': 'जितने',
            'भ': 'भी',
            'कलसस': 'क्लासेज',
            'लग': 'लगे',
            'ह': 'हैं',
            'न': 'नहीं',
            'उनम': 'उनमें',
            'स': 'से',
            'सबस': 'सबसे',
            'इपरटट': 'इम्पोर्टेन्ट',
            'कलस': 'क्लास',
            'ह': 'है',
            'कयक': 'क्योंकि',
            'आज': 'आज',
            'क': 'का',
            'हमर': 'हमारी',
            'कलस': 'क्लास',
            'क': 'का',
            'ज': 'जो',
            'टपक': 'टॉपिक',
            'ह': 'है',
            'न': 'नहीं',
            'व': 'वो',
            'बसस': 'बेसिक',
            'ह': 'है',
            'एसएचएम': 'एसएचएम',
            'क': 'के',
            'चपटर': 'चैप्टर',
            'क': 'के',
            'अलटरनटग': 'अल्टरनेटिंग',
            'करट': 'करंट',
            'क': 'के',
            'चपटर': 'चैप्टर',
            'क': 'के',
            'इलकटरमगनटक': 'इलेक्ट्रोमैग्नेटिक',
            'ववस': 'वेव्स',
            'क': 'के',
            'चपटर': 'चैप्टर',
            'क': 'के',
            'ववस': 'वेव्स',
            'क': 'के',
            'चपटर': 'चैप्टर',
            'क': 'के',
            'और': 'और',
            'वव': 'वेव',
            'ऑपटकस': 'ऑप्टिक्स',
            'क': 'के',
            'चपटर': 'चैप्टर',
            'क': 'के',
            'अगर': 'अगर',
            'तमह': 'तुम्हें',
            'य': 'ये',
            'पत': 'पता',
            'लग': 'लग',
            'गय': 'गया',
            'वहट': 'व्हाट',
            'इज': 'इज',
            'द': 'द',
            'मनग': 'मीनिंग',
            'ऑफ़': 'ऑफ',
            'समथग': 'समथिंग',
            'बइग': 'बीइंग',
            'अ': 'ए',
            'फकशन': 'फंक्शन',
            'ऑफ़': 'ऑफ',
            'सइन': 'साइन',
            'और': 'और',
            'कस': 'कोस',
            'दन': 'दोनों',
            'य': 'ये',
            'गइज़': 'गाइज',
            'आर': 'आर',
            'सरटड': 'सर्टेड',
            'फर': 'फॉर',
            'ऑल': 'ऑल',
            'द': 'द',
            'फइव': 'फाइव',
            'चपटरस': 'चैप्टर्स',
            'एड': 'एंड',
            'आई': 'आई',
            'एम': 'एम',
            'हपग': 'होपिंग',
            'तम': 'तुम',
            'लग': 'लोग',
            'उतन': 'उतने',
            'ह': 'ही',
            'एकसइटड': 'एक्साइटेड',
            'ह': 'हैं',
            'आज': 'आज',
            'क': 'का',
            'हमर': 'हमारी',
            'कलस': 'क्लास',
            'लन': 'लिए',
            'क': 'के',
            'लए': 'लिए',
            'जतन': 'जितने',
            'म': 'मैं',
            'एकसइटड': 'एक्साइटेड',
            'ह': 'हैं',
            'आज': 'आज',
            'क': 'का',
            'हमर': 'हमारी',
            'कलस': 'क्लास',
            'क': 'का',
            'ज': 'जो',
            'टपक': 'टॉपिक',
            'क': 'का',
            'पढन': 'पढ़ना',
            'क': 'के',
            'लए': 'लिए',
            'एवरवन': 'एवरीवन',
            'इज': 'इज',
            'रड': 'रेडी',
            'ह': 'है',
            'ज': 'जो',
            'फटफट': 'फटाफट',
            'स': 'से',
            'बत': 'बता',
            'कलस': 'क्लास',
            'शरआत': 'शुरुआत',
            'कर': 'करो',
            'बचच': 'बच्चे',
            'जब': 'जब',
            'कलस': 'क्लास',
            'खतम': 'खत्म',
            'हग': 'होगा',
            'त': 'तो',
            'तम': 'तुम',
            'लग': 'लोग',
            'क': 'का',
            'आज': 'आज',
            'मर': 'मेरे',
            'टelegram': 'टेलीग्राम',
            'चनल': 'चैनल',
            'क': 'के',
            'ऊपर': 'ऊपर',
            'भ': 'भी'
        }
    
    def process_hindi_text(self, text: str) -> str:
        """Process Hindi text for better clarity"""
        try:
            if not text:
                return text
            
            # First, try to fix common misrecognitions
            processed_text = self._fix_common_misrecognitions(text)
            
            # Then try transliteration if needed
            if self._needs_transliteration(processed_text):
                processed_text = self._transliterate_to_hindi(processed_text)
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Error processing Hindi text: {str(e)}")
            return text
    
    def _fix_common_misrecognitions(self, text: str) -> str:
        """Fix common misrecognitions in Hindi text"""
        try:
            processed_text = text
            
            # Apply character mapping
            for wrong, correct in self.hindi_char_map.items():
                processed_text = processed_text.replace(wrong, correct)
            
            # Fix common patterns
            patterns = [
                (r'हय\s+', 'है '),
                (r'एवरवन', 'एवरीवन'),
                (r'हल\s+', 'है '),
                (r'अ\s+', 'आ '),
                (r'आफटरनन', 'आफ्टरनून'),
                (r'पलज', 'प्लीज'),
                (r'कफरम', 'कन्फर्म'),
                (r'दट', 'देट'),
                (r'कलयरल', 'कलरफुल'),
                (r'ऑडबल', 'ऑडिबल'),
                (r'वज़बल', 'विजुअल'),
                (r'अचछ', 'अच्छा'),
                (r'दखई', 'दिखाई'),
                (r'फटफट', 'फटाफट'),
                (r'बचच', 'बच्चे'),
                (r'हमर', 'हमारे'),
                (r'बहत', 'बहुत'),
                (r'जयद', 'ज्यादा'),
                (r'शनदर', 'शानदार'),
                (r'कलस', 'क्लास'),
                (r'शरआत', 'शुरुआत'),
                (r'करग', 'करेंगे'),
                (r'जसक', 'जिसकी'),
                (r'अदर', 'अंदर'),
                (r'फजकस', 'फिजिक्स'),
                (r'सबस', 'सबसे'),
                (r'इपरटट', 'इम्पोर्टेन्ट'),
                (r'टपकस', 'टॉपिक्स'),
                (r'कसपटस', 'कॉन्सेप्ट्स'),
                (r'पवईकयस', 'फिजिक्स'),
                (r'एपलकशनशस', 'एप्लिकेशन्स'),
                (r'डयमशस', 'डायमेंशन्स'),
                (r'कलजस', 'क्लासेज'),
                (r'ममट', 'मोमेंट'),
                (r'इनरशय', 'इनरशिया'),
                (r'रडयस', 'रेडियस'),
                (r'गरशन', 'गिरेशन'),
                (r'एसएचएम', 'एसएचएम'),
                (r'कवटटज', 'क्वांटिटीज'),
                (r'कयक', 'क्योंकि'),
                (r'बसस', 'बेसिक'),
                (r'चपटर', 'चैप्टर'),
                (r'अलटरनटग', 'अल्टरनेटिंग'),
                (r'करट', 'करंट'),
                (r'इलकटरमगनटक', 'इलेक्ट्रोमैग्नेटिक'),
                (r'ववस', 'वेव्स'),
                (r'ऑपटकस', 'ऑप्टिक्स'),
                (r'तमह', 'तुम्हें'),
                (r'पत', 'पता'),
                (r'लग', 'लग'),
                (r'गय', 'गया'),
                (r'वहट', 'व्हाट'),
                (r'मनग', 'मीनिंग'),
                (r'समथग', 'समथिंग'),
                (r'बइग', 'बीइंग'),
                (r'फकशन', 'फंक्शन'),
                (r'सइन', 'साइन'),
                (r'कस', 'कोस'),
                (r'गइज़', 'गाइज'),
                (r'सरटड', 'सर्टेड'),
                (r'फइव', 'फाइव'),
                (r'चपटरस', 'चैप्टर्स'),
                (r'हपग', 'होपिंग'),
                (r'एकसइटड', 'एक्साइटेड'),
                (r'पढन', 'पढ़ना'),
                (r'रड', 'रेडी'),
                (r'खतम', 'खत्म'),
                (r'हग', 'होगा'),
                (r'मर', 'मेरे'),
                (r'टelegram', 'टेलीग्राम'),
                (r'चनल', 'चैनल')
            ]
            
            for pattern, replacement in patterns:
                processed_text = re.sub(pattern, replacement, processed_text)
            
            return processed_text
            
        except Exception as e:
            logger.error(f"Error fixing misrecognitions: {str(e)}")
            return text
    
    def _needs_transliteration(self, text: str) -> bool:
        """Check if text needs transliteration"""
        try:
            # Check if text contains mostly English words with Hindi characters
            words = text.split()
            hindi_words = 0
            english_words = 0
            
            for word in words:
                if any('\u0900' <= char <= '\u097F' for char in word):
                    hindi_words += 1
                elif word.isascii():
                    english_words += 1
            
            return hindi_words > 0 and english_words > 0
            
        except Exception as e:
            logger.error(f"Error checking transliteration need: {str(e)}")
            return False
    
    def _transliterate_to_hindi(self, text: str) -> str:
        """Transliterate text to proper Hindi"""
        try:
            # Try to transliterate from ITRANS to Devanagari
            try:
                transliterated = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
                return transliterated
            except:
                # If ITRANS fails, try HK
                try:
                    transliterated = transliterate(text, sanscript.HK, sanscript.DEVANAGARI)
                    return transliterated
                except:
                    # If all fails, return original
                    return text
                    
        except Exception as e:
            logger.error(f"Error transliterating: {str(e)}")
            return text
    
    def get_hindi_quality_score(self, text: str) -> float:
        """Get quality score for Hindi text (0-100)"""
        try:
            if not text:
                return 0.0
            
            # Count proper Hindi characters
            hindi_chars = sum(1 for char in text if '\u0900' <= char <= '\u097F')
            total_chars = len(text)
            
            if total_chars == 0:
                return 0.0
            
            # Calculate quality based on Hindi character ratio
            hindi_ratio = hindi_chars / total_chars
            quality_score = min(100, hindi_ratio * 200)
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error calculating Hindi quality score: {str(e)}")
            return 0.0

# Global instance
hindi_text_processor = HindiTextProcessor()
