import re
from typing import List, Dict
import arabic_reshaper
from bidi.algorithm import get_display

class ArabicPreprocessor:
    def __init__(self):
        self.patterns = {
            'diacritics': re.compile(r'[\u064B-\u065F\u0670]'),
            'special_chars': re.compile(r'[^\w\s]'),
            'numbers': re.compile(r'[0-9]'),
            'tatweel': re.compile(r'\u0640'),
        }
        
    def normalize_text(self, text: str) -> str:
        """Normalize Arabic text by removing diacritics and special characters."""
        text = self.remove_diacritics(text)
        text = self.normalize_alef(text)
        text = self.normalize_tah_marbota(text)
        text = self.remove_tatweel(text)
        return text
    
    def remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritical marks from text."""
        return self.patterns['diacritics'].sub('', text)
    
    def normalize_alef(self, text: str) -> str:
        """Normalize different forms of Alef to a single form."""
        return re.sub('[إأآا]', 'ا', text)
    
    def normalize_tah_marbota(self, text: str) -> str:
        """Normalize Tah Marbota to Ha."""
        return text.replace('ة', 'ه')
    
    def remove_tatweel(self, text: str) -> str:
        """Remove tatweel character from text."""
        return self.patterns['tatweel'].sub('', text)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Arabic text into words."""
        tokens = text.split()
        return [self.normalize_text(token) for token in tokens]
