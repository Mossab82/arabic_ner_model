"""
Arabic language patterns module.
Defines patterns and rules for feature extraction from classical Arabic text.
"""

from typing import Dict, List, Set
import re
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ArabicPatterns:
    """
    Manages patterns and rules for Arabic text analysis.
    
    Provides patterns for morphological analysis, entity recognition,
    and contextual feature extraction.
    
    Attributes:
        patterns (Dict): Dictionary of compiled regex patterns
        rules (Dict): Dictionary of linguistic rules
    """
    
    def __init__(self, patterns_file: str = None):
        """
        Initialize patterns manager.
        
        Args:
            patterns_file (str, optional): Path to JSON file containing patterns
        """
        self.patterns = {
            'honorifics': [
                'الملك', 'السلطان', 'الأمير', 'الوزير',
                'الشيخ', 'الحكيم', 'العالم', 'السيد'
            ],
            'locations': [
                'مدينة', 'قرية', 'بلد', 'جبل',
                'وادي', 'بحر', 'نهر', 'قصر'
            ],
            'mythical': [
                'الجن', 'العفريت', 'الساحر', 'الطائر',
                'السحري', 'العجيب', 'الخارق'
            ],
            'narrative_markers': [
                'قال', 'حكى', 'روى', 'أخبر',
                'كان', 'يحكى', 'زعموا'
            ]
        }
        
        self.rules = self._load_rules(patterns_file)
        self._compile_patterns()
        
    def _load_rules(self, patterns_file: str = None) -> Dict:
        """Load rules from file or use defaults."""
        if patterns_file and Path(patterns_file).exists():
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading patterns file: {e}")
                return self._get_default_rules()
        return self._get_default_rules()
    
    def _get_default_rules(self) -> Dict:
        """Get default linguistic rules."""
        return {
            'name_patterns': [
                r'بن\s+\w+',
                r'ابن\s+\w+',
                r'أبو\s+\w+',
                r'أم\s+\w+',
            ],
            'location_patterns': [
                r'مدينة\s+\w+',
                r'بلاد\s+\w+',
                r'جبل\s+\w+',
                r'وادي\s+\w+',
            ],
            'object_patterns': [
                r'خاتم\s+\w+',
                r'سيف\s+\w+',
                r'مصباح\s+\w+',
                r'كتاب\s+\w+',
            ]
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.compiled_patterns = {}
        for category, patterns in self.rules.items():
            self.compiled_patterns[category] = [
                re.compile(pattern) for pattern in patterns
            ]
    
    def match_pattern(self, text: str, pattern_type: str) -> bool:
        """
        Check if text matches a specific pattern type.
        
        Args:
            text (str): Text to check
            pattern_type (str): Type of pattern to match
            
        Returns:
            bool: Whether the text matches the pattern
        """
        if pattern_type not in self.compiled_patterns:
            return False
            
        return any(
            pattern.search(text)
            for pattern in self.compiled_patterns[pattern_type]
        )
    
    def get_matching_patterns(self, text: str) -> Dict[str, List[str]]:
        """
        Get all patterns that match the text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, List[str]]: Dictionary of matching patterns by category
        """
        matches = {}
        for category, patterns in self.compiled_patterns.items():
            category_matches = []
            for pattern in patterns:
                found = pattern.findall(text)
                if found:
                    category_matches.extend(found)
            if category_matches:
                matches[category] = category_matches
        return matches
    
    def is_honorific(self, word: str) -> bool:
        """Check if word is an honorific title."""
        return word in self.patterns['honorifics']
    
    def is_location_indicator(self, word: str) -> bool:
        """Check if word indicates a location."""
        return word in self.patterns['locations']
    
    def is_mythical_indicator(self, word: str) -> bool:
        """Check if word indicates a mythical entity."""
        return word in self.patterns['mythical']
    
    def is_narrative_marker(self, word: str) -> bool:
        """Check if word is a narrative marker."""
        return word in self.patterns['narrative_markers']
