from typing import List, Dict
import numpy as np
from collections import defaultdict
import re

class FeatureExtractor:
    def __init__(self, config: Dict):
        self.config = config
        self.window_size = config['window_size']
        self.patterns = self._load_patterns()
        
    def _load_patterns(self) -> Dict:
        """Load predefined patterns for feature extraction."""
        return {
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
            ]
        }
        
    def extract_features(self, tokens: List[str], position: int) -> Dict:
        """Extract features for a token at given position."""
        features = {}
        
        # Current token features
        token = tokens[position]
        features.update(self._get_token_features(token))
        
        # Context features
        features.update(self._get_context_features(tokens, position))
        
        # Pattern features
        features.update(self._get_pattern_features(token))
        
        return features
    
    def _get_token_features(self, token: str) -> Dict:
        """Extract features from the token itself."""
        return {
            'word': token,
            'length': len(token),
            'prefix_3': token[:3] if len(token) >= 3 else token,
            'suffix_3': token[-3:] if len(token) >= 3 else token,
            'is_honorific': token in self.patterns['honorifics'],
            'has_location_indicator': any(loc in token for loc in self.patterns['locations']),
            'has_mythical_indicator': any(myth in token for myth in self.patterns['mythical'])
        }
    
    def _get_context_features(self, tokens: List[str], position: int) -> Dict:
        """Extract features from surrounding context."""
        features = {}
        window = range(
            max(0, position - self.window_size),
            min(len(tokens), position + self.window_size + 1)
        )
        
        for i in window:
            if i != position:
                offset = i - position
                features[f'word_{offset}'] = tokens[i]
                
        return features
    
    def _get_pattern_features(self, token: str) -> Dict:
        """Extract pattern-based features."""
        return {
            'starts_with_al': token.startswith('ال'),
            'ends_with_in': token.endswith('ين'),
            'ends_with_an': token.endswith('ان'),
            'contains_ibn': 'ابن' in token
        }
