"""
Core feature extraction module for Arabic NER.
Provides basic feature extraction functionality for classical Arabic text.
"""

from typing import List, Dict, Optional
import numpy as np
from .patterns import ArabicPatterns
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Extract features from Arabic text for NER.
    
    Implements basic feature extraction methods including morphological,
    contextual, and pattern-based features.
    
    Attributes:
        patterns (ArabicPatterns): Pattern matcher instance
        window_size (int): Context window size
        use_position (bool): Whether to use positional features
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature extractor.
        
        Args:
            config (Dict, optional): Configuration parameters
        """
        self.config = config or {}
        self.patterns = ArabicPatterns()
        self.window_size = self.config.get('window_size', 3)
        self.use_position = self.config.get('use_position', True)
        
        logger.info(
            "Initialized FeatureExtractor with window_size=%d, use_position=%s",
            self.window_size,
            self.use_position
        )

    def extract_features(self, tokens: List[str], position: int) -> Dict:
        """
        Extract features for a token at given position.
        
        Args:
            tokens (List[str]): List of tokens
            position (int): Position of target token
            
        Returns:
            Dict: Dictionary of features
            
        Example:
            >>> extractor = FeatureExtractor()
            >>> features = extractor.extract_features(['قال', 'الملك', 'شهريار'], 1)
        """
        if not tokens or position >= len(tokens):
            raise ValueError("Invalid position or empty tokens")
            
        features = {}
        token = tokens[position]
        
        # Basic token features
        features.update(self._get_token_features(token))
        
        # Context features
        features.update(self._get_context_features(tokens, position))
        
        # Pattern features
        features.update(self._get_pattern_features(token))
        
        # Position features
        if self.use_position:
            features.update(self._get_position_features(position, len(tokens)))
        
        return features

    def _get_token_features(self, token: str) -> Dict:
        """Extract features from the token itself."""
        return {
            'word': token,
            'word.length': len(token),
            'word.prefix3': token[:3] if len(token) >= 3 else token,
            'word.suffix3': token[-3:] if len(token) >= 3 else token,
            'word.has_alef': 'ا' in token,
            'word.has_hamza': any(c in token for c in 'ءأؤإئ'),
            'word.starts_with_al': token.startswith('ال'),
            'word.is_honorific': self.patterns.is_honorific(token),
            'word.is_location': self.patterns.is_location_indicator(token),
            'word.is_mythical': self.patterns.is_mythical_indicator(token)
        }

    def _get_context_features(self, tokens: List[str], position: int) -> Dict:
        """Extract features from surrounding context."""
        features = {}
        
        # Window around current token
        start = max(0, position - self.window_size)
        end = min(len(tokens), position + self.window_size + 1)
        
        for i in range(start, end):
            if i != position:
                offset = i - position
                token = tokens[i]
                prefix = 'prev' if offset < 0 else 'next'
                abs_offset = abs(offset)
                
                features.update({
                    f'{prefix}{abs_offset}.word': token,
                    f'{prefix}{abs_offset}.is_honorific': self.patterns.is_honorific(token),
                    f'{prefix}{abs_offset}.is_location': self.patterns.is_location_indicator(token)
                })
        
        return features

    def _get_pattern_features(self, token: str) -> Dict:
        """Extract pattern-based features."""
        patterns = self.patterns.get_matching_patterns(token)
        
        features = {}
        for pattern_type, matches in patterns.items():
            features[f'pattern.{pattern_type}'] = len(matches) > 0
            if matches:
                features[f'pattern.{pattern_type}.match'] = matches[0]
        
        return features

    def _get_position_features(self, position: int, length: int) -> Dict:
        """Extract position-based features."""
        return {
            'position.absolute': position,
            'position.relative': position / length,
            'position.is_first': position == 0,
            'position.is_last': position == length - 1
        }

    def batch_extract_features(self, sentences: List[List[str]]) -> List[List[Dict]]:
        """
        Extract features for multiple sentences.
        
        Args:
            sentences (List[List[str]]): List of tokenized sentences
            
        Returns:
            List[List[Dict]]: Features for each token in each sentence
        """
        features = []
        for sentence in sentences:
            sentence_features = []
            for i in range(len(sentence)):
                token_features = self.extract_features(sentence, i)
                sentence_features.append(token_features)
            features.append(sentence_features)
        return features

    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names."""
        # Extract features for a sample token to get all possible features
        sample_features = self.extract_features(['sample'], 0)
        return list(sample_features.keys())
