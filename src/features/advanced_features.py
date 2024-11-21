"""
Advanced feature extraction module for Arabic NER.
Implements sophisticated feature extraction methods including BERT embeddings,
stylometric features, and narrative structure analysis.
"""

from typing import List, Dict, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from collections import defaultdict
import re
import logging
from .extractor import FeatureExtractor

logger = logging.getLogger(__name__)

class AdvancedFeatureExtractor(FeatureExtractor):
    """
    Advanced feature extraction for classical Arabic NER.
    
    Extends base feature extractor with BERT embeddings, stylometric analysis,
    and narrative structure features.
    
    Attributes:
        bert_tokenizer: BERT tokenizer for Arabic
        bert_model: BERT model for contextual embeddings
        narrative_patterns: Patterns for narrative structure analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize advanced feature extractor.
        
        Args:
            config (Dict, optional): Configuration parameters
        """
        super().__init__(config)
        
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize BERT components
        self.bert_tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
        self.bert_model = AutoModel.from_pretrained("asafaya/bert-base-arabic").to(self.device)
        self.bert_model.eval()
        
        # Load narrative patterns
        self.narrative_patterns = self._load_narrative_patterns()
        
        logger.info("Initialized AdvancedFeatureExtractor on device: %s", self.device)

    def _load_narrative_patterns(self) -> Dict:
        """Load patterns for narrative structure analysis."""
        return {
            'dialogue_markers': [
                'قال', 'قالت', 'أجاب', 'صاح',
                'همس', 'تحدث', 'روى', 'حكى'
            ],
            'scene_transitions': [
                'وفي اليوم التالي', 'وبعد ذلك',
                'وفي الصباح', 'وعندما حل المساء'
            ],
            'character_introductions': [
                'كان هناك', 'يحكى أن',
                'عاش في قديم الزمان', 'في ما مضى'
            ],
            'spatial_markers': [
                'في قصر', 'في مدينة',
                'على جبل', 'في وادي'
            ],
            'temporal_markers': [
                'في ذلك الزمان', 'في عهد',
                'في يوم من الأيام', 'منذ زمن بعيد'
            ]
        }

    def extract_contextual_embeddings(self, text: str) -> np.ndarray:
        """
        Extract BERT embeddings for text.
        
        Args:
            text (str): Input text
            
        Returns:
            np.ndarray: Contextual embeddings
        """
        with torch.no_grad():
            inputs = self.bert_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.bert_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
        return embeddings

    def extract_narrative_features(self, text: str) -> Dict[str, float]:
        """
        Extract features related to narrative structure.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Narrative feature scores
        """
        features = defaultdict(float)
        
        # Check for various narrative elements
        for pattern_type, patterns in self.narrative_patterns.items():
            pattern_count = sum(text.count(pattern) for pattern in patterns)
            features[f'narrative_{pattern_type}_count'] = pattern_count
            features[f'narrative_{pattern_type}_density'] = pattern_count / len(text.split())
        
        # Analyze dialogue structure
        features['dialogue_ratio'] = self._calculate_dialogue_ratio(text)
        
        # Detect scene changes
        features['scene_change_density'] = self._detect_scene_changes(text)
        
        return dict(features)

    def extract_stylometric_features(self, text: str) -> Dict[str, float]:
        """
        Extract stylometric features from text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, float]: Stylometric feature scores
        """
        words = text.split()
        sentences = text.split('.')
        
        features = {
            'avg_word_length': np.mean([len(w) for w in words]),
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'vocabulary_richness': len(set(words)) / len(words),
            'punctuation_density': len(re.findall(r'[،؛؟!]', text)) / len(words),
            'honorific_density': sum(self.patterns.is_honorific(w) for w in words) / len(words),
            'location_reference_density': sum(self.patterns.is_location_indicator(w) for w in words) / len(words)
        }
        
        return features

    def extract_morphological_features(self, word: str) -> Dict[str, bool]:
        """
        Extract detailed morphological features.
        
        Args:
            word (str): Input word
            
        Returns:
            Dict[str, bool]: Morphological features
        """
        features = {
            'has_definite_article': word.startswith('ال'),
            'ends_with_feminine': word.endswith('ة') or word.endswith('ات'),
            'has_plural_form': any(word.endswith(suffix) for suffix in ['ون', 'ين', 'ات']),
            'has_dual_form': word.endswith('ان') or word.endswith('ين'),
            'has_possessive': any(word.endswith(suffix) for suffix in ['ه', 'ها', 'هم', 'هن']),
            'has_prefix': any(word.startswith(prefix) for prefix in ['ب', 'ل', 'ك', 'و', 'ف']),
            'root_pattern': self._extract_root_pattern(word)
        }
        
        return features

    def _calculate_dialogue_ratio(self, text: str) -> float:
        """Calculate ratio of dialogue to narrative text."""
        dialogue_markers = sum(text.count(marker) for marker in self.narrative_patterns['dialogue_markers'])
        total_sentences = len(text.split('.'))
        return dialogue_markers / max(total_sentences, 1)

    def _detect_scene_changes(self, text: str) -> float:
        """Detect and quantify scene changes in text."""
        scene_transitions = sum(
            text.count(transition)
            for transition in self.narrative_patterns['scene_transitions']
        )
        total_paragraphs = len(text.split('\n'))
        return scene_transitions / max(total_paragraphs, 1)

    def _extract_root_pattern(self, word: str) -> str:
        """Extract root pattern from word."""
        # Remove common prefixes and suffixes
        prefixes = ['ال', 'وال', 'بال', 'كال', 'فال']
        suffixes = ['ة', 'ات', 'ون', 'ين', 'ان', 'ها', 'هم', 'هن']
        
        processed_word = word
        for prefix in prefixes:
            if processed_word.startswith(prefix):
                processed_word = processed_word[len(prefix):]
                break
        
        for suffix in suffixes:
            if processed_word.endswith(suffix):
                processed_word = processed_word[:-len(suffix)]
                break
        
        # Extract consonants as potential root letters
        consonants = re.findall(r'[^اويةء]', processed_word)
        if len(consonants) >= 3:
            return ''.join(consonants[:3])
        return processed_word

    def get_full_feature_set(self, text: str, position: int = None) -> Dict:
        """
        Get complete set of features including basic and advanced features.
        
        Args:
            text (str): Input text
            position (int, optional): Token position for basic features
            
        Returns:
            Dict: Complete feature set
        """
        features = {}
        
        # Get basic features if position is provided
        if position is not None:
            tokens = text.split()
            features.update(super().extract_features(tokens, position))
        
        # Add advanced features
        features.update({
            'contextual_embedding': self.extract_contextual_embeddings(text)[0],
            'narrative_features': self.extract_narrative_features(text),
            'stylometric_features': self.extract_stylometric_features(text),
            'morphological_features': self.extract_morphological_features(text.split()[position] if position is not None else text)
        })
        
        return features
