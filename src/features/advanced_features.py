from typing import List, Dict
import numpy as np
from collections import defaultdict
import re
import torch
from transformers import AutoTokenizer, AutoModel

class AdvancedFeatureExtractor:
    def __init__(self, config: Dict):
        self.config = config
        self.bert_tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
        self.bert_model = AutoModel.from_pretrained("asafaya/bert-base-arabic")
        self.narrative_patterns = self._load_narrative_patterns()
        
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
            ]
        }
    
    def extract_contextual_embeddings(self, text: str) -> np.ndarray:
        """Extract BERT embeddings for text."""
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    def extract_narrative_features(self, text: str) -> Dict:
        """Extract features related to narrative structure."""
        features = defaultdict(int)
        
        # Check for dialogue markers
        for marker in self.narrative_patterns['dialogue_markers']:
            features['dialogue_count'] += len(re.findall(marker, text))
            
        # Check for scene transitions
        for transition in self.narrative_patterns['scene_transitions']:
            features['transition_count'] += len(re.findall(transition, text))
            
        # Check for character introductions
        for intro in self.narrative_patterns['character_introductions']:
            features['introduction_count'] += len(re.findall(intro, text))
            
        return dict(features)
    
    def extract_stylometric_features(self, text: str) -> Dict:
        """Extract stylometric features from text."""
        words = text.split()
        sentences = text.split('.')
        
        features = {
            'avg_word_length': np.mean([len(w) for w in words]),
            'avg_sentence_length': len(words) / len(sentences),
            'vocabulary_richness': len(set(words)) / len(words),
            'punctuation_density': len(re.findall(r'[،؛؟!]', text)) / len(words)
        }
        
        return features
    
    def extract_morphological_features(self, word: str) -> Dict:
        """Extract detailed morphological features."""
        features = {
            'has_definite_article': word.startswith('ال'),
            'ends_with_feminine': word.endswith('ة') or word.endswith('ات'),
            'has_plural_form': any(word.endswith(suffix) for suffix in ['ون', 'ين', 'ات']),
            'root_pattern': self._extract_root_pattern(word),
            'length': len(word)
        }
        
        return features
    
    def _extract_root_pattern(self, word: str) -> str:
        """Extract root pattern from word using rule-based approach."""
        # Simplified root extraction logic
        consonants = re.findall(r'[^اوىيةء]', word)
        if len(consonants) >= 3:
            return ''.join(consonants[:3])
        return word
