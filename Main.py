# Import required libraries
import pandas as pd
import numpy as np
import re
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# NLP libraries
import spacy
from sklearn_crfsuite import CRF
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from sklearn.metrics import precision_recall_fscore_support
import torch

class ArabicTextPreprocessor:
    """Handle Arabic text preprocessing and normalization."""
   
    def __init__(self):
        self.arabic_diacritics = re.compile("""
            [\u0617-\u061A\u064B-\u0652\u0657-\u065F\u0670\u06D6-\u06ED\u08F0-\u08F3]
        """, re.VERBOSE)
   
    def remove_diacritics(self, text: str) -> str:
        """Remove Arabic diacritical marks."""
        return self.arabic_diacritics.sub('', text)
   
    def normalize_alef(self, text: str) -> str:
        """Normalize different forms of Alef."""
        return text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
   
    def normalize_text(self, text: str) -> str:
        """Apply full text normalization."""
        text = self.remove_diacritics(text)
        text = self.normalize_alef(text)
        text = ' '.join(text.split())  # normalize whitespace
        return text

class DataAnnotator:
    """Handle data annotation for the One Thousand and One Nights dataset."""
   
    def __init__(self, input_file: str):
        self.data = pd.read_csv(input_file)
        self.entity_types = ['PERSON', 'LOCATION', 'MYTHICAL', 'HISTORICAL']
        self.annotated_data = []
   
    def prepare_data(self) -> List[Dict]:
        """Convert raw text to annotated format."""
        preprocessor = ArabicTextPreprocessor()
       
        processed_data = []
        for _, row in self.data.iterrows():
            text = preprocessor.normalize_text(row['text'])
            tokens = text.split()
           
            # Initialize with O tags
            annotations = [{'token': token, 'tag': 'O'} for token in tokens]
            processed_data.append({
                'text': text,
                'tokens': annotations,
                'id': row.get('id', len(processed_data))
            })
       
        return processed_data
   
    def save_annotations(self, output_file: str):
        """Save annotated data to file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.annotated_data, f, ensure_ascii=False, indent=2)

class RuleBasedNER:
    """Rule-based component for Arabic NER."""
   
    def __init__(self):
        self.name_patterns = [
            r'(عبد|ابو|ابن)\s+\w+',  # Common name patterns
            r'(سلطان|وزير|شيخ)\s+\w+',  # Title patterns
        ]
        self.location_patterns = [
            r'مدينة\s+\w+',  # City patterns
            r'بلاد\s+\w+',  # Country/region patterns
        ]
        self.compiled_patterns = self._compile_patterns()
   
    def _compile_patterns(self) -> Dict:
        """Compile regex patterns."""
        patterns = {
            'PERSON': [re.compile(p) for p in self.name_patterns],
            'LOCATION': [re.compile(p) for p in self.location_patterns]
        }
        return patterns
   
    def extract_entities(self, text: str) -> List[Tuple[str, str, int, int]]:
        """Extract entities using rule-based patterns."""
        entities = []
        for entity_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entities.append((
                        match.group(),
                        entity_type,
                        match.start(),
                        match.end()
                    ))
        return entities

class CRFModel:
    """CRF-based model for Arabic NER."""
   
    def __init__(self):
        self.model = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
   
    def extract_features(self, tokens: List[str]) -> List[Dict]:
        """Extract features for CRF model."""
        features = []
        for i, token in enumerate(tokens):
            token_features = {
                'word': token,
                'is_first': i == 0,
                'is_last': i == len(tokens) - 1,
                'is_capitalized': token[0].isupper(),
                'is_all_caps': token.isupper(),
                'prefix-1': token[0],
                'prefix-2': token[:2] if len(token) > 1 else token[0],
                'suffix-1': token[-1],
                'suffix-2': token[-2:] if len(token) > 1 else token[-1],
            }
            features.append(token_features)
        return features
   
    def train(self, X_train: List[List[Dict]], y_train: List[List[str]]):
        """Train the CRF model."""
        self.model.fit(X_train, y_train)
   
    def predict(self, X_test: List[List[Dict]]) -> List[List[str]]:
        """Make predictions using the trained model."""
        return self.model.predict(X_test)

class TransformerNER:
    """Transformer-based model for Arabic NER."""
   
    def __init__(self, model_name: str = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
   
    def predict(self, text: str) -> List[Dict]:
        """Make predictions using the transformer model."""
        return self.nlp(text)

class ModelEvaluator:
    """Evaluate and compare different NER models."""
   
    @staticmethod
    def calculate_metrics(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
        """Calculate precision, recall, and F1 score."""
        # Flatten the lists
        y_true_flat = [item for sublist in y_true for item in sublist]
        y_pred_flat = [item for sublist in y_pred for item in sublist]
       
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_flat, y_pred_flat, average='weighted'
        )
       
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

class NERPipeline:
    """Main pipeline for Arabic NER."""
   
    def __init__(self):
        self.preprocessor = ArabicTextPreprocessor()
        self.rule_based = RuleBasedNER()
        self.crf_model = CRFModel()
        self.transformer_model = TransformerNER()
   
    def train(self, training_data: List[Dict]):
        """Train the hybrid NER model."""
        # Prepare features and labels for CRF
        X_train = [self.crf_model.extract_features(sample['tokens'])
                  for sample in training_data]
        y_train = [[token['tag'] for token in sample['tokens']]
                  for sample in training_data]
       
        # Train CRF model
        self.crf_model.train(X_train, y_train)
   
    def predict(self, text: str) -> List[Dict]:
        """Make predictions using the hybrid approach."""
        # Preprocess text
        normalized_text = self.preprocessor.normalize_text(text)
       
        # Get rule-based predictions
        rule_entities = self.rule_based.extract_entities(normalized_text)
       
        # Get CRF predictions
        tokens = normalized_text.split()
        features = self.crf_model.extract_features(tokens)
        crf_predictions = self.crf_model.predict([features])[0]
       
        # Get transformer predictions
        transformer_entities = self.transformer_model.predict(normalized_text)
       
        # Combine predictions (simple majority voting)
        combined_predictions = []
        for i, token in enumerate(tokens):
            predictions = {
                'token': token,
                'rule_based': self._get_rule_based_tag(token, rule_entities),
                'crf': crf_predictions[i],
                'transformer': self._get_transformer_tag(token, transformer_entities)
            }
            combined_predictions.append(predictions)
       
        return combined_predictions
   
    def _get_rule_based_tag(self, token: str, entities: List[Tuple]) -> str:
        """Get tag from rule-based predictions."""
        for entity, tag, _, _ in entities:
            if token in entity:
                return tag
        return 'O'
   
    def _get_transformer_tag(self, token: str, entities: List[Dict]) -> str:
        """Get tag from transformer predictions."""
        for entity in entities:
            if token in entity['word']:
                return entity['entity']
        return 'O'

def main():
    # Initialize pipeline
    pipeline = NERPipeline()
   
    # Load and preprocess data
    annotator = DataAnnotator('one_thousand_nights.csv')
    training_data = annotator.prepare_data()
   
    # Train the model
    pipeline.train(training_data)
   
    # Example prediction
    text = """
    قال الملك شهريار لشهرزاد: حدثيني حديثاً عجيباً
    """
    predictions = pipeline.predict(text)
   
    # Save the model
    with open('arabic_ner_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

if __name__ == "__main__":
    main()
