"""
Rule-based model implementation for Arabic NER.
Implements pattern matching and rule-based entity recognition.
"""

from typing import List, Dict, Optional, Set, Tuple
import re
from pathlib import Path
import json
import logging
from src.features.patterns import ArabicPatterns

logger = logging.getLogger(__name__)

class RuleBasedModel:
    """
    Rule-based model for Arabic Named Entity Recognition.
    
    Implements rule-based approach using patterns, gazetteers,
    and linguistic rules for entity recognition.
    
    Attributes:
        patterns (ArabicPatterns): Pattern matcher
        gazetteers (Dict): Entity gazetteers
        rules (Dict): Recognition rules
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize rule-based model.
        
        Args:
            config (Dict, optional): Model configuration
        """
        self.config = config or {}
        self.patterns = ArabicPatterns()
        self.gazetteers = self._load_gazetteers()
        self.rules = self._load_rules()
        
        logger.info(
            "Initialized RuleBasedModel with %d gazetteers and %d rules",
            len(self.gazetteers),
            len(self.rules)
        )

    def _load_gazetteers(self) -> Dict[str, Set[str]]:
        """
        Load entity gazetteers from files.
        
        Returns:
            Dict[str, Set[str]]: Gazetteer entries by entity type
        """
        gazetteers = {
            'PERSON': {
                'شهريار', 'شهرزاد', 'علاء الدين', 'السندباد',
                'هارون الرشيد', 'جعفر البرمكي', 'مسرور'
            },
            'LOCATION': {
                'بغداد', 'البصرة', 'دمشق', 'القاهرة',
                'جبل قاف', 'وادي السحر', 'مدينة النحاس'
            },
            'ORGANIZATION': {
                'ديوان الخلافة', 'مجلس القضاء', 'دار الحكمة',
                'خزانة الكتب', 'بيت الحكمة'
            },
            'MYTHICAL': {
                'العفريت', 'الجني', 'طائر الرخ', 'العنقاء',
                'مارد المصباح', 'ملك الجان'
            },
            'OBJECT': {
                'المصباح السحري', 'خاتم سليمان', 'البساط السحري',
                'سيف الملوك', 'تاج الملك'
            }
        }
        
        # Add custom gazetteers from config if provided
        if 'gazetteers_path' in self.config:
            try:
                with open(self.config['gazetteers_path'], 'r', encoding='utf-8') as f:
                    custom_gazetteers = json.load(f)
                for entity_type, entries in custom_gazetteers.items():
                    if entity_type in gazetteers:
                        gazetteers[entity_type].update(entries)
                    else:
                        gazetteers[entity_type] = set(entries)
            except Exception as e:
                logger.warning("Error loading custom gazetteers: %s", str(e))
        
        return gazetteers

    def _load_rules(self) -> Dict[str, List[Dict]]:
        """
        Load recognition rules.
        
        Returns:
            Dict[str, List[Dict]]: Rules by entity type
        """
        rules = {
            'PERSON': [
                {
                    'pattern': r'الملك\s+(\w+)',
                    'group': 1,
                    'prefix': 'B-PERSON'
                },
                {
                    'pattern': r'السلطان\s+(\w+)',
                    'group': 1,
                    'prefix': 'B-PERSON'
                },
                {
                    'pattern': r'الأمير\s+(\w+)',
                    'group': 1,
                    'prefix': 'B-PERSON'
                }
            ],
            'LOCATION': [
                {
                    'pattern': r'مدينة\s+(\w+)',
                    'group': 1,
                    'prefix': 'B-LOCATION'
                },
                {
                    'pattern': r'في\s+(\w+)',
                    'group': 1,
                    'prefix': 'B-LOCATION',
                    'conditions': ['is_location_in_gazetteer']
                }
            ],
            'MYTHICAL': [
                {
                    'pattern': r'جني\s+(\w+)',
                    'group': 1,
                    'prefix': 'B-MYTHICAL'
                },
                {
                    'pattern': r'عفريت\s+(\w+)',
                    'group': 1,
                    'prefix': 'B-MYTHICAL'
                }
            ]
        }
        
        # Add custom rules from config if provided
        if 'rules_path' in self.config:
            try:
                with open(self.config['rules_path'], 'r', encoding='utf-8') as f:
                    custom_rules = json.load(f)
                for entity_type, rule_list in custom_rules.items():
                    if entity_type in rules:
                        rules[entity_type].extend(rule_list)
                    else:
                        rules[entity_type] = rule_list
            except Exception as e:
                logger.warning("Error loading custom rules: %s", str(e))
        
        return rules

    def predict(self, sentences: List[List[str]]) -> List[List[str]]:
        """
        Predict entity labels for input sentences.
        
        Args:
            sentences (List[List[str]]): Input sentences
            
        Returns:
            List[List[str]]: Predicted labels
        """
        predictions = []
        for sentence in sentences:
            sentence_labels = self._label_sentence(sentence)
            predictions.append(sentence_labels)
        return predictions

    def _label_sentence(self, sentence: List[str]) -> List[str]:
        """
        Label tokens in a sentence.
        
        Args:
            sentence (List[str]): Input sentence tokens
            
        Returns:
            List[str]: Entity labels
        """
        text = ' '.join(sentence)
        labels = ['O'] * len(sentence)
        
        # Apply gazetteer matching
        for entity_type, entries in self.gazetteers.items():
            for entry in entries:
                if entry in text:
                    entry_tokens = entry.split()
                    for i in range(len(sentence) - len(entry_tokens) + 1):
                        if sentence[i:i+len(entry_tokens)] == entry_tokens:
                            labels[i] = f'B-{entity_type}'
                            for j in range(i+1, i+len(entry_tokens)):
                                labels[j] = f'I-{entity_type}'
        
        # Apply rule-based matching
        for entity_type, rule_list in self.rules.items():
            for rule in rule_list:
                matches = re.finditer(rule['pattern'], text)
                for match in matches:
                    if 'conditions' in rule:
                        if not self._check_conditions(match.group(rule['group']), rule['conditions']):
                            continue
                    
                    start_idx = len(text[:match.start()].split())
                    entity_tokens = match.group(rule['group']).split()
                    
                    if start_idx < len(labels):
                        labels[start_idx] = rule['prefix']
                        for i in range(start_idx + 1, min(start_idx + len(entity_tokens), len(labels))):
                            labels[i] = f'I-{entity_type}'
        
        return labels

    def _check_conditions(self, text: str, conditions: List[str]) -> bool:
        """Check if text meets specified conditions."""
        for condition in conditions:
            if condition == 'is_location_in_gazetteer':
                if text not in self.gazetteers['LOCATION']:
                    return False
        return True

    def evaluate(self, X_test: List[List[str]], y_test: List[List[str]]) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X_test (List[List[str]]): Test sentences
            y_test (List[List[str]]): True labels
            
        Returns:
            Dict: Evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': self._compute_accuracy(y_test, y_pred),
            'entity_metrics': self._compute_entity_metrics(y_test, y_pred)
        }
        
        return metrics

    def _compute_accuracy(self, y_true: List[List[str]], y_pred: List[List[str]]) -> float:
        """Compute token-level accuracy."""
        correct = 0
        total = 0
        for true_seq, pred_seq in zip(y_true, y_pred):
            for true_label, pred_label in zip(true_seq, pred_seq):
                if true_label == pred_label:
                    correct += 1
                total += 1
        return correct / total

    def _compute_entity_metrics(self, y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
        """Compute metrics for each entity type."""
        entity_types = set()
        for seq in y_true:
            entity_types.update(label.split('-')[1] for label in seq if '-' in label)
        
        metrics = {}
        for entity_type in entity_types:
            metrics[entity_type] = self._compute_entity_type_metrics(y_true, y_pred, entity_type)
        
        return metrics

    def _compute_entity_type_metrics(self, y_true: List[List[str]], y_pred: List[List[str]], 
                                   entity_type: str) -> Dict:
        """Compute metrics for a specific entity type."""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for true_seq, pred_seq in zip(y_true, y_pred):
            for true_label, pred_label in zip(true_seq, pred_seq):
                is_true_entity = true_label.endswith(entity_type)
                is_pred_entity = pred_label.endswith(entity_type)
                
                if is_true_entity and is_pred_entity:
                    true_positives += 1
                elif is_pred_entity:
                    false_positives += 1
                elif is_true_entity:
                    false_negatives += 1
        
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
