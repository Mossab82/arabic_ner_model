"""
Evaluation utilities for Arabic NER.
Implements detailed evaluation metrics and analysis tools.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
import pandas as pd
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class EntityEvaluator:
    """
    Evaluator for Named Entity Recognition results.
    
    Provides detailed evaluation metrics, error analysis,
    and performance reporting functionality.
    """
    
    def __init__(self, labels: Optional[List[str]] = None):
        """
        Initialize evaluator.
        
        Args:
            labels (List[str], optional): List of entity labels
        """
        self.labels = labels or ['PERSON', 'LOCATION', 'ORGANIZATION', 'MYTHICAL', 'OBJECT']
        self.results = {}
        logger.info("Initialized EntityEvaluator with %d labels", len(self.labels))

    def compute_metrics(self, y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true (List[List[str]]): True labels
            y_pred (List[List[str]]): Predicted labels
            
        Returns:
            Dict: Evaluation metrics
        """
        metrics = {}
        
        # Overall metrics
        metrics['overall'] = self._compute_overall_metrics(y_true, y_pred)
        
        # Per-entity metrics
        metrics['entity'] = self._compute_entity_metrics(y_true, y_pred)
        
        # Confusion matrix
        metrics['confusion_matrix'] = self._compute_confusion_matrix(y_true, y_pred)
        
        # Error analysis
        metrics['error_analysis'] = self._analyze_errors(y_true, y_pred)
        
        self.results = metrics
        return metrics

    def _compute_overall_metrics(self, y_true: List[List[str]], 
                               y_pred: List[List[str]]) -> Dict:
        """Compute overall performance metrics."""
        true_entities = self._extract_entities(y_true)
        pred_entities = self._extract_entities(y_pred)
        
        correct = len(true_entities.intersection(pred_entities))
        total_pred = len(pred_entities)
        total_true = len(true_entities)
        
        precision = correct / max(total_pred, 1)
        recall = correct / max(total_true, 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': self._compute_accuracy(y_true, y_pred)
        }

    def _compute_entity_metrics(self, y_true: List[List[str]], 
                              y_pred: List[List[str]]) -> Dict:
        """Compute metrics for each entity type."""
        metrics = {}
        
        for label in self.labels:
            true_entities = self._extract_entities(y_true, label)
            pred_entities = self._extract_entities(y_pred, label)
            
            correct = len(true_entities.intersection(pred_entities))
            total_pred = len(pred_entities)
            total_true = len(true_entities)
            
            precision = correct / max(total_pred, 1)
            recall = correct / max(total_true, 1)
            f1 = 2 * (precision * recall) / max(precision + recall, 1)
            
            metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': total_true
            }
            
        return metrics

    def _compute_confusion_matrix(self, y_true: List[List[str]], 
                                y_pred: List[List[str]]) -> pd.DataFrame:
        """Compute confusion matrix for entity labels."""
        labels = ['O'] + [f'B-{label}' for label in self.labels]
        matrix = pd.DataFrame(0, index=labels, columns=labels)
        
        for true_seq, pred_seq in zip(y_true, y_pred):
            for true_label, pred_label in zip(true_seq, pred_seq):
                matrix.loc[true_label, pred_label] += 1
                
        return matrix

    def _analyze_errors(self, y_true: List[List[str]], 
                       y_pred: List[List[str]]) -> Dict:
        """Perform detailed error analysis."""
        error_analysis = {
            'error_types': defaultdict(int),
            'error_examples': [],
            'entity_errors': defaultdict(lambda: defaultdict(int))
        }
        
        for sent_idx, (true_seq, pred_seq) in enumerate(zip(y_true, y_pred)):
            for token_idx, (true_label, pred_label) in enumerate(zip(true_seq, pred_seq)):
                if true_label != pred_label:
                    error_type = self._classify_error(true_label, pred_label)
                    error_analysis['error_types'][error_type] += 1
                    
                    # Store error example
                    error_analysis['error_examples'].append({
                        'sentence_id': sent_idx,
                        'token_id': token_idx,
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'error_type': error_type
                    })
                    
                    # Track entity-specific errors
                    if true_label != 'O':
                        true_entity = true_label.split('-')[1]
                        error_analysis['entity_errors'][true_entity]['total'] += 1
                        error_analysis['entity_errors'][true_entity][error_type] += 1
        
        return dict(error_analysis)

    def _classify_error(self, true_label: str, pred_label: str) -> str:
        """Classify type of prediction error."""
        if true_label == 'O':
            return 'false_positive'
        elif pred_label == 'O':
            return 'false_negative'
        elif true_label.split('-')[1] != pred_label.split('-')[1]:
            return 'wrong_entity_type'
        else:
            return 'boundary_error'

    def _extract_entities(self, sequences: List[List[str]], 
                         entity_type: Optional[str] = None) -> set:
        """Extract entity spans from sequences."""
        entities = set()
        
        for sent_idx, sequence in enumerate(sequences):
            current_entity = []
            current_type = None
            
            for token_idx, label in enumerate(sequence):
                if label.startswith('B-'):
                    if current_entity:
                        if not entity_type or current_type == entity_type:
                            entities.add((sent_idx, tuple(current_entity)))
                        current_entity = []
                    current_type = label.split('-')[1]
                    current_entity.append(token_idx)
                    
                elif label.startswith('I-'):
                    if current_entity:
                        current_entity.append(token_idx)
                        
                else:  # O label
                    if current_entity:
                        if not entity_type or current_type == entity_type:
                            entities.add((sent_idx, tuple(current_entity)))
                        current_entity = []
                    current_type = None
            
            if current_entity and (not entity_type or current_type == entity_type):
                entities.add((sent_idx, tuple(current_entity)))
                
        return entities

    def _compute_accuracy(self, y_true: List[List[str]], 
                         y_pred: List[List[str]]) -> float:
        """Compute token-level accuracy."""
        correct = sum(
            sum(t == p for t, p in zip(true_seq, pred_seq))
            for true_seq, pred_seq in zip(y_true, y_pred)
        )
        total = sum(len(seq) for seq in y_true)
        return correct / total

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate detailed evaluation report.
        
        Args:
            output_path (str, optional): Path to save report
            
        Returns:
            str: Formatted report text
        """
        if not self.results:
            raise ValueError("No evaluation results available. Run compute_metrics first.")
        
        report = []
        report.append("Named Entity Recognition Evaluation Report")
        report.append("=" * 50)
        
        # Overall metrics
        report.append("\nOverall Metrics:")
        report.append("-" * 20)
        overall = self.results['overall']
        report.append(f"Accuracy: {overall['accuracy']:.4f}")
        report.append(f"Precision: {overall['precision']:.4f}")
        report.append(f"Recall: {overall['recall']:.4f}")
        report.append(f"F1 Score: {overall['f1']:.4f}")
        
        # Entity-specific metrics
        report.append("\nEntity-specific Metrics:")
        report.append("-" * 20)
        for entity, metrics in self.results['entity'].items():
            report.append(f"\n{entity}:")
            report.append(f"  Precision: {metrics['precision']:.4f}")
            report.append(f"  Recall: {metrics['recall']:.4f}")
            report.append(f"  F1 Score: {metrics['f1']:.4f}")
            report.append(f"  Support: {metrics['support']}")
        
        # Error analysis
        report.append("\nError Analysis:")
        report.append("-" * 20)
        error_types = self.results['error_analysis']['error_types']
        for error_type, count in error_types.items():
            report.append(f"{error_type}: {count}")
        
        report_text = "\n".join(report)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report_text)
            logger.info("Evaluation report saved to %s", output_path)
        
        return report_text

    def save_results(self, path: str):
        """
        Save evaluation results to file.
        
        Args:
            path (str): Save path
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert confusion matrix to dict for JSON serialization
            results_copy = self.results.copy()
            results_copy['confusion_matrix'] = self.results['confusion_matrix'].to_dict()
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results_copy, f, ensure_ascii=False, indent=2)
                
            logger.info("Evaluation results saved to %s", path)
            
        except Exception as e:
            logger.error("Error saving evaluation results: %s", str(e))
            raise
