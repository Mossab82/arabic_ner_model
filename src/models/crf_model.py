"""
CRF model implementation for Arabic NER.
Implements Conditional Random Fields for sequence labeling.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn_crfsuite import CRF
import torch
from src.features.extractor import FeatureExtractor
from src.features.advanced_features import AdvancedFeatureExtractor
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

class CRFModel:
    """
    CRF-based model for Arabic Named Entity Recognition.
    
    Implements a CRF model with advanced feature extraction
    and optimized training procedures.
    
    Attributes:
        model (CRF): Underlying CRF model
        feature_extractor (AdvancedFeatureExtractor): Feature extractor
        config (Dict): Model configuration
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize CRF model.
        
        Args:
            config (Dict, optional): Model configuration
        """
        self.config = config or {
            'algorithm': 'lbfgs',
            'c1': 0.1,
            'c2': 0.1,
            'max_iterations': 100,
            'all_possible_transitions': True,
            'verbose': True
        }
        
        self.model = CRF(
            algorithm=self.config['algorithm'],
            c1=self.config['c1'],
            c2=self.config['c2'],
            max_iterations=self.config['max_iterations'],
            all_possible_transitions=self.config['all_possible_transitions'],
            verbose=self.config['verbose']
        )
        
        self.feature_extractor = AdvancedFeatureExtractor(self.config)
        logger.info("Initialized CRF model with config: %s", self.config)

    def prepare_features(self, sentences: List[List[str]]) -> List[List[Dict]]:
        """
        Prepare features for each token in each sentence.
        
        Args:
            sentences (List[List[str]]): List of tokenized sentences
            
        Returns:
            List[List[Dict]]: Features for each token
        """
        features = []
        for sentence in sentences:
            sentence_features = []
            for i in range(len(sentence)):
                token_features = self.feature_extractor.extract_features(sentence, i)
                sentence_features.append(token_features)
            features.append(sentence_features)
        return features

    def fit(self, X: List[List[str]], y: List[List[str]], validation_data: Optional[Tuple] = None):
        """
        Train the CRF model.
        
        Args:
            X (List[List[str]]): List of tokenized sentences
            y (List[List[str]]): List of label sequences
            validation_data (Tuple, optional): Validation data (X_val, y_val)
        """
        try:
            logger.info("Starting model training with %d sentences", len(X))
            
            # Prepare features
            X_features = self.prepare_features(X)
            
            # Prepare validation features if provided
            if validation_data:
                X_val, y_val = validation_data
                X_val_features = self.prepare_features(X_val)
                
                # Training with validation monitoring
                best_f1 = 0
                patience = self.config.get('patience', 3)
                patience_counter = 0
                
                for epoch in range(self.config['max_iterations']):
                    self.model.fit(X_features, y, X_val_features, y_val)
                    val_pred = self.model.predict(X_val_features)
                    current_f1 = self._compute_f1(y_val, val_pred)
                    
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        logger.info("Early stopping triggered at epoch %d", epoch)
                        break
            else:
                # Training without validation
                self.model.fit(X_features, y)
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error("Error during model training: %s", str(e))
            raise

    def predict(self, X: List[List[str]]) -> List[List[str]]:
        """
        Predict entity labels for input sentences.
        
        Args:
            X (List[List[str]]): Input sentences
            
        Returns:
            List[List[str]]: Predicted labels
        """
        try:
            X_features = self.prepare_features(X)
            predictions = self.model.predict(X_features)
            return predictions
        except Exception as e:
            logger.error("Error during prediction: %s", str(e))
            raise

    def evaluate(self, X_test: List[List[str]], y_test: List[List[str]]) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X_test (List[List[str]]): Test sentences
            y_test (List[List[str]]): True labels
            
        Returns:
            Dict: Evaluation metrics
        """
        try:
            y_pred = self.predict(X_test)
            
            metrics = {
                'accuracy': self._compute_accuracy(y_test, y_pred),
                'f1': self._compute_f1(y_test, y_pred),
                'precision': self._compute_precision(y_test, y_pred),
                'recall': self._compute_recall(y_test, y_pred)
            }
            
            # Compute per-entity metrics
            entity_metrics = self._compute_entity_metrics(y_test, y_pred)
            metrics['entity_metrics'] = entity_metrics
            
            return metrics
            
        except Exception as e:
            logger.error("Error during evaluation: %s", str(e))
            raise

    def _compute_accuracy(self, y_true: List[List[str]], y_pred: List[List[str]]) -> float:
        """Compute token-level accuracy."""
        correct = sum(
            sum(yt == yp for yt, yp in zip(y_true_seq, y_pred_seq))
            for y_true_seq, y_pred_seq in zip(y_true, y_pred)
        )
        total = sum(len(seq) for seq in y_true)
        return correct / total

    def _compute_f1(self, y_true: List[List[str]], y_pred: List[List[str]]) -> float:
        """Compute macro F1 score."""
        precision = self._compute_precision(y_true, y_pred)
        recall = self._compute_recall(y_true, y_pred)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def _compute_precision(self, y_true: List[List[str]], y_pred: List[List[str]]) -> float:
        """Compute precision score."""
        true_positives = 0
        predicted_positives = 0
        
        for y_true_seq, y_pred_seq in zip(y_true, y_pred):
            for yt, yp in zip(y_true_seq, y_pred_seq):
                if yp != 'O':
                    predicted_positives += 1
                    if yt == yp:
                        true_positives += 1
        
        return true_positives / max(predicted_positives, 1)

    def _compute_recall(self, y_true: List[List[str]], y_pred: List[List[str]]) -> float:
        """Compute recall score."""
        true_positives = 0
        actual_positives = 0
        
        for y_true_seq, y_pred_seq in zip(y_true, y_pred):
            for yt, yp in zip(y_true_seq, y_pred_seq):
                if yt != 'O':
                    actual_positives += 1
                    if yt == yp:
                        true_positives += 1
        
        return true_positives / max(actual_positives, 1)

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
        
        for y_true_seq, y_pred_seq in zip(y_true, y_pred):
            for yt, yp in zip(y_true_seq, y_pred_seq):
                is_true_entity = yt.endswith(entity_type)
                is_pred_entity = yp.endswith(entity_type)
                
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

    def save(self, path: str):
        """
        Save model to disk.
        
        Args:
            path (str): Save path
        """
        try:
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            model_data = {
                'model': self.model,
                'config': self.config
            }
            
            joblib.dump(model_data, save_path)
            logger.info("Model saved successfully to %s", path)
            
        except Exception as e:
            logger.error("Error saving model: %s", str(e))
            raise

    @classmethod
    def load(cls, path: str) -> 'CRFModel':
        """
        Load model from disk.
        
        Args:
            path (str): Load path
            
        Returns:
            CRFModel: Loaded model instance
        """
        try:
            model_data = joblib.load(path)
            instance = cls(model_data['config'])
            instance.model = model_data['model']
            logger.info("Model loaded successfully from %s", path)
            return instance
            
        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise
