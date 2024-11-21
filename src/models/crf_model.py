from typing import List, Dict, Tuple
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from src.features.extractor import FeatureExtractor

class CRFModel:
    def __init__(self, config: Dict):
        self.config = config
        self.feature_extractor = FeatureExtractor(config)
        self.model = sklearn_crfsuite.CRF(
            algorithm=config['algorithm'],
            c1=config['c1'],
            c2=config['c2'],
            max_iterations=config['max_iterations'],
            all_possible_transitions=config['all_possible_transitions'],
        )
        
    def prepare_features(self, sentences: List[List[str]]) -> List[List[Dict]]:
        """Prepare features for each token in each sentence."""
        X = []
        for sentence in sentences:
            sentence_features = []
            for i in range(len(sentence)):
                features = self.feature_extractor.extract_features(sentence, i)
                sentence_features.append(features)
            X.append(sentence_features)
        return X
    
    def fit(self, X: List[List[str]], y: List[List[str]]):
        """Train the CRF model."""
        X_features = self.prepare_features(X)
        self.model.fit(X_features, y)
        
    def predict(self, X: List[List[str]]) -> List[List[str]]:
        """Predict entity labels for input sentences."""
        X_features = self.prepare_features(X)
        return self.model.predict(X_features)
    
    def evaluate(self, X_test: List[List[str]], y_test: List[List[str]]) -> Dict:
        """Evaluate model performance."""
        y_pred = self.predict(X_test)
        
        labels = list(self.model.classes_)
        labels.remove('O')  # Remove non-entity class
        
        metrics_dict = {
            'accuracy': metrics.flat_accuracy_score(y_test, y_pred),
            'f1': metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels),
            'precision': metrics.flat_precision_score(y_test, y_pred, average='weighted', labels=labels),
            'recall': metrics.flat_recall_score(y_test, y_pred, average='weighted', labels=labels)
        }
        
        return metrics_dict
