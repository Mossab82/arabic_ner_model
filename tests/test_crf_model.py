# tests/test_crf_model.py
```python
"""
Unit tests for the CRF model module.
"""

import pytest
from src.models.crf_model import CRFModel
import numpy as np
from pathlib import Path
import tempfile

@pytest.fixture
def sample_data():
    """Create sample training data."""
    X = [
        ['قال', 'الملك', 'شهريار', 'في', 'قصره'],
        ['وجد', 'علاء', 'الدين', 'المصباح', 'السحري'],
        ['في', 'مدينة', 'بغداد', 'القديمة']
    ]
    y = [
        ['O', 'B-PERSON', 'I-PERSON', 'O', 'O'],
        ['O', 'B-PERSON', 'I-PERSON', 'B-OBJECT', 'I-OBJECT'],
        ['O', 'B-LOCATION', 'I-LOCATION', 'O']
    ]
    return X, y

@pytest.fixture
def crf_model():
    """Create CRF model instance."""
    config = {
        'algorithm': 'lbfgs',
        'c1': 0.1,
        'c2': 0.1,
        'max_iterations': 50,
        'all_possible_transitions': True,
        'verbose': False
    }
    return CRFModel(config)

def test_model_initialization(crf_model):
    """Test model initialization."""
    assert crf_model.model is not None
    assert crf_model.feature_extractor is not None
    assert crf_model.config['algorithm'] == 'lbfgs'

def test_feature_preparation(crf_model, sample_data):
    """Test feature preparation."""
    X, _ = sample_data
    features = crf_model.prepare_features(X)
    assert len(features) == len(X)
    assert all(isinstance(sent_features, list) for sent_features in features)
    assert all(isinstance(token_features, dict) for sent in features for token_features in sent)

def test_model_training(crf_model, sample_data):
    """Test model training."""
    X, y = sample_data
    crf_model.fit(X, y)
    
    # Test predictions
    predictions = crf_model.predict(X)
    assert len(predictions) == len(X)
    assert all(len(pred) == len(sent) for pred, sent in zip(predictions, X))

def test_model_evaluation(crf_model, sample_data):
    """Test model evaluation."""
    X, y = sample_data
    crf_model.fit(X, y)
    metrics = crf_model.evaluate(X, y)
    
    assert 'accuracy' in metrics
    assert 'f1' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'entity_metrics' in metrics
    
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['f1'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1

def test_model_save_load(crf_model, sample_data):
    """Test model saving and loading."""
    X, y = sample_data
    crf_model.fit(X, y)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pkl"
        
        # Save model
        crf_model.save(str(model_path))
        assert model_path.exists()
        
        # Load model
        loaded_model = CRFModel.load(str(model_path))
        assert loaded_model is not None
        
        # Compare predictions
        original_preds = crf_model.predict(X)
        loaded_preds = loaded_model.predict(X)
        assert original_preds == loaded_preds

def test_entity_metrics(crf_model, sample_data):
    """Test entity-specific metrics calculation."""
    X, y = sample_data
    crf_model.fit(X, y)
    metrics = crf_model.evaluate(X, y)
    
    assert 'entity_metrics' in metrics
    assert 'PERSON' in metrics['entity_metrics']
    assert 'LOCATION' in metrics['entity_metrics']
    assert 'OBJECT' in metrics['entity_metrics']
    
    for entity_type in metrics['entity_metrics']:
        entity_scores = metrics['entity_metrics'][entity_type]
        assert 'precision' in entity_scores
        assert 'recall' in entity_scores
        assert 'f1' in entity_scores

def test_validation_monitoring(crf_model, sample_data):
    """Test training with validation monitoring."""
    X, y = sample_data
    
    # Split data for validation
    val_size = len(X) // 3
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]
    
    # Train with validation data
    crf_model.fit(X_train, y_train, validation_data=(X_val, y_val))
    
    # Evaluate on validation set
    val_metrics = crf_model.evaluate(X_val, y_val)
    assert all(0 <= val_metrics[metric] <= 1 for metric in ['accuracy', 'f1', 'precision', 'recall'])

def test_error_handling(crf_model):
    """Test error handling in model operations."""
    # Test empty input
    with pytest.raises(ValueError):
        crf_model.predict([[]])
    
    # Test invalid input type
    with pytest.raises(TypeError):
        crf_model.fit(None, None)
    
    # Test invalid file path
    with pytest.raises(Exception):
        crf_model.load("nonexistent_path.pkl")
