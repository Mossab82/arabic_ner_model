"""
Unit tests for the rule-based model module.
"""

import pytest
from src.models.rule_based import RuleBasedModel
import json
from pathlib import Path

@pytest.fixture
def sample_data():
    """Create sample test data."""
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
def rule_based_model():
    """Create rule-based model instance."""
    return RuleBasedModel()

@pytest.fixture
def custom_gazetteers(tmp_path):
    """Create custom gazetteers file."""
    gazetteers = {
        'PERSON': ['زبيدة', 'جعفر'],
        'LOCATION': ['سامراء', 'الكوفة'],
        'MYTHICAL': ['سندباد', 'طائر الرخ']
    }
    
    gazetteer_path = tmp_path / "gazetteers.json"
    with open(gazetteer_path, 'w', encoding='utf-8') as f:
        json.dump(gazetteers, f, ensure_ascii=False)
    
    return str(gazetteer_path)

def test_model_initialization(rule_based_model):
    """Test model initialization."""
    assert rule_based_model.patterns is not None
    assert rule_based_model.gazetteers is not None
    assert rule_based_model.rules is not None
    
    assert 'PERSON' in rule_based_model.gazetteers
    assert 'LOCATION' in rule_based_model.gazetteers
    assert 'MYTHICAL' in rule_based_model.gazetteers

def test_gazetteer_loading(custom_gazetteers):
    """Test loading custom gazetteers."""
    model = RuleBasedModel({'gazetteers_path': custom_gazetteers})
    assert 'زبيدة' in model.gazetteers['PERSON']
    assert 'سامراء' in model.gazetteers['LOCATION']
    assert 'سندباد' in model.gazetteers['MYTHICAL']

def test_prediction(rule_based_model, sample_data):
    """Test entity prediction."""
    X, _ = sample_data
    predictions = rule_based_model.predict(X)
    
    assert len(predictions) == len(X)
    assert all(len(pred) == len(sent) for pred, sent in zip(predictions, X))
    assert all(all(label.startswith(('B-', 'I-', 'O')) for label in sent) for sent in predictions)

def test_evaluation(rule_based_model, sample_data):
    """Test model evaluation."""
    X, y = sample_data
    metrics = rule_based_model.evaluate(X, y)
    
    assert 'accuracy' in metrics
    assert 'entity_metrics' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    
    for entity_type in metrics['entity_metrics']:
        entity_scores = metrics['entity_metrics'][entity_type]
        assert 0 <= entity_scores['precision'] <= 1
        assert 0 <= entity_scores['recall'] <= 1
        assert 0 <= entity_scores['f1'] <= 1

def test_rule_matching(rule_based_model):
    """Test rule-based matching."""
    text = ['قال', 'الملك', 'شهريار']
    labels = rule_based_model._label_sentence(text)
    
    assert len(labels) == len(text)
    assert 'B-PERSON' in labels or 'I-PERSON' in labels

def test_gazetteer_matching(rule_based_model):
    """Test gazetteer-based matching."""
    text = ['في', 'مدينة', 'بغداد']
    labels = rule_based_model._label_sentence(text)
    
    assert len(labels) == len(text)
    assert 'B-LOCATION' in labels

def test_condition_checking(rule_based_model):
    """Test condition checking in rules."""
    assert rule_based_model._check_conditions('بغداد', ['is_location_in_gazetteer'])
    assert not rule_based_model._check_conditions('invalid_location', ['is_location_in_gazetteer'])

def test_error_handling(rule_based_model):
    """Test error handling."""
    # Test empty input
    assert rule_based_model.predict([[]]) == [[]]
    
    # Test invalid input type
    with pytest.raises(AttributeError):
        rule_based_model.predict(None)
    
    # Test missing gazetteer file
    model = RuleBasedModel({'gazetteers_path': 'nonexistent.json'})
    assert model.gazetteers == rule_based_model.gazetteers

def test_metrics_calculation(rule_based_model):
    """Test metrics calculation."""
    y_true = [['O', 'B-PERSON', 'I-PERSON']]
    y_pred = [['O', 'B-PERSON', 'O']]
    
    metrics = rule_based_model._compute_entity_metrics(y_true, y_pred)
    assert 'PERSON' in metrics
    assert all(key in metrics['PERSON'] for key in ['precision', 'recall', 'f1'])
