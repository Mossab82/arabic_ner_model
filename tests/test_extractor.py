"""
Unit tests for the feature extractor module.
"""

import pytest
from src.features.extractor import FeatureExtractor

@pytest.fixture
def extractor():
    """Create feature extractor instance for testing."""
    return FeatureExtractor()

def test_token_features(extractor):
    """Test basic token feature extraction."""
    features = extractor._get_token_features('الملك')
    assert features['word'] == 'الملك'
    assert features['word.starts_with_al'] == True
    assert features['word.is_honorific'] == True

def test_context_features(extractor):
    """Test context feature extraction."""
    tokens = ['قال', 'الملك', 'شهريار']
    features = extractor._get_context_features(tokens, 1)
    assert 'prev1.word' in features
    assert 'next1.word' in features

def test_pattern_features(extractor):
    """Test pattern-based feature extraction."""
    features = extractor._get_pattern_features('مدينة بغداد')
    assert any('location_patterns' in key for key in features.keys())

def test_position_features(extractor):
    """Test position feature extraction."""
    features = extractor._get_position_features(0, 3)
    assert features['position.is_first'] == True
    assert features['position.is_last'] == False

def test_full_feature_extraction(extractor):
    """Test complete feature extraction process."""
    tokens = ['قال', 'الملك', 'شهريار']
    features = extractor.extract_features(tokens, 1)
    assert 'word' in features
    assert 'position.absolute' in features
    assert len(features) > 10

def test_batch_extraction(extractor):
    """Test batch feature extraction."""
    sentences = [['قال', 'الملك'], ['في', 'مدينة', 'بغداد']]
    features = extractor.batch_extract_features(sentences)
    assert len(features) == 2
    assert len(features[0]) == 2
    assert len(features[1]) == 3

def test_invalid_input(extractor):
    """Test handling of invalid input."""
    with pytest.raises(ValueError):
        extractor.extract_features([], 0)
    with pytest.raises(ValueError):
        extractor.extract_features(['token'], 1)

def test_feature_names(extractor):
    """Test feature names retrieval."""
    names = extractor.get_feature_names()
    assert isinstance(names, list)
    assert len(names) > 0
    assert 'word' in names
