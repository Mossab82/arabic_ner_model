"""
Unit tests for the advanced feature extractor module.
"""

import pytest
import torch
import numpy as np
from src.features.advanced_features import AdvancedFeatureExtractor

@pytest.fixture
def advanced_extractor():
    """Create advanced feature extractor instance for testing."""
    return AdvancedFeatureExtractor()

def test_contextual_embeddings(advanced_extractor):
    """Test BERT embedding extraction."""
    text = "قال الملك شهريار"
    embeddings = advanced_extractor.extract_contextual_embeddings(text)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[1] == 768  # BERT base hidden size

def test_narrative_features(advanced_extractor):
    """Test narrative feature extraction."""
    text = "قال الملك شهريار في يوم من الأيام"
    features = advanced_extractor.extract_narrative_features(text)
    assert 'narrative_dialogue_markers_count' in features
    assert 'dialogue_ratio' in features
    assert features['narrative_dialogue_markers_count'] > 0

def test_stylometric_features(advanced_extractor):
    """Test stylometric feature extraction."""
    text = "كان الملك شهريار في قصره الكبير، وكان معه وزيره."
    features = advanced_extractor.extract_stylometric_features(text)
    assert 'avg_word_length' in features
    assert 'vocabulary_richness' in features
    assert 0 <= features['vocabulary_richness'] <= 1

def test_morphological_features(advanced_extractor):
    """Test morphological feature extraction."""
    word = "الملوك"
    features = advanced_extractor.extract_morphological_features(word)
    assert features['has_definite_article'] == True
    assert 'root_pattern' in features

def test_full_feature_set(advanced_extractor):
    """Test complete feature extraction."""
    text = "قال الملك شهريار"
    features = advanced_extractor.get_full_feature_set(text, position=1)
    assert 'contextual_embedding' in features
    assert 'narrative_features' in features
    assert 'stylometric_features' in features
    assert 'morphological_features' in features

def test_scene_change_detection(advanced_extractor):
    """Test scene change detection."""
    text = "وفي اليوم التالي\nوعندما حل المساء"
    scene_density = advanced_extractor._detect_scene_changes(text)
    assert scene_density > 0

def test_dialogue_ratio(advanced_extractor):
    """Test dialogue ratio calculation."""
    text = "قال الملك. همس الوزير. تحدث الحكيم."
    ratio = advanced_extractor._calculate_dialogue_ratio(text)
    assert 0 <= ratio <= 1

def test_root_pattern_extraction(advanced_extractor):
    """Test root pattern extraction."""
    assert len(advanced_extractor._extract_root_pattern("الكاتب")) <= 3
    assert len(advanced_extractor._extract_root_pattern("مكتوب")) <= 3

def test_device_handling(advanced_extractor):
    """Test device handling for BERT model."""
    assert hasattr(advanced_extractor, 'device')
    assert isinstance(advanced_extractor.device, torch.device)
