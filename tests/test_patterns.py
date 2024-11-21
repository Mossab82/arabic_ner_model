"""
Unit tests for the Arabic patterns module.
"""

import pytest
from src.features.patterns import ArabicPatterns

@pytest.fixture
def patterns():
    """Create patterns instance for testing."""
    return ArabicPatterns()

def test_honorific_detection(patterns):
    """Test honorific title detection."""
    assert patterns.is_honorific('الملك')
    assert patterns.is_honorific('السلطان')
    assert not patterns.is_honorific('رجل')

def test_pattern_matching(patterns):
    """Test pattern matching functionality."""
    text = "قال الملك شهريار في مدينة بغداد"
    matches = patterns.get_matching_patterns(text)
    assert 'name_patterns' in matches or 'location_patterns' in matches

def test_location_indicator(patterns):
    """Test location indicator detection."""
    assert patterns.is_location_indicator('مدينة')
    assert patterns.is_location_indicator('قصر')
    assert not patterns.is_location_indicator('كتاب')

def test_mythical_indicator(patterns):
    """Test mythical entity indicator detection."""
    assert patterns.is_mythical_indicator('العفريت')
    assert patterns.is_mythical_indicator('السحري')
    assert not patterns.is_mythical_indicator('الرجل')

def test_narrative_marker(patterns):
    """Test narrative marker detection."""
    assert patterns.is_narrative_marker('قال')
    assert patterns.is_narrative_marker('حكى')
    assert not patterns.is_narrative_marker('ذهب')

def test_custom_patterns_loading(tmp_path):
    """Test loading custom patterns from file."""
    patterns_file = tmp_path / "patterns.json"
    patterns_file.write_text('{"test_patterns": ["test\\s+\\w+"]}')
    patterns = ArabicPatterns(str(patterns_file))
    assert 'test_patterns' in patterns.rules

def test_invalid_patterns_file():
    """Test handling of invalid patterns file."""
    patterns = ArabicPatterns("nonexistent.json")
    assert patterns.rules == patterns._get_default_rules()
