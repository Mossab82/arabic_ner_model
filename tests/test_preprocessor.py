"""
Unit tests for the Arabic preprocessor module.
"""

import pytest
from src.data.preprocessor import ArabicPreprocessor
import os

@pytest.fixture
def preprocessor():
    """Create a preprocessor instance for testing."""
    return ArabicPreprocessor()

def test_normalize_text(preprocessor):
    """Test text normalization functionality."""
    # Test diacritics removal
    input_text = "الْعَرَبِيَّة"
    expected = "العربية"
    assert preprocessor.normalize_text(input_text) == expected
    
    # Test alef normalization
    input_text = "إأآا"
    expected = "اااا"
    assert preprocessor.normalize_text(input_text) == expected

def test_tokenize(preprocessor):
    """Test tokenization functionality."""
    input_text = "السلام عليكم ورحمة الله"
    expected = ['السلام', 'عليكم', 'ورحمة', 'الله']
    assert preprocessor.tokenize(input_text) == expected

def test_invalid_input(preprocessor):
    """Test handling of invalid input."""
    with pytest.raises(TypeError):
        preprocessor.normalize_text(None)
    with pytest.raises(TypeError):
        preprocessor.normalize_text(123)

def test_file_processing(preprocessor, tmp_path):
    """Test file processing functionality."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_content = "الْعَرَبِيَّة\nإأآا"
    test_file.write_text(test_content, encoding='utf-8')
    
    # Test processing
    output_file = tmp_path / "output.txt"
    result = preprocessor.process_file(str(test_file), str(output_file))
    
    assert result == "العربية\nاااا"
    assert output_file.read_text(encoding='utf-8') == "العربية\nاااا"

def test_file_not_found(preprocessor):
    """Test handling of non-existent files."""
    with pytest.raises(FileNotFoundError):
        preprocessor.process_file("nonexistent.txt")
