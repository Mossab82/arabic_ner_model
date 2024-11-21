import re
from typing import List, Dict, Optional
import arabic_reshaper
from bidi.algorithm import get_display
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ArabicPreprocessor:
    """
    Preprocessor for classical Arabic text.
    
    Handles text normalization, character conversion, and special character processing
    specifically designed for classical Arabic literature.
    
    Attributes:
        patterns (Dict[str, re.Pattern]): Compiled regex patterns for text processing
        config (Dict): Configuration parameters for preprocessing
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the preprocessor with configuration settings.
        
        Args:
            config (Dict, optional): Configuration dictionary containing preprocessing parameters
        """
        self.config = config or {}
        self.patterns = {
            'diacritics': re.compile(r'[\u064B-\u065F\u0670]'),
            'special_chars': re.compile(r'[^\w\s]'),
            'numbers': re.compile(r'[0-9]'),
            'tatweel': re.compile(r'\u0640'),
            'whitespace': re.compile(r'\s+')
        }
        logger.info("Initialized ArabicPreprocessor with %d patterns", len(self.patterns))

    def normalize_text(self, text: str) -> str:
        """
        Normalize Arabic text by applying various preprocessing steps.
        
        Args:
            text (str): Input Arabic text
            
        Returns:
            str: Normalized text
            
        Example:
            >>> preprocessor = ArabicPreprocessor()
            >>> preprocessor.normalize_text("الْعَرَبِيَّة")
            'العربية'
        """
        if not isinstance(text, str):
            raise TypeError("Input must be a string")
            
        try:
            # Apply preprocessing steps
            text = self.remove_diacritics(text)
            text = self.normalize_alef(text)
            text = self.normalize_tah_marbota(text)
            text = self.remove_tatweel(text)
            text = self.normalize_whitespace(text)
            
            logger.debug("Successfully normalized text of length %d", len(text))
            return text
        except Exception as e:
            logger.error("Error normalizing text: %s", str(e))
            raise

    def remove_diacritics(self, text: str) -> str:
        """
        Remove Arabic diacritical marks from text.
        
        Args:
            text (str): Input text with diacritics
            
        Returns:
            str: Text with diacritics removed
        """
        return self.patterns['diacritics'].sub('', text)

    def normalize_alef(self, text: str) -> str:
        """
        Normalize different forms of Alef to a single form.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized Alef characters
        """
        return re.sub('[إأآا]', 'ا', text)

    def normalize_tah_marbota(self, text: str) -> str:
        """
        Normalize Tah Marbota to Ha.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized Tah Marbota
        """
        return text.replace('ة', 'ه')

    def remove_tatweel(self, text: str) -> str:
        """
        Remove tatweel character from text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with tatweel characters removed
        """
        return self.patterns['tatweel'].sub('', text)

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace characters in text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized whitespace
        """
        return self.patterns['whitespace'].sub(' ', text).strip()

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Arabic text into words.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of tokens
            
        Example:
            >>> preprocessor = ArabicPreprocessor()
            >>> preprocessor.tokenize("السلام عليكم")
            ['السلام', 'عليكم']
        """
        normalized_text = self.normalize_text(text)
        tokens = normalized_text.split()
        return tokens

    def process_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """
        Process an entire text file.
        
        Args:
            file_path (str): Path to input file
            output_path (str, optional): Path to save processed text
            
        Returns:
            str: Processed text
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            IOError: If there's an error reading/writing files
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            processed_text = self.normalize_text(text)
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(processed_text)
                logger.info("Processed text saved to %s", output_path)
                
            return processed_text
            
        except FileNotFoundError:
            logger.error("Input file not found: %s", file_path)
            raise
        except Exception as e:
            logger.error("Error processing file: %s", str(e))
            raise
