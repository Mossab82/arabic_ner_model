"""
Dataset handling module for Arabic NER.
Provides dataset loading, batching, and preprocessing functionality.
"""

from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from .preprocessor import ArabicPreprocessor

logger = logging.getLogger(__name__)

class ArabicNERDataset(Dataset):
    """
    Dataset class for Arabic Named Entity Recognition.
    
    Handles loading and preprocessing of classical Arabic text data
    for named entity recognition tasks.
    
    Attributes:
        texts (List[str]): List of text segments
        labels (List[List[str]]): List of entity labels for each token
        preprocessor (ArabicPreprocessor): Text preprocessor instance
    """
    
    def __init__(
        self, 
        data_path: str, 
        preprocessor: Optional[ArabicPreprocessor] = None,
        max_length: int = 512,
        label_map: Optional[Dict[str, int]] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to dataset file
            preprocessor (ArabicPreprocessor, optional): Preprocessor instance
            max_length (int): Maximum sequence length
            label_map (Dict[str, int], optional): Mapping of labels to indices
        """
        self.data_path = Path(data_path)
        self.preprocessor = preprocessor or ArabicPreprocessor()
        self.max_length = max_length
        self.label_map = label_map or self._create_default_label_map()
        
        self.texts, self.labels = self._load_data()
        logger.info(
            "Loaded dataset with %d samples from %s",
            len(self.texts),
            self.data_path
        )

    def _create_default_label_map(self) -> Dict[str, int]:
        """Create default label mapping."""
        return {
            'O': 0,
            'B-PERSON': 1,
            'I-PERSON': 2,
            'B-LOCATION': 3,
            'I-LOCATION': 4,
            'B-ORGANIZATION': 5,
            'I-ORGANIZATION': 6,
            'B-MYTHICAL': 7,
            'I-MYTHICAL': 8,
            'B-OBJECT': 9,
            'I-OBJECT': 10
        }

    def _load_data(self) -> Tuple[List[str], List[List[str]]]:
        """
        Load and preprocess data from file.
        
        Returns:
            Tuple[List[str], List[List[str]]]: Texts and their labels
            
        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
        """
        try:
            df = pd.read_csv(self.data_path)
            
            if 'text' not in df.columns or 'labels' not in df.columns:
                raise ValueError("Data file must contain 'text' and 'labels' columns")
                
            texts = []
            labels = []
            
            for _, row in df.iterrows():
                processed_text = self.preprocessor.normalize_text(row['text'])
                text_tokens = self.preprocessor.tokenize(processed_text)
                
                if len(text_tokens) > self.max_length:
                    text_tokens = text_tokens[:self.max_length]
                    
                texts.append(text_tokens)
                labels.append(row['labels'].split())
                
            return texts, labels
            
        except Exception as e:
            logger.error("Error loading data: %s", str(e))
            raise

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing tokens and labels
        """
        tokens = self.texts[idx]
        labels = self.labels[idx]
        
        # Convert labels to indices
        label_indices = [self.label_map[label] for label in labels]
        
        # Pad sequences if necessary
        if len(tokens) < self.max_length:
            padding_length = self.max_length - len(tokens)
            tokens = tokens + ['PAD'] * padding_length
            label_indices = label_indices + [0] * padding_length
        
        return {
            'tokens': torch.tensor(tokens),
            'labels': torch.tensor(label_indices)
        }

    def get_label_distribution(self) -> Dict[str, int]:
        """
        Get distribution of labels in the dataset.
        
        Returns:
            Dict[str, int]: Count of each label type
        """
        distribution = {}
        for label_seq in self.labels:
            for label in label_seq:
                distribution[label] = distribution.get(label, 0) + 1
        return distribution

    def split_dataset(self, 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     seed: int = 42) -> Tuple['ArabicNERDataset', 'ArabicNERDataset', 'ArabicNERDataset']:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            seed (int): Random seed
            
        Returns:
            Tuple[ArabicNERDataset, ArabicNERDataset, ArabicNERDataset]:
                Train, validation, and test datasets
        """
        np.random.seed(seed)
        indices = np.random.permutation(len(self))
        
        train_size = int(len(self) * train_ratio)
        val_size = int(len(self) * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        return (
            self._create_subset(train_indices),
            self._create_subset(val_indices),
            self._create_subset(test_indices)
        )

    def _create_subset(self, indices: np.ndarray) -> 'ArabicNERDataset':
        """Create a subset of the dataset using given indices."""
        subset = ArabicNERDataset(
            self.data_path,
            self.preprocessor,
            self.max_length,
            self.label_map
        )
        subset.texts = [self.texts[i] for i in indices]
        subset.labels = [self.labels[i] for i in indices]
        return subset
