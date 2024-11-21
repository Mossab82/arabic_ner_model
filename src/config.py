import os
from pathlib import Path
# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Data paths
RAW_DATA_PATH = DATA_DIR / "raw"
PROCESSED_DATA_PATH = DATA_DIR / "processed"

# Model configuration
CRF_CONFIG = {
    'algorithm': 'lbfgs',
    'c1': 0.1,  # L1 regularization
    'c2': 0.1,  # L2 regularization
    'max_iterations': 100,
    'all_possible_transitions': True,
    'early_stopping': True,
    'tolerance': 1e-4
}

# Feature extraction configuration
FEATURE_CONFIG = {
    'window_size': 3,
    'use_morphology': True,
    'use_position': True,
    'ngram_range': (1, 3)
}

# Entity types
ENTITY_TYPES = [
    'PERSON',
    'LOCATION',
    'ORGANIZATION',
    'MYTHICAL',
    'OBJECT'
]
