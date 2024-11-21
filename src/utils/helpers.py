"""
Helper functions for Arabic NER project.
"""

from typing import List, Dict, Tuple, Union
import re
import json
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def setup_logger(log_file: str = None) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_file (str, optional): Path to log file
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger('arabic_ner')
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_data(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Load data from file in various formats.
    
    Args:
        file_path (str): Path to data file
        
    Returns:
        Tuple[List[List[str]], List[List[str]]]: Tokens and labels
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data['tokens'], data['labels']
    
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        tokens = [sent.split() for sent in df['text']]
        labels = [labs.split() for labs in df['labels']]
        return tokens, labels
    
    elif file_path.suffix == '.conll':
        tokens, labels = [], []
        current_tokens, current_labels = [], []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    token, label = line.split('\t')
                    current_tokens.append(token)
                    current_labels.append(label)
                elif current_tokens:
                    tokens.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens, current_labels = [], []
                    
        if current_tokens:
            tokens.append(current_tokens)
            labels.append(current_labels)
            
        return tokens, labels
    
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

def save_predictions(tokens: List[List[str]], 
                    predictions: List[List[str]], 
                    output_path: str,
                    format: str = 'conll') -> None:
    """
    Save predictions to file.
    
    Args:
        tokens (List[List[str]]): Input tokens
        predictions (List[List[str]]): Predicted labels
        output_path (str): Output file path
        format (str): Output format ('conll', 'json', or 'csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'conll':
        with open(output_path, 'w', encoding='utf-8') as f:
            for sent_tokens, sent_preds in zip(tokens, predictions):
                for token, label in zip(sent_tokens, sent_preds):
                    f.write(f"{token}\t{label}\n")
                f.write("\n")
                
    elif format == 'json':
        data = {
            'tokens': tokens,
            'predictions': predictions,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_sentences': len(tokens),
                'num_tokens': sum(len(sent) for sent in tokens)
            }
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    elif format == 'csv':
        df = pd.DataFrame({
            'text': [' '.join(sent) for sent in tokens],
            'predictions': [' '.join(preds) for preds in predictions]
        })
        df.to_csv(output_path, index=False)
        
    else:
        raise ValueError(f"Unsupported output format: {format}")

def preprocess_text(text: str) -> str:
    """
    Preprocess Arabic text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    # Remove diacritics
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # Normalize alef
    text = re.sub('[إأآا]', 'ا', text)
    
    # Normalize tah marbota
    text = text.replace('ة', 'ه')
    
    # Remove tatweel
    text = re.sub(r'\u0640', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def batch_generator(data: List, 
                   batch_size: int, 
                   shuffle: bool = True) -> List:
    """
    Generate batches from data.
    
    Args:
        data (List): Input data
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        
    Yields:
        List: Data batch
    """
    indices = list(range(len(data)))
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, len(data), batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        yield [data[i] for i in batch_indices]

def calculate_metrics(predictions: List[List[str]], 
                     gold_labels: List[List[str]]) -> Dict:
    """
    Calculate evaluation metrics.
    
    Args:
        predictions (List[List[str]]): Predicted labels
        gold_labels (List[List[str]]): True labels
        
    Returns:
        Dict: Evaluation metrics
    """
    correct = 0
    total = 0
    entity_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for pred_seq, gold_seq in zip(predictions, gold_labels):
        for pred, gold in zip(pred_seq, gold_seq):
            if pred == gold:
                correct += 1
            total += 1
            
            if gold != 'O':
                entity_type = gold.split('-')[1]
                if pred == gold:
                    entity_metrics[entity_type]['tp'] += 1
                else:
                    entity_metrics[entity_type]['fn'] += 1
            
            if pred != 'O' and pred != gold:
                entity_type = pred.split('-')[1]
                entity_metrics[entity_type]['fp'] += 1
    
    # Calculate metrics for each entity type
    results = {
        'accuracy': correct / total,
        'entity_types': {}
    }
    
    for entity_type, counts in entity_metrics.items():
        precision = counts['tp'] / (counts['tp'] + counts['fp']) if counts['tp'] + counts['fp'] > 0 else 0
        recall = counts['tp'] / (counts['tp'] + counts['fn']) if counts['tp'] + counts['fn'] > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        results['entity_types'][entity_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': counts['tp'] + counts['fn']
        }
    
    return results
