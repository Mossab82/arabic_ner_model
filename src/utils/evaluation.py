from typing import List, Dict
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(y_true: List[List[str]], y_pred: List[List[str]], labels: List[str]) -> Dict:
    """Compute detailed evaluation metrics."""
    # Flatten the predictions and true labels
    y_true_flat = [item for sublist in y_true for item in sublist]
    y_pred_flat = [item for sublist in y_pred for item in sublist]
    
    # Generate classification report
    report = classification_report(y_true_flat, y_pred_flat, labels=labels, output_dict=True)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)
    
    return {
        'classification_report': report,
        'confusion_matrix': conf_matrix
    }

def plot_confusion_matrix(conf_matrix: np.ndarray, labels: List[str], output_path: str):
    """Plot confusion matrix as a heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
