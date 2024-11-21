from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

class ModelOptimizer:
    def __init__(self, base_model, config: Dict):
        self.base_model = base_model
        self.config = config
        self.best_params = None
        self.performance_metrics = {}
        
    def optimize_hyperparameters(self, X_train: List, y_train: List, 
                               n_iter: int = 100) -> Dict:
        """Optimize model hyperparameters using randomized search."""
        param_dist = {
            'c1': uniform(0.05, 0.5),
            'c2': uniform(0.05, 0.5),
            'max_iterations': randint(50, 200),
            'feature_window_size': randint(2, 5)
        }
        
        random_search = RandomizedSearchCV(
            self.base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=5,
            scoring='f1_weighted',
            n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        self.best_params = random_search.best_params_
        
        return self.best_params
    
    def optimize_inference(self, model_path: str) -> None:
        """Optimize model for inference using quantization and pruning."""
        if torch.cuda.is_available():
            self.base_model.to('cuda')
            
        # Quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.base_model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        # Pruning
        parameters_to_prune = []
        for module in self.base_model.modules():
            if isinstance(module, nn.Linear):
                parameters_to_prune.append((module, 'weight'))
                
        torch.nn.utils.prune.global_unstructured(
            parameters_to_prune,
            pruning_method=torch.nn.utils.prune.L1Unstructured,
            amount=0.2,
        )
        
        # Save optimized model
        torch.save(quantized_model.state_dict(), model_path)
        
    def benchmark_performance(self, X_test: List, batch_size: int = 32) -> Dict:
        """Benchmark model inference performance."""
        test_loader = DataLoader(X_test, batch_size=batch_size)
        
        # Measure inference time
        start_time = time.time()
        with torch.no_grad():
            for batch in test_loader:
                _ = self.base_model(batch)
        end_time = time.time()
        
        # Calculate metrics
        inference_time = end_time - start_time
        throughput = len(X_test) / inference_time
        
        self.performance_metrics = {
            'total_inference_time': inference_time,
            'throughput': throughput,
            'latency_per_sample': inference_time / len(X_test)
        }
        
        return self.performance_metrics
