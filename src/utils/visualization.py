"""
Visualization utilities for Arabic NER.
Provides tools for visualizing entity recognition results and model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class ResultVisualizer:
    """
    Visualization tools for NER results.
    
    Provides methods for creating various visualizations
    including confusion matrices, entity distributions,
    and performance metrics.
    """
    
    def __init__(self, style: str = 'seaborn'):
        """
        Initialize visualizer.
        
        Args:
            style (str): Matplotlib style to use
        """
        plt.style.use(style)
        self.colors = plt.cm.Set3(np.linspace(0, 1, 10))
        logger.info("Initialized ResultVisualizer with style: %s", style)

    def plot_confusion_matrix(self, 
                            confusion_matrix: pd.DataFrame,
                            output_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot confusion matrix heatmap.
        
        Args:
            confusion_matrix (pd.DataFrame): Confusion matrix
            output_path (str, optional): Path to save plot
            figsize (Tuple[int, int]): Figure size
        """
        plt.figure(figsize=figsize)
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='YlOrRd',
            square=True,
            xticklabels=confusion_matrix.columns,
            yticklabels=confusion_matrix.index
        )
        
        plt.title('Entity Recognition Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info("Saved confusion matrix plot to %s", output_path)
        
        plt.close()

    def plot_entity_distribution(self,
                               entity_counts: Dict[str, int],
                               output_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot entity type distribution.
        
        Args:
            entity_counts (Dict[str, int]): Entity counts
            output_path (str, optional): Path to save plot
            figsize (Tuple[int, int]): Figure size
        """
        plt.figure(figsize=figsize)
        
        # Create bar plot
        bars = plt.bar(entity_counts.keys(), entity_counts.values())
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height):,}',
                ha='center',
                va='bottom'
            )
        
        plt.title('Distribution of Entity Types')
        plt.xlabel('Entity Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info("Saved entity distribution plot to %s", output_path)
        
        plt.close()

    def plot_performance_metrics(self,
                               metrics: Dict[str, Dict[str, float]],
                               output_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (12, 6)) -> None:
        """
        Plot performance metrics for each entity type.
        
        Args:
            metrics (Dict[str, Dict[str, float]]): Performance metrics
            output_path (str, optional): Path to save plot
            figsize (Tuple[int, int]): Figure size
        """
        entities = list(metrics.keys())
        metrics_types = ['precision', 'recall', 'f1']
        
        x = np.arange(len(entities))
        width = 0.25
        
        plt.figure(figsize=figsize)
        
        # Plot bars for each metric
        for i, metric in enumerate(metrics_types):
            values = [metrics[entity][metric] for entity in entities]
            plt.bar(x + i*width, values, width, label=metric.capitalize())
        
        plt.xlabel('Entity Type')
        plt.ylabel('Score')
        plt.title('Performance Metrics by Entity Type')
        plt.xticks(x + width, entities, rotation=45)
        plt.legend()
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info("Saved performance metrics plot to %s", output_path)
        
        plt.close()

    def plot_error_analysis(self,
                           error_analysis: Dict,
                           output_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Create comprehensive error analysis visualization.
        
        Args:
            error_analysis (Dict): Error analysis results
            output_path (str, optional): Path to save plot
            figsize (Tuple[int, int]): Figure size
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Error Types Distribution',
                'Entity-specific Errors',
                'Error Rate Over Time',
                'Error Patterns'
            )
        )
        
        # Plot 1: Error Types Distribution (Pie Chart)
        error_types = error_analysis['error_types']
        fig.add_trace(
            go.Pie(
                labels=list(error_types.keys()),
                values=list(error_types.values()),
                name="Error Types"
            ),
            row=1, col=1
        )
        
        # Plot 2: Entity-specific Errors (Stacked Bar Chart)
        entity_errors = error_analysis['entity_errors']
        entities = list(entity_errors.keys())
        error_categories = ['false_positive', 'false_negative', 'wrong_entity_type', 'boundary_error']
        
        for error_cat in error_categories:
            values = [entity_errors[entity].get(error_cat, 0) for entity in entities]
            fig.add_trace(
                go.Bar(
                    name=error_cat,
                    x=entities,
                    y=values
                ),
                row=1, col=2
            )
        
        # Plot 3: Error Rate Over Time (Line Plot)
        error_examples = error_analysis['error_examples']
        error_times = defaultdict(list)
        for example in error_examples:
            error_times[example['error_type']].append(example['sentence_id'])
            
        for error_type, times in error_times.items():
            hist, bins = np.histogram(times, bins=20)
            fig.add_trace(
                go.Scatter(
                    x=bins[:-1],
                    y=hist,
                    name=error_type,
                    mode='lines'
                ),
                row=2, col=1
            )
        
        # Plot 4: Error Patterns (Heatmap)
        error_patterns = defaultdict(lambda: defaultdict(int))
        for example in error_examples:
            error_patterns[example['true_label']][example['pred_label']] += 1
            
        labels = sorted(set(
            [ex['true_label'] for ex in error_examples] +
            [ex['pred_label'] for ex in error_examples]
        ))
        pattern_matrix = [[error_patterns[true][pred] for pred in labels] for true in labels]
        
        fig.add_trace(
            go.Heatmap(
                z=pattern_matrix,
                x=labels,
                y=labels,
                colorscale='YlOrRd'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text="Comprehensive Error Analysis",
            showlegend=True
        )
        
        if output_path:
            fig.write_html(output_path)
            logger.info("Saved error analysis plot to %s", output_path)
            
        fig.show()

    def plot_entity_network(self,
                           sentences: List[List[str]],
                           entities: List[List[str]],
                           output_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 12)) -> None:
        """
        Create entity relationship network visualization.
        
        Args:
            sentences (List[List[str]]): Input sentences
            entities (List[List[str]]): Entity labels
            output_path (str, optional): Path to save plot
            figsize (Tuple[int, int]): Figure size
        """
        # Create graph
        G = nx.Graph()
        
        # Track entity co-occurrences
        entity_pairs = defaultdict(int)
        current_entities = set()
        
        for sent_tokens, sent_entities in zip(sentences, entities):
            # Extract entities in current sentence
            current_entities.clear()
            current_entity = []
            current_type = None
            
            for token, label in zip(sent_tokens, sent_entities):
                if label.startswith('B-'):
                    if current_entity:
                        entity_text = ' '.join(current_entity)
                        current_entities.add((entity_text, current_type))
                    current_entity = [token]
                    current_type = label.split('-')[1]
                elif label.startswith('I-'):
                    current_entity.append(token)
                elif current_entity:
                    entity_text = ' '.join(current_entity)
                    current_entities.add((entity_text, current_type))
                    current_entity = []
                    current_type = None
            
            if current_entity:
                entity_text = ' '.join(current_entity)
                current_entities.add((entity_text, current_type))
            
            # Add co-occurrence edges
            entities_list = list(current_entities)
            for i in range(len(entities_list)):
                for j in range(i + 1, len(entities_list)):
                    entity_pairs[(entities_list[i], entities_list[j])] += 1
        
        # Add nodes and edges to graph
        for (entity1, entity2), weight in entity_pairs.items():
            G.add_edge(
                f"{entity1[0]} ({entity1[1]})",
                f"{entity2[0]} ({entity2[1]})",
                weight=weight
            )
        
        plt.figure(figsize=figsize)
        
        # Calculate node positions using force-directed layout
        pos = nx.spring_layout(G, k=1/np.sqrt(len(G.nodes())), iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            node_color='lightblue',
            node_size=[G.degree(node) * 100 for node in G.nodes()],
            alpha=0.7
        )
        
        # Draw edges with varying thickness based on weight
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos,
            width=[w/max(edge_weights) * 2 for w in edge_weights],
            alpha=0.5
        )
        
        # Add labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=8,
            font_weight='bold'
        )
        
        plt.title("Entity Co-occurrence Network")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            logger.info("Saved entity network plot to %s", output_path)
        
        plt.close()

    def create_interactive_dashboard(self,
                                   metrics: Dict,
                                   error_analysis: Dict,
                                   output_path: str) -> None:
        """
        Create interactive HTML dashboard with all visualizations.
        
        Args:
            metrics (Dict): Performance metrics
            error_analysis (Dict): Error analysis results
            output_path (str): Path to save dashboard
        """
        # Create dashboard using plotly
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Entity Recognition Performance',
                'Confusion Matrix',
                'Error Analysis',
                'Entity Distribution',
                'Performance Trends',
                'Entity Network'
            ),
            specs=[
                [{"type": "bar"}, {"type": "heatmap"}],
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # Add performance metrics
        entities = list(metrics['entity'].keys())
        metrics_types = ['precision', 'recall', 'f1']
        
        for i, metric in enumerate(metrics_types):
            values = [metrics['entity'][entity][metric] for entity in entities]
            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=entities,
                    y=values,
                    text=values,
                    textposition='auto',
                ),
                row=1, col=1
            )
        
        # Add confusion matrix
        fig.add_trace(
            go.Heatmap(
                z=metrics['confusion_matrix'].values,
                x=metrics['confusion_matrix'].columns,
                y=metrics['confusion_matrix'].index,
                colorscale='YlOrRd'
            ),
            row=1, col=2
        )
        
        # Add error analysis pie chart
        error_types = error_analysis['error_types']
        fig.add_trace(
            go.Pie(
                labels=list(error_types.keys()),
                values=list(error_types.values()),
                name="Error Types"
            ),
            row=2, col=1
        )
        
        # Add entity distribution
        entity_counts = {
            entity: metrics['entity'][entity]['support']
            for entity in entities
        }
        fig.add_trace(
            go.Bar(
                x=list(entity_counts.keys()),
                y=list(entity_counts.values()),
                name="Entity Distribution"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1600,
            title_text="Named Entity Recognition Analysis Dashboard",
            showlegend=True
        )
        
        # Save dashboard
        fig.write_html(output_path)
        logger.info("Saved interactive dashboard to %s", output_path)
