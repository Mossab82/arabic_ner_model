import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from collections import defaultdict
import community
import numpy as np

class CharacterNetworkAnalyzer:
    def __init__(self):
        self.G = nx.Graph()
        self.centrality_scores = {}
        
    def build_network(self, text_segments: List[str], character_entities: List[List[str]]):
        """Build character interaction network from annotated text."""
        for segment, entities in zip(text_segments, character_entities):
            characters = [e for e in entities if e.startswith('PERSON') or e.startswith('MYTHICAL')]
            # Add edges between characters appearing in same segment
            for i in range(len(characters)):
                for j in range(i + 1, len(characters)):
                    if self.G.has_edge(characters[i], characters[j]):
                        self.G[characters[i]][characters[j]]['weight'] += 1
                    else:
                        self.G.add_edge(characters[i], characters[j], weight=1)
    
    def calculate_centrality_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality metrics for characters."""
        metrics = {
            'eigenvector': nx.eigenvector_centrality_numpy(self.G, weight='weight'),
            'betweenness': nx.betweenness_centrality(self.G, weight='weight'),
            'degree': nx.degree_centrality(self.G),
            'pagerank': nx.pagerank(self.G, weight='weight')
        }
        
        # Normalize scores
        for metric_name, scores in metrics.items():
            max_score = max(scores.values())
            metrics[metric_name] = {k: v/max_score for k, v in scores.items()}
            
        return metrics
    
    def detect_communities(self) -> Dict[str, int]:
        """Detect character communities using Louvain algorithm."""
        return community.best_partition(self.G, weight='weight')
    
    def visualize_network(self, output_path: str, min_edge_weight: int = 2):
        """Create visualization of character network."""
        plt.figure(figsize=(15, 15))
        
        # Filter edges by weight
        edges = [(u, v) for (u, v, d) in self.G.edges(data=True) if d['weight'] >= min_edge_weight]
        G_filtered = self.G.edge_subgraph(edges)
        
        # Calculate layout
        pos = nx.spring_layout(G_filtered, k=1/np.sqrt(G_filtered.number_of_nodes()))
        
        # Get communities
        communities = self.detect_communities()
        colors = [communities[node] for node in G_filtered.nodes()]
        
        # Draw network
        nx.draw_networkx_nodes(G_filtered, pos, node_color=colors, 
                             node_size=[G_filtered.degree(n) * 100 for n in G_filtered.nodes()],
                             alpha=0.7, cmap=plt.cm.rainbow)
        
        edge_weights = [G_filtered[u][v]['weight'] for u, v in G_filtered.edges()]
        nx.draw_networkx_edges(G_filtered, pos, alpha=0.5, 
                             width=[w/max(edge_weights) * 3 for w in edge_weights])
        
        plt.title("Character Interaction Network", size=16)
        plt.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
