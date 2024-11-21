from typing import List, Dict, Tuple
import pandas as pd
import folium
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class SpatialTemporalAnalyzer:
    def __init__(self):
        self.location_data = defaultdict(lambda: {'count': 0, 'coordinates': None, 'type': None})
        self.temporal_data = defaultdict(lambda: defaultdict(int))
        
    def load_gazetteer(self, gazetteer_path: str):
        """Load location coordinates and types from gazetteer."""
        self.gazetteer = pd.read_csv(gazetteer_path)
        
    def analyze_locations(self, text_segments: List[str], location_entities: List[List[str]]):
        """Analyze location distribution and patterns."""
        for segment, entities in zip(text_segments, location_entities):
            for entity in entities:
                if entity.startswith('LOCATION'):
                    self.location_data[entity]['count'] += 1
                    if entity in self.gazetteer.index:
                        self.location_data[entity].update({
                            'coordinates': (
                                self.gazetteer.loc[entity, 'latitude'],
                                self.gazetteer.loc[entity, 'longitude']
                            ),
                            'type': self.gazetteer.loc[entity, 'type']
                        })
    
    def create_location_map(self, output_path: str):
        """Create interactive map of story locations."""
        m = folium.Map(location=[25, 45], zoom_start=5)  # Centered on Middle East
        
        # Add locations to map
        for loc, data in self.location_data.items():
            if data['coordinates']:
                folium.CircleMarker(
                    location=data['coordinates'],
                    radius=np.log(data['count'] + 1) * 5,
                    popup=f"{loc} (mentioned {data['count']} times)",
                    color='red',
                    fill=True
                ).add_to(m)
        
        m.save(output_path)
    
    def analyze_location_distribution(self) -> Dict[str, Dict[str, int]]:
        """Analyze distribution of location types across the narrative."""
        distribution = defaultdict(lambda: defaultdict(int))
        
        for loc, data in self.location_data.items():
            if data['type']:
                distribution['all'][data['type']] += data['count']
                
        return dict(distribution)
    
    def plot_location_distribution(self, output_path: str):
        """Create visualization of location type distribution."""
        distribution = self.analyze_location_distribution()
        
        plt.figure(figsize=(12, 6))
        df = pd.DataFrame.from_dict(distribution['all'], orient='index')
        df.plot(kind='bar')
        plt.title('Distribution of Location Types')
        plt.xlabel('Location Type')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
