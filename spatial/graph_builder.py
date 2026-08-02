import os
import sys
import networkx as nx
from loguru import logger

# Inject current workspace directory into system path to resolve module imports
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from spatial.distance import BIMGeometryEngine
from spatial.adjacency import BIMSpatialAdjacencyMatrix
from spatial.storey_relation import BIMStoreySpatialFilter

class BIMTopologyGraphBuilder:
    def __init__(self):
        logger.info("Compiling Section 3.5 Spatial Context Extraction Topological Pipeline Core...")
        self.geom = BIMGeometryEngine()
        self.adj = BIMSpatialAdjacencyMatrix()
        self.storey = BIMStoreySpatialFilter()
        self.composite_spatial_graph = nx.Graph()

    def build_extract_spatial_context(self, element_pool_data: list):
        """Processes database arrays to extract geometric proximity constraints."""
        logger.info(f"Extracting spatial adjacency interfaces across {len(element_pool_data)} components...")
        
        for i in range(len(element_pool_data)):
            for j in range(i + 1, len(element_pool_data)):
                el_A = element_pool_data[i]
                el_B = element_pool_data[j]
                
                if self.storey.verify_storey_concurrency(el_A, el_B):
                    d = self.geom.calculate_element_distance(el_A["bbox"], el_B["bbox"])
                    if d <= 0.05:
                        self.adj.register_proximity_edge(el_A["guid"], el_B["guid"], d)
                        self.composite_spatial_graph.add_edge(el_A["guid"], el_B["guid"], clearance=d)
                        
        logger.success(f"Spatial Context Network compiled with {self.composite_spatial_graph.number_of_edges()} structural interfaces.")
        return self.composite_spatial_graph

if __name__ == '__main__':
    builder = BIMTopologyGraphBuilder()
    mock_data = [
        {"guid": "W1", "storey": "L1", "bbox": (0,0,0, 1,1,1)},
        {"guid": "D1", "storey": "L1", "bbox": (1.01,0,0, 2,1,1)},
        {"guid": "W2", "storey": "L2", "bbox": (0,0,5, 1,1,6)}
    ]
    builder.build_extract_spatial_context(mock_data)
