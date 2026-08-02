import networkx as nx
from loguru import logger

class BIMSpatialAdjacencyMatrix:
    def __init__(self):
        self.adjacency_graph = nx.Graph()
        logger.info("Spatial Adjacency Topological Graph Matrix initialized.")

    def register_proximity_edge(self, guid_A: str, guid_B: str, spatial_clearance: float):
        """Registers a structural adjacency link if elements share a physical boundary interface."""
        if spatial_clearance <= 0.05: # Interface boundary threshold (50mm)
            self.adjacency_graph.add_edge(guid_A, guid_B, distance=spatial_clearance, type="TOUCHING")
            
    def get_touching_neighbors(self, target_guid: str) -> list:
        if self.adjacency_graph.has_node(target_guid):
            return list(self.adjacency_graph.neighbors(target_guid))
        return []

if __name__ == '__main__':
    adj = BIMSpatialAdjacencyMatrix()
    adj.register_proximity_edge("WALL_01", "DOOR_02", 0.002)
    logger.success(f"Adjacency matrix engine verified. Mapped neighbors: {adj.get_touching_neighbors('WALL_01')}")
