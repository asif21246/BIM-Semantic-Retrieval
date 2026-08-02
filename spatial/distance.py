import numpy as np
from loguru import logger

class BIMGeometryEngine:
    def __init__(self, tolerance_mm: float = 5.0):
        self.tolerance = tolerance_mm / 1000.0  # Convert metrics to meters
        logger.info(f"Spatial Geometry Engine initialized with a clearance threshold of {tolerance_mm}mm.")

    def calculate_element_distance(self, bbox_coords_A: tuple, bbox_coords_B: tuple) -> float:
        """
        Computes the absolute Euclidean proximity distance between two 3D spatial bounding volumes.
        Format: (min_x, min_y, min_z, max_x, max_y, max_z)
        """
        minA, maxA = np.array(bbox_coords_A[:3]), np.array(bbox_coords_A[3:])
        minB, maxB = np.array(bbox_coords_B[:3]), np.array(bbox_coords_B[3:])
        
        # Calculate axis-aligned delta overlap constraints
        deltas = np.maximum(0, np.maximum(minA - maxB, minB - maxA))
        distance = float(np.linalg.norm(deltas))
        return distance

if __name__ == '__main__':
    engine = BIMGeometryEngine()
    # Test simulation run: Element A at origin, Element B shifted 20mm away
    dist = engine.calculate_element_distance((0,0,0, 1,1,1), (1.02,0,0, 2,1,1))
    logger.success(f"Spatial coordinate distance calculation active. Evaluated clearance delta: {dist:.4f}m")
