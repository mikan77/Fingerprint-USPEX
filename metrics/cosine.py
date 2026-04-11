import numpy as np
from core.base import BaseMetric


class CosineMetric(BaseMetric):
    """
    Cosine distance between vectors.
    
    Formula: distance = 0.5 * (1 - cos(theta))
             where cos(theta) = (a dot b) / (||a|| * ||b||)
    
    Value range: [0, 1]
    - 0.0: vectors are identical or point in the same direction
    - 0.5: vectors are orthogonal
    - 1.0: vectors point in opposite directions
    
    Returns a distance value; lower means more similar.
    """
    
    def __init__(self, epsilon: float = 1e-10, **kwargs):
        """
        Args:
            epsilon: Threshold for division-by-zero protection
        """
        self.epsilon = epsilon
        self._name = "cosine"
    
    def calculate(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculate cosine distance between two vectors.
        
        Args:
            vec_a: First vector
            vec_b: Second vector
            
        Returns:
            float: Cosine distance in the [0, 1] range
        """
        vec_a = np.asarray(vec_a).flatten().astype(float)
        vec_b = np.asarray(vec_b).flatten().astype(float)
        
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a < self.epsilon and norm_b < self.epsilon:
            return 0.0
        if norm_a < self.epsilon or norm_b < self.epsilon:
            return 1.0
        
        cosine_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)  # Numerical stability.
        
        # Formula: 0.5 * (1 - cosine_similarity).
        distance = 0.5 * (1.0 - cosine_sim)
        
        return float(distance)
    
    def calculate_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate a pairwise cosine distance matrix.
        
        Args:
            vectors: Vector matrix [n_samples, n_features]
            
        Returns:
            np.ndarray: Symmetric distance matrix [n, n]
        """
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        n = vectors.shape[0]
        
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        zero_mask = norms[:, 0] < self.epsilon
        safe_norms = np.where(~zero_mask[:, None], norms, 1.0)
        vectors_norm = vectors / safe_norms
        
        # Cosine similarity matrix: dot product of normalized vectors.
        similarity_matrix = vectors_norm @ vectors_norm.T
        similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
        
        # Convert to distance: 0.5 * (1 - similarity).
        distance_matrix = 0.5 * (1.0 - similarity_matrix)
        zero_pair_mask = zero_mask[:, None] | zero_mask[None, :]
        both_zero_mask = zero_mask[:, None] & zero_mask[None, :]
        distance_matrix[zero_pair_mask] = 1.0
        distance_matrix[both_zero_mask] = 0.0
        np.fill_diagonal(distance_matrix, 0.0)
        
        return distance_matrix
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def is_similarity(self) -> bool:
        return False  # Distance metric, not a similarity metric.
