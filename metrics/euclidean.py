# metrics/euclidean.py
from typing import List
import numpy as np
from core.base import BaseMetric


class EuclideanMetric(BaseMetric):
    """
    Euclidean distance between vectors.
    
    Formula: d = ||a - b|| = sqrt(sum((a_i - b_i)^2))
    
    Returns a distance value; lower means more similar.
    """
    
    def __init__(self, normalize: bool = True, **kwargs):
        """
        Args:
            normalize: Whether to normalize vectors before distance calculation
        """
        self.normalize = normalize
        self._name = "euclidean"
    
    def calculate(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        vec_a = np.asarray(vec_a).flatten()
        vec_b = np.asarray(vec_b).flatten()
        
        if self.normalize:
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a > 1e-10 and norm_b > 1e-10:
                vec_a = vec_a / norm_a
                vec_b = vec_b / norm_b
        
        return float(np.linalg.norm(vec_a - vec_b))
    
    def calculate_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """
        Efficiently calculate a pairwise Euclidean distance matrix.
        vectors: [n_samples, n_features]
        """
        vectors = np.asarray(vectors)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        n = vectors.shape[0]
        
        if self.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms = np.where(norms > 1e-10, norms, 1.0)
            vectors = vectors / norms
        
        # Vectorized identity: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a dot b.
        sq_norms = np.sum(vectors ** 2, axis=1)
        dist_sq = sq_norms.reshape(-1, 1) + sq_norms.reshape(1, -1) - 2 * vectors @ vectors.T
        dist_sq = np.maximum(dist_sq, 0)  # Numerical stability.
        
        return np.sqrt(dist_sq)
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def is_similarity(self) -> bool:
        return False
