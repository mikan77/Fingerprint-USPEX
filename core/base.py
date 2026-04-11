# core/base.py
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from ase import Atoms


class BaseDescriptor(ABC):
    """Abstract base class for structure descriptors."""
    
    @abstractmethod
    def __init__(self, species: List[str], **kwargs):
        """Initialize the descriptor."""
        pass
    
    @abstractmethod
    def create(self, atoms: Atoms) -> np.ndarray:
        """
        Generate a descriptor for one structure.
        
        Args:
            atoms: ASE Atoms object
            
        Returns:
            np.ndarray: Descriptor vector, 1D or 2D
        """
        pass
    
    @abstractmethod
    def create_batch(self, atoms_list: List[Atoms]) -> np.ndarray:
        """
        Generate descriptors for a list of structures.
        
        Returns:
            np.ndarray: Descriptor matrix [n_structures, n_features]
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Descriptor name for logging."""
        pass

    @property
    def params(self) -> dict:
        """Descriptor parameters for caching and logging."""
        return {}


class BaseMetric(ABC):
    """Abstract base class for distance/similarity metrics."""
    
    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the metric."""
        pass
    
    @abstractmethod
    def calculate(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculate distance/similarity between two vectors.
        
        Args:
            vec_a: First vector
            vec_b: Second vector
            
        Returns:
            float: Metric value
        """
        pass
    
    @abstractmethod
    def calculate_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate a pairwise metric matrix.
        
        Args:
            vectors: Vector matrix [n, features]
            
        Returns:
            np.ndarray: Symmetric matrix [n, n]
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name for logging."""
        pass
    
    @property
    @abstractmethod
    def is_similarity(self) -> bool:
        """True if the metric returns similarity, false if it returns distance."""
        pass
