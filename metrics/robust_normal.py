# metrics/robust_normal.py
import numpy as np
from sklearn.preprocessing import RobustScaler
from core.base import BaseMetric


class RobustNormalMetric(BaseMetric):
    """
    Metric that applies RobustScaler normalization followed by a sigmoid transformation
    to the values of a base metric (e.g., cosine distance).

    Normalization formula:
        X_scaled = (X - median) / (Q3 - Q1)
    Values are then mapped to the [0, 1] range using the standard logistic sigmoid:
        X_rescaled = 1 / (1 + exp(-X_scaled))
    """

    def __init__(self, base_metric: BaseMetric, **kwargs):
        """
        Initialize the robust normalization wrapper.
        
        Args:
            base_metric: Instance of a BaseMetric to wrap.
            **kwargs: Additional arguments passed to BaseMetric.
        """
        super().__init__(**kwargs)
        self.base_metric = base_metric
        self._name = f"RobustNormal_{base_metric.name}"

    def calculate(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculate the metric for a single pair of vectors.
        
        Note: RobustScaler requires a dataset to compute median and IQR.
        For a single pair, statistical normalization is not mathematically defined.
        This method returns the raw base metric value. Use `calculate_matrix` for proper normalization.
        """
        return self.base_metric.calculate(vec_a, vec_b)

    def calculate_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate the pairwise metric matrix, apply robust normalization, 
        and rescale values to [0, 1] via sigmoid.
        
        Args:
            vectors: Descriptor matrix of shape [n_structures, n_features]
            
        Returns:
            np.ndarray: Normalized symmetric matrix [n, n] in range ~[0, 1]
        """
        # 1. Compute the base distance/similarity matrix
        base_matrix = self.base_metric.calculate_matrix(vectors)

        # 2. Reshape to [n_samples, 1] format required by sklearn transformers
        original_shape = base_matrix.shape
        flat_values = base_matrix.flatten().reshape(-1, 1)

        # 3. Normalize: subtract median and scale by IQR (Q3 - Q1)
        scaler = RobustScaler()
        scaled_values = scaler.fit_transform(flat_values).flatten()

        # 4. Apply standard sigmoid transformation to map to [0, 1]
        # Formula: 1 / (1 + exp(-x))
        rescaled_values = 1.0 / (1.0 + np.exp(-scaled_values))

        # 5. Reshape back to the original matrix dimensions
        return rescaled_values.reshape(original_shape)

    @property
    def name(self) -> str:
        """Return the descriptive name of this metric."""
        return self._name

    @property
    def is_similarity(self) -> bool:
        """Return True since values are mapped to [0, 1] and treated as similarity scores."""
        return True