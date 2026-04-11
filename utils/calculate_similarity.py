# utils/calculate_similarity.py
from typing import Optional, Union
import numpy as np
import pandas as pd


def convert_distance_to_similarity(
    matrix: np.ndarray,
    method: str = "inverse"
) -> np.ndarray:
    """
    Convert a distance matrix to a similarity matrix.
    
    Args:
        matrix: Distance matrix
        method: 'inverse' (1/(1+d)), 'gaussian' (exp(-d^2)), 'linear' (1-d)
    """
    if method == "inverse":
        return 1.0 / (1.0 + matrix)
    elif method == "gaussian":
        return np.exp(-matrix ** 2)
    elif method == "linear":
        return np.maximum(0, 1.0 - matrix)
    else:
        raise ValueError(f"Unknown method: {method}")


def filter_unique_pairs(
    matrix: Union[np.ndarray, pd.DataFrame],
    threshold: Optional[float] = None,
    return_indices: bool = True
):
    """
    Return unique pairs (i < j), optionally filtered by threshold.
    
    Returns:
        List[Tuple] or DataFrame with columns [id1, id2, value]
    """
    if isinstance(matrix, pd.DataFrame):
        values = matrix.values
        index = matrix.index.tolist()
        columns = matrix.columns.tolist()
    else:
        values = matrix
        index = list(range(len(matrix)))
        columns = list(range(len(matrix)))
    
    n = len(values)
    pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            val = values[i, j]
            if threshold is None or val >= threshold:
                if return_indices:
                    pairs.append((index[i], columns[j], val))
                else:
                    pairs.append((i + 1, j + 1, val))  # 1-based indexing
    
    if return_indices:
        return pd.DataFrame(pairs, columns=["id1", "id2", "value"])
    return pairs


def normalize_matrix(
    matrix: np.ndarray,
    method: str = "minmax",
    axis: Optional[int] = None
) -> np.ndarray:
    """Normalize a value matrix."""
    if method == "minmax":
        min_val = matrix.min() if axis is None else matrix.min(axis=axis, keepdims=True)
        max_val = matrix.max() if axis is None else matrix.max(axis=axis, keepdims=True)
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        return (matrix - min_val) / range_val
    elif method == "zscore":
        mean = matrix.mean() if axis is None else matrix.mean(axis=axis, keepdims=True)
        std = matrix.std() if axis is None else matrix.std(axis=axis, keepdims=True)
        std = np.where(std == 0, 1, std)
        return (matrix - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")
