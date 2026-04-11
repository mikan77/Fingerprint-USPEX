from typing import Iterable, Any

import numpy as np


def descriptor_to_numpy(result: Any) -> np.ndarray:
    """Converts dense or sparse descriptor output to a numpy array."""
    if hasattr(result, "toarray"):
        result = result.toarray()
    elif hasattr(result, "todense"):
        result = result.todense()
    return np.asarray(result, dtype=float)


def format_species(species: Iterable[str]) -> str:
    return "-".join(str(item) for item in species)
