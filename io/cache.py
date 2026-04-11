# io/cache.py
import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional


class DiskCache:
    """Simple descriptor cache on disk."""
    
    def __init__(self, cache_dir: str = ".cache/comparators"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_key(self, data: Any) -> str:
        """Create a hash key for input data."""
        if isinstance(data, (str, Path)):
            try:
                content = str(data) + str(Path(data).stat().st_mtime)
            except (OSError, FileNotFoundError):
                content = str(data)
        else:
            content = str(data)
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, key: Any, default: Optional[Any] = None) -> Optional[Any]:
        """Get a value from the cache."""
        cache_key = self._get_key(key)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return default
        return default
    
    def set(self, key: Any, value: Any):
        """Save a value to the cache."""
        cache_key = self._get_key(key)
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_path, 'wb') as f:
            pickle.dump(value, f)
    
    def clear(self):
        """Clear the whole cache."""
        for file in self.cache_dir.glob("*.pkl"):
            file.unlink()
