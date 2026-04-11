# utils/write_file.py
from typing import Optional
import pandas as pd
from pathlib import Path


class FileWriter:
    """Generic writer for comparison results."""
    
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def write_csv(self, df: pd.DataFrame, **kwargs):
        """Save a DataFrame to CSV."""
        df.to_csv(self.filepath, index=True, **kwargs)
    
    def write_txt(
        self,
        df: pd.DataFrame,
        unique_pairs_only: bool = True,
        method_name: Optional[str] = None,
        include_header: bool = False,
        precision: int = 6
    ):
        """
        Save pairwise comparisons to TXT.
        
        Format: {id1} {id2} {value}
        """
        lines = []
        
        if include_header and method_name:
            lines.append(f"# Method: {method_name}\n")
            lines.append(f"# Format: id1 id2 value\n")
        
        n = len(df)
        for i in range(n):
            start_j = i + 1 if unique_pairs_only else 0
            for j in range(start_j, n):
                if i == j and unique_pairs_only:
                    continue
                id1 = df.index[i]
                id2 = df.columns[j]
                value = df.iloc[i, j]
                
                # Keep method_name only in the optional header.
                line = f"{id1} {id2} {value:.{precision}f}\n"
                lines.append(line)
        
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
    
    def write_combined(
        self,
        results: dict,
        unique_pairs_only: bool = True
    ):
        """
        Write multiple methods to one file with additional columns.
        
        Args:
            results: Dict[str, pd.DataFrame] {method_name: dataframe}
        """
        if not results:
            return
        
        # Use indices from the first DataFrame.
        first_df = next(iter(results.values()))
        n = len(first_df)
        
        lines = []
        # Header with method names.
        header = "id1 id2 " + " ".join(results.keys()) + "\n"
        lines.append(header)
        
        for i in range(n):
            start_j = i + 1 if unique_pairs_only else 0
            for j in range(start_j, n):
                if i == j and unique_pairs_only:
                    continue
                id1 = first_df.index[i]
                id2 = first_df.columns[j]
                
                values = []
                for method_name, df in results.items():
                    val = df.iloc[i, j]
                    values.append(f"{val:.6f}")
                
                line = f"{id1} {id2} " + " ".join(values) + "\n"
                lines.append(line)
        
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
