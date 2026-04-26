# core/comparator.py
import hashlib
import json
from typing import List, Union, Optional, Dict
import numpy as np
import pandas as pd
from pathlib import Path

from ase import Atoms

from pymatgen.core import Structure

from core.base import BaseDescriptor, BaseMetric
from utils.write_file import FileWriter


class StructureComparator:
    """
    Generic structure comparator.
    
    Uses a strategy pattern: any descriptor plus any metric.
    Supports caching, batch processing, and parallel execution.
    """
    
    def __init__(
        self,
        descriptor: BaseDescriptor,
        metric: BaseMetric,
        remove_species: Optional[List[str]] = None,
        cache_enabled: bool = True,
        n_jobs: int = -1,
        descriptor_normalization: Optional[str] = None,
    ):
        """
        Args:
            descriptor: BaseDescriptor instance
            metric: BaseMetric instance
            remove_species: Elements to remove before descriptor calculation
            cache_enabled: Whether to cache descriptors
            n_jobs: Number of CPU cores for parallel execution (-1 = all cores)
            descriptor_normalization: Optional descriptor normalization before
                metric calculation. None disables normalization. Valid values:
                None/'none', 'l2', 'zscore', 'minmax'
        """
        self.descriptor = descriptor
        self.metric = metric
        self.remove_species = remove_species or []
        self.cache_enabled = cache_enabled
        self.n_jobs = n_jobs
        self.descriptor_normalization = descriptor_normalization
        self._cache: Dict[str, np.ndarray] = {}
        
        # Pass n_jobs to the descriptor if it exposes this attribute.
        if hasattr(self.descriptor, 'n_jobs'):
            self.descriptor.n_jobs = n_jobs
    
    def _load_structure(self, path: Union[str, Path]) -> Structure:
        """
        Load a structure from a file, e.g. CIF, POSCAR, XYZ.
        
        Args:
            path: Path to the structure file
            
        Returns:
            Structure: pymatgen Structure object
        """
        return Structure.from_file(str(path))
    
    def _to_ase(self, structure: Structure) -> Atoms:
        """
        Convert pymatgen Structure to ASE Atoms.
        
        Args:
            structure: pymatgen Structure object
            
        Returns:
            Atoms: ASE Atoms object
        """
        return structure.to_ase_atoms()
    
    def _remove_species_from_ase(self, atoms: Atoms, species: List[str]) -> Atoms:
        """
        Remove selected elements from ASE Atoms.
        
        Args:
            atoms: ASE Atoms object
            species: Elements to remove
            
        Returns:
            Atoms: Filtered ASE Atoms object
        """
        if not species:
            return atoms.copy()
        
        # Build a mask: True for atoms that should be kept.
        mask = [atom.symbol not in species for atom in atoms]
        
        # Filter atoms.
        filtered_atoms = atoms[mask]
        
        return filtered_atoms
    
    def _get_cache_key(self, path: str) -> str:
        """
        Generate a cache key from path and file metadata.
        
        Args:
            path: File path
            
        Returns:
            str: Unique cache key
        """
        file_path = Path(path)
        try:
            stat = file_path.stat()
            file_payload = {
                "path": str(file_path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        except OSError:
            file_payload = {"path": str(file_path)}

        payload = {
            "file": file_payload,
            "descriptor": self.descriptor.name,
            "descriptor_params": getattr(self.descriptor, "params", {}),
            "remove_species": sorted(self.remove_species),
        }
        payload_json = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    def _validate_descriptor_species(self, atoms: Atoms, path: str) -> None:
        """Validate that descriptor species are consistent with preprocessing."""
        descriptor_species = getattr(self.descriptor, "species", None)
        if not descriptor_species:
            return

        descriptor_species = {str(species) for species in descriptor_species}
        removed_species = set(self.remove_species)
        removed_in_descriptor = descriptor_species & removed_species
        if removed_in_descriptor:
            raise ValueError(
                f"Descriptor {self.descriptor.name} contains removed elements "
                f"{sorted(removed_in_descriptor)}. Remove them from species or do "
                f"not use remove_species for file {path}."
            )

        actual_species = set(atoms.get_chemical_symbols())
        missing_species = actual_species - descriptor_species
        if missing_species:
            raise ValueError(
                f"After preprocessing, file {path} contains elements "
                f"{sorted(missing_species)} that are not present in descriptor.species "
                f"{sorted(descriptor_species)}."
            )

    @staticmethod
    def _stack_descriptors(descriptors: List[np.ndarray]) -> np.ndarray:
        """Build a 2D matrix and fail clearly on incompatible fingerprint lengths."""
        vectors = [np.asarray(desc, dtype=float).reshape(-1) for desc in descriptors]
        try:
            return np.vstack(vectors)
        except ValueError as exc:
            lengths = [vec.size for vec in vectors]
            raise ValueError(
                "Descriptors have different lengths and cannot be compared as "
                f"one matrix. Lengths: {lengths}"
            ) from exc

    def _normalize_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        """Normalize fingerprint vectors before passing them to the metric."""
        method = self.descriptor_normalization
        if method is None:
            return descriptors

        method = method.lower()
        if method == "none":
            return descriptors
        descriptors = np.asarray(descriptors, dtype=float)

        if method == "l2":
            norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
            norms = np.where(norms > 1e-12, norms, 1.0)
            return descriptors / norms
        if method == "zscore":
            mean = descriptors.mean(axis=0, keepdims=True)
            std = descriptors.std(axis=0, keepdims=True)
            std = np.where(std > 1e-12, std, 1.0)
            return (descriptors - mean) / std
        if method == "minmax":
            min_values = descriptors.min(axis=0, keepdims=True)
            max_values = descriptors.max(axis=0, keepdims=True)
            value_range = max_values - min_values
            value_range = np.where(value_range > 1e-12, value_range, 1.0)
            return (descriptors - min_values) / value_range

        raise ValueError(
            f"Unsupported descriptor normalization: {self.descriptor_normalization}"
        )
    
    def _compute_descriptors(self, paths: List[Union[str, Path]]) -> np.ndarray:
        """
        Compute descriptors for all structures.
        
        Args:
            paths: Structure file paths
            
        Returns:
            np.ndarray: Descriptor matrix [n_structures, n_features]
        """
        descriptors: List[Optional[np.ndarray]] = []
        atoms_list: List[Atoms] = []
        cache_keys: List[str] = []
        
        # 1. Check cache and collect atoms that still need descriptor calculation.
        for path in paths:
            path_str = str(path)
            cache_key = self._get_cache_key(path_str)
            cache_keys.append(cache_key)
            
            if self.cache_enabled and cache_key in self._cache:
                descriptors.append(self._cache[cache_key])
            else:
                # Load and preprocess the structure.
                struct = self._load_structure(path_str)
                if self.remove_species:
                    updated_struct = struct.remove_species(self.remove_species)
                    if updated_struct is not None:
                        struct = updated_struct
                
                atoms = self._to_ase(struct)
                
                if len(atoms) == 0:
                    raise ValueError(
                        f"Structure {path_str} became empty after removing {self.remove_species}"
                    )
                self._validate_descriptor_species(atoms, path_str)
                
                atoms_list.append(atoms)
                descriptors.append(None)  # Placeholder for later filling.
        
        # 2. Compute missing descriptors.
        if atoms_list:
            # Use batch mode when available.
            if hasattr(self.descriptor, 'create_batch'):
                missing_indices = [i for i, d in enumerate(descriptors) if d is None]
                batch_result = self.descriptor.create_batch(atoms_list)
                
                for idx, atoms_idx in enumerate(missing_indices):
                    desc = batch_result[idx]
                    descriptors[atoms_idx] = desc
                    
                    if self.cache_enabled:
                        self._cache[cache_keys[atoms_idx]] = desc
            else:
                # Sequential calculation fallback.
                desc_idx = 0
                for i, desc in enumerate(descriptors):
                    if desc is None:
                        result = self.descriptor.create(atoms_list[desc_idx])
                        descriptors[i] = result
                        
                        if self.cache_enabled:
                            self._cache[cache_keys[i]] = result
                        desc_idx += 1
        
        descriptor_matrix = self._stack_descriptors(descriptors)
        return self._normalize_descriptors(descriptor_matrix)
    
    def compare(self, paths: List[Union[str, Path]]) -> np.ndarray:
        """
        Compare a list of structures.
        
        Args:
            paths: Structure file paths
            
        Returns:
            np.ndarray: [n, n] metric matrix
        """
        if not paths:
            raise ValueError("Path list cannot be empty.")
        
        descriptors = self._compute_descriptors(paths)
        return self.metric.calculate_matrix(descriptors)
    
    def compare_to_dataframe(
        self,
        paths: List[Union[str, Path]],
        index_start: int = 1
    ) -> pd.DataFrame:
        """
        Compare structures and return the result as pandas DataFrame.
        
        Args:
            paths: Structure file paths
            index_start: Starting index for rows/columns
            
        Returns:
            pd.DataFrame: Distance/similarity matrix
        """
        matrix = self.compare(paths)
        indices = list(range(index_start, index_start + len(paths)))
        return pd.DataFrame(matrix, index=indices, columns=indices)
    
    def compare_and_save(
        self,
        paths: List[Union[str, Path]],
        output_file: str,
        format: str = "txt",
        unique_pairs_only: bool = True,
        include_header: bool = False
    ) -> pd.DataFrame:
        """
        Compare structures and save the result to a file.
        
        Args:
            paths: Structure file paths
            output_file: Output file path
            format: 'txt' or 'csv'
            unique_pairs_only: Save only i < j pairs
            include_header: Whether to add a header with method names
            
        Returns:
            pd.DataFrame: Result matrix
        """
        df = self.compare_to_dataframe(paths)
        
        writer = FileWriter(output_file)
        
        if format == "csv":
            writer.write_csv(df)
        elif format == "txt":
            writer.write_txt(
                df,
                unique_pairs_only=unique_pairs_only,
                method_name=self.metric.name,
                include_header=include_header
            )
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Results saved to {output_file}")
        return df
    
    def clear_cache(self):
        """Clear the in-memory descriptor cache."""
        self._cache.clear()
    
    def get_cache_size(self) -> int:
        """
        Return cache size.
        
        Returns:
            int: Number of cached structures
        """
        return len(self._cache)
    
    @property
    def descriptor_name(self) -> str:
        """Descriptor name."""
        return self.descriptor.name
    
    @property
    def metric_name(self) -> str:
        """Metric name."""
        return self.metric.name
class EnsembleComparator:
    """
    Combines multiple descriptor-metric pipelines into a single weighted similarity/distance matrix.
    Supports optional RobustScaler normalization per pipeline before weighted aggregation.
    """

    def __init__(
        self,
        configs: List[Tuple[BaseDescriptor, BaseMetric, float]],
        use_robust_scaling: bool = False,
        cache_enabled: bool = True,
        n_jobs: int = -1,
        remove_species: Optional[List[str]] = None,
    ):
        """
        Args:
            configs: List of tuples (descriptor, metric, raw_weight).
            use_robust_scaling: Apply RobustScaler (median/IQR normalization) to each matrix before combining.
            cache_enabled: Enable in-memory descriptor caching.
            n_jobs: Number of CPU cores for parallel execution.
            remove_species: Elements to remove before descriptor calculation.
        """
        if not configs:
            raise ValueError("At least one (descriptor, metric, weight) configuration is required.")
            
        # Validate and normalize weights to sum to 1.0
        raw_weights = [cfg[2] for cfg in configs]
        total_weight = sum(raw_weights)
        if total_weight == 0:
            raise ValueError("Weights cannot sum to zero.")
        self.weights = [w / total_weight for w in raw_weights]
        
        self.use_robust_scaling = use_robust_scaling
        self.comparators = [
            StructureComparator(
                descriptor=cfg[0],
                metric=cfg[1],
                cache_enabled=cache_enabled,
                n_jobs=n_jobs,
                remove_species=remove_species,
                descriptor_normalization=None,  # Handled internally if needed
            )
            for cfg in configs
        ]
        self._name = "Ensemble_" + "_".join(cfg[1].name for cfg in configs)

    def compare(self, paths: List[Union[str, Path]]) -> np.ndarray:
        """
        Compute individual metric matrices, optionally scale them, and return the weighted combination.
        
        Args:
            paths: Structure file paths.
            
        Returns:
            np.ndarray: Combined symmetric matrix [n, n].
        """
        if not paths:
            raise ValueError("Path list cannot be empty.")
            
        matrices = []
        for comp in self.comparators:
            mat = comp.compare(paths)
            
            if self.use_robust_scaling:
                # RobustScaler operates on 2D arrays. Flatten, scale, reshape to preserve symmetry.
                scaler = RobustScaler()
                flat_scaled = scaler.fit_transform(mat.flatten().reshape(-1, 1)).flatten()
                mat = flat_scaled.reshape(mat.shape)
                
            matrices.append(mat)
            
        # Weighted linear combination
        combined = np.zeros_like(matrices[0])
        for w, m in zip(self.weights, matrices):
            combined += w * m
            
        return combined

    def compare_to_dataframe(
        self,
        paths: List[Union[str, Path]],
        index_start: int = 1
    ) -> pd.DataFrame:
        """Compare structures and return the combined result as a pandas DataFrame."""
        matrix = self.compare(paths)
        indices = list(range(index_start, index_start + len(paths)))
        return pd.DataFrame(matrix, index=indices, columns=indices)

    def compare_and_save(
        self,
        paths: List[Union[str, Path]],
        output_file: str,
        format: str = "txt",
        unique_pairs_only: bool = True,
        include_header: bool = False
    ) -> pd.DataFrame:
        """Compare structures and save the combined result to a file."""
        df = self.compare_to_dataframe(paths)
        writer = FileWriter(output_file)
        
        if format == "csv":
            writer.write_csv(df)
        elif format == "txt":
            writer.write_txt(
                df,
                unique_pairs_only=unique_pairs_only,
                method_name=self._name,
                include_header=include_header
            )
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        print(f"Combined results saved to {output_file}")
        return df
