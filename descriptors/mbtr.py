# descriptors/mbtr.py
from typing import List
import numpy as np
from ase import Atoms
from dscribe.descriptors import MBTR as DscribeMBTR
from core.base import BaseDescriptor
from descriptors._utils import descriptor_to_numpy, format_species


class MBTRDescriptor(BaseDescriptor):
    """
    MBTR (Many-Body Tensor Representation) descriptor via DScribe.
    
    Supports k-body terms: k=1 atom types, k=2 distances, k=3 angles.
    """
    
    def __init__(
        self,
        species: List[str],
        geometry: dict,
        grid: dict,
        weighting: dict,
        periodic: bool = True,
        sparse: bool = False,
        flatten: bool = True,
        n_jobs: int = 1,
        **kwargs
    ):
        self.species = species
        self.geometry = geometry
        self.grid = grid
        self.weighting = weighting
        self.periodic = periodic
        self.sparse = sparse
        self.flatten = flatten
        self.n_jobs = n_jobs
        self._name = (
            f"MBTR_species-{format_species(species)}_geom-{geometry.get('function', 'unknown')}"
            f"_grid{grid.get('n', 'unknown')}_periodic{periodic}_sparse{sparse}"
        )
        
        self._descriptor = DscribeMBTR(
            species=species,
            geometry=geometry,
            grid=grid,
            weighting=weighting,
            periodic=periodic,
            sparse=sparse,
            **kwargs
        )
    
    def create(self, atoms: Atoms) -> np.ndarray:
        result = self._descriptor.create(atoms)
        result = descriptor_to_numpy(result)
        return result.flatten() if self.flatten and result.ndim > 1 else result
    
    def create_batch(self, atoms_list: List[Atoms]) -> np.ndarray:
        result = self._descriptor.create(
            atoms_list,     
            n_jobs=self.n_jobs,
            only_physical_cores=True
        )
        result = descriptor_to_numpy(result)
        return result.reshape(result.shape[0], -1)
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> dict:
        return {
            "species": list(self.species),
            "geometry": self.geometry,
            "grid": self.grid,
            "weighting": self.weighting,
            "periodic": self.periodic,
            "sparse": self.sparse,
            "flatten": self.flatten,
        }
