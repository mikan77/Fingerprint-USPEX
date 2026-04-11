# descriptors/ovf_by_dscribe.py
from typing import List
import numpy as np
from ase import Atoms
from dscribe.descriptors import ValleOganov as DescribeValleOganov
from core.base import BaseDescriptor
from descriptors._utils import descriptor_to_numpy, format_species


class OVFDescriptors(BaseDescriptor):
    """
    OVF (Oganov-Valle Fingerprint) descriptor via DScribe.
    
    Computes a fingerprint from ordered distance/angle functions.
    """
    
    def __init__(
        self,
        species: List[str],
        function: str,
        n: int,
        sigma: float,
        r_cut: float,
        sparse: bool = False,
        flatten: bool = True,
        n_jobs: int = 1,
        **kwargs
    ):
        """
        Args:
            species: Chemical element list
            function: Function type ("distance", "angle", "torsion")
            n: Number of bins/components
            sigma: Gaussian width
            r_cut: Cutoff radius in Angstrom
            sparse: Whether to use sparse output
            flatten: Whether to flatten the result to a 1D vector
        """
        self.species = species
        self.function = function
        self.n = n
        self.sigma = sigma
        self.r_cut = r_cut
        self.sparse = sparse
        self.flatten = flatten
        self._name = (
            f"OVF_species-{format_species(species)}_function-{function}"
            f"_n{n}_sigma{sigma}_R{r_cut}_sparse{sparse}"
        )
        self.n_jobs = n_jobs
        
        self._descriptor = DescribeValleOganov(
            species=species,
            function=function,
            n=n,
            sparse=sparse,
            sigma=sigma,
            r_cut=r_cut,
            **kwargs
        )
    
    def create(self, atoms: Atoms) -> np.ndarray:
        """Create a descriptor for one structure."""
        result = self._descriptor.create(atoms)
        result = descriptor_to_numpy(result)
        return result.flatten() if self.flatten and result.ndim > 1 else result
    
    def create_batch(self, atoms_list: List[Atoms]) -> np.ndarray:
        """
        Compute descriptors for all structures in one call.
        DScribe optimizes internal calculations for batch processing.
        """
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
            "function": self.function,
            "n": self.n,
            "sigma": self.sigma,
            "r_cut": self.r_cut,
            "sparse": self.sparse,
            "flatten": self.flatten,
        }
