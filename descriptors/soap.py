# descriptors/soap.py
from typing import List
import numpy as np
from ase import Atoms
from dscribe.descriptors import SOAP as DscribeSOAP
from core.base import BaseDescriptor
from descriptors._utils import descriptor_to_numpy, format_species


class SOAPDescriptor(BaseDescriptor):
    """
    SOAP (Smooth Overlap of Atomic Positions) descriptor via DScribe.
    
    Returns a structure-level vector by averaging or pooling atomic descriptors.
    """
    
    def __init__(
        self,
        species: List[str],
        r_cut: float = 12.0,
        n_max: int = 12,
        l_max: int = 12,
        sigma: float = 0.03,
        rbf: str = "gto",
        periodic: bool = True,
        sparse: bool = False,
        average: str = "inner",  # 'inner', 'outer', 'off'
        pooling: str = "mean_std",
        n_jobs: int = 1,
        **kwargs
    ):
        self.species = species
        self.r_cut = r_cut
        self.n_max = n_max
        self.l_max = l_max
        self.sigma = sigma
        self.rbf = rbf
        self.periodic = periodic
        self.sparse = sparse
        self.average = average
        self.pooling = pooling
        self.n_jobs = n_jobs
        if average == "off" and pooling not in {"mean", "mean_std"}:
            raise ValueError("SOAP average='off' supports pooling='mean' or 'mean_std'.")
        self._name = (
            f"SOAP_species-{format_species(species)}_r{r_cut}_n{n_max}_l{l_max}"
            f"_sigma{sigma}_rbf{rbf}_avg{average}_pool{pooling}_periodic{periodic}"
        )
        
        self._descriptor = DscribeSOAP(
            species=species,
            rbf=rbf,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            sigma=sigma,
            periodic=periodic,
            sparse=sparse,
            average=average,
            **kwargs
        )
    
    def create(self, atoms: Atoms) -> np.ndarray:
        """Create a descriptor for one structure."""
        result = self._descriptor.create(atoms)
        return self._to_fingerprint(result)
    
    def create_batch(self, atoms_list: List[Atoms]) -> np.ndarray:
        if self.average == "off":
            return np.vstack([self.create(atoms) for atoms in atoms_list])

        result = self._descriptor.create(
            atoms_list,     
            n_jobs=self.n_jobs,
            only_physical_cores=True
        )
        result = descriptor_to_numpy(result)
        return result.reshape(result.shape[0], -1)

    def _to_fingerprint(self, result) -> np.ndarray:
        result = descriptor_to_numpy(result)
        if self.average == "off" and result.ndim > 1:
            mean = result.mean(axis=0)
            if self.pooling == "mean":
                return mean
            return np.concatenate([mean, result.std(axis=0)])
        return result.reshape(-1)
    
    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> dict:
        return {
            "species": list(self.species),
            "r_cut": self.r_cut,
            "n_max": self.n_max,
            "l_max": self.l_max,
            "sigma": self.sigma,
            "rbf": self.rbf,
            "periodic": self.periodic,
            "sparse": self.sparse,
            "average": self.average,
            "pooling": self.pooling,
        }
