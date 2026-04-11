from collections import Counter
from typing import List

import numpy as np
from ase import Atoms

from core.base import BaseDescriptor
from descriptors._utils import format_species


class CellDescriptor(BaseDescriptor):
    """Simple global fingerprint based on cell parameters and composition."""

    def __init__(
        self,
        species: List[str],
        include_cell: bool = True,
        include_composition: bool = True,
        include_density: bool = True,
        **kwargs
    ):
        self.species = species
        self.include_cell = include_cell
        self.include_composition = include_composition
        self.include_density = include_density
        self._name = (
            f"Cell_species-{format_species(species)}_cell{include_cell}"
            f"_composition{include_composition}_density{include_density}"
        )

    def create(self, atoms: Atoms) -> np.ndarray:
        features = []

        if self.include_cell:
            lengths = atoms.cell.lengths()
            angles = atoms.cell.angles()
            volume = atoms.get_volume()
            volume_per_atom = volume / len(atoms) if len(atoms) else 0.0
            features.extend(lengths)
            features.extend(angles)
            features.append(volume_per_atom)

        if self.include_density:
            volume = atoms.get_volume()
            mass = float(np.sum(atoms.get_masses()))
            density = mass / volume if volume > 1e-12 else 0.0
            features.append(density)

        if self.include_composition:
            counts = Counter(atoms.get_chemical_symbols())
            total = len(atoms) or 1
            features.extend(counts.get(species, 0) / total for species in self.species)

        return np.asarray(features, dtype=float)

    def create_batch(self, atoms_list: List[Atoms]) -> np.ndarray:
        return np.vstack([self.create(atoms) for atoms in atoms_list])

    @property
    def name(self) -> str:
        return self._name

    @property
    def params(self) -> dict:
        return {
            "species": list(self.species),
            "include_cell": self.include_cell,
            "include_composition": self.include_composition,
            "include_density": self.include_density,
        }
