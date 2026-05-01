import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms

from core.comparator import StructureComparator, combine_distance_matrices
from descriptors.cell import CellDescriptor
from descriptors.soap import SOAPDescriptor
from metrics.cosine import CosineMetric


class CosineMetricTests(unittest.TestCase):
    def test_zero_vectors_are_consistent(self):
        metric = CosineMetric()
        vectors = np.array([[0.0, 0.0], [1.0, 0.0]])

        self.assertEqual(metric.calculate(vectors[0], vectors[0]), 0.0)
        self.assertEqual(metric.calculate(vectors[0], vectors[1]), 1.0)

        matrix = metric.calculate_matrix(vectors)
        self.assertEqual(matrix[0, 0], 0.0)
        self.assertEqual(matrix[1, 1], 0.0)
        self.assertEqual(matrix[0, 1], 1.0)


class ComparatorTests(unittest.TestCase):
    def test_stack_descriptors_rejects_different_lengths(self):
        with self.assertRaisesRegex(ValueError, "different lengths"):
            StructureComparator._stack_descriptors([
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0, 3.0]),
            ])

    def test_descriptor_normalization_l2_zscore_minmax(self):
        comparator = StructureComparator(
            descriptor=CellDescriptor(["C"]),
            metric=CosineMetric(),
            descriptor_normalization="l2",
        )

        l2 = comparator._normalize_descriptors(np.array([[3.0, 4.0], [0.0, 0.0]]))
        np.testing.assert_allclose(l2, np.array([[0.6, 0.8], [0.0, 0.0]]))

        comparator.descriptor_normalization = "zscore"
        zscore = comparator._normalize_descriptors(np.array([[1.0, 2.0], [3.0, 2.0]]))
        np.testing.assert_allclose(zscore, np.array([[-1.0, 0.0], [1.0, 0.0]]))

        comparator.descriptor_normalization = "minmax"
        minmax = comparator._normalize_descriptors(np.array([[1.0, 2.0], [3.0, 2.0]]))
        np.testing.assert_allclose(minmax, np.array([[0.0, 0.0], [1.0, 0.0]]))

    def test_cache_key_depends_on_descriptor_params_and_removed_species(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "structure.cif"
            path.write_text("data_test\n", encoding="utf-8")

            comp_a = StructureComparator(CellDescriptor(["C"]), CosineMetric())
            comp_b = StructureComparator(CellDescriptor(["C", "O"]), CosineMetric())
            comp_c = StructureComparator(
                CellDescriptor(["C"]),
                CosineMetric(),
                remove_species=["H"],
            )

            key_a = comp_a._get_cache_key(str(path))

            self.assertNotEqual(key_a, comp_b._get_cache_key(str(path)))
            self.assertNotEqual(key_a, comp_c._get_cache_key(str(path)))

    def test_remove_species_and_descriptor_species_are_not_silent(self):
        atoms = Atoms("C", positions=[[0.0, 0.0, 0.0]], cell=[4.0, 4.0, 4.0], pbc=True)

        removed_species_comparator = StructureComparator(
            CellDescriptor(["C", "H"]),
            CosineMetric(),
            remove_species=["H"],
        )
        with self.assertRaisesRegex(ValueError, "contains removed elements"):
            removed_species_comparator._validate_descriptor_species(atoms, "structure.cif")

        missing_species_comparator = StructureComparator(
            CellDescriptor(["O"]),
            CosineMetric(),
        )
        with self.assertRaisesRegex(ValueError, "not present in descriptor.species"):
            missing_species_comparator._validate_descriptor_species(atoms, "structure.cif")

    def test_combine_distance_matrices_with_robust_sigmoid_scaling(self):
        rdf = np.array([
            [0.0, 0.10, 0.40],
            [0.10, 0.0, 0.25],
            [0.40, 0.25, 0.0],
        ])
        adf = np.array([
            [0.0, 0.30, 0.90],
            [0.30, 0.0, 0.60],
            [0.90, 0.60, 0.0],
        ])

        combined = combine_distance_matrices(
            [rdf, adf],
            weights=[2.0, 1.0],
            use_robust_scaling=True,
        )

        self.assertEqual(combined.shape, rdf.shape)
        np.testing.assert_allclose(combined, combined.T)
        np.testing.assert_allclose(np.diag(combined), np.zeros(3))
        np.testing.assert_allclose(combined[1, 2], 0.5)
        self.assertTrue(np.all(combined >= 0.0))
        self.assertTrue(np.all(combined <= 1.0))


class DescriptorTests(unittest.TestCase):
    def test_cell_descriptor_contains_cell_density_and_composition(self):
        atoms = Atoms(
            "CO",
            positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            cell=[2.0, 3.0, 4.0],
            pbc=True,
        )
        descriptor = CellDescriptor(["C", "O"])
        fingerprint = descriptor.create(atoms)

        self.assertEqual(fingerprint.shape, (10,))
        np.testing.assert_allclose(fingerprint[:3], [2.0, 3.0, 4.0])
        np.testing.assert_allclose(fingerprint[3:6], [90.0, 90.0, 90.0])
        np.testing.assert_allclose(fingerprint[6], 12.0)
        np.testing.assert_allclose(fingerprint[-2:], [0.5, 0.5])

    def test_soap_average_off_pooling_is_order_and_translation_invariant(self):
        atoms = Atoms(
            "COH",
            positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.0, 0.0]],
            cell=[5.0, 5.0, 5.0],
            pbc=False,
        )
        descriptor = SOAPDescriptor(
            species=["C", "O", "H"],
            r_cut=3.0,
            n_max=2,
            l_max=1,
            sigma=0.2,
            periodic=False,
            average="off",
            pooling="mean_std",
        )

        reordered = atoms[[2, 0, 1]]
        shifted = atoms.copy()
        shifted.positions += [0.4, 0.2, 0.1]

        np.testing.assert_allclose(descriptor.create(atoms), descriptor.create(reordered))
        np.testing.assert_allclose(descriptor.create(atoms), descriptor.create(shifted))
        self.assertIn("sigma0.2", descriptor.name)
        self.assertIn("avgoff", descriptor.name)

    def test_soap_sparse_output_is_converted_to_numpy(self):
        atoms = Atoms(
            "CO",
            positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
            cell=[5.0, 5.0, 5.0],
            pbc=False,
        )
        descriptor = SOAPDescriptor(
            species=["C", "O"],
            r_cut=3.0,
            n_max=2,
            l_max=1,
            sigma=0.2,
            periodic=False,
            average="inner",
            sparse=True,
        )

        fingerprint = descriptor.create(atoms)
        self.assertIsInstance(fingerprint, np.ndarray)
        self.assertEqual(fingerprint.ndim, 1)


if __name__ == "__main__":
    unittest.main()
