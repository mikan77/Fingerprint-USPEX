import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from backends.ovf_runner import OVFRunner


PYTHON_EXE = Path(sys.executable)


def make_runner(**kwargs) -> OVFRunner:
    return OVFRunner(cry_path=PYTHON_EXE, ovf_path=PYTHON_EXE, **kwargs)


def poscar_text(symbols="C O", counts="1 1") -> str:
    return f"""test POSCAR
1.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
{symbols}
{counts}
Direct
0.0 0.0 0.0
0.5 0.5 0.5
"""


class OVFRunnerArgsTests(unittest.TestCase):
    def test_default_binary_candidates_prefer_descriptors_movf(self):
        first_candidate = OVFRunner._default_binary_candidates("cry")[0]

        self.assertIn("descriptors", first_candidate.parts)
        self.assertIn("mOVF", first_candidate.parts)

    def test_build_default_ovf_args(self):
        runner = make_runner()

        args = runner._build_ovf_args(Path("rebuilt_POSCARS"))

        self.assertEqual(args[-4:], ["d", "c", "0", "0"])
        self.assertIn("b", args)
        self.assertIn("0.05", args)
        self.assertIn("s", args)
        self.assertIn("0.02", args)
        self.assertIn("m", args)
        self.assertIn("3", args)

    def test_build_angle_pair_ovf_args(self):
        runner = make_runner(
            fingerprint_type="a",
            metric="euclidean",
            pair=(1, 2),
            cutoff_multiplier=0.5,
            molecular_weight=0.0,
            auto_min_triangle_pair=("C", "N"),
            exclude_z=1,
        )

        args = runner._build_ovf_args(Path("rebuilt_POSCARS"))

        self.assertIn("u", args)
        self.assertIn("*0.5", args)
        self.assertIn("M", args)
        self.assertIn("0", args)
        self.assertIn("A", args)
        self.assertIn("C", args)
        self.assertIn("N", args)
        self.assertEqual(args[-6:], ["a", "x", "1", "e", "1", "2"])


class OVFRunnerParsingTests(unittest.TestCase):
    def test_parse_all_pairs_stdout(self):
        runner = make_runner()
        stdout = """Structure_1   Structure_2   Cosine_Distance
1   2   0.12345
1   3   0.23456
"""

        dataframe = runner._parse_ovf_stdout(stdout)

        self.assertEqual(list(dataframe.columns), ["structure_1", "structure_2", "cosine_distance"])
        self.assertEqual(len(dataframe), 2)
        self.assertEqual(dataframe.loc[0, "structure_1"], 1)
        self.assertEqual(dataframe.loc[0, "structure_2"], 2)
        self.assertAlmostEqual(dataframe.loc[0, "cosine_distance"], 0.12345)

    def test_parse_single_pair_with_angle_stdout(self):
        runner = make_runner(fingerprint_type="a", metric="euclidean")
        stdout = (
            "Comparison between structure 2 and 3    Euclidean Distance = 1.23456"
            "    Euclidean Distance Angle = 2.34567\n"
            "Structure 2: QuasiEntropy = 0.00000   S-Order = 0.00000\n"
        )

        dataframe = runner._parse_ovf_stdout(stdout)

        self.assertEqual(
            list(dataframe.columns),
            ["structure_1", "structure_2", "euclidean_distance", "euclidean_angle_distance"],
        )
        self.assertEqual(dataframe.loc[0, "structure_1"], 2)
        self.assertEqual(dataframe.loc[0, "structure_2"], 3)
        self.assertAlmostEqual(dataframe.loc[0, "euclidean_distance"], 1.23456)
        self.assertAlmostEqual(dataframe.loc[0, "euclidean_angle_distance"], 2.34567)

    def test_pairs_to_matrix(self):
        runner = make_runner()
        pairs = pd.DataFrame(
            {
                "structure_1": [1, 1, 2],
                "structure_2": [2, 3, 3],
                "cosine_distance": [0.1, 0.2, 0.3],
            }
        )

        matrix = runner._pairs_to_matrix(pairs, n_structures=3)

        np.testing.assert_allclose(
            matrix,
            np.array([
                [0.0, 0.1, 0.2],
                [0.1, 0.0, 0.3],
                [0.2, 0.3, 0.0],
            ]),
        )


class OVFRunnerPOSCARTests(unittest.TestCase):
    def test_gather_poscars_rejects_different_composition_or_order(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            first = tmp_path / "POSCAR_1"
            second = tmp_path / "POSCAR_2"
            output = tmp_path / "gathered_POSCARS"
            first.write_text(poscar_text("C O", "1 1"), encoding="utf-8")
            second.write_text(poscar_text("O C", "1 1"), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "identical composition and atom-type order"):
                OVFRunner._gather_poscars([first, second], output)

    def test_gather_poscars_does_not_insert_blank_separator(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            first = tmp_path / "POSCAR_1"
            second = tmp_path / "POSCAR_2"
            output = tmp_path / "gathered_POSCARS"
            first.write_text(poscar_text(), encoding="utf-8")
            second.write_text(poscar_text(), encoding="utf-8")

            OVFRunner._gather_poscars([first, second], output)

            gathered = output.read_text(encoding="utf-8")
            self.assertEqual(gathered.count("test POSCAR"), 2)
            self.assertNotIn("\n\n", gathered)


class OVFRunnerSubprocessTests(unittest.TestCase):
    def test_compare_uses_isolated_fake_cry_and_ovf(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            poscar_1 = tmp_path / "POSCAR_1"
            poscar_2 = tmp_path / "POSCAR_2"
            fake_cry = tmp_path / "fake_cry.sh"
            fake_ovf = tmp_path / "fake_ovf.sh"
            poscar_1.write_text(poscar_text(), encoding="utf-8")
            poscar_2.write_text(poscar_text(), encoding="utf-8")
            fake_cry.write_text(
                "#!/bin/sh\ncat \"$1\"\ntouch Data_for_ovf.bin\n",
                encoding="utf-8",
            )
            fake_ovf.write_text(
                "#!/bin/sh\nprintf 'Structure_1   Structure_2   Cosine_Distance\\n1   2   0.12500\\n'\n",
                encoding="utf-8",
            )
            os.chmod(fake_cry, 0o755)
            os.chmod(fake_ovf, 0o755)

            runner = OVFRunner(cry_path=fake_cry, ovf_path=fake_ovf)

            matrix = runner.compare([poscar_1, poscar_2])

            np.testing.assert_allclose(matrix, np.array([[0.0, 0.125], [0.125, 0.0]]))

    def test_compare_components_returns_rdf_and_adf_matrices(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            poscar_1 = tmp_path / "POSCAR_1"
            poscar_2 = tmp_path / "POSCAR_2"
            fake_cry = tmp_path / "fake_cry.sh"
            fake_ovf = tmp_path / "fake_ovf.sh"
            poscar_1.write_text(poscar_text(), encoding="utf-8")
            poscar_2.write_text(poscar_text(), encoding="utf-8")
            fake_cry.write_text(
                "#!/bin/sh\ncat \"$1\"\ntouch Data_for_ovf.bin\n",
                encoding="utf-8",
            )
            fake_ovf.write_text(
                "#!/bin/sh\n"
                "printf 'Structure_1   Structure_2   Cosine_Distance   Cosine_Distance_Angle\\n'\n"
                "printf '1   2   0.12500   0.25000\\n'\n",
                encoding="utf-8",
            )
            os.chmod(fake_cry, 0o755)
            os.chmod(fake_ovf, 0o755)

            runner = OVFRunner(
                cry_path=fake_cry,
                ovf_path=fake_ovf,
                fingerprint_type="a",
            )

            components = runner.compare_components([poscar_1, poscar_2])

            self.assertEqual(set(components), {"rdf", "adf"})
            np.testing.assert_allclose(components["rdf"], np.array([[0.0, 0.125], [0.125, 0.0]]))
            np.testing.assert_allclose(components["adf"], np.array([[0.0, 0.25], [0.25, 0.0]]))


if __name__ == "__main__":
    unittest.main()
