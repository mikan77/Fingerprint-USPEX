from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


class OVFExecutionError(RuntimeError):
    """Error raised when an external OVF command fails or cannot be parsed."""


class OVFRunner:
    """
    Wrapper around the cry/ovf executables.

    This runner is not a regular BaseDescriptor: the C ovf program computes the
    fingerprint and directly returns distances between structures.
    """

    _FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    _ALL_PAIR_RE = re.compile(
        rf"^\s*(?P<i>\d+)\s+(?P<j>\d+)\s+(?P<distance>{_FLOAT_RE})"
        rf"(?:\s+(?P<angle>{_FLOAT_RE}))?\s*$"
    )
    _SINGLE_PAIR_RE = re.compile(
        rf"Comparison between structure\s+(?P<i>\d+)\s+and\s+(?P<j>\d+)"
        rf".*?(?P<metric>Cosine|Euclidean)\s+Distance\s*=\s*"
        rf"(?P<distance>{_FLOAT_RE})"
        rf"(?:\s+(?P=metric)\s+Distance\s+Angle\s*=\s*"
        rf"(?P<angle>{_FLOAT_RE}))?",
        re.IGNORECASE,
    )

    def __init__(
        self,
        cry_path: Optional[PathLike] = None,
        ovf_path: Optional[PathLike] = None,
        fingerprint_type: str = "d",
        metric: str = "cosine",
        pair: Optional[Tuple[int, int]] = None,
        cutoff_radius: Optional[float] = None,
        cutoff_multiplier: Optional[float] = None,
        rdf_bin_size: Optional[float] = 0.05,
        adf_bin_size: Optional[float] = None,
        rdf_sigma: Optional[float] = 0.02,
        adf_sigma: Optional[float] = None,
        max_triangle_side: Optional[float] = 3.0,
        molecular_weight: Optional[float] = None,
        auto_min_triangle_pair: Optional[Tuple[Union[str, int], Union[str, int]]] = None,
        exclude_z: Optional[int] = None,
        work_dir: Optional[PathLike] = None,
        keep_work_dir: bool = False,
        timeout: Optional[float] = None,
        omp_num_threads: Optional[int] = None,
        distance_component: str = "rdf",
        allow_experimental_torsion: bool = False,
    ):
        self.cry_path = self._resolve_binary(cry_path, "cry")
        self.ovf_path = self._resolve_binary(ovf_path, "ovf")

        self.fingerprint_type = fingerprint_type.lower()
        if self.fingerprint_type not in {"d", "a", "t"}:
            raise ValueError("fingerprint_type must be 'd', 'a', or 't'.")
        if self.fingerprint_type == "t" and not allow_experimental_torsion:
            raise ValueError(
                "fingerprint_type='t' is marked as NOT YET in ovf.c. "
                "Pass allow_experimental_torsion=True if you want to run it explicitly."
            )

        self.metric = metric.lower()
        if self.metric not in {"cosine", "euclidean"}:
            raise ValueError("metric must be 'cosine' or 'euclidean'.")

        self.pair = pair
        if pair is not None:
            if len(pair) != 2 or pair[0] < 1 or pair[1] < 1:
                raise ValueError("pair must be a tuple of two 1-based POSCAR indices.")

        if cutoff_radius is not None and cutoff_multiplier is not None:
            raise ValueError("cutoff_radius and cutoff_multiplier cannot be set together.")
        self.cutoff_radius = cutoff_radius
        self.cutoff_multiplier = cutoff_multiplier

        self.rdf_bin_size = rdf_bin_size
        self.adf_bin_size = adf_bin_size
        self.rdf_sigma = rdf_sigma
        self.adf_sigma = adf_sigma
        self.max_triangle_side = max_triangle_side
        self.molecular_weight = molecular_weight
        self.auto_min_triangle_pair = auto_min_triangle_pair
        if auto_min_triangle_pair is not None:
            if len(auto_min_triangle_pair) != 2:
                raise ValueError("auto_min_triangle_pair must contain exactly two items.")
            if molecular_weight is None:
                raise ValueError("auto_min_triangle_pair requires molecular_weight, i.e. M mode.")

        self.exclude_z = exclude_z
        if exclude_z is not None and exclude_z < 1:
            raise ValueError("exclude_z must be an atomic number >= 1.")

        self.work_dir = Path(work_dir).expanduser() if work_dir is not None else None
        self.keep_work_dir = keep_work_dir
        self.timeout = timeout
        self.omp_num_threads = omp_num_threads
        if omp_num_threads is not None and omp_num_threads < 1:
            raise ValueError("omp_num_threads must be >= 1.")

        self.distance_component = distance_component.lower()
        if self.distance_component not in {"rdf", "angle"}:
            raise ValueError("distance_component must be 'rdf' or 'angle'.")
        if self.distance_component == "angle" and self.fingerprint_type != "a":
            raise ValueError("distance_component='angle' requires fingerprint_type='a'.")

        self.last_stdout = ""
        self.last_stderr = ""
        self.last_work_dir: Optional[Path] = None
        self.last_rebuilt_poscars: Optional[Path] = None

    def compare_pairs(self, poscar_paths: Sequence[PathLike]) -> pd.DataFrame:
        """
        Return an ovf distance table.

        pair=None returns all pairs. pair=(i, j) returns only the selected pair.
        """
        poscar_paths = self._validate_poscar_paths(poscar_paths)
        if self.pair is not None and max(self.pair) > len(poscar_paths):
            raise ValueError(
                f"pair={self.pair} is outside the list of {len(poscar_paths)} POSCAR files."
            )

        run_dir = Path(tempfile.mkdtemp(prefix="ovf_run_", dir=self.work_dir))
        self.last_work_dir = run_dir

        try:
            gathered_poscars = run_dir / "gathered_POSCARS"
            rebuilt_poscars = run_dir / "rebuilt_POSCARS"
            self._gather_poscars(poscar_paths, gathered_poscars)

            cry_result = self._run_command(
                self._build_cry_args(gathered_poscars),
                cwd=run_dir,
                stdout_path=rebuilt_poscars,
            )
            if not rebuilt_poscars.exists() or rebuilt_poscars.stat().st_size == 0:
                raise OVFExecutionError(
                    "cry completed successfully but did not create a non-empty rebuilt_POSCARS."
                )
            self.last_stderr = cry_result.stderr or ""
            self.last_rebuilt_poscars = rebuilt_poscars

            data_for_ovf = run_dir / "Data_for_ovf.bin"
            if self.molecular_weight is not None and not data_for_ovf.exists():
                raise OVFExecutionError(
                    "M mode requires Data_for_ovf.bin, but cry did not create it."
                )

            ovf_result = self._run_command(
                self._build_ovf_args(rebuilt_poscars),
                cwd=run_dir,
            )
            self.last_stdout = ovf_result.stdout or ""
            self.last_stderr = ovf_result.stderr or ""
            return self._parse_ovf_stdout(self.last_stdout)
        finally:
            if not self.keep_work_dir:
                shutil.rmtree(run_dir, ignore_errors=True)

    def compare(self, poscar_paths: Sequence[PathLike]) -> np.ndarray:
        """Return a symmetric distance matrix with shape [n_structures, n_structures]."""
        poscar_paths = self._validate_poscar_paths(poscar_paths)
        pairs = self.compare_pairs(poscar_paths)
        return self._pairs_to_matrix(pairs, n_structures=len(poscar_paths))

    def compare_to_dataframe(
        self,
        poscar_paths: Sequence[PathLike],
        index_start: int = 1,
    ) -> pd.DataFrame:
        """Return the distance matrix as a pandas.DataFrame."""
        matrix = self.compare(poscar_paths)
        labels = list(range(index_start, index_start + len(matrix)))
        return pd.DataFrame(matrix, index=labels, columns=labels)

    def compare_components(self, poscar_paths: Sequence[PathLike]) -> Dict[str, np.ndarray]:
        """
        Return RDF and ADF distance matrices for fingerprint_type='a'.

        compare() returns only one component selected via distance_component.
        Use this method when downstream analysis needs both fingerprint parts.
        """
        if self.fingerprint_type != "a":
            raise ValueError("compare_components requires fingerprint_type='a'.")

        poscar_paths = self._validate_poscar_paths(poscar_paths)
        pairs = self.compare_pairs(poscar_paths)
        n_structures = len(poscar_paths)
        return {
            "rdf": self._pairs_to_matrix_by_column(
                pairs,
                n_structures=n_structures,
                column=self.distance_column,
            ),
            "adf": self._pairs_to_matrix_by_column(
                pairs,
                n_structures=n_structures,
                column=self.angle_distance_column,
            ),
        }

    def _build_cry_args(self, gathered_poscars: PathLike) -> List[str]:
        return [
            str(self.cry_path),
            str(gathered_poscars),
            "r",
            "0",
            "v",
            "0",
            "w",
        ]

    def _build_ovf_args(self, rebuilt_poscars: PathLike) -> List[str]:
        args = [str(self.ovf_path), str(rebuilt_poscars)]

        if self.cutoff_radius is not None:
            args.extend(["u", self._format_value(self.cutoff_radius)])
        if self.cutoff_multiplier is not None:
            args.extend(["u", f"*{self._format_value(self.cutoff_multiplier)}"])
        if self.rdf_bin_size is not None:
            args.extend(["b", self._format_value(self.rdf_bin_size)])
        if self.adf_bin_size is not None:
            args.extend(["B", self._format_value(self.adf_bin_size)])
        if self.molecular_weight is not None:
            args.extend(["M", self._format_value(self.molecular_weight)])
        if self.rdf_sigma is not None:
            args.extend(["s", self._format_value(self.rdf_sigma)])
        if self.adf_sigma is not None:
            args.extend(["S", self._format_value(self.adf_sigma)])
        if self.max_triangle_side is not None:
            args.extend(["m", self._format_value(self.max_triangle_side)])
        if self.auto_min_triangle_pair is not None:
            z1, z2 = self.auto_min_triangle_pair
            args.extend(["A", str(z1), str(z2)])

        args.append(self.fingerprint_type)

        if self.exclude_z is not None:
            args.extend(["x", str(self.exclude_z)])

        metric_arg = "c" if self.metric == "cosine" else "e"
        if self.pair is None:
            args.extend([metric_arg, "0", "0"])
        else:
            args.extend([metric_arg, str(self.pair[0]), str(self.pair[1])])

        return args

    @staticmethod
    def _format_value(value: Union[int, float]) -> str:
        return f"{value:g}"

    @classmethod
    def _parse_poscar_signature(cls, path: Path) -> Tuple[Tuple[str, ...], Tuple[int, ...]]:
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 7:
            raise ValueError(f"POSCAR {path} is too short: at least 7 lines are expected.")

        symbols = tuple(lines[5].split())
        count_tokens = lines[6].split()
        try:
            counts = tuple(int(token) for token in count_tokens)
        except ValueError as exc:
            raise ValueError(
                f"POSCAR {path} must use VASP5 format: line 6 contains elements, "
                "line 7 contains atom counts."
            ) from exc

        if not symbols or not counts or len(symbols) != len(counts):
            raise ValueError(
                f"POSCAR {path} has invalid element/count lines: "
                f"{symbols} / {counts}."
            )
        return symbols, counts

    @classmethod
    def _gather_poscars(cls, poscar_paths: Sequence[Path], output_path: Path) -> None:
        reference_signature = None
        reference_path = None

        with output_path.open("w", encoding="utf-8") as output_file:
            for path in poscar_paths:
                signature = cls._parse_poscar_signature(path)
                if reference_signature is None:
                    reference_signature = signature
                    reference_path = path
                elif signature != reference_signature:
                    raise ValueError(
                        "ovf expects identical composition and atom-type order across all POSCAR files. "
                        f"{path} has {signature}, while {reference_path} has "
                        f"{reference_signature}."
                    )

                text = path.read_text(encoding="utf-8").strip()
                if not text:
                    raise ValueError(f"POSCAR {path} is empty.")
                output_file.write(text)
                output_file.write("\n")

    def _parse_ovf_stdout(self, stdout: str) -> pd.DataFrame:
        columns = ["structure_1", "structure_2", self.distance_column]
        if self.fingerprint_type == "a":
            columns.append(self.angle_distance_column)

        rows = []
        saw_distance_output = False
        for line in stdout.splitlines():
            if "Structure_1" in line or "Comparison between structure" in line:
                saw_distance_output = True

            pair_match = self._ALL_PAIR_RE.match(line)
            if pair_match:
                row = {
                    "structure_1": int(pair_match.group("i")),
                    "structure_2": int(pair_match.group("j")),
                    self.distance_column: float(pair_match.group("distance")),
                }
                if pair_match.group("angle") is not None:
                    row[self.angle_distance_column] = float(pair_match.group("angle"))
                rows.append(row)
                continue

            single_match = self._SINGLE_PAIR_RE.search(line)
            if single_match:
                row = {
                    "structure_1": int(single_match.group("i")),
                    "structure_2": int(single_match.group("j")),
                    self.distance_column: float(single_match.group("distance")),
                }
                if single_match.group("angle") is not None:
                    row[self.angle_distance_column] = float(single_match.group("angle"))
                rows.append(row)

        if not rows and not saw_distance_output:
            raise OVFExecutionError(
                "Could not parse distances from ovf stdout. "
                f"First output characters: {stdout[:500]!r}"
            )

        dataframe = pd.DataFrame(rows, columns=columns)
        for column in columns:
            if column not in dataframe:
                dataframe[column] = pd.Series(dtype=float)
        return dataframe[columns]

    def _pairs_to_matrix(self, pairs: pd.DataFrame, n_structures: int) -> np.ndarray:
        return self._pairs_to_matrix_by_column(
            pairs,
            n_structures=n_structures,
            column=self._selected_distance_column(),
        )

    @staticmethod
    def _pairs_to_matrix_by_column(
        pairs: pd.DataFrame,
        n_structures: int,
        column: str,
    ) -> np.ndarray:
        matrix = np.zeros((n_structures, n_structures), dtype=float)
        if column not in pairs.columns:
            raise ValueError(f"The ovf result does not contain column {column!r}.")

        for _, row in pairs.iterrows():
            i = int(row["structure_1"]) - 1
            j = int(row["structure_2"]) - 1
            if i < 0 or j < 0 or i >= n_structures or j >= n_structures:
                raise ValueError(
                    f"OVF returned pair ({i + 1}, {j + 1}) outside range 1..{n_structures}."
                )
            distance = float(row[column])
            matrix[i, j] = distance
            matrix[j, i] = distance

        return matrix

    def _selected_distance_column(self) -> str:
        if self.distance_component == "angle":
            return self.angle_distance_column
        return self.distance_column

    @property
    def distance_column(self) -> str:
        return f"{self.metric}_distance"

    @property
    def angle_distance_column(self) -> str:
        return f"{self.metric}_angle_distance"

    @staticmethod
    def _validate_poscar_paths(poscar_paths: Sequence[PathLike]) -> List[Path]:
        if not poscar_paths:
            raise ValueError("At least one POSCAR file must be provided.")

        paths = [Path(path).expanduser() for path in poscar_paths]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"POSCAR file not found: {path}")
            if not path.is_file():
                raise ValueError(f"Expected a POSCAR file, got: {path}")
        return [path.resolve() for path in paths]

    def _run_command(
        self,
        command: Sequence[str],
        cwd: Path,
        stdout_path: Optional[Path] = None,
    ) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        if self.omp_num_threads is not None:
            env["OMP_NUM_THREADS"] = str(self.omp_num_threads)

        stdout_handle = None
        try:
            if stdout_path is not None:
                stdout_handle = stdout_path.open("w", encoding="utf-8")
                result = subprocess.run(
                    command,
                    cwd=str(cwd),
                    stdout=stdout_handle,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout,
                    env=env,
                    check=False,
                )
            else:
                result = subprocess.run(
                    command,
                    cwd=str(cwd),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=env,
                    check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise OVFExecutionError(
                f"Command exceeded timeout={self.timeout}: {' '.join(command)}"
            ) from exc
        finally:
            if stdout_handle is not None:
                stdout_handle.close()

        if result.returncode != 0:
            raise OVFExecutionError(
                f"Command exited with code {result.returncode}: {' '.join(command)}\n"
                f"stderr:\n{result.stderr or ''}\nstdout:\n{result.stdout or ''}"
            )
        return result

    @classmethod
    def _resolve_binary(cls, explicit_path: Optional[PathLike], kind: str) -> Path:
        if explicit_path is not None:
            explicit = Path(explicit_path).expanduser()
            if explicit.exists():
                cls._ensure_executable(explicit, kind)
                return explicit.resolve()

            found = shutil.which(str(explicit_path))
            if found is not None:
                found_path = Path(found)
                cls._ensure_executable(found_path, kind)
                return found_path.resolve()

            raise FileNotFoundError(f"Executable {kind} not found: {explicit_path}")

        for candidate in cls._default_binary_candidates(kind):
            if candidate.exists():
                cls._ensure_executable(candidate, kind)
                return candidate.resolve()

        for name in cls._default_binary_names(kind):
            found = shutil.which(name)
            if found is not None:
                found_path = Path(found)
                cls._ensure_executable(found_path, kind)
                return found_path.resolve()

        candidates = ", ".join(str(path) for path in cls._default_binary_candidates(kind))
        raise FileNotFoundError(
            f"Executable {kind} not found. Checked: {candidates}; "
            f"also checked PATH names {cls._default_binary_names(kind)}."
        )

    @classmethod
    def _default_binary_candidates(cls, kind: str) -> List[Path]:
        project_root = Path(__file__).resolve().parents[1]
        legacy_source_root = project_root.parent / "version_6.3.5"
        binary_dirs = [
            project_root / "descriptors" / "mOVF",
            legacy_source_root,
        ]
        return [
            binary_dir / name
            for binary_dir in binary_dirs
            for name in cls._default_binary_names(kind)
        ]

    @staticmethod
    def _default_binary_names(kind: str) -> List[str]:
        system = platform.system()
        if system == "Darwin":
            return [f"{kind}_macos", kind]
        if system == "Windows":
            return [f"{kind}.exe", kind]
        return [kind]

    @staticmethod
    def _ensure_executable(path: Path, kind: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"{kind} was found but is not a file: {path}")
        if not os.access(path, os.X_OK):
            raise PermissionError(f"{kind} was found but is not executable: {path}")


__all__ = ["OVFExecutionError", "OVFRunner"]
