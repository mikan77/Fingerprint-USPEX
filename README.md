# Comparators

English | [Russian](README.ru.md)

Python tools for calculating fingerprint descriptors of crystalline structures and building pairwise distance matrices. The default workflow is to load a set of structures, compute a descriptor for each structure, and compare the resulting vectors with a selected metric. A separate `mOVF` backend runs the native `cry` and `ovf` executables, because that OVF implementation computes distances inside the C program.

## Project Layout

```text
Comparators/
  backends/
    ovf_runner.py          # Python wrapper around descriptors/mOVF/cry and ovf
  core/
    base.py                # BaseDescriptor and BaseMetric interfaces
    comparator.py          # StructureComparator: descriptor + metric + cache
  descriptors/
    _utils.py              # Shared descriptor helpers
    cell.py                # CellDescriptor: cell, density, composition
    mbtr.py                # MBTRDescriptor via DScribe
    ovf_by_dscribe.py      # OVFDescriptors via DScribe ValleOganov
    soap.py                # SOAPDescriptor via DScribe
    mOVF/
      cry                  # Linux x86_64 executable
      ovf                  # Linux x86_64 executable
      cry_macos            # macOS arm64 executable
      ovf_macos            # macOS arm64 executable
  metrics/
    cosine.py              # CosineMetric
    euclidean.py           # EuclideanMetric
  tests/
    test_fingerprints.py
    test_ovf_runner.py
  utils/
    calculate_similarity.py
    density.py
    sorting.py
    write_file.py
```

There are two OVF-related implementations in this project:

- `descriptors/ovf_by_dscribe.py` provides `OVFDescriptors`, a DScribe-based descriptor that returns a fingerprint vector. It is used with `StructureComparator` and a Python metric such as `CosineMetric` or `EuclideanMetric`.
- `backends/ovf_runner.py` provides `OVFRunner`, a wrapper around the external `cry` and `ovf` executables in `descriptors/mOVF`. The native `ovf` program computes the Oganov-Valle fingerprint and directly returns pairwise distances, so `OVFRunner` is not a `BaseDescriptor` subclass.

## Installation

Recommended conda setup:

```bash
conda create -n sci -c conda-forge python=3.12 numpy pandas ase pymatgen dscribe
conda activate sci
```

Alternative pip setup:

```bash
python -m pip install -r requirements.txt
```

Run examples and tests from the `Comparators/` project root. The current code uses local top-level imports such as `from core.comparator import StructureComparator`.

```bash
cd /path/to/Comparators
python -m unittest discover -s tests
```

## Standard Workflow: Descriptor + Metric

`StructureComparator` loads structures with `pymatgen`, converts them to `ASE Atoms`, computes descriptors, and passes the descriptor matrix to a metric.

```python
from core.comparator import StructureComparator
from descriptors.soap import SOAPDescriptor
from metrics.cosine import CosineMetric

paths = ["POSCAR_1", "POSCAR_2", "POSCAR_3"]

descriptor = SOAPDescriptor(
    species=["C", "O", "H"],
    r_cut=6.0,
    n_max=8,
    l_max=6,
    sigma=0.2,
    periodic=True,
    average="inner",
)

comparator = StructureComparator(
    descriptor=descriptor,
    metric=CosineMetric(),
    descriptor_normalization=None,
)

matrix = comparator.compare(paths)
df = comparator.compare_to_dataframe(paths)
```

`descriptor_normalization` is optional:

- `None` or `"none"`: do not normalize descriptor vectors;
- `"l2"`: L2-normalize each row;
- `"zscore"`: apply column-wise z-score normalization;
- `"minmax"`: apply column-wise min-max normalization.

### Combining Descriptor Pipelines

`EnsembleComparator` combines multiple Python descriptor + metric pipelines into one weighted distance matrix:

```python
from core.comparator import EnsembleComparator
from descriptors.cell import CellDescriptor
from descriptors.soap import SOAPDescriptor
from metrics.cosine import CosineMetric

comparator = EnsembleComparator(
    configs=[
        (CellDescriptor(["C", "O"]), CosineMetric(), 1.0),
        (SOAPDescriptor(["C", "O"]), CosineMetric(), 2.0),
    ],
    use_robust_scaling=True,
)

matrix = comparator.compare(paths)
```

Weights are normalized internally, so weights `[1.0, 2.0]` become `[1/3, 2/3]`. With `use_robust_scaling=True`, each component matrix is processed independently:

```text
x_robust = (x - median) / IQR
x_final = 1 / (1 + exp(-x_robust))
combined = sum(weight_i * x_final_i)
```

The diagonal is not included in robust-sigmoid scaling and is left unchanged. For valid distance matrices this keeps the diagonal at `0.0` without letting diagonal zeros affect the median/IQR.

## Descriptors

### CellDescriptor

`CellDescriptor` is a simple global fingerprint that does not use DScribe. It can include:

- cell lengths `a, b, c`;
- cell angles `alpha, beta, gamma`;
- `volume_per_atom`;
- density;
- composition fractions for the requested `species` list.

Example:

```python
from descriptors.cell import CellDescriptor

descriptor = CellDescriptor(
    species=["C", "O"],
    include_cell=True,
    include_composition=True,
    include_density=True,
)
```

### SOAPDescriptor

`SOAPDescriptor` uses `dscribe.descriptors.SOAP`. It is suitable for describing local atomic environments and can be used as a structure-level fingerprint when the atomic descriptors are averaged or pooled.

Important parameters:

- `species`: full list of elements that may remain after preprocessing;
- `r_cut`, `n_max`, `l_max`, `sigma`: SOAP parameters;
- `periodic`: whether to use periodic boundary conditions;
- `average`: DScribe averaging mode, `"inner"`, `"outer"`, or `"off"`;
- `pooling`: if `average="off"`, use `"mean"` or `"mean_std"` to obtain a fixed-length fingerprint.

### MBTRDescriptor

`MBTRDescriptor` uses `dscribe.descriptors.MBTR`. It supports k-body terms through the DScribe `geometry`, `grid`, and `weighting` parameters.

Example:

```python
from descriptors.mbtr import MBTRDescriptor

descriptor = MBTRDescriptor(
    species=["C", "O"],
    geometry={"function": "distance"},
    grid={"min": 0.0, "max": 8.0, "n": 200, "sigma": 0.1},
    weighting={"function": "exp", "scale": 0.5, "threshold": 1e-3},
    periodic=True,
)
```

### OVFDescriptors via DScribe

`OVFDescriptors` uses `dscribe.descriptors.ValleOganov`. This is the Python/DScribe variant that returns a fingerprint vector.

Example:

```python
from descriptors.ovf_by_dscribe import OVFDescriptors

descriptor = OVFDescriptors(
    species=["C", "O"],
    function="distance",
    n=200,
    sigma=0.05,
    r_cut=8.0,
)
```

## Metrics

### CosineMetric

`CosineMetric` returns cosine distance:

```text
distance = 0.5 * (1 - cosine_similarity)
```

Lower values mean more similar structures. Zero vectors are handled explicitly: two zero vectors have distance `0.0`, while a zero vector compared with a non-zero vector has distance `1.0`.

### EuclideanMetric

`EuclideanMetric` returns Euclidean distance. By default `normalize=True`, so vectors are normalized before distance calculation.

## mOVF: Native cry + ovf

The `descriptors/mOVF/` directory contains two executable pairs:

```text
cry, ovf             # Linux x86_64
cry_macos, ovf_macos # macOS arm64
```

`cry` rebuilds POSCAR structures for the native OVF workflow. The equivalent manual command is:

```bash
./cry gathered_POSCARS r 0 v 0 w > rebuilt_POSCARS
```

Meaning:

- `gathered_POSCARS`: one file with multiple POSCAR blocks written one after another;
- `r 0`: rebuild all structures;
- `v 0`: print all rebuilt structures in POSCAR/VASP format;
- `w`: write `Data_for_ovf.bin`, required for molecular mode `M`;
- `> rebuilt_POSCARS`: save `cry` stdout to a rebuilt POSCAR file.

`ovf` then reads `rebuilt_POSCARS` and computes Oganov-Valle fingerprint distances. `OVFRunner` automates both steps and runs them in an isolated temporary working directory, so `Data_for_ovf.bin` cannot collide between separate calculations.

### mOVF Requirements

1. On macOS, `cry_macos` and `ovf_macos` are used. These binaries are built for arm64. If running them fails with a `libgomp.1.dylib` error, install GCC with Homebrew:

```bash
brew install gcc
```

2. On Linux, `cry` and `ovf` are used. The repository contains x86_64 ELF binaries; other architectures require rebuilding.

3. Windows binaries are not included. For Windows users, the simplest route is WSL/Linux or a separate build of `cry.exe` and `ovf.exe`, passed explicitly through `OVFRunner`.

4. The files must be executable:

```bash
chmod +x descriptors/mOVF/cry descriptors/mOVF/ovf descriptors/mOVF/cry_macos descriptors/mOVF/ovf_macos
```

5. POSCAR files in one run must have identical composition and identical atom-type order. For example, `C O` / `1 2` and `O C` / `2 1` must not be mixed in the same run.

6. VASP5-style POSCAR input is expected: one line with element symbols followed by one line with atom counts.

### Binary Discovery

If `cry_path` and `ovf_path` are not provided explicitly, `OVFRunner` searches in this order:

1. `Comparators/descriptors/mOVF/cry_macos` and `ovf_macos` on macOS;
2. `Comparators/descriptors/mOVF/cry` and `ovf` on Linux;
3. legacy fallback `../version_6.3.5/`;
4. system `PATH`.

Explicit paths always take priority:

```python
from backends import OVFRunner

runner = OVFRunner(
    cry_path="descriptors/mOVF/cry_macos",
    ovf_path="descriptors/mOVF/ovf_macos",
)
```

### Basic mOVF Example

```python
from backends import OVFRunner

poscars = ["POSCAR_1", "POSCAR_2", "POSCAR_3"]

runner = OVFRunner(
    fingerprint_type="d",
    metric="cosine",
)

matrix = runner.compare(poscars)
pairs = runner.compare_pairs(poscars)
```

`compare()` returns a square symmetric `numpy.ndarray`. `compare_pairs()` returns a table:

```text
structure_1  structure_2  cosine_distance
```

### RDF + ADF

Use `fingerprint_type="a"` when both the RDF/distance component and ADF/angle component are needed:

```python
runner = OVFRunner(
    fingerprint_type="a",
    metric="cosine",
)

pairs = runner.compare_pairs(poscars)
components = runner.compare_components(poscars)

rdf_matrix = components["rdf"]
adf_matrix = components["adf"]
```

In this mode `compare_pairs()` returns both columns:

```text
structure_1  structure_2  cosine_distance  cosine_angle_distance
```

If only one component is needed as a matrix:

```python
runner = OVFRunner(
    fingerprint_type="a",
    metric="cosine",
    distance_component="angle",
)

adf_matrix = runner.compare(poscars)
```

`distance_component="rdf"` selects the normal distance/RDF part. `distance_component="angle"` selects the angle/ADF part and is valid only with `fingerprint_type="a"`.

### Combining Native mOVF RDF and ADF Distances

`OVFRunner` is not a `BaseDescriptor`: the native `ovf` executable computes fingerprints and distances internally. For this reason, RDF/ADF output from `ovf_macos` should be combined as ready-made distance matrices rather than passed to `EnsembleComparator`.

Use `combine_distance_matrices` for this case:

```python
from backends import OVFRunner
from core.comparator import combine_distance_matrices

runner = OVFRunner(
    fingerprint_type="a",
    metric="cosine",
)

components = runner.compare_components(poscars)

combined = combine_distance_matrices(
    matrices=[components["rdf"], components["adf"]],
    weights=[0.5, 0.5],
    use_robust_scaling=True,
)
```

With `use_robust_scaling=True`, RDF and ADF matrices are scaled independently with `RobustScaler`, mapped through sigmoid to `(0, 1)`, and then combined by normalized weights. Diagonal values are excluded from scaling and left unchanged.

### OVFRunner Parameters

```python
OVFRunner(
    cry_path=None,
    ovf_path=None,
    fingerprint_type="d",
    metric="cosine",
    pair=None,
    cutoff_radius=None,
    cutoff_multiplier=None,
    rdf_bin_size=0.05,
    adf_bin_size=None,
    rdf_sigma=0.02,
    adf_sigma=None,
    max_triangle_side=3.0,
    molecular_weight=None,
    auto_min_triangle_pair=None,
    exclude_z=None,
    work_dir=None,
    keep_work_dir=False,
    timeout=None,
    omp_num_threads=None,
    distance_component="rdf",
    allow_experimental_torsion=False,
)
```

Parameters mapped directly to native `ovf` commands:

| Parameter | ovf command | Meaning |
| --- | --- | --- |
| `fingerprint_type="d"` | `d` | distance/RDF fingerprint only |
| `fingerprint_type="a"` | `a` | distance/RDF + angle/ADF fingerprint |
| `metric="cosine"` | `c` | cosine distance |
| `metric="euclidean"` | `e` | Euclidean distance |
| `pair=None` | `c 0 0` or `e 0 0` | all pairs |
| `pair=(1, 2)` | `c 1 2` or `e 1 2` | only structures 1 and 2 |
| `cutoff_radius=10.0` | `u 10` | explicit cutoff radius |
| `cutoff_multiplier=0.5` | `u *0.5` | multiplier of the automatic cutoff |
| `rdf_bin_size=0.05` | `b 0.05` | RDF bin size |
| `adf_bin_size=...` | `B ...` | ADF bin size |
| `rdf_sigma=0.02` | `s 0.02` | RDF Gaussian kernel HWHM |
| `adf_sigma=...` | `S ...` | ADF Gaussian kernel HWHM |
| `max_triangle_side=3.0` | `m 3` | maximum triangle side for angles |
| `molecular_weight=0.0` | `M 0` | molecular fingerprint mode; `None` does not enable `M` |
| `auto_min_triangle_pair=("C", "N")` | `A C N` | auto-set `m` from a minimum intermolecular distance |
| `exclude_z=1` | `x 1` | exclude atomic number 1, usually H |

`fingerprint_type="t"` is blocked by default because torsion mode is marked as `NOT YET` in the C source. It can be enabled only explicitly with `allow_experimental_torsion=True`.

Technical parameters:

- `work_dir`: parent directory for temporary run directories;
- `keep_work_dir=True`: keep the temporary run directory for debugging;
- `timeout`: subprocess timeout;
- `omp_num_threads`: sets the `OMP_NUM_THREADS` environment variable.

## Tests

```bash
conda run -n sci python -m unittest discover -s tests
```

The tests cover:

- zero-vector handling in the cosine metric;
- descriptor-vector normalization;
- cache keys that include descriptor parameters;
- consistency between `remove_species` and `descriptor.species`;
- SOAP invariance under atom reordering and translation;
- parsing native `ovf` stdout;
- POSCAR gathering for mOVF;
- binary auto-discovery from `descriptors/mOVF`;
- isolated `OVFRunner` execution through a temporary working directory.

## Limitations

- `StructureComparator` expects fixed-length descriptors. If a descriptor returns different fingerprint lengths for different structures, it raises an explicit error.
- `OVFRunner` is intended for POSCAR/VASP5 input. For other formats, use `StructureComparator` with `pymatgen` or convert structures to POSCAR first.
- For `mOVF`, all structures in one run must have identical element and atom-count lines.
- The external `cry` and `ovf` binaries are platform-dependent. If a bundled binary does not run on your system, rebuild it for your architecture or pass a compatible executable through `cry_path` and `ovf_path`.
