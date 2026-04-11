# Comparators

[English](README.md) | Русский

Python-инструменты для расчета fingerprint-дескрипторов кристаллических структур и построения попарных матриц расстояний. Основной сценарий: взять набор структур, посчитать для каждой структуры дескриптор и сравнить структуры выбранной метрикой. Отдельный backend `mOVF` запускает нативные исполняемые файлы `cry` и `ovf`, потому что эта версия OVF сама считает расстояния внутри C-кода.

## Структура проекта

```text
Comparators/
  backends/
    ovf_runner.py          # Python-обертка над descriptors/mOVF/cry и ovf
  core/
    base.py                # базовые интерфейсы BaseDescriptor и BaseMetric
    comparator.py          # StructureComparator: дескриптор + метрика + кэш
  descriptors/
    _utils.py              # общие утилиты для дескрипторов
    cell.py                # CellDescriptor: ячейка, плотность, состав
    mbtr.py                # MBTRDescriptor через DScribe
    ovf_by_dscribe.py      # OVFDescriptors через DScribe ValleOganov
    soap.py                # SOAPDescriptor через DScribe
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

Важно: `descriptors/ovf_by_dscribe.py` и `backends/ovf_runner.py` - это разные реализации OVF.

- `OVFDescriptors` использует Python-библиотеку DScribe, возвращает fingerprint-вектор и дальше сравнивается обычной Python-метрикой через `StructureComparator`.
- `OVFRunner` использует внешние исполняемые файлы `cry` и `ovf` из `descriptors/mOVF`. C-программа `ovf` сама считает fingerprint и сразу возвращает расстояния. Поэтому `OVFRunner` не является наследником `BaseDescriptor`.

## Установка

Рекомендуемый вариант через conda:

```bash
conda create -n sci -c conda-forge python=3.12 numpy pandas ase pymatgen dscribe
conda activate sci
```

Альтернатива через pip:

```bash
python -m pip install -r requirements.txt
```

Запускать примеры и тесты лучше из корня проекта `Comparators/`, потому что модули сейчас импортируются как локальные top-level пакеты:

```bash
cd /path/to/Comparators
python -m unittest discover -s tests
```

## Обычный workflow: дескриптор + метрика

`StructureComparator` загружает структуры через `pymatgen`, переводит их в `ASE Atoms`, считает дескрипторы и передает матрицу fingerprint-векторов в выбранную метрику.

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

`descriptor_normalization` является опциональным. Доступные значения:

- `None` или `"none"` - не нормализовать fingerprint-векторы;
- `"l2"` - L2-нормализация каждой строки;
- `"zscore"` - z-score по колонкам;
- `"minmax"` - min-max по колонкам.

## Дескрипторы

### CellDescriptor

`CellDescriptor` - простой глобальный fingerprint без DScribe. Он может включать:

- параметры ячейки `a, b, c`;
- углы `alpha, beta, gamma`;
- `volume_per_atom`;
- плотность;
- доли элементов из заданного списка `species`.

Пример:

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

`SOAPDescriptor` использует `dscribe.descriptors.SOAP`. Подходит для локального окружения атомов и хорошо работает как fingerprint для структур, если результат усредняется или агрегируется.

Важные параметры:

- `species` - полный список элементов, которые могут встречаться после предобработки;
- `r_cut`, `n_max`, `l_max`, `sigma` - параметры SOAP;
- `periodic` - учитывать ли периодические граничные условия;
- `average` - режим усреднения DScribe: `"inner"`, `"outer"` или `"off"`;
- `pooling` - если `average="off"`, используется `"mean"` или `"mean_std"`, чтобы получить fingerprint фиксированной длины.

### MBTRDescriptor

`MBTRDescriptor` использует `dscribe.descriptors.MBTR`. Поддерживает k-body признаки через параметры DScribe `geometry`, `grid`, `weighting`.

Пример:

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

### OVFDescriptors через DScribe

`OVFDescriptors` использует `dscribe.descriptors.ValleOganov`. Это Python/DScribe-вариант, который возвращает fingerprint-вектор.

Пример:

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

## Метрики

### CosineMetric

`CosineMetric` возвращает косинусное расстояние:

```text
distance = 0.5 * (1 - cosine_similarity)
```

Меньше значит структуры ближе. Нулевые векторы обрабатываются явно: два нулевых вектора дают расстояние `0.0`, нулевой и ненулевой - `1.0`.

### EuclideanMetric

`EuclideanMetric` возвращает евклидово расстояние. По умолчанию `normalize=True`, то есть векторы нормализуются перед расчетом расстояния.

## mOVF: native cry + ovf

Папка `descriptors/mOVF/` содержит две пары исполняемых файлов:

```text
cry, ovf             # Linux x86_64
cry_macos, ovf_macos # macOS arm64
```

`cry` перестраивает POSCAR-структуры для корректной работы native OVF. В ручном режиме команда выглядит так:

```bash
./cry gathered_POSCARS r 0 v 0 w > rebuilt_POSCARS
```

Что здесь происходит:

- `gathered_POSCARS` - один файл, где несколько POSCAR-структур записаны подряд;
- `r 0` - перестроить все структуры;
- `v 0` - напечатать все перестроенные структуры в POSCAR/VASP-формате;
- `w` - создать `Data_for_ovf.bin`, который нужен для molecular mode `M`;
- `> rebuilt_POSCARS` - stdout `cry` сохраняется в новый POSCAR-файл.

`ovf` затем читает `rebuilt_POSCARS` и считает Oganov-Valle fingerprint distances. `OVFRunner` автоматизирует оба шага и запускает программы во временной рабочей папке, чтобы `Data_for_ovf.bin` не конфликтовал между разными расчетами.

### Требования для mOVF

1. На macOS используются `cry_macos` и `ovf_macos`. Эти бинарники собраны под arm64. Если при запуске появится ошибка про `libgomp.1.dylib`, установите GCC через Homebrew:

```bash
brew install gcc
```

2. На Linux используются `cry` и `ovf`. В репозитории лежат x86_64 ELF-бинарники; на другой архитектуре их нужно пересобрать.

3. Windows-бинарники не входят в проект. Для Windows-пользователей самый простой путь - WSL/Linux или отдельная сборка `cry.exe` и `ovf.exe` с явной передачей путей в `OVFRunner`.

4. Файлы должны быть исполняемыми:

```bash
chmod +x descriptors/mOVF/cry descriptors/mOVF/ovf descriptors/mOVF/cry_macos descriptors/mOVF/ovf_macos
```

5. POSCAR-файлы для одного запуска должны иметь одинаковый состав и одинаковый порядок типов атомов. Например, `C O` / `1 2` и `O C` / `2 1` в одном запуске смешивать нельзя.

6. Сейчас ожидается VASP5-style POSCAR: строка элементов, затем строка количеств атомов.

### Как OVFRunner ищет cry и ovf

Если `cry_path` и `ovf_path` не заданы явно, `OVFRunner` ищет бинарники в таком порядке:

1. `Comparators/descriptors/mOVF/cry_macos` и `ovf_macos` на macOS;
2. `Comparators/descriptors/mOVF/cry` и `ovf` на Linux;
3. legacy fallback `../version_6.3.5/`;
4. системный `PATH`.

Явные пути всегда имеют приоритет:

```python
from backends import OVFRunner

runner = OVFRunner(
    cry_path="descriptors/mOVF/cry_macos",
    ovf_path="descriptors/mOVF/ovf_macos",
)
```

### Базовый пример mOVF

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

`compare()` возвращает квадратную симметричную матрицу `numpy.ndarray`. `compare_pairs()` возвращает таблицу:

```text
structure_1  structure_2  cosine_distance
```

### RDF + ADF

Если нужен не только RDF/distance-компонент, но и ADF/angle-компонент, включите `fingerprint_type="a"`:

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

В этом режиме `compare_pairs()` возвращает обе колонки:

```text
structure_1  structure_2  cosine_distance  cosine_angle_distance
```

Если нужен только один компонент как матрица:

```python
runner = OVFRunner(
    fingerprint_type="a",
    metric="cosine",
    distance_component="angle",
)

adf_matrix = runner.compare(poscars)
```

`distance_component="rdf"` означает брать обычную distance/RDF-часть. `distance_component="angle"` означает брать angle/ADF-часть, и работает только при `fingerprint_type="a"`.

### Параметры OVFRunner

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

Параметры, которые напрямую соответствуют ключам `ovf`:

| Параметр | Команда ovf | Значение |
| --- | --- | --- |
| `fingerprint_type="d"` | `d` | только distance/RDF fingerprint |
| `fingerprint_type="a"` | `a` | distance/RDF + angle/ADF fingerprint |
| `metric="cosine"` | `c` | cosine distance |
| `metric="euclidean"` | `e` | Euclidean distance |
| `pair=None` | `c 0 0` или `e 0 0` | все пары |
| `pair=(1, 2)` | `c 1 2` или `e 1 2` | только пара структур 1 и 2 |
| `cutoff_radius=10.0` | `u 10` | явный cutoff radius |
| `cutoff_multiplier=0.5` | `u *0.5` | множитель автоматического cutoff |
| `rdf_bin_size=0.05` | `b 0.05` | размер RDF bin |
| `adf_bin_size=...` | `B ...` | размер ADF bin |
| `rdf_sigma=0.02` | `s 0.02` | HWHM Gaussian kernel для RDF |
| `adf_sigma=...` | `S ...` | HWHM Gaussian kernel для ADF |
| `max_triangle_side=3.0` | `m 3` | максимальная сторона треугольника для углов |
| `molecular_weight=0.0` | `M 0` | molecular fingerprint mode; `None` не включает `M` |
| `auto_min_triangle_pair=("C", "N")` | `A C N` | auto `m` по минимальному межмолекулярному расстоянию |
| `exclude_z=1` | `x 1` | исключить атомный номер 1, обычно H |

`fingerprint_type="t"` заблокирован по умолчанию, потому что в C-коде torsion mode помечен как `NOT YET`. Его можно включить только явно через `allow_experimental_torsion=True`.

Технические параметры:

- `work_dir` - где создавать временную папку запуска;
- `keep_work_dir=True` - оставить временную папку для отладки;
- `timeout` - лимит времени для subprocess;
- `omp_num_threads` - задает переменную окружения `OMP_NUM_THREADS`.

## Тесты

```bash
conda run -n sci python -m unittest discover -s tests
```

Тесты проверяют:

- устойчивую обработку нулевых векторов в cosine metric;
- нормализацию fingerprint-векторов;
- cache key с учетом параметров дескриптора;
- согласование `remove_species` и `descriptor.species`;
- инвариантность SOAP при перестановке атомов и трансляции;
- разбор stdout `ovf`;
- сборку POSCAR-файлов для mOVF;
- auto-discovery бинарников из `descriptors/mOVF`;
- изолированный запуск `OVFRunner` через временную рабочую папку.

## Ограничения

- `StructureComparator` работает с дескрипторами фиксированной длины. Если дескриптор возвращает разные длины fingerprint для разных структур, будет выброшена явная ошибка.
- `OVFRunner` рассчитан на POSCAR/VASP5 input. Для других форматов используйте `StructureComparator` с `pymatgen`, либо заранее конвертируйте структуры в POSCAR.
- Для `mOVF` все структуры в одном запуске должны иметь одинаковые строки элементов и количеств атомов.
- Внешние бинарники `cry` и `ovf` зависят от платформы. Если бинарник не запускается на вашей системе, его нужно пересобрать под вашу архитектуру или передать путь к совместимой версии через `cry_path` и `ovf_path`.
