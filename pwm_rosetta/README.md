# pwm_rosetta

**Hybrid PWM generation for transcription factor binding sites — AF3 wild-type structure + PyRosetta SNP scanning.**

Instead of running AlphaFold 3 for every single-nucleotide mutation (expensive), `pwm_rosetta` runs AF3 **once** for the wild-type complex and then uses PyRosetta to generate all single-base mutations with local minimisation. This makes it **~50× faster** than a full AF3-per-mutation approach while remaining physically grounded.

```
Wild-type PDB (AF3 or your own)
         │
         ▼
  PyRosetta: mutate every base × 3 alternatives
         │
         ▼
  Interface ΔΔG per mutation  →  Boltzmann PPM  →  PWM / sequence logo
```

> **Self-contained** — Rosetta weight files, flags, and the XML scoring protocol are all bundled inside the package. No external `dbp_design` installation or `--rosetta_dir` path is required.

---

## Features

- **No `dbp_design` dependency** — all Rosetta scoring files bundled in the package
- **AF3 optional** — if you already have a PDB, PyRosetta alone is sufficient
- **Double-stranded DNA** — complementary strand is mutated automatically (Watson-Crick pairing)
- **Clean Python API** — import and call individual steps without touching the CLI
- **Pure-Python PPM/PWM** — `pwm_from_hybrid_csv` and `plot` require only numpy/pandas/matplotlib; no PyRosetta needed for post-processing
- **`logomaker` visualisation** — sequence logos with per-position information content

---

## Installation

### Prerequisites

| Dependency | Required for |
|-----------|-------------|
| Python ≥ 3.9 | everything |
| [PyRosetta](https://www.pyrosetta.org/downloads) | mutation scanning & ΔΔG |
| `gemmi` | CIF → PDB conversion |
| `biopython` | chain renaming |
| `numpy`, `pandas`, `matplotlib`, `scipy` | always |
| `tqdm` | progress bar |
| `logomaker` | sequence logo plots |
| AF3 / `af3cli` (Python 3.11+) | only if you don't have a PDB yet |

### Install from source

```bash
git clone https://github.com/Layne-Huang/pwm_rosetta.git
cd pwm_rosetta
pip install -e .
```

This registers the `pwm-hybrid` command-line tool and the `pwm_hybrid` Python package.
The Rosetta weight/flag files ship inside the package — no extra paths to set.

### Optional: install visualisation extras

```bash
pip install logomaker
```

---

## Quick start

### 1 — From an existing PDB (most common)

```python
from pwm_hybrid import pwm_from_hybrid_csv, plot, HAS_AF3
print(f"AF3 available: {HAS_AF3}")  # False is fine — not needed here
```

**Run the CLI:**

```bash
pwm-hybrid \
  -pdb /path/to/wt_complex.pdb \
  -output_dir ./results/
```

This writes `results/pwm_results_hybrid.csv`.

**Then generate and plot the PWM:**

```python
from pwm_hybrid import pwm_from_hybrid_csv, plot

seq, energies, PPM, PWM = pwm_from_hybrid_csv(
    'results/pwm_results_hybrid.csv',
    tau=1.5          # temperature: lower → sharper motif
)

plot(PPM, seq=seq, outpath='results/', filename_prefix='myTF')
```

---

### 2 — Full pipeline (AF3 + Rosetta), Python API

```python
import pyrosetta
from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags, init_pyrosetta
from pwm_hybrid.pipeline import generate_pwm_hybrid
from pwm_hybrid import pwm_from_hybrid_csv, plot

# Initialise PyRosetta using the bundled flags
pyrosetta.init(get_pyrosetta_init_flags())
init_pyrosetta()   # loads bundled weights; no path arguments needed

df = generate_pwm_hybrid(
    protein_seq='MKTAYIAKQRQISFVKSHFSRQ...',
    dna_seq='GCAGTATGCATA',
    output_dir='./results/',
    minimize_local=True,
    use_relax=False
)

seq, energies, PPM, PWM = pwm_from_hybrid_csv('results/pwm_results_hybrid.csv')
plot(PPM, seq=seq, outpath='results/')
```

---

### 3 — Skip AF3: provide your own PDB

```python
df = generate_pwm_hybrid(
    protein_seq=None,
    dna_seq=None,          # extracted automatically from PDB
    output_dir='./results/',
    wt_pdb='/path/to/complex.pdb'
)
```

---

## CLI reference

```
pwm-hybrid [-h] -output_dir OUTPUT_DIR
           [-pdb PDB]
           [-protein_seq PROTEIN_SEQ] [-dna_seq DNA_SEQ]
           [-template TEMPLATE] [-zf_count ZF_COUNT]
           [-no_minimize] [-relax]
           [-psipred_exe PSIPRED_EXE]
```

| Flag | Description |
|------|-------------|
| `-pdb` | Wild-type PDB (skip AF3) |
| `-protein_seq` | Protein sequence (used with AF3) |
| `-dna_seq` | DNA sequence (used with AF3) |
| `-output_dir` | Output directory **(required)** |
| `-psipred_exe` | Path to `runpsipred_single` (for SSPrediction filters). Defaults to `PSIPRED_EXE` env var, then `/software/psipred4/runpsipred_single`. Pass `''` to disable. |
| `-no_minimize` | Skip local minimisation around mutations |
| `-relax` | Use full FastRelax (slower, more thorough) |
| `-template` | AF3 template JSON |
| `-zf_count` | Number of zinc fingers (passed to AF3) |

### psipred (optional)

The SSPrediction Rosetta filter requires `psipred`. The ΔΔG calculation
(what generates the PWM) does **not** require it. If `psipred` is not
available the filter is silently disabled:

```bash
# Explicitly disable psipred
pwm-hybrid -pdb complex.pdb -output_dir ./out/ -psipred_exe ''

# Or set the path via environment variable
export PSIPRED_EXE=/opt/psipred/runpsipred_single
pwm-hybrid -pdb complex.pdb -output_dir ./out/
```

---

## Python API reference

### `pwm_hybrid.rosetta.init.get_pyrosetta_init_flags()`

Returns the standard `pyrosetta.init()` flags string (using the bundled
`RM8B_flags` file).  Call this before `pyrosetta.init()`:

```python
import pyrosetta
from pwm_hybrid.rosetta.init import get_pyrosetta_init_flags
pyrosetta.init(get_pyrosetta_init_flags())
```

### `pwm_hybrid.rosetta.init.init_pyrosetta(psipred_exe=None)`

Load the bundled scoring protocol into module globals (`ddg_filter`,
`sfxn`, `fast_relax`).  Must be called after `pyrosetta.init()`.

### `pwm_hybrid.pwm_from_hybrid_csv`

```python
seq, energies, PPM, PWM = pwm_from_hybrid_csv(
    csv_path,           # path to pwm_results_hybrid.csv
    tau=1.5,            # Boltzmann temperature (>0)
    value='mut_ddg'     # column: 'mut_ddg' or 'ddG'
)
```

Returns DNA sequence, raw energy matrix `(L×4)`, PPM `(L×4)`, and PWM
`(L×4)` in ACGT order.

### `pwm_hybrid.plot`

```python
plot(
    ppm,                        # (L×4) array
    seq=None,                   # optional sequence string
    outpath='.',                # save directory
    filename_prefix='motif',    # output filename prefix
    highlight_positions=None,   # list of 1-based positions to shade
    highlight_color='red',
    position_range=None,        # (start, end) 1-based slice
    show_only_positions=None    # list of 1-based positions
)
```

Saves `<outpath>/<filename_prefix>_logo.png`.

### `pwm_hybrid.pwm_from_energies_with_temperature`

```python
energies, PPM, PWM = pwm_from_energies_with_temperature(
    WT_seq, WT_energy, mutation_dict,
    tau=1.0, value='ddg'
)
```

Lower-level function: build a PPM from a pre-assembled mutation DataFrame.

---

## Output files

| File | Description |
|------|-------------|
| `pwm_results_hybrid.csv` | Per-mutation ΔΔG table |
| `<prefix>_logo.png` | Sequence information logo |

### `pwm_results_hybrid.csv` columns

| Column | Description |
|--------|-------------|
| `position` | 1-based DNA position |
| `original` | Wild-type base |
| `mutant` | Mutant base |
| `wt_ddg` | Wild-type interface ΔG |
| `mut_ddg` | Mutant interface ΔG |
| `ddG` | `mut_ddg − wt_ddg` |
| `metal_free` | 1 if no metal ions in structure |

---

## Package structure

```
pwm_hybrid/
├── __init__.py              # Public API
├── _version.py
├── constants.py             # DNA_BASES, BASE_MAPPING, COMPLEMENTARY, …
├── af3/
│   ├── __init__.py          # HAS_AF3 flag + graceful ImportError
│   └── folding.py           # re-export from multiflow/evaluation/AF3.py
├── rosetta/
│   ├── _xml_loader.py       # bundled xml_loader (scoring protocol)
│   ├── _data/
│   │   └── flags_and_weights/
│   │       ├── RM8B_flags                               # pyrosetta.init() flags
│   │       ├── RM8B_torsional.wts                       # score function weights
│   │       └── no_ref.rosettacon2018.beta_nov16_constrained.txt  # relax script
│   ├── init.py              # init_pyrosetta() — deferred, uses bundled files
│   ├── structure.py         # convert_cif_to_pdb, rename_chains, …
│   ├── mutations.py         # mutate_dna_base, minimize_around_mutation
│   └── scoring.py           # calculate_interface_ddg, calculate_ddg_with_relax
├── pwm/
│   ├── energies.py          # pwm_from_hybrid_csv, pwm_from_energies_with_temperature
│   └── viz.py               # plot, makeLogo, plotPWM, seqToOneHot
├── pipeline.py              # generate_pwm_hybrid() orchestration
└── cli.py                   # pwm-hybrid entry point
```

---

## How it works

1. **Wild-type structure** — either provided as a PDB or generated by AF3.
2. **Chain preparation** — `rename_chains` maps protein → chain A, DNA → chain B (metals appended to A).
3. **Mutation scanning** — for every DNA position, all 3 alternative bases are introduced. The complementary strand is mutated automatically.
4. **Local minimisation** — side chains and backbone within 6 Å of the mutation site are relaxed (or full FastRelax if `-relax` is set).
5. **ΔΔG calculation** — `ddg_filter.compute(pose)` using the bundled `RM8B_torsional.wts` score function (3 repeats with outlier removal).
6. **Boltzmann PPM** — energies are converted to probabilities via softmax with temperature τ, then to log-odds PWM.

---

## Bundled Rosetta files

The following files from the `dbp_design` scoring protocol are bundled inside the package and require **no external installation**:

| Bundled file | Purpose |
|---|---|
| `RM8B_flags` | `pyrosetta.init()` flags (atom charges, DNA parameters, score corrections) |
| `RM8B_torsional.wts` | Score function weights (beta_nov16, DNA dihedrals) |
| `no_ref.rosettacon2018.beta_nov16_constrained.txt` | FastRelax ramp script |
| `_xml_loader.py` | RosettaScripts XML protocol (ddg filter, FastRelax mover) |

---

## Citation / acknowledgements

If you use this tool, please cite the relevant PyRosetta and AlphaFold 3 papers.
The ΔΔG protocol follows the `eval_pdb.py` workflow from the `dbp_design` pipeline.
