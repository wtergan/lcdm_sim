# `lcdm_sim`

Modular LCDM particle-mesh (PM) simulation toolkit for learning, experimentation,
and incremental engine development.

This project is a package-first refactor of earlier notebook-based PM simulation
work. The goal is to preserve the teaching value of the notebooks while moving
the core physics and simulation pipeline into testable, reusable Python modules.

## Project Goals

- Build a modular LCDM PM simulation engine (GRF -> Zel'dovich ICs -> CIC ->
  Poisson/forces -> leapfrog integration).
- Keep the physics pipeline inspectable and easy to validate.
- Provide diagnostics and plotting utilities (static + interactive).
- Support package-backed teaching notebooks instead of monolithic notebook code.
- Prepare for later browser-emulator/export workflows.

## Current Status

Implemented on this branch:

- Phase 1: package skeleton, typed config, CLI scaffold
- Phase 2: physics core (`cosmology`, `spectra`, `grf`, `zeldovich`, `cic`,
  `potential`, `forces`, FFT abstraction)
- Phase 3: integrator + orchestration + HDF5 snapshot I/O
- Phase 4: diagnostics + static/interactive plotting
- Validation module and CLI `validate` integration
- Package-backed teaching notebooks in `notebooks/`

Still in progress / future work:

- CLI `plot` command implementation
- CLI `export-web-dataset` implementation
- richer reference-comparison validation modes
- browser-emulator dataset/export workflow polish

## Codebase Structure

```text
lcdm_sim/
├── configs/                     # Preset simulation configs (smoke/medium/large)
├── docs/                        # Reproducibility and project docs
├── notebooks/                   # Package-backed teaching notebooks
├── src/lcdm_sim/
│   ├── config.py                # Typed config loading
│   ├── types.py                 # Shared dataclasses (fields, particles, runs)
│   ├── cosmology.py             # H(a), growth factor/rate
│   ├── spectra.py               # Transfer function + P(k) utilities
│   ├── fft_backend.py           # SciPy / optional pyFFTW FFT backend wrapper
│   ├── grf.py                   # Gaussian random field generation
│   ├── zeldovich.py             # Zel'dovich IC generation
│   ├── cic.py                   # CIC density deposition / force gather
│   ├── potential.py             # Poisson solve
│   ├── forces.py                # Acceleration grids / unit conversion helpers
│   ├── integrators.py           # KDK leapfrog in scale factor a
│   ├── simulation.py            # End-to-end orchestration
│   ├── io_hdf5.py               # Snapshot save/load
│   ├── diagnostics.py           # Stats + comparisons + power spectrum estimates
│   ├── plotting_static.py       # Matplotlib plots
│   ├── plotting_interactive.py  # Plotly plots (optional dependency)
│   ├── validation.py            # Validation suite + run-dir validation
│   └── cli.py                   # CLI entrypoints
└── tests/                       # Phase-by-phase test coverage
```

## Dependencies / Requirements

### Python

- Python `>=3.11`

### Runtime dependencies (current branch)

The code currently relies on a manually managed environment (the `pyproject.toml`
metadata is intentionally minimal at this stage). Typical runtime/testing
dependencies used on this branch:

- `numpy`
- `scipy`
- `PyYAML`
- `h5py` (for snapshot I/O and validation from run directories)
- `matplotlib` (static plots)
- `plotly` (interactive plots)

Optional / performance-oriented:

- `pyfftw` (optional FFT backend)
- `numba` (reserved for future acceleration paths)

Development / testing:

- `pytest`
- `ruff`
- `nbformat` (used to generate/check notebooks in tests/tooling)

### Versioning note

This branch may be tested against a deliberately pinned historical dependency set.
Do not assume the latest package versions are required. Prefer adapting code to
the existing environment for reproducibility unless an upgrade is explicitly
requested.

## Quickstart

### 1) Create/activate an environment (example with `uv`)

```bash
uv venv .venv
source .venv/bin/activate
```

### 2) Install project and supporting libraries

Minimal editable install:

```bash
uv pip install -e .
```

Typical local development/test environment (example):

```bash
uv pip install numpy scipy pyyaml h5py matplotlib plotly pytest ruff nbformat pyfftw numba
```

### 3) Run tests

```bash
PYTHONPATH=src pytest -q tests
```

## Usage (Current)

### CLI scaffold

```bash
PYTHONPATH=src python -m lcdm_sim.cli --help
PYTHONPATH=src python -m lcdm_sim.cli run --config configs/smoke.yaml
PYTHONPATH=src python -m lcdm_sim.cli validate --run-dir <run_dir>
```

Notes:

- `validate` is implemented and writes a JSON report under `metrics/`.
- `plot` and `export-web-dataset` commands are currently scaffolds/placeholders.

### Python API (end-to-end example)

```python
from pathlib import Path
from lcdm_sim.config import load_simulation_config
from lcdm_sim.simulation import run_simulation
from lcdm_sim.validation import run_validation_suite

cfg = load_simulation_config("configs/smoke.yaml")
result = run_simulation(cfg, output_dir=Path("outputs/demo"), num_snapshots=3, save_snapshots=True)
report = run_validation_suite(result, cfg)
print(report.ok, report.summary)
```

## Outputs

When snapshot saving is enabled, runs write HDF5 snapshots under:

- `<run_dir>/snapshots/snapshot_XXXX.h5`

Validation reports are written to:

- `<run_dir>/metrics/validation_report.json`

Plot helpers write PNG/HTML artifacts to user-specified paths.

## Notebooks

See `notebooks/README.md` for the package-backed teaching notebooks. These
notebooks import from `lcdm_sim` and are intended to replace inline/monolithic
implementations for ongoing development and learning.

## Reproducibility

See `docs/reproducibility.md` for an environment-agnostic workflow covering:

- environment setup
- test execution
- smoke runs
- deterministic behavior notes

## Testing

The test suite is organized by implementation phase and validates:

- config/CLI scaffolding
- physics core correctness and shape/finite checks
- integrator + orchestration + HDF5 I/O
- diagnostics + plotting artifact generation
- validation suite behavior
- notebook refactor presence/import usage
