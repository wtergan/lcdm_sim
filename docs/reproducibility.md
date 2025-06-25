# Reproducibility Guide

This document describes a path-agnostic workflow for reproducing test runs and
smoke simulations for the modular `lcdm_sim` package.

## Scope

Use this guide for:

- local development on a fresh machine or branch
- verifying behavior after code changes
- reproducing test and smoke-run results before committing

## Environment Setup

### Option A: `uv` (recommended)

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
uv pip install numpy scipy pyyaml h5py matplotlib plotly pytest ruff nbformat pyfftw numba
```

### Option B: `venv` + `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install numpy scipy pyyaml h5py matplotlib plotly pytest ruff nbformat pyfftw numba
```

Notes:

- The project metadata is intentionally minimal during this phase of development,
  so supporting scientific/plotting/test dependencies are installed explicitly.
- If you are using a historically pinned environment, keep those versions and
  adapt code/tests rather than upgrading packages unless required.

## Baseline Verification

From the repository root (or a Git worktree for this repository):

```bash
PYTHONPATH=src pytest -q tests
```

Expected result on a fully provisioned environment:

- All tests pass
- Plotly/HDF5 tests run (not skipped) if `plotly` and `h5py` are installed

## Smoke Runs

### CLI smoke check

```bash
PYTHONPATH=src python -m lcdm_sim.cli --help
PYTHONPATH=src python -m lcdm_sim.cli run --config configs/smoke.yaml
```

### Validation from a run directory

```bash
PYTHONPATH=src python -m lcdm_sim.cli validate --run-dir <run_dir>
```

This writes:

- `<run_dir>/metrics/validation_report.json`

## Notebook Usage

The notebooks in `notebooks/` are package-backed teaching notebooks. They are
designed to:

- import from `lcdm_sim`
- avoid re-defining core PM engine functions inline
- remain readable for physics-learning workflows

Each notebook includes a startup cell that adds `src/` to `sys.path` relative to
the current working directory (repo root or the `notebooks/` folder).

## Determinism Notes

- GRF generation is deterministic for a fixed `random_seed`.
- Validation and simulation tests use small fixed-size configs for stable runtime
  and repeatable outputs.
- Optional plotting outputs may differ in metadata (file generation timestamps)
  even when numerical results match.

## Known Environment-Specific Warnings

You may see third-party warnings depending on pinned versions (for example,
`matplotlib`/`pyparsing` deprecation warnings). These do not necessarily indicate
problems in `lcdm_sim` itself.

## Recommended Pre-Commit Check

```bash
ruff check src tests --fix
ruff format src tests
python -m compileall src tests
PYTHONPATH=src pytest -q tests
```
