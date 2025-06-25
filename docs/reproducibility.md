# Reproducibility Guide

## Environment
Use the existing `uv`-managed environment for this project:

```bash
source /home/gilgames/envs/lcdm-sim/bin/activate
```

## Run Tests
From the repo root (or the `lcdm-revival` worktree):

```bash
PYTHONPATH=src pytest -q tests
```

## Run a Smoke Simulation (Python API)
```bash
PYTHONPATH=src python -m lcdm_sim.cli run --config configs/smoke.yaml
```

## Notebook Usage
The refactored notebooks in `notebooks/` are package-backed. They add `src/` to `sys.path` and import from `lcdm_sim` instead of re-defining core PM functions inline.

## Determinism Notes
- GRF generation is deterministic for a fixed `random_seed` in the config.
- Validation and simulation tests use small fixed-size configs for stable runtime and repeatable output.
