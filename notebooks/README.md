# `notebooks`

Package-backed teaching notebooks for the modular `lcdm_sim` particle-mesh
pipeline.

## Purpose

These notebooks are the learning-facing companion to the package code in
`src/lcdm_sim/`. They demonstrate the simulation workflow while importing the
engine modules instead of re-implementing the core PM logic inline.

## Notebook Sequence

- `01_intro_parameters.ipynb`
  - typed config objects
  - grid parameters
  - LCDM cosmology helpers (`H(a)`, `D(a)`, `f(a)`)
- `02_initial_conditions.ipynb`
  - transfer function / power spectrum usage
  - GRF generation
  - Zel'dovich initial conditions
- `03_cic_forces_and_diagnostics.ipynb`
  - CIC density deposition
  - force interpolation
  - diagnostics and quick visual checks
- `04_full_pm_simulation.ipynb`
  - end-to-end simulation run
  - validation suite usage
  - plotting helpers for output artifacts

## Usage Notes

- Run from the repo root or from the `notebooks/` directory.
- Each notebook includes a setup cell that resolves `src/` and adds it to
  `sys.path`.
- These notebooks are intended to evolve with the package API; if an API changes,
  update the notebooks to import the new interface rather than duplicating logic.
