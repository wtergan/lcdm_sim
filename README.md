# lcdm_sim

Modular LCDM particle-mesh simulation toolkit (work in progress).

## Current state
- Phases 1-4 core engine implemented (config, physics core, simulation loop, plotting/diagnostics)
- Validation module implemented
- Package-backed teaching notebooks added in `notebooks/`

## Quickstart
```bash
source /home/gilgames/envs/lcdm-sim/bin/activate
PYTHONPATH=src pytest -q tests
```

See `docs/reproducibility.md` for a reproducible testing workflow.
