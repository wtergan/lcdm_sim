# Documentation

This directory holds reproducibility notes and versioned result assets for the
package-backed LCDM particle-mesh simulation.

## Contents

- `reproducibility.md`: environment, test, smoke-run, and validation workflow.
- `assets/simulations/gallery-32/`: lightweight deterministic gallery generated
  from `configs/gallery/gallery_32.yaml`.
- `assets/simulations/gallery-128/`: README showcase figures and analysis
  generated from the validated `configs/gallery/gallery_128.yaml` run.

The gallery images are committed so the README can show the actual simulation
output without requiring readers to run the code first.

Each simulation gallery uses the same generated layout:

```text
gallery-<resolution>/
├── plot_manifest.json
├── summaries/                # evolution and run-level overview plots
├── snapshots/
│   ├── density/              # density slices and projections by step
│   └── particles/            # projected particle positions by step
└── analysis/                 # machine-readable quantitative summaries
```
