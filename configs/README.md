# Simulation Configurations

Configuration presets are grouped by intended use:

```text
configs/
├── smoke.yaml              # short local health check
├── gallery/                # matched README and benchmark gallery runs
│   ├── gallery_32.yaml     # lightweight gallery
│   ├── gallery_64.yaml     # intermediate benchmark
│   └── gallery_128.yaml    # README showcase
└── scaling/                # larger exploratory run presets
    ├── medium.yaml
    └── large_local.yaml
```

The three `gallery/` configurations keep the physical box, time interval,
snapshot count, and random seed aligned so resolution timing comparisons remain
meaningful.
