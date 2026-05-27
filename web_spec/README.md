# Web Dataset Contract

`lcdm_sim export-web-dataset` produces compact visualization bundles for
reference playback in `lcdm_explorer`. It does not export raw HDF5 snapshots
and it does not claim that a browser renderer reproduces the PM computation.

## Version 1 Bundle

```text
<output>/
  manifest.json
  frames/
    density_0000.u8
    density_0001.u8
    ...
```

`manifest.json` uses `schema_version: 1` and
`format: "lcdm-density-volume"`. It contains:

- required source-run provenance and affirmative validation summary;
- shared volume dimensions, box size, units, byte encoding and scalar
  transform metadata;
- ordered frame entries with simulation step, scale factor `a`, derived
  redshift `z`, relative byte-file path and payload size;
- density standard deviation computed directly from each exported frame.

## Density Encoding

Each frame is a C-order, single-channel `uint8` volume with exactly one byte
per grid cell. The dataset applies one global run-level mapping:

```text
transformed = log1p(max(overdensity, -0.999999))
encoded = round(255 * (transformed - transformed_min)
                / (transformed_max - transformed_min))
```

The manifest records the floor and transformed range required to interpret the
bytes. This encoding is a compact Explore visualization candidate; browser
visual checks should decide whether a higher precision export is warranted.

## Failure Boundary

Export returns a non-zero CLI status when the run contains no snapshots,
missing density snapshots, inconsistent density dimensions, no finite
quantization range, absent/failed validation metadata, or snapshots whose run
identifier does not match the validated run summary. The browser consumer must
separately reject unsupported schema versions and unapproved provenance.
