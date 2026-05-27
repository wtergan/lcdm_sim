"""Browser dataset export for validated LCDM density-volume playback.

This module converts saved density snapshots into compact byte volumes plus a
versioned manifest. It does not run or modify the particle-mesh simulation;
the consumer is expected to display provenance and validate schema support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .io_hdf5 import load_snapshot_hdf5


class WebDatasetExportError(RuntimeError):
    """Raised when a saved run cannot be exported as a browser dataset."""


def _read_json_if_present(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _redshift(a: float) -> float:
    return (1.0 / a) - 1.0


def _quantize_density_volumes(
    density_arrays: list[np.ndarray],
) -> tuple[list[np.ndarray], dict[str, float | str]]:
    floor = -0.999999
    transformed = [np.log1p(np.maximum(array, floor)) for array in density_arrays]
    transformed_min = float(min(float(array.min()) for array in transformed))
    transformed_max = float(max(float(array.max()) for array in transformed))
    span = transformed_max - transformed_min
    if not np.isfinite(span) or span <= 0.0:
        raise WebDatasetExportError("density snapshots have no finite exportable range")

    quantized = [
        np.rint((array - transformed_min) * (255.0 / span)).astype(np.uint8)
        for array in transformed
    ]
    transform: dict[str, float | str] = {
        "name": "log1p_overdensity",
        "input_floor": floor,
        "transformed_min": transformed_min,
        "transformed_max": transformed_max,
    }
    return quantized, transform


def export_web_dataset(run_dir: str | Path, output_dir: str | Path) -> Path:
    """Export ordered density snapshots into a versioned browser bundle."""
    source_dir = Path(run_dir)
    destination = Path(output_dir)
    snapshot_paths = sorted((source_dir / "snapshots").glob("snapshot_*.h5"))
    if not snapshot_paths:
        raise WebDatasetExportError(
            f"no snapshots found under {source_dir / 'snapshots'}"
        )

    snapshots = [load_snapshot_hdf5(path) for path in snapshot_paths]
    if any(snapshot.density_field is None for snapshot in snapshots):
        raise WebDatasetExportError("all exported snapshots must contain density data")

    metrics_dir = source_dir / "metrics"
    run_summary = _read_json_if_present(metrics_dir / "run_summary.json")
    validation_report = _read_json_if_present(metrics_dir / "validation_report.json")
    if not isinstance(run_summary, dict) or not run_summary.get("run_id"):
        raise WebDatasetExportError("reference export requires a validated run summary")
    if (
        not isinstance(validation_report, dict)
        or validation_report.get("ok") is not True
    ):
        raise WebDatasetExportError(
            "reference export requires validated run provenance"
        )
    run_id = str(run_summary["run_id"])
    if any(snapshot.metadata.get("run_id") != run_id for snapshot in snapshots):
        raise WebDatasetExportError(
            "snapshot run id does not match validated run summary"
        )

    density_arrays = [
        np.asarray(snapshot.density_field.data, dtype=np.float32)
        for snapshot in snapshots
        if snapshot.density_field is not None
    ]
    dimensions = list(density_arrays[0].shape)
    if any(list(array.shape) != dimensions for array in density_arrays):
        raise WebDatasetExportError("density snapshot dimensions are inconsistent")

    quantized_arrays, transform = _quantize_density_volumes(density_arrays)

    frames_dir = destination / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    for index, (snapshot, density, volume) in enumerate(
        zip(snapshots, density_arrays, quantized_arrays)
    ):
        relative_path = Path("frames") / f"density_{index:04d}.u8"
        payload = volume.tobytes(order="C")
        (destination / relative_path).write_bytes(payload)
        frames.append(
            {
                "index": index,
                "step": int(snapshot.step),
                "a": float(snapshot.a),
                "z": _redshift(float(snapshot.a)),
                "path": relative_path.as_posix(),
                "byte_length": len(payload),
                "density_std": float(np.std(density)),
            }
        )

    density_field = snapshots[0].density_field
    if density_field is None:  # pragma: no cover - guarded before export
        raise WebDatasetExportError("density data disappeared during export")
    manifest = {
        "schema_version": 1,
        "format": "lcdm-density-volume",
        "scenario_id": source_dir.name,
        "provenance": {
            "source": "lcdm_sim",
            "run_id": run_id,
            "config": run_summary.get("config"),
            "validation": {
                "status": "validated",
                "summary": validation_report.get("summary"),
            },
        },
        "volume": {
            "dimensions": dimensions,
            "box_size_mpc_h": float(density_field.box_size_mpc_h),
            "units": str(density_field.units),
            "voxel_encoding": "uint8",
            "layout": "C",
            "scalar_transform": transform,
        },
        "frames": frames,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return manifest_path
