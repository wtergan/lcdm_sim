"""HDF5 snapshot serialization for PM simulation states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .types import MeshField, ParticleState, Snapshot


class HDF5IOError(RuntimeError):
    """Raised when HDF5 snapshot I/O fails or h5py is unavailable."""


def _require_h5py():
    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on env
        raise HDF5IOError("h5py is required for snapshot I/O") from exc
    return h5py


def _coerce_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    return dict(metadata)


def save_snapshot_hdf5(
    path: str | Path,
    snapshot: Snapshot,
    metadata: dict[str, Any] | None = None,
) -> Path:
    h5py = _require_h5py()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    merged_metadata = _coerce_metadata(snapshot.metadata)
    merged_metadata.update(_coerce_metadata(metadata))

    with h5py.File(out_path, "w") as f:
        f.attrs["schema_version"] = 1
        f.attrs["step"] = int(snapshot.step)
        f.attrs["a"] = float(snapshot.particle_state.a)
        f.attrs["mass"] = float(snapshot.particle_state.mass)
        f.attrs["metadata_json"] = json.dumps(merged_metadata)

        pgrp = f.create_group("particles")
        pgrp.create_dataset(
            "positions", data=np.asarray(snapshot.particle_state.positions, dtype=float)
        )
        pgrp.create_dataset(
            "velocities",
            data=np.asarray(snapshot.particle_state.velocities, dtype=float),
        )

        if snapshot.density_field is not None:
            dgrp = f.create_group("density")
            dgrp.create_dataset(
                "data", data=np.asarray(snapshot.density_field.data, dtype=float)
            )
            dgrp.attrs["box_size_mpc_h"] = float(snapshot.density_field.box_size_mpc_h)
            dgrp.attrs["units"] = str(snapshot.density_field.units)

    return out_path


def load_snapshot_hdf5(path: str | Path) -> Snapshot:
    h5py = _require_h5py()
    in_path = Path(path)

    with h5py.File(in_path, "r") as f:
        step = int(f.attrs["step"])
        a = float(f.attrs["a"])
        mass = float(f.attrs.get("mass", 1.0))
        metadata_json = f.attrs.get("metadata_json", "{}")
        if isinstance(metadata_json, bytes):
            metadata_json = metadata_json.decode("utf-8")
        metadata = json.loads(str(metadata_json)) if metadata_json else {}

        positions = np.asarray(f["particles"]["positions"], dtype=float)
        velocities = np.asarray(f["particles"]["velocities"], dtype=float)
        state = ParticleState(
            positions=positions, velocities=velocities, mass=mass, a=a
        )

        density_field = None
        if "density" in f:
            dgrp = f["density"]
            density_field = MeshField(
                data=np.asarray(dgrp["data"], dtype=float),
                box_size_mpc_h=float(dgrp.attrs["box_size_mpc_h"]),
                units=str(dgrp.attrs.get("units", "overdensity")),
            )

    return Snapshot(
        step=step, particle_state=state, density_field=density_field, metadata=metadata
    )
