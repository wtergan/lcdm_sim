from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("h5py")


def test_hdf5_snapshot_roundtrip_with_optional_density_and_metadata(tmp_path):
    from lcdm_sim.io_hdf5 import load_snapshot_hdf5, save_snapshot_hdf5
    from lcdm_sim.types import MeshField, ParticleState, Snapshot

    state = ParticleState(
        positions=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float),
        velocities=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float),
        mass=2.5,
        a=0.125,
    )
    density = MeshField(data=np.ones((4, 4, 4), dtype=float), box_size_mpc_h=10.0)
    snapshot = Snapshot(
        step=3, particle_state=state, density_field=density, metadata={"tag": "unit"}
    )

    path = Path(tmp_path) / "snap.h5"
    save_snapshot_hdf5(path, snapshot, metadata={"run_id": "abc123", "seed": 7})
    loaded = load_snapshot_hdf5(path)

    assert loaded.step == 3
    assert np.isclose(loaded.particle_state.a, 0.125)
    assert np.allclose(loaded.particle_state.positions, state.positions)
    assert np.allclose(loaded.particle_state.velocities, state.velocities)
    assert loaded.density_field is not None
    assert loaded.density_field.data.shape == (4, 4, 4)
    assert loaded.metadata["run_id"] == "abc123"
    assert loaded.metadata["seed"] == 7
    assert loaded.metadata["tag"] == "unit"
