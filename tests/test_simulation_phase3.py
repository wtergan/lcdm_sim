from pathlib import Path

import pytest


pytest.importorskip("h5py")


def test_run_simulation_generates_snapshots_history_and_hdf5_outputs(
    tiny_config, tmp_path
):
    from lcdm_sim.simulation import run_simulation

    result = run_simulation(
        tiny_config,
        output_dir=tmp_path,
        num_snapshots=3,
        history_stride=2,
        save_snapshots=True,
    )

    assert result.initial_density.data.shape == (tiny_config.grid.n_grid_1d,) * 3
    assert result.initial_state.positions.shape[1] == 3
    assert len(result.step_durations_s) == tiny_config.integrator.num_steps
    assert all(dt >= 0.0 for dt in result.step_durations_s)
    assert len(result.snapshots) >= 3
    assert result.snapshots[0].step == 0
    assert result.snapshots[-1].step == tiny_config.integrator.num_steps
    assert len(result.history) >= 2
    assert result.output_dir is not None

    snap_dir = Path(result.output_dir) / "snapshots"
    files = sorted(snap_dir.glob("snapshot_*.h5"))
    assert len(files) == len(result.snapshots)
