"""Simulation orchestration for the modular PM pipeline."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from uuid import uuid4

import numpy as np

from .cic import deposit_density, interpolate_forces_to_particles
from .fft_backend import get_fft_backend
from .forces import compute_comoving_acceleration_grid
from .grf import generate_grf
from .integrators import leapfrog_kdk_a_step
from .io_hdf5 import save_snapshot_hdf5
from .types import RunResult, Snapshot
from .zeldovich import initial_conditions_from_density


def _snapshot_step_indices(num_steps: int, num_snapshots: int | None) -> list[int]:
    if num_steps < 0:
        raise ValueError("num_steps must be non-negative")
    if num_snapshots is None:
        num_snapshots = min(num_steps + 1, 5)
    num = max(2, min(int(num_snapshots), num_steps + 1)) if num_steps > 0 else 1
    indices = np.linspace(0, num_steps, num=num, dtype=int)
    return sorted(set(int(i) for i in indices))


def _make_run_id() -> str:
    return f"run-{uuid4().hex[:10]}"


def _snapshot_path(output_dir: Path, step: int) -> Path:
    return output_dir / "snapshots" / f"snapshot_{step:04d}.h5"


def _history_row(
    step: int, state, density, elapsed_s: float, step_time_s: float
) -> dict[str, float | int]:
    speeds = np.linalg.norm(state.velocities, axis=1)
    return {
        "step": int(step),
        "a": float(state.a),
        "elapsed_s": float(elapsed_s),
        "step_time_s": float(step_time_s),
        "density_mean": float(np.mean(density.data)),
        "density_std": float(np.std(density.data)),
        "velocity_rms": float(np.sqrt(np.mean(speeds * speeds)))
        if speeds.size
        else 0.0,
    }


def run_simulation(
    config,
    *,
    output_dir: str | Path | None = None,
    num_snapshots: int | None = None,
    history_stride: int = 1,
    save_snapshots: bool = False,
):
    """Run an end-to-end PM simulation using the current core modules.

    Returns a `RunResult` with thinned history and scheduled snapshots.
    """

    if history_stride <= 0:
        raise ValueError("history_stride must be positive")

    run_id = _make_run_id()
    out_dir_path: Path | None = None
    if output_dir is not None:
        out_dir_path = Path(output_dir)
        (out_dir_path / "snapshots").mkdir(parents=True, exist_ok=True)

    backend = get_fft_backend(
        config.performance.fft_backend,
        workers=config.performance.fft_workers,
    )

    total_t0 = perf_counter()
    initial_density = generate_grf(config, fft_backend=backend)
    state = initial_conditions_from_density(
        initial_density, config, fft_backend=backend
    )
    initial_state = state

    snapshot_steps = set(
        _snapshot_step_indices(config.integrator.num_steps, num_snapshots)
    )
    snapshots: list[Snapshot] = []
    history: list[dict[str, float | int]] = []
    step_durations_s: list[float] = []

    def particle_accel_fn(current_state):
        delta = deposit_density(current_state, config.grid)
        accel_grid = compute_comoving_acceleration_grid(
            delta, config, fft_backend=backend
        )
        return interpolate_forces_to_particles(accel_grid, current_state, config.grid)

    def maybe_capture(step: int, *, step_time_s: float):
        if (
            step not in snapshot_steps
            and (step % history_stride) != 0
            and step != config.integrator.num_steps
        ):
            return
        density = deposit_density(state, config.grid)
        elapsed = perf_counter() - total_t0
        if step in snapshot_steps:
            density_for_snap = density if config.output.save_density else None
            snapshot = Snapshot(
                step=int(step),
                particle_state=state,
                density_field=density_for_snap,
                metadata={"run_id": run_id, "fft_backend": backend.name},
            )
            snapshots.append(snapshot)
            if save_snapshots and out_dir_path is not None:
                save_snapshot_hdf5(
                    _snapshot_path(out_dir_path, step),
                    snapshot,
                    metadata={
                        "num_steps": int(config.integrator.num_steps),
                        "random_seed": int(config.random_seed),
                    },
                )
        if step % history_stride == 0 or step == config.integrator.num_steps:
            history.append(
                _history_row(
                    step, state, density, elapsed_s=elapsed, step_time_s=step_time_s
                )
            )

    # Step 0 capture (initial conditions before integration)
    maybe_capture(0, step_time_s=0.0)

    for step in range(1, int(config.integrator.num_steps) + 1):
        step_t0 = perf_counter()
        state = leapfrog_kdk_a_step(state, config, particle_accel_fn)
        step_dt = perf_counter() - step_t0
        step_durations_s.append(float(max(step_dt, 0.0)))
        maybe_capture(step, step_time_s=step_dt)

    total_runtime_s = perf_counter() - total_t0
    return RunResult(
        config=config,
        initial_density=initial_density,
        initial_state=initial_state,
        snapshots=snapshots,
        history=history,
        step_durations_s=step_durations_s,
        total_runtime_s=float(max(total_runtime_s, 0.0)),
        run_id=run_id,
        output_dir=str(out_dir_path) if out_dir_path is not None else None,
    )
