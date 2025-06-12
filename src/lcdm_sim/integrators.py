"""Time integration routines for the PM simulation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .cosmology import hubble
from .types import ParticleState, SimulationConfig


ParticleAccelerationFn = Callable[[ParticleState], np.ndarray]


def step_size_a(config: SimulationConfig) -> float:
    """Return the canonical linear step size in scale factor a."""

    n = int(config.integrator.num_steps)
    if n <= 0:
        raise ValueError("Integrator num_steps must be positive.")
    a0 = float(config.cosmology.a_initial)
    a1 = float(config.cosmology.a_final)
    if a1 <= a0:
        raise ValueError("cosmology.a_final must be greater than a_initial.")
    return (a1 - a0) / n


def _dadt(a: float, config: SimulationConfig) -> float:
    h = float(hubble(a, config.cosmology))
    return a * h


def _dx_da(velocities: np.ndarray, a: float, config: SimulationConfig) -> np.ndarray:
    dadt = _dadt(a, config)
    if dadt <= 0 or not np.isfinite(dadt):
        raise ValueError("Invalid da/dt conversion for drift step.")
    return velocities / dadt


def _dv_da(accelerations: np.ndarray, a: float, config: SimulationConfig) -> np.ndarray:
    dadt = _dadt(a, config)
    if dadt <= 0 or not np.isfinite(dadt):
        raise ValueError("Invalid da/dt conversion for kick step.")
    return accelerations / dadt


def _wrap_positions(positions: np.ndarray, box_size_mpc_h: float) -> np.ndarray:
    return np.mod(positions, float(box_size_mpc_h))


def leapfrog_kdk_a_step(
    state: ParticleState,
    config: SimulationConfig,
    accel_fn: ParticleAccelerationFn,
    da: float | None = None,
) -> ParticleState:
    """Advance one KDK leapfrog step in scale factor a.

    The state stores comoving positions and velocities (dx/dt). The acceleration
    function is expected to return comoving accelerations on particles for the
    supplied state. The drift/kick are performed using dt = da / (a H(a)).
    """

    da_use = float(step_size_a(config) if da is None else da)
    if da_use <= 0:
        raise ValueError("da must be positive")

    a0 = float(state.a)
    a1 = a0 + da_use
    a_max = float(config.cosmology.a_final)
    if a1 > a_max:
        da_use = a_max - a0
        a1 = a_max
    if da_use <= 0:
        return state

    acc0 = np.asarray(accel_fn(state), dtype=float)
    if acc0.shape != state.positions.shape:
        raise ValueError("accel_fn must return array shaped like state.positions")

    v_half = state.velocities + 0.5 * da_use * _dv_da(acc0, a0, config)

    a_mid = a0 + 0.5 * da_use
    x_new = state.positions + da_use * _dx_da(v_half, a_mid, config)
    x_new = _wrap_positions(x_new, config.grid.box_size_mpc_h)

    probe_state = ParticleState(
        positions=x_new,
        velocities=v_half,
        mass=state.mass,
        a=a1,
    )
    acc1 = np.asarray(accel_fn(probe_state), dtype=float)
    if acc1.shape != state.positions.shape:
        raise ValueError("accel_fn must return array shaped like state.positions")

    v_new = v_half + 0.5 * da_use * _dv_da(acc1, a1, config)
    return ParticleState(positions=x_new, velocities=v_new, mass=state.mass, a=a1)
