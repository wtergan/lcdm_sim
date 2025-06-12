"""Cloud-in-cell density deposition and grid-to-particle interpolation."""

from __future__ import annotations

import numpy as np

from ._sampling import periodic_trilinear_sample_vector
from .types import AccelerationField, GridConfig, MeshField, ParticleState


def _cic_base_indices(positions: np.ndarray, n: int, box_size_mpc_h: float):
    dx = box_size_mpc_h / float(n)
    grid_coords = np.mod(positions, box_size_mpc_h) / dx
    base = np.floor(grid_coords).astype(np.int64)
    frac = grid_coords - base
    base %= n
    nxt = (base + 1) % n
    return base, nxt, frac


def deposit_density(
    particles: ParticleState, grid_cfg: GridConfig, scheme: str = "cic"
) -> MeshField:
    if scheme.lower() != "cic":
        raise ValueError("Only CIC scheme is currently supported")

    n = int(grid_cfg.n_grid_1d)
    L = float(grid_cfg.box_size_mpc_h)
    base, nxt, frac = _cic_base_indices(particles.positions, n, L)
    tx, ty, tz = frac[:, 0], frac[:, 1], frac[:, 2]

    mass_grid = np.zeros((n, n, n), dtype=float)
    mass = float(particles.mass)

    for ox in (0, 1):
        wx = tx if ox else (1.0 - tx)
        ix = nxt[:, 0] if ox else base[:, 0]
        for oy in (0, 1):
            wy = ty if oy else (1.0 - ty)
            iy = nxt[:, 1] if oy else base[:, 1]
            for oz in (0, 1):
                wz = tz if oz else (1.0 - tz)
                iz = nxt[:, 2] if oz else base[:, 2]
                weights = mass * wx * wy * wz
                np.add.at(mass_grid, (ix, iy, iz), weights)

    mean_mass = float(mass_grid.mean())
    overdensity = (
        np.divide(
            mass_grid, mean_mass, out=np.zeros_like(mass_grid), where=mean_mass != 0.0
        )
        - 1.0
    )
    overdensity -= float(overdensity.mean())
    return MeshField(data=overdensity, box_size_mpc_h=L, units="overdensity")


def interpolate_forces_to_particles(
    accel_grid: AccelerationField | np.ndarray,
    particles: ParticleState,
    grid_cfg: GridConfig,
    scheme: str = "cic",
) -> np.ndarray:
    if scheme.lower() != "cic":
        raise ValueError("Only CIC scheme is currently supported")

    if isinstance(accel_grid, AccelerationField):
        field = accel_grid.stacked()
    else:
        field = np.asarray(accel_grid, dtype=float)
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError(
            "accel_grid must be AccelerationField or array with shape (N,N,N,3)"
        )

    return periodic_trilinear_sample_vector(
        field, particles.positions, float(grid_cfg.box_size_mpc_h)
    )
