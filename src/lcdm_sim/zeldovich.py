"""Zel'dovich approximation helpers for displacement and velocity initial conditions."""

from __future__ import annotations

import numpy as np

from ._fourier import rfft_wavenumber_meshes, safe_inverse_k2
from ._sampling import periodic_trilinear_sample_vector
from .cosmology import growth_rate, hubble
from .fft_backend import FFTBackend, get_fft_backend
from .types import MeshField, ParticleState, SimulationConfig


def _resolve_backend(
    config: SimulationConfig, fft_backend: FFTBackend | None
) -> FFTBackend:
    if fft_backend is not None:
        return fft_backend
    return get_fft_backend(
        config.performance.fft_backend, workers=config.performance.fft_workers
    )


def generate_particle_lattice(grid_cfg) -> np.ndarray:
    n = int(grid_cfg.n_particles_1d)
    L = float(grid_cfg.box_size_mpc_h)
    dx = L / n
    coords = (np.arange(n, dtype=float) + 0.5) * dx
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def wrap_positions(positions: np.ndarray, box_size_mpc_h: float) -> np.ndarray:
    return np.mod(positions, box_size_mpc_h)


def displacement_velocity_fields_from_density(
    delta: MeshField,
    config: SimulationConfig,
    fft_backend: FFTBackend | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    backend = _resolve_backend(config, fft_backend)
    n = delta.n_grid_1d
    L = delta.box_size_mpc_h

    delta_k = backend.rfftn(delta.data)
    kx, ky, kz = rfft_wavenumber_meshes(n, L, backend)
    inv_k2 = safe_inverse_k2(kx, ky, kz)

    disp_components = []
    for kcomp in (kx, ky, kz):
        psi_k = 1j * kcomp * inv_k2 * delta_k
        psi = backend.irfftn(psi_k, s=delta.data.shape).real.astype(float)
        disp_components.append(psi)

    displacement = np.stack(disp_components, axis=-1)

    a0 = float(config.cosmology.a_initial)
    v_scale = (
        float(hubble(a0, config.cosmology))
        * a0
        * float(growth_rate(a0, config.cosmology))
    )
    velocity = displacement * v_scale
    return displacement, velocity


def initial_conditions_from_density(
    delta: MeshField,
    config: SimulationConfig,
    rng: np.random.Generator | int | None = None,
    fft_backend: FFTBackend | None = None,
) -> ParticleState:
    del rng  # reserved for future jitter/noise options

    displacement_grid, velocity_grid = displacement_velocity_fields_from_density(
        delta, config, fft_backend=fft_backend
    )
    lattice = generate_particle_lattice(config.grid)
    L = float(config.grid.box_size_mpc_h)

    disp_particles = periodic_trilinear_sample_vector(displacement_grid, lattice, L)
    vel_particles = periodic_trilinear_sample_vector(velocity_grid, lattice, L)
    positions = wrap_positions(lattice + disp_particles, L)

    return ParticleState(
        positions=positions.astype(float),
        velocities=vel_particles.astype(float),
        mass=1.0,
        a=float(config.cosmology.a_initial),
    )
