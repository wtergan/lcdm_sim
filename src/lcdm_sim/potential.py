"""Poisson solver for gravitational potential on the PM grid."""

from __future__ import annotations


from ._fourier import rfft_wavenumber_meshes, safe_inverse_k2
from .fft_backend import FFTBackend, get_fft_backend
from .types import MeshField, SimulationConfig


def _resolve_backend(
    config: SimulationConfig, fft_backend: FFTBackend | None
) -> FFTBackend:
    if fft_backend is not None:
        return fft_backend
    return get_fft_backend(
        config.performance.fft_backend, workers=config.performance.fft_workers
    )


def solve_poisson_potential(
    delta: MeshField,
    config: SimulationConfig,
    fft_backend: FFTBackend | None = None,
) -> MeshField:
    backend = _resolve_backend(config, fft_backend)
    n = delta.n_grid_1d
    L = delta.box_size_mpc_h

    delta_k = backend.rfftn(delta.data)
    kx, ky, kz = rfft_wavenumber_meshes(n, L, backend)
    inv_k2 = safe_inverse_k2(kx, ky, kz)

    phi_k = -delta_k * inv_k2
    phi_k[0, 0, 0] = 0.0 + 0.0j
    phi = backend.irfftn(phi_k, s=delta.data.shape).real.astype(float)
    phi -= float(phi.mean())
    return MeshField(data=phi, box_size_mpc_h=L, units="potential_comoving")
