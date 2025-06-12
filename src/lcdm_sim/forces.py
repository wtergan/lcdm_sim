"""Acceleration-grid computation and comoving->physical conversion helpers."""

from __future__ import annotations


from ._fourier import rfft_wavenumber_meshes, safe_inverse_k2
from .fft_backend import FFTBackend, get_fft_backend
from .types import AccelerationField, MeshField, SimulationConfig


def _resolve_backend(
    config: SimulationConfig, fft_backend: FFTBackend | None
) -> FFTBackend:
    if fft_backend is not None:
        return fft_backend
    return get_fft_backend(
        config.performance.fft_backend, workers=config.performance.fft_workers
    )


def compute_comoving_acceleration_grid(
    delta: MeshField,
    config: SimulationConfig,
    fft_backend: FFTBackend | None = None,
) -> AccelerationField:
    backend = _resolve_backend(config, fft_backend)
    n = delta.n_grid_1d
    L = delta.box_size_mpc_h

    delta_k = backend.rfftn(delta.data)
    kx, ky, kz = rfft_wavenumber_meshes(n, L, backend)
    inv_k2 = safe_inverse_k2(kx, ky, kz)

    acc_components = []
    for kcomp in (kx, ky, kz):
        acc_k = 1j * kcomp * inv_k2 * delta_k
        acc = backend.irfftn(acc_k, s=delta.data.shape).real.astype(float)
        acc_components.append(acc)

    return AccelerationField(
        ax=acc_components[0],
        ay=acc_components[1],
        az=acc_components[2],
        box_size_mpc_h=L,
        units="comoving",
    )


def convert_comoving_to_physical_acceleration(
    accel: AccelerationField,
    a: float,
    power: float = 2.0,
) -> AccelerationField:
    if a <= 0:
        raise ValueError("Scale factor a must be positive")
    factor = 1.0 / (float(a) ** float(power))
    return AccelerationField(
        ax=accel.ax * factor,
        ay=accel.ay * factor,
        az=accel.az * factor,
        box_size_mpc_h=accel.box_size_mpc_h,
        units="physical",
    )
