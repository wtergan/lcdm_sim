"""Gaussian random field generation for LCDM initial density fluctuations."""

from __future__ import annotations

import numpy as np

from ._fourier import rfft_wavenumber_meshes
from .fft_backend import FFTBackend, get_fft_backend
from .spectra import power_spectrum
from .types import MeshField, SimulationConfig


def _resolve_backend(
    config: SimulationConfig, fft_backend: FFTBackend | None
) -> FFTBackend:
    if fft_backend is not None:
        return fft_backend
    return get_fft_backend(
        config.performance.fft_backend, workers=config.performance.fft_workers
    )


def _resolve_rng(
    config: SimulationConfig, rng: np.random.Generator | int | None
) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        return np.random.default_rng(config.random_seed)
    return np.random.default_rng(int(rng))


def generate_grf(
    config: SimulationConfig,
    fft_backend: FFTBackend | None = None,
    rng: np.random.Generator | int | None = None,
) -> MeshField:
    """Generate a real-space overdensity field using a sigma8-normalized power spectrum.

    This uses a complex half-spectrum (rFFT shape), inverse FFTs to real space,
    and then rescales the realization variance to the configured sigma8 as a
    practical early-development normalization proxy.
    """

    backend = _resolve_backend(config, fft_backend)
    prng = _resolve_rng(config, rng)

    n = config.grid.n_grid_1d
    L = config.grid.box_size_mpc_h
    half_shape = (n, n, n // 2 + 1)

    kx, ky, kz = rfft_wavenumber_meshes(n, L, backend)
    k_mag = np.sqrt(kx * kx + ky * ky + kz * kz)
    pk = power_spectrum(k_mag, config.cosmology, a=config.cosmology.a_initial)

    noise = (
        prng.normal(size=half_shape) + 1j * prng.normal(size=half_shape)
    ) / np.sqrt(2.0)

    volume = L**3
    amplitude = np.sqrt(np.clip(pk / max(volume, 1e-30), 0.0, None))
    delta_k = noise * amplitude
    delta_k[0, 0, 0] = 0.0 + 0.0j

    # Nyquist/zero planes should be purely real for strict Hermitian consistency.
    delta_k[:, :, 0] = delta_k[:, :, 0].real + 0.0j
    if n % 2 == 0:
        delta_k[:, :, -1] = delta_k[:, :, -1].real + 0.0j

    delta = backend.irfftn(delta_k, s=(n, n, n)).astype(float)
    delta -= float(np.mean(delta))

    std = float(np.std(delta))
    if std > 0 and np.isfinite(std):
        delta *= float(config.cosmology.sigma8) / std

    return MeshField(data=delta, box_size_mpc_h=L, units="overdensity")
