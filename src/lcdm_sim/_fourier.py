"""Internal Fourier-grid helpers shared across modules."""

from __future__ import annotations

import numpy as np

from .fft_backend import FFTBackend


def rfft_wavenumber_meshes(
    n: int, box_size_mpc_h: float, backend: FFTBackend
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = box_size_mpc_h / float(n)
    kx_1d = 2.0 * np.pi * backend.fftfreq(n, d=dx)
    ky_1d = 2.0 * np.pi * backend.fftfreq(n, d=dx)
    kz_1d = 2.0 * np.pi * backend.rfftfreq(n, d=dx)
    return np.meshgrid(kx_1d, ky_1d, kz_1d, indexing="ij")


def safe_inverse_k2(kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> np.ndarray:
    k2 = kx * kx + ky * ky + kz * kz
    return np.divide(1.0, k2, out=np.zeros_like(k2, dtype=float), where=k2 > 0)
