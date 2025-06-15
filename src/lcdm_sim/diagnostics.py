"""Derived diagnostics and lightweight analysis helpers for PM outputs."""

from __future__ import annotations

import numpy as np

from .types import AccelerationField, MeshField


def density_stats(field: MeshField) -> dict[str, object]:
    data = np.asarray(field.data, dtype=float)
    return {
        "shape": list(data.shape),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "rms": float(np.sqrt(np.mean(data * data))),
        "units": field.units,
        "box_size_mpc_h": float(field.box_size_mpc_h),
    }


def acceleration_stats(accel: AccelerationField) -> dict[str, float | str]:
    mag = np.sqrt(accel.ax * accel.ax + accel.ay * accel.ay + accel.az * accel.az)
    return {
        "mag_mean": float(np.mean(mag)),
        "mag_std": float(np.std(mag)),
        "mag_rms": float(np.sqrt(np.mean(mag * mag))),
        "mag_max": float(np.max(np.abs(mag))),
        "ax_rms": float(np.sqrt(np.mean(accel.ax * accel.ax))),
        "ay_rms": float(np.sqrt(np.mean(accel.ay * accel.ay))),
        "az_rms": float(np.sqrt(np.mean(accel.az * accel.az))),
        "units": accel.units,
    }


def estimate_power_spectrum(
    field: MeshField, nbins: int = 16
) -> dict[str, list[float] | list[int]]:
    if nbins <= 0:
        raise ValueError("nbins must be positive")
    data = np.asarray(field.data, dtype=float)
    n = data.shape[0]
    L = float(field.box_size_mpc_h)
    dx = L / float(n)

    delta_k = np.fft.fftn(data)
    power = (np.abs(delta_k) ** 2) / float(data.size**2)

    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kz = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    kxg, kyg, kzg = np.meshgrid(kx, ky, kz, indexing="ij")
    kmag = np.sqrt(kxg * kxg + kyg * kyg + kzg * kzg)

    mask = kmag > 0
    k_vals = kmag[mask].ravel()
    p_vals = power[mask].ravel()
    if k_vals.size == 0:
        return {"k_centers": [], "power": [], "counts": []}

    k_edges = np.linspace(float(k_vals.min()), float(k_vals.max()), nbins + 1)
    bin_idx = np.digitize(k_vals, k_edges) - 1

    centers: list[float] = []
    pk_bins: list[float] = []
    counts: list[int] = []
    for i in range(nbins):
        sel = bin_idx == i
        count = int(np.count_nonzero(sel))
        if count == 0:
            continue
        centers.append(float(np.mean(k_vals[sel])))
        pk_bins.append(float(np.mean(p_vals[sel])))
        counts.append(count)

    return {"k_centers": centers, "power": pk_bins, "counts": counts}


def compare_cic_to_initial(initial: MeshField, cic: MeshField) -> dict[str, float]:
    if initial.data.shape != cic.data.shape:
        raise ValueError("Initial and CIC fields must have matching shapes")
    a = np.asarray(initial.data, dtype=float).ravel()
    b = np.asarray(cic.data, dtype=float).ravel()
    diff = b - a

    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std == 0.0 or b_std == 0.0:
        corr = 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(corr):
            corr = 0.0

    return {
        "mean_bias": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "corrcoef": corr,
        "std_ratio": float(b_std / a_std) if a_std > 0 else 0.0,
    }
