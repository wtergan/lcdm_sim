"""Cosmology helper functions for LCDM scale-factor evolution."""

from __future__ import annotations

import numpy as np

from .types import CosmologyConfig


ArrayLike = np.ndarray | float | list[float] | tuple[float, ...]


def _as_array(a: ArrayLike) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if np.any(arr <= 0):
        raise ValueError("Scale factor a must be positive.")
    return arr


def omega_k(cfg: CosmologyConfig) -> float:
    return 1.0 - float(cfg.omega_m) - float(cfg.omega_lambda)


def e_of_a(a: ArrayLike, cfg: CosmologyConfig) -> np.ndarray:
    arr = _as_array(a)
    ok = omega_k(cfg)
    return np.sqrt(cfg.omega_m / arr**3 + ok / arr**2 + cfg.omega_lambda)


def hubble(a: ArrayLike, cfg: CosmologyConfig) -> np.ndarray:
    return cfg.h0 * e_of_a(a, cfg)


def omega_m_of_a(a: ArrayLike, cfg: CosmologyConfig) -> np.ndarray:
    arr = _as_array(a)
    E = e_of_a(arr, cfg)
    return (cfg.omega_m / arr**3) / (E**2)


def growth_rate(a: ArrayLike, cfg: CosmologyConfig, gamma: float = 0.55) -> np.ndarray:
    om_a = omega_m_of_a(a, cfg)
    return np.power(np.clip(om_a, 1e-12, None), gamma)


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y)
    if y.size < 2:
        return out
    dx = np.diff(x)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)
    return out


def growth_factor(
    a: ArrayLike, cfg: CosmologyConfig, method: str = "ode"
) -> np.ndarray | float:
    """Return linear growth factor D(a), normalized to D(1)=1.

    The implementation uses the standard LCDM integral form. The `method='ode'`
    name is preserved for notebook/API continuity.
    """

    arr = _as_array(a)
    scalar = np.isscalar(a)
    method_norm = method.lower()
    if method_norm not in {"ode", "integral", "approx"}:
        raise ValueError(f"Unsupported growth_factor method: {method}")

    # Integral solution grid spanning requested domain up to at least a=1.
    a_min = max(1e-4, float(np.min(arr)) * 0.5)
    a_max = max(1.0, float(np.max(arr)))
    a_grid = np.geomspace(a_min, a_max, 4096)

    E = e_of_a(a_grid, cfg)
    integrand = 1.0 / (a_grid**3 * E**3)
    integral = _cumulative_trapezoid(integrand, a_grid)
    D_raw = 2.5 * cfg.omega_m * E * integral

    if method_norm == "approx":
        # Approximate growth for quick comparisons; still normalized below.
        D_raw = a_grid * np.power(np.clip(omega_m_of_a(a_grid, cfg), 1e-12, None), 0.55)

    D_interp = np.interp(arr, a_grid, D_raw)
    D1 = float(np.interp(1.0, a_grid, D_raw))
    if not np.isfinite(D1) or D1 == 0.0:
        raise ValueError("Failed to normalize growth factor at a=1.")
    D = D_interp / D1

    if scalar:
        return float(D)
    return D
