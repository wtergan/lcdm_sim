"""Transfer functions and power-spectrum utilities."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .cosmology import growth_factor
from .types import CosmologyConfig


ArrayLike = np.ndarray | float | list[float] | tuple[float, ...]


def _as_array(k: ArrayLike) -> np.ndarray:
    arr = np.asarray(k, dtype=float)
    if np.any(arr < 0):
        raise ValueError("Wavenumber k must be non-negative.")
    return arr


def eisenstein_hu_transfer(k: ArrayLike, cfg: CosmologyConfig) -> np.ndarray:
    """No-wiggle transfer approximation in the Eisenstein-Hu / BBKS spirit.

    Input `k` is assumed to be in h/Mpc units.
    """

    k_arr = _as_array(k)
    h = cfg.h0 / 100.0
    om = max(cfg.omega_m, 1e-12)
    ob = max(min(cfg.omega_b, om), 0.0)

    gamma_eff = om * h * np.exp(-ob - (np.sqrt(2.0 * h) * ob / om))
    q = np.divide(k_arr, gamma_eff, out=np.zeros_like(k_arr), where=gamma_eff > 0)

    num = np.log1p(2.34 * q)
    den = 2.34 * q
    L0 = np.divide(num, den, out=np.ones_like(q), where=q > 0)
    C0 = np.power(
        1.0 + 3.89 * q + (16.1 * q) ** 2 + (5.46 * q) ** 3 + (6.71 * q) ** 4, -0.25
    )
    T = L0 * C0
    T = np.where(k_arr == 0.0, 1.0, T)
    return np.clip(T, 0.0, None)


def _unnormalized_power_spectrum(
    k: np.ndarray, cfg: CosmologyConfig, a: float
) -> np.ndarray:
    T = eisenstein_hu_transfer(k, cfg)
    D = float(growth_factor(float(a), cfg, method="ode"))
    k_safe = np.where(k > 0, k, 1.0)
    tilt = np.power(k_safe / cfg.k_pivot_mpc, cfg.n_s)
    P = cfg.a_s * tilt * T**2 * D**2
    P = np.where(k > 0, P, 0.0)
    return np.clip(P, 0.0, None)


def tophat_window(x: ArrayLike) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    x2 = x_arr * x_arr
    x3 = x2 * x_arr
    numer = np.sin(x_arr) - x_arr * np.cos(x_arr)
    w = np.divide(3.0 * numer, x3, out=np.ones_like(x_arr), where=np.abs(x_arr) > 1e-12)
    return w


def sigma_r_from_power_spectrum(
    k: np.ndarray, pk: np.ndarray, radius_mpc_h: float
) -> float:
    if radius_mpc_h <= 0:
        raise ValueError("radius_mpc_h must be positive")
    if k.ndim != 1 or pk.ndim != 1 or k.shape != pk.shape:
        raise ValueError("k and pk must be 1D arrays of equal shape")
    if k.size < 2:
        return 0.0
    mask = k > 0
    k_use = k[mask]
    pk_use = pk[mask]
    if k_use.size < 2:
        return 0.0
    w = tophat_window(k_use * radius_mpc_h)
    integrand = pk_use * (w**2) * (k_use**2)
    trapz_fn = getattr(np, "trapezoid", None)
    if trapz_fn is None:  # pragma: no cover - legacy NumPy fallback
        trapz_fn = np.trapz
    sigma2 = trapz_fn(integrand, k_use) / (2.0 * np.pi**2)
    return float(np.sqrt(max(sigma2, 0.0)))


def _sigma8_cache_key(cfg: CosmologyConfig) -> tuple[float, ...]:
    return (
        float(cfg.h0),
        float(cfg.omega_m),
        float(cfg.omega_lambda),
        float(cfg.omega_b),
        float(cfg.sigma8),
        float(cfg.n_s),
        float(cfg.a_s),
        float(cfg.k_pivot_mpc),
    )


@lru_cache(maxsize=64)
def _sigma8_norm_factor_cached(key: tuple[float, ...]) -> float:
    cfg = CosmologyConfig(
        h0=key[0],
        omega_m=key[1],
        omega_lambda=key[2],
        omega_b=key[3],
        sigma8=key[4],
        n_s=key[5],
        a_s=key[6],
        k_pivot_mpc=key[7],
        a_initial=0.01,
        a_final=1.0,
    )
    k = np.logspace(-4, 2, 4096)
    pk0 = _unnormalized_power_spectrum(k, cfg, a=1.0)
    sigma8_unscaled = sigma_r_from_power_spectrum(k, pk0, radius_mpc_h=8.0)
    if not np.isfinite(sigma8_unscaled) or sigma8_unscaled <= 0:
        raise ValueError("Could not compute sigma8 normalization factor.")
    return (cfg.sigma8 / sigma8_unscaled) ** 2


def sigma8_normalization_factor(cfg: CosmologyConfig) -> float:
    return float(_sigma8_norm_factor_cached(_sigma8_cache_key(cfg)))


def power_spectrum(
    k: ArrayLike, cfg: CosmologyConfig, a: float | None = None
) -> np.ndarray:
    k_arr = _as_array(k)
    a_use = float(cfg.a_initial if a is None else a)
    pk = _unnormalized_power_spectrum(k_arr, cfg, a=a_use)
    pk *= sigma8_normalization_factor(cfg)
    return np.where(k_arr > 0, np.clip(pk, 0.0, None), 0.0)
