"""Internal periodic trilinear sampling helpers (CIC-compatible weights)."""

from __future__ import annotations

import numpy as np


def _cic_indices_and_fractions(positions: np.ndarray, n: int, box_size_mpc_h: float):
    dx = box_size_mpc_h / float(n)
    grid_coords = np.mod(positions, box_size_mpc_h) / dx
    base = np.floor(grid_coords).astype(np.int64)
    frac = grid_coords - base
    base %= n
    nxt = (base + 1) % n
    return base, nxt, frac


def periodic_trilinear_sample_vector(
    field: np.ndarray, positions: np.ndarray, box_size_mpc_h: float
) -> np.ndarray:
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError("field must have shape (N, N, N, 3)")
    n = field.shape[0]
    base, nxt, frac = _cic_indices_and_fractions(positions, n, box_size_mpc_h)
    tx, ty, tz = frac[:, 0], frac[:, 1], frac[:, 2]

    out = np.zeros((positions.shape[0], 3), dtype=float)
    for ox in (0, 1):
        wx = tx if ox else (1.0 - tx)
        ix = nxt[:, 0] if ox else base[:, 0]
        for oy in (0, 1):
            wy = ty if oy else (1.0 - ty)
            iy = nxt[:, 1] if oy else base[:, 1]
            for oz in (0, 1):
                wz = tz if oz else (1.0 - tz)
                iz = nxt[:, 2] if oz else base[:, 2]
                w = (wx * wy * wz)[:, None]
                out += field[ix, iy, iz] * w
    return out
