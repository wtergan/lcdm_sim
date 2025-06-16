"""Static Matplotlib visualizations for PM diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .types import AccelerationField, MeshField, ParticleState


def _plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _slice_index(n: int, index: int | None) -> int:
    if index is None:
        return n // 2
    return int(index) % n


def _slice_2d(data: np.ndarray, axis: int = 2, index: int | None = None) -> np.ndarray:
    axis_norm = int(axis) % 3
    idx = _slice_index(data.shape[axis_norm], index)
    if axis_norm == 0:
        return data[idx, :, :]
    if axis_norm == 1:
        return data[:, idx, :]
    return data[:, :, idx]


def plot_density_slice(
    field: MeshField,
    output_path: str | Path,
    *,
    axis: int = 2,
    index: int | None = None,
    title: str = "Density Slice",
    cmap: str = "viridis",
) -> Path:
    plt = _plt()
    out = _ensure_parent(output_path)
    arr = _slice_2d(np.asarray(field.data, dtype=float), axis=axis, index=index)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    im = ax.imshow(arr.T, origin="lower", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("x-cell")
    ax.set_ylabel("y-cell")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_grf_slice(
    field: MeshField,
    output_path: str | Path,
    *,
    axis: int = 2,
    index: int | None = None,
) -> Path:
    return plot_density_slice(
        field, output_path, axis=axis, index=index, title="GRF Slice", cmap="magma"
    )


def plot_density_projection(
    field: MeshField,
    output_path: str | Path,
    *,
    axis: int = 2,
    log_scale: bool = True,
) -> Path:
    plt = _plt()
    out = _ensure_parent(output_path)
    data = np.asarray(field.data, dtype=float)
    proj = np.sum(data, axis=int(axis) % 3)
    display = proj
    if log_scale:
        shifted = proj - float(np.min(proj)) + 1e-12
        display = np.log10(shifted)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    im = ax.imshow(display.T, origin="lower", cmap="viridis")
    ax.set_title("Density Projection")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_particle_scatter(
    state: ParticleState,
    output_path: str | Path,
    *,
    dims: tuple[int, int] = (0, 1),
    max_points: int = 10000,
    seed: int = 0,
) -> Path:
    plt = _plt()
    out = _ensure_parent(output_path)

    pts = np.asarray(state.positions, dtype=float)
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], size=int(max_points), replace=False)
        pts = pts[idx]

    d0, d1 = int(dims[0]) % 3, int(dims[1]) % 3
    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    ax.scatter(pts[:, d0], pts[:, d1], s=2, alpha=0.35, linewidths=0)
    ax.set_title("Particle Distribution")
    ax.set_xlabel(f"x{d0 + 1}")
    ax.set_ylabel(f"x{d1 + 1}")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_acceleration_quiver(
    accel: AccelerationField,
    output_path: str | Path,
    *,
    axis: int = 2,
    index: int | None = None,
    stride: int = 4,
) -> Path:
    plt = _plt()
    out = _ensure_parent(output_path)
    axis_norm = int(axis) % 3
    stride_use = max(1, int(stride))

    grid = accel.stacked()
    vec = _slice_2d(grid, axis=axis_norm, index=index)
    mag = np.sqrt(np.sum(vec * vec, axis=-1))

    if axis_norm == 0:
        u = vec[:, :, 1]
        v = vec[:, :, 2]
    elif axis_norm == 1:
        u = vec[:, :, 0]
        v = vec[:, :, 2]
    else:
        u = vec[:, :, 0]
        v = vec[:, :, 1]

    yy, xx = np.mgrid[0 : u.shape[0], 0 : u.shape[1]]
    sl = (slice(None, None, stride_use), slice(None, None, stride_use))

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    im = ax.imshow(mag.T, origin="lower", cmap="cividis", alpha=0.75)
    ax.quiver(xx[sl], yy[sl], u[sl].T, v[sl].T, color="white", scale=None)
    ax.set_title("Acceleration Quiver")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_acceleration_histogram(
    accel: AccelerationField,
    output_path: str | Path,
    *,
    bins: int = 50,
) -> Path:
    plt = _plt()
    out = _ensure_parent(output_path)
    mag = np.sqrt(
        accel.ax * accel.ax + accel.ay * accel.ay + accel.az * accel.az
    ).ravel()

    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    ax.hist(mag, bins=int(bins), alpha=0.8)
    ax.set_title("Acceleration Magnitude Histogram")
    ax.set_xlabel("|a|")
    ax.set_ylabel("Count")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_power_spectrum(
    k: np.ndarray,
    pk: np.ndarray,
    output_path: str | Path,
    *,
    pk_ref: np.ndarray | None = None,
    label: str = "P(k)",
) -> Path:
    plt = _plt()
    out = _ensure_parent(output_path)
    k_arr = np.asarray(k, dtype=float)
    pk_arr = np.asarray(pk, dtype=float)
    mask = (k_arr > 0) & np.isfinite(k_arr) & np.isfinite(pk_arr) & (pk_arr >= 0)

    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    ax.loglog(k_arr[mask], pk_arr[mask], marker="o", ms=3, lw=1.25, label=label)
    if pk_ref is not None:
        pk_ref_arr = np.asarray(pk_ref, dtype=float)
        mask_ref = mask & np.isfinite(pk_ref_arr) & (pk_ref_arr >= 0)
        ax.loglog(
            k_arr[mask_ref], pk_ref_arr[mask_ref], lw=1.0, ls="--", label="Reference"
        )
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("P(k)")
    ax.set_title("Power Spectrum")
    ax.legend(loc="best")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
