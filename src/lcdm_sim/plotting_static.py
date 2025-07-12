"""Static Matplotlib visualizations for PM diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .diagnostics import estimate_power_spectrum
from .types import AccelerationField, MeshField, ParticleState, Snapshot


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


def _evenly_spaced_items(items, max_items: int):
    if len(items) <= max_items:
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=max_items, dtype=int)
    return [items[int(i)] for i in sorted(set(indices))]


def plot_density_evolution(
    snapshots: list[Snapshot],
    output_path: str | Path,
    *,
    axis: int = 2,
    index: int | None = None,
    max_snapshots: int = 6,
    cmap: str = "viridis",
) -> Path:
    """Plot density slices from several saved snapshots as one evolution panel."""

    density_snapshots = [s for s in snapshots if s.density_field is not None]
    if not density_snapshots:
        raise ValueError("At least one snapshot with a density field is required")

    chosen = _evenly_spaced_items(density_snapshots, max(1, int(max_snapshots)))
    slices = [
        _slice_2d(np.asarray(s.density_field.data, dtype=float), axis=axis, index=index)
        for s in chosen
        if s.density_field is not None
    ]

    finite_values = np.concatenate([arr[np.isfinite(arr)].ravel() for arr in slices])
    if finite_values.size:
        vmin, vmax = np.percentile(finite_values, [2.0, 98.0])
    else:
        vmin, vmax = 0.0, 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = None, None

    plt = _plt()
    out = _ensure_parent(output_path)
    ncols = len(chosen)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(max(3.0 * ncols, 4.0), 3.35),
        constrained_layout=True,
        squeeze=False,
    )

    last_im = None
    for ax, snapshot, arr in zip(axes[0], chosen, slices):
        last_im = ax.imshow(arr.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"step {snapshot.step}\na={snapshot.a:.3f}")
        ax.set_xticks([])
        ax.set_yticks([])

    if last_im is not None:
        fig.colorbar(last_im, ax=list(axes[0]), shrink=0.72, label="overdensity")
    fig.suptitle("Density Field Evolution")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_particle_evolution(
    snapshots: list[Snapshot],
    output_path: str | Path,
    *,
    dims: tuple[int, int] = (0, 1),
    max_points: int = 12000,
    max_snapshots: int = 6,
    seed: int = 0,
) -> Path:
    """Plot projected particle positions across saved simulation stages."""

    if not snapshots:
        raise ValueError("At least one snapshot is required")
    chosen = _evenly_spaced_items(snapshots, max(1, int(max_snapshots)))
    d0, d1 = int(dims[0]) % 3, int(dims[1]) % 3
    n_particles = chosen[0].particle_state.positions.shape[0]
    sample_idx: np.ndarray | None = None
    if n_particles > max_points and all(
        snapshot.particle_state.positions.shape[0] == n_particles for snapshot in chosen
    ):
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(n_particles, size=int(max_points), replace=False)

    projected: list[np.ndarray] = []
    for snapshot in chosen:
        pts = np.asarray(snapshot.particle_state.positions, dtype=float)
        if sample_idx is not None:
            pts = pts[sample_idx]
        elif pts.shape[0] > max_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(pts.shape[0], size=int(max_points), replace=False)
            pts = pts[idx]
        projected.append(pts[:, (d0, d1)])

    finite = np.concatenate(projected, axis=0)
    mins = np.nanmin(finite, axis=0)
    maxs = np.nanmax(finite, axis=0)
    padding = np.maximum((maxs - mins) * 0.02, 1e-6)

    plt = _plt()
    out = _ensure_parent(output_path)
    ncols = len(chosen)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(max(3.0 * ncols, 4.0), 3.35),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, snapshot, pts in zip(axes[0], chosen, projected):
        ax.scatter(pts[:, 0], pts[:, 1], s=1.5, alpha=0.3, linewidths=0)
        ax.set_title(f"step {snapshot.step}\na={snapshot.a:.3f}")
        ax.set_xlim(mins[0] - padding[0], maxs[0] + padding[0])
        ax.set_ylim(mins[1] - padding[1], maxs[1] + padding[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle("Particle Distribution Evolution")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_power_spectrum_evolution(
    snapshots: list[Snapshot],
    output_path: str | Path,
    *,
    nbins: int = 16,
    max_snapshots: int = 6,
) -> Path:
    """Plot binned density power spectra for saved simulation stages."""

    density_snapshots = [s for s in snapshots if s.density_field is not None]
    if not density_snapshots:
        raise ValueError("At least one snapshot with a density field is required")
    chosen = _evenly_spaced_items(density_snapshots, max(1, int(max_snapshots)))

    plt = _plt()
    out = _ensure_parent(output_path)
    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    for snapshot in chosen:
        spectrum = estimate_power_spectrum(snapshot.density_field, nbins=nbins)
        k = np.asarray(spectrum["k_centers"], dtype=float)
        power = np.asarray(spectrum["power"], dtype=float)
        mask = (k > 0) & (power > 0) & np.isfinite(k) & np.isfinite(power)
        ax.loglog(
            k[mask],
            power[mask],
            marker="o",
            ms=2.5,
            lw=1.2,
            label=f"a={snapshot.a:.3f}",
        )

    ax.set_title("Density Power Spectrum Evolution")
    ax.set_xlabel("k [h/Mpc]")
    ax.set_ylabel("P(k)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(loc="best")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_history_summary(
    history: list[dict[str, float | int]],
    output_path: str | Path,
) -> Path:
    """Plot compact run-history diagnostics saved by the CLI."""

    if not history:
        raise ValueError("At least one history row is required")

    plt = _plt()
    out = _ensure_parent(output_path)
    steps = np.asarray([row["step"] for row in history], dtype=float)
    a_vals = np.asarray([row["a"] for row in history], dtype=float)
    density_std = np.asarray([row["density_std"] for row in history], dtype=float)
    velocity_rms = np.asarray([row["velocity_rms"] for row in history], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)
    axes[0].plot(steps, a_vals, marker="o", lw=1.35)
    axes[0].set_title("Scale Factor")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("a")

    density_line = axes[1].plot(
        steps, density_std, marker="o", lw=1.35, color="tab:blue", label="density std"
    )
    axes_twin = axes[1].twinx()
    velocity_line = axes_twin.plot(
        steps,
        velocity_rms,
        marker="s",
        lw=1.15,
        color="tab:orange",
        label="velocity RMS",
    )
    axes[1].set_title("Run Diagnostics")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("density std", color="tab:blue")
    axes_twin.set_ylabel("velocity RMS", color="tab:orange")
    axes[1].legend(density_line + velocity_line, ["density std", "velocity RMS"])

    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out
