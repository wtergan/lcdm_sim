"""Optional Plotly-based interactive visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _plotly_go():
    try:
        import plotly.graph_objects as go  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError("plotly is required for interactive plotting") from exc
    return go


def _ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def plot_particles_3d_interactive(
    state, output_path: str | Path, *, max_points: int = 20000, seed: int = 0
) -> Path:
    go = _plotly_go()
    out = _ensure_parent(output_path)

    pts = np.asarray(state.positions, dtype=float)
    if pts.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(pts.shape[0], size=int(max_points), replace=False)
        pts = pts[idx]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker={"size": 2, "opacity": 0.45},
            )
        ]
    )
    fig.update_layout(title="Particles (3D)", scene={"aspectmode": "cube"})
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def plot_density_slice_explorer_interactive(
    field, output_path: str | Path, *, axis: int = 2
) -> Path:
    go = _plotly_go()
    out = _ensure_parent(output_path)

    data = np.asarray(field.data, dtype=float)
    axis_norm = int(axis) % 3
    slices = [np.take(data, i, axis=axis_norm) for i in range(data.shape[axis_norm])]

    frames = [
        go.Frame(data=[go.Heatmap(z=s.T, colorscale="Viridis")], name=str(i))
        for i, s in enumerate(slices)
    ]
    fig = go.Figure(data=frames[0].data, frames=frames)
    steps = [
        {
            "label": str(i),
            "method": "animate",
            "args": [
                [str(i)],
                {"mode": "immediate", "frame": {"duration": 0, "redraw": True}},
            ],
        }
        for i in range(len(slices))
    ]
    fig.update_layout(
        title="Density Slice Explorer",
        sliders=[{"active": len(slices) // 2, "steps": steps}],
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 120, "redraw": True}}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                            },
                        ],
                    },
                ],
            }
        ],
    )
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def plot_evolution_summary_interactive(
    history_or_result, output_path: str | Path
) -> Path:
    go = _plotly_go()
    out = _ensure_parent(output_path)

    history = history_or_result
    if hasattr(history_or_result, "history"):
        history = history_or_result.history
    history = list(history)
    if not history:
        raise ValueError("history must contain at least one row")

    a_vals = [float(row.get("a", row.get("scale_factor", 0.0))) for row in history]
    traces = []
    for key in ("density_std", "velocity_rms", "step_time_s"):
        values = [row.get(key) for row in history]
        if any(v is not None for v in values):
            y = [float(v) if v is not None else np.nan for v in values]
            traces.append(go.Scatter(x=a_vals, y=y, mode="lines+markers", name=key))

    if not traces:
        raise ValueError("history rows do not contain supported summary metrics")

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Evolution Summary",
        xaxis_title="Scale factor a",
        yaxis_title="Metric value",
    )
    fig.write_html(out, include_plotlyjs="cdn")
    return out
