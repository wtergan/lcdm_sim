from pathlib import Path

import pytest


plotly = pytest.importorskip("plotly")


def test_interactive_plotting_writes_html_artifacts(tiny_config, tmp_path):
    from lcdm_sim.grf import generate_grf
    from lcdm_sim.plotting_interactive import (
        plot_density_slice_explorer_interactive,
        plot_evolution_summary_interactive,
        plot_particles_3d_interactive,
    )
    from lcdm_sim.zeldovich import initial_conditions_from_density

    delta0 = generate_grf(tiny_config)
    state = initial_conditions_from_density(delta0, tiny_config)

    p1 = plot_particles_3d_interactive(
        state, Path(tmp_path) / "particles3d.html", max_points=256
    )
    p2 = plot_density_slice_explorer_interactive(
        delta0, Path(tmp_path) / "slice_explorer.html"
    )
    history = [
        {"step": 0, "a": 0.1, "density_std": 0.2, "velocity_rms": 0.01},
        {"step": 1, "a": 0.15, "density_std": 0.25, "velocity_rms": 0.02},
        {"step": 2, "a": 0.2, "density_std": 0.3, "velocity_rms": 0.03},
    ]
    p3 = plot_evolution_summary_interactive(history, Path(tmp_path) / "evolution.html")

    for p in (p1, p2, p3):
        assert Path(p).exists()
        assert Path(p).stat().st_size > 0
