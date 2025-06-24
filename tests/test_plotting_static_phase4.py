from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("matplotlib")


def test_static_plotting_writes_png_artifacts(tiny_config, tmp_path):
    from lcdm_sim.forces import compute_comoving_acceleration_grid
    from lcdm_sim.grf import generate_grf
    from lcdm_sim.plotting_static import (
        plot_acceleration_histogram,
        plot_acceleration_quiver,
        plot_density_projection,
        plot_grf_slice,
        plot_particle_scatter,
        plot_power_spectrum,
    )
    from lcdm_sim.spectra import power_spectrum
    from lcdm_sim.zeldovich import initial_conditions_from_density

    delta0 = generate_grf(tiny_config)
    state = initial_conditions_from_density(delta0, tiny_config)
    accel = compute_comoving_acceleration_grid(delta0, tiny_config)

    paths = [
        plot_grf_slice(delta0, Path(tmp_path) / "grf_slice.png"),
        plot_density_projection(delta0, Path(tmp_path) / "projection.png"),
        plot_particle_scatter(state, Path(tmp_path) / "particles.png", max_points=200),
        plot_acceleration_quiver(accel, Path(tmp_path) / "accel_quiver.png", stride=2),
        plot_acceleration_histogram(accel, Path(tmp_path) / "accel_hist.png"),
    ]

    k = np.logspace(-2, 0, 16)
    pk = power_spectrum(k, tiny_config.cosmology, a=tiny_config.cosmology.a_initial)
    paths.append(plot_power_spectrum(k, pk, Path(tmp_path) / "pk.png"))

    for p in paths:
        assert Path(p).exists()
        assert Path(p).stat().st_size > 0
