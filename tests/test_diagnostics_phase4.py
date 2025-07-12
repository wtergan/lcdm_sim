import numpy as np


def test_diagnostics_stats_and_power_spectrum_estimate_are_finite(tiny_config):
    from lcdm_sim.cic import deposit_density
    from lcdm_sim.diagnostics import (
        acceleration_stats,
        compare_cic_to_initial,
        density_stats,
        estimate_power_spectrum,
        summarize_density_snapshots,
    )
    from lcdm_sim.forces import compute_comoving_acceleration_grid
    from lcdm_sim.grf import generate_grf
    from lcdm_sim.zeldovich import initial_conditions_from_density

    delta0 = generate_grf(tiny_config)
    state = initial_conditions_from_density(delta0, tiny_config)
    delta_cic = deposit_density(state, tiny_config.grid)
    accel = compute_comoving_acceleration_grid(delta_cic, tiny_config)

    d_stats = density_stats(delta_cic)
    a_stats = acceleration_stats(accel)
    ps = estimate_power_spectrum(delta_cic, nbins=6)
    cmp = compare_cic_to_initial(delta0, delta_cic)
    from lcdm_sim.types import Snapshot

    summary = summarize_density_snapshots(
        [Snapshot(step=0, particle_state=state, density_field=delta_cic)], nbins=6
    )

    assert d_stats["shape"] == list(delta_cic.data.shape)
    assert np.isfinite(d_stats["std"])
    assert np.isfinite(a_stats["mag_rms"])
    assert len(ps["k_centers"]) == len(ps["power"]) == len(ps["counts"])
    assert np.all(np.asarray(ps["power"]) >= 0.0)
    assert np.isfinite(cmp["rmse"])
    assert "corrcoef" in cmp
    assert summary[0]["step"] == 0
    assert np.isfinite(summary[0]["fraction_cells_delta_gt_1"])
    assert len(summary[0]["power_spectrum"]["power"]) > 0
