import numpy as np


def test_kdk_step_advances_scale_factor_wraps_positions_and_preserves_shapes(
    tiny_config,
):
    from lcdm_sim.grf import generate_grf
    from lcdm_sim.integrators import leapfrog_kdk_a_step, step_size_a
    from lcdm_sim.zeldovich import initial_conditions_from_density

    delta0 = generate_grf(tiny_config)
    state0 = initial_conditions_from_density(delta0, tiny_config)

    def zero_accel_fn(state):
        return np.zeros_like(state.positions)

    state1 = leapfrog_kdk_a_step(state0, tiny_config, zero_accel_fn)

    da = step_size_a(tiny_config)
    assert np.isclose(state1.a, state0.a + da)
    assert state1.positions.shape == state0.positions.shape
    assert state1.velocities.shape == state0.velocities.shape
    assert np.all(np.isfinite(state1.positions))
    assert np.all(np.isfinite(state1.velocities))
    assert np.all(state1.positions >= 0.0)
    assert np.all(state1.positions < tiny_config.grid.box_size_mpc_h)
