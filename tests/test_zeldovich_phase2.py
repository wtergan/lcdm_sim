import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lcdm_sim.config import load_simulation_config  # noqa: E402

CFG = load_simulation_config(ROOT / "configs" / "smoke.yaml")


def test_zeldovich_initial_conditions_have_expected_shapes_and_bounds():
    from lcdm_sim.grf import generate_grf
    from lcdm_sim.zeldovich import (
        initial_conditions_from_density,
        displacement_velocity_fields_from_density,
    )

    delta = generate_grf(CFG)
    disp, vel = displacement_velocity_fields_from_density(delta, CFG)
    state = initial_conditions_from_density(delta, CFG)

    ngrid = CFG.grid.n_grid_1d
    nparts = CFG.grid.n_particles_1d**3
    assert disp.shape == (ngrid, ngrid, ngrid, 3)
    assert vel.shape == (ngrid, ngrid, ngrid, 3)
    assert state.positions.shape == (nparts, 3)
    assert state.velocities.shape == (nparts, 3)
    assert np.all(np.isfinite(state.positions))
    assert np.all(np.isfinite(state.velocities))
    L = CFG.grid.box_size_mpc_h
    assert np.all(state.positions >= 0)
    assert np.all(state.positions < L)
