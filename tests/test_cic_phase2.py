import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lcdm_sim.config import load_simulation_config  # noqa: E402

CFG = load_simulation_config(ROOT / "configs" / "smoke.yaml")


def test_cic_density_and_force_interpolation_shapes_and_mass_conservation():
    from lcdm_sim.grf import generate_grf
    from lcdm_sim.zeldovich import initial_conditions_from_density
    from lcdm_sim.cic import deposit_density, interpolate_forces_to_particles
    from lcdm_sim.types import AccelerationField

    delta0 = generate_grf(CFG)
    state = initial_conditions_from_density(delta0, CFG)
    delta = deposit_density(state, CFG.grid)

    assert delta.data.shape == (CFG.grid.n_grid_1d,) * 3
    assert np.all(np.isfinite(delta.data))
    assert abs(float(delta.data.mean())) < 1e-6

    n = CFG.grid.n_grid_1d
    zeros = np.zeros((n, n, n), dtype=float)
    accel = AccelerationField(
        ax=zeros,
        ay=zeros.copy(),
        az=zeros.copy(),
        box_size_mpc_h=CFG.grid.box_size_mpc_h,
    )
    gathered = interpolate_forces_to_particles(accel, state, CFG.grid)
    assert gathered.shape == state.positions.shape
    assert np.allclose(gathered, 0.0)
