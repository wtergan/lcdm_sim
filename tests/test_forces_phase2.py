import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lcdm_sim.config import load_simulation_config  # noqa: E402
from lcdm_sim.types import MeshField  # noqa: E402

CFG = load_simulation_config(ROOT / "configs" / "smoke.yaml")


def test_poisson_and_acceleration_grid_are_finite_and_directional():
    from lcdm_sim.potential import solve_poisson_potential
    from lcdm_sim.forces import (
        compute_comoving_acceleration_grid,
        convert_comoving_to_physical_acceleration,
    )

    n = CFG.grid.n_grid_1d
    L = CFG.grid.box_size_mpc_h
    x = np.arange(n) * (L / n)
    xx = x[:, None, None]
    mode = np.sin(2 * np.pi * xx / L)
    delta = np.broadcast_to(mode, (n, n, n)).copy()
    delta -= delta.mean()

    mesh = MeshField(data=delta, box_size_mpc_h=L)
    phi = solve_poisson_potential(mesh, CFG)
    accel = compute_comoving_acceleration_grid(mesh, CFG)
    phys = convert_comoving_to_physical_acceleration(accel, a=CFG.cosmology.a_initial)

    assert phi.data.shape == (n, n, n)
    assert np.all(np.isfinite(phi.data))
    assert np.all(np.isfinite(accel.ax))
    assert np.all(np.isfinite(accel.ay))
    assert np.all(np.isfinite(accel.az))
    assert np.max(np.abs(accel.ay)) < 1e-8
    assert np.max(np.abs(accel.az)) < 1e-8
    assert np.max(np.abs(accel.ax)) > 0
    assert np.all(np.isfinite(phys.ax))
