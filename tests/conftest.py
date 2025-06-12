import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture()
def tiny_config():
    from lcdm_sim.config import simulation_config_from_dict

    data = {
        "grid": {"n_particles_1d": 8, "n_grid_1d": 8, "box_size_mpc_h": 32.0},
        "cosmology": {
            "h0": 67.66,
            "omega_m": 0.3097,
            "omega_lambda": 0.6903,
            "sigma8": 0.2,
            "n_s": 0.96,
            "a_initial": 0.1,
            "a_final": 0.2,
        },
        "integrator": {"num_steps": 4, "method": "kdk_a"},
        "output": {
            "output_root": "outputs",
            "save_density": True,
            "save_plots": False,
        },
        "performance": {"fft_backend": "scipy", "use_numba": False, "fft_workers": 1},
        "validation": {"enable_invariants": True, "enable_reference_compare": False},
        "random_seed": 7,
    }
    return simulation_config_from_dict(data)
