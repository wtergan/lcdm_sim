import json
import sys
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class ConfigLoaderTests(unittest.TestCase):
    def test_loads_json_compatible_yaml_into_typed_config(self):
        from lcdm_sim.config import load_simulation_config

        payload = {
            "grid": {
                "n_particles_1d": 64,
                "n_grid_1d": 64,
                "box_size_mpc_h": 100.0,
            },
            "cosmology": {
                "h0": 67.66,
                "omega_m": 0.3097,
                "omega_lambda": 0.6903,
                "sigma8": 0.811,
                "n_s": 0.96,
                "a_initial": 0.01,
                "a_final": 1.0,
            },
            "integrator": {
                "num_steps": 20,
                "method": "kdk_a",
            },
            "output": {
                "output_root": "outputs",
                "save_density": False,
                "save_plots": False,
            },
            "performance": {
                "fft_backend": "scipy",
                "use_numba": False,
            },
            "validation": {
                "enable_invariants": True,
                "enable_reference_compare": False,
            },
            "random_seed": 38,
        }

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "smoke.yaml"
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")

            cfg = load_simulation_config(cfg_path)

        self.assertEqual(cfg.grid.n_particles_1d, 64)
        self.assertEqual(cfg.grid.n_grid_1d, 64)
        self.assertAlmostEqual(cfg.cosmology.h0, 67.66)
        self.assertEqual(cfg.integrator.method, "kdk_a")
        self.assertFalse(cfg.output.save_density)
        self.assertEqual(cfg.random_seed, 38)


if __name__ == "__main__":
    unittest.main()
