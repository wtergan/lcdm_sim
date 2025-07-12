import json
import io
import importlib.util
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class CliTests(unittest.TestCase):
    def test_cli_help_lists_subcommands(self):
        from lcdm_sim.cli import main

        buf = io.StringIO()
        with self.assertRaises(SystemExit) as cm, redirect_stdout(buf):
            main(["--help"])

        self.assertEqual(cm.exception.code, 0)
        help_text = buf.getvalue()
        self.assertIn("run", help_text)
        self.assertIn("plot", help_text)
        self.assertIn("validate", help_text)
        self.assertIn("export-web-dataset", help_text)

    @unittest.skipIf(importlib.util.find_spec("h5py") is None, "h5py not installed")
    def test_run_subcommand_writes_snapshots_and_metrics(self):
        from lcdm_sim.cli import main

        payload = {
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
            "performance": {
                "fft_backend": "scipy",
                "use_numba": False,
                "fft_workers": 1,
            },
            "validation": {
                "enable_invariants": True,
                "enable_reference_compare": False,
            },
            "random_seed": 123,
        }

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "smoke.yaml"
            run_dir = Path(tmp) / "run"
            cfg_path.write_text(json.dumps(payload), encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main(
                    [
                        "run",
                        "--config",
                        str(cfg_path),
                        "--out-dir",
                        str(run_dir),
                        "--num-snapshots",
                        "3",
                    ]
                )

            self.assertEqual(code, 0)
            out = buf.getvalue()
            self.assertIn("run:", out)
            self.assertIn("8^3 particles", out)
            self.assertIn("steps=4", out)
            self.assertTrue((run_dir / "metrics" / "history.json").exists())
            self.assertTrue((run_dir / "metrics" / "validation_report.json").exists())
            self.assertEqual(len(list((run_dir / "snapshots").glob("*.h5"))), 3)

    @unittest.skipIf(importlib.util.find_spec("h5py") is None, "h5py not installed")
    @unittest.skipIf(
        importlib.util.find_spec("matplotlib") is None, "matplotlib not installed"
    )
    def test_plot_subcommand_writes_png_manifest(self):
        from lcdm_sim.cli import main
        from lcdm_sim.simulation import run_simulation
        from lcdm_sim.config import simulation_config_from_dict

        cfg = simulation_config_from_dict(
            {
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
                "performance": {
                    "fft_backend": "scipy",
                    "use_numba": False,
                    "fft_workers": 1,
                },
                "validation": {
                    "enable_invariants": True,
                    "enable_reference_compare": False,
                },
                "random_seed": 123,
            }
        )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            plot_dir = Path(tmp) / "plots"
            run_simulation(
                cfg, output_dir=run_dir, num_snapshots=3, save_snapshots=True
            )

            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main(
                    ["plot", "--run-dir", str(run_dir), "--out-dir", str(plot_dir)]
                )

            self.assertEqual(code, 0)
            self.assertIn("plot:", buf.getvalue())
            manifest = plot_dir / "plot_manifest.json"
            self.assertTrue(manifest.exists())
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(data["artifacts"]), 1)
            self.assertTrue((plot_dir / "summaries" / "density_evolution.png").exists())
            self.assertTrue(
                (plot_dir / "summaries" / "particle_evolution.png").exists()
            )
            self.assertTrue(
                (plot_dir / "summaries" / "power_spectrum_evolution.png").exists()
            )
            analysis = json.loads(
                (plot_dir / "analysis" / "snapshot_analysis.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(analysis["snapshots"]), 3)
            self.assertEqual(
                len(list((plot_dir / "snapshots" / "particles").glob("*.png"))), 3
            )


if __name__ == "__main__":
    unittest.main()
