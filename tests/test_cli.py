import json
import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class CliTests(unittest.TestCase):
    def test_cli_help_lists_subcommands(self):
        from lcdm_sim.cli import main

        buf = io.StringIO()
        with self.assertRaises(SystemExit) as cm, redirect_stdout(buf):
            main(['--help'])

        self.assertEqual(cm.exception.code, 0)
        help_text = buf.getvalue()
        self.assertIn('run', help_text)
        self.assertIn('plot', help_text)
        self.assertIn('validate', help_text)
        self.assertIn('export-web-dataset', help_text)

    def test_run_subcommand_loads_config_and_returns_zero(self):
        from lcdm_sim.cli import main

        payload = {
            'grid': {'n_particles_1d': 64, 'n_grid_1d': 64, 'box_size_mpc_h': 100.0},
            'cosmology': {
                'h0': 67.66, 'omega_m': 0.3097, 'omega_lambda': 0.6903,
                'sigma8': 0.811, 'n_s': 0.96, 'a_initial': 0.01, 'a_final': 1.0,
            },
            'integrator': {'num_steps': 5, 'method': 'kdk_a'},
            'output': {'output_root': 'outputs', 'save_density': False, 'save_plots': False},
            'performance': {'fft_backend': 'scipy', 'use_numba': False},
            'validation': {'enable_invariants': True, 'enable_reference_compare': False},
            'random_seed': 123,
        }

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / 'smoke.yaml'
            cfg_path.write_text(json.dumps(payload), encoding='utf-8')
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main(['run', '--config', str(cfg_path)])

        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertIn('run (stub)', out)
        self.assertIn('64^3 particles', out)
        self.assertIn('steps=5', out)


if __name__ == '__main__':
    unittest.main()
