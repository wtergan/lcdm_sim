import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lcdm_sim.config import load_simulation_config  # noqa: E402

CFG = load_simulation_config(ROOT / "configs" / "smoke.yaml")


def test_generate_grf_is_deterministic_and_finite():
    from lcdm_sim.grf import generate_grf

    f1 = generate_grf(CFG)
    f2 = generate_grf(CFG)

    assert f1.data.shape == (CFG.grid.n_grid_1d, CFG.grid.n_grid_1d, CFG.grid.n_grid_1d)
    assert np.all(np.isfinite(f1.data))
    assert np.allclose(f1.data, f2.data)
    assert abs(float(f1.data.mean())) < 1e-2
