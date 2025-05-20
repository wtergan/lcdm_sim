import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lcdm_sim.config import load_simulation_config  # noqa: E402

CFG = load_simulation_config(ROOT / "configs" / "smoke.yaml")


def test_hubble_and_growth_rate_are_finite_positive():
    from lcdm_sim.cosmology import hubble, growth_rate

    a = np.array([0.01, 0.1, 0.5, 1.0])
    H = hubble(a, CFG.cosmology)
    f = growth_rate(a, CFG.cosmology)

    assert np.all(np.isfinite(H))
    assert np.all(H > 0)
    assert np.all(np.isfinite(f))
    assert np.all((f > 0) & (f < 2.0))


def test_growth_factor_normalizes_to_one_at_a_equal_one_and_is_monotonic():
    from lcdm_sim.cosmology import growth_factor

    a = np.linspace(0.01, 1.0, 64)
    D = growth_factor(a, CFG.cosmology, method="ode")

    assert np.all(np.isfinite(D))
    assert D.shape == a.shape
    assert np.isclose(D[-1], 1.0, rtol=1e-3, atol=1e-3)
    assert np.all(np.diff(D) >= -1e-6)
