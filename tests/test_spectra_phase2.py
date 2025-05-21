import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lcdm_sim.config import load_simulation_config  # noqa: E402

CFG = load_simulation_config(ROOT / "configs" / "smoke.yaml")


def test_transfer_function_and_power_spectrum_are_finite_and_nonnegative():
    from lcdm_sim.spectra import eisenstein_hu_transfer, power_spectrum

    k = np.concatenate(([0.0], np.logspace(-3, 1, 128)))
    T = eisenstein_hu_transfer(k, CFG.cosmology)
    P = power_spectrum(k, CFG.cosmology, a=CFG.cosmology.a_initial)

    assert np.all(np.isfinite(T))
    assert np.all(T >= 0)
    assert np.all(np.isfinite(P))
    assert np.all(P >= 0)
    assert P[0] == 0.0


def test_sigma8_normalization_factor_is_finite_positive():
    from lcdm_sim.spectra import sigma8_normalization_factor

    factor = sigma8_normalization_factor(CFG.cosmology)
    assert np.isfinite(factor)
    assert factor > 0
