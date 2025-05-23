import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_scipy_fft_backend_roundtrip():
    from lcdm_sim.fft_backend import get_fft_backend

    backend = get_fft_backend("scipy")
    x = np.arange(64, dtype=float).reshape(4, 4, 4)
    X = backend.fftn(x)
    xr = backend.ifftn(X).real
    assert x.shape == xr.shape
    assert np.allclose(x, xr, atol=1e-10)


def test_pyfftw_backend_hook_falls_back_or_works():
    from lcdm_sim.fft_backend import get_fft_backend

    backend = get_fft_backend("pyfftw")
    x = np.random.default_rng(0).normal(size=(8, 8, 8))
    X = backend.rfftn(x)
    xr = backend.irfftn(X, s=x.shape)
    assert xr.shape == x.shape
    assert np.allclose(x, xr, atol=1e-8)
