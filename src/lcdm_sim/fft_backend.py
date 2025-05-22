"""FFT backend abstraction with SciPy baseline and optional pyFFTW hook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FFTBackend:
    name: str
    _fft_module: Any
    workers: int | None = None
    fallback_reason: str | None = None

    def _kwargs(self) -> dict[str, int]:
        if self.workers is None:
            return {}
        return {"workers": int(self.workers)}

    def fftn(self, x, **kwargs):
        return self._fft_module.fftn(x, **(self._kwargs() | kwargs))

    def ifftn(self, x, **kwargs):
        return self._fft_module.ifftn(x, **(self._kwargs() | kwargs))

    def rfftn(self, x, **kwargs):
        return self._fft_module.rfftn(x, **(self._kwargs() | kwargs))

    def irfftn(self, x, **kwargs):
        return self._fft_module.irfftn(x, **(self._kwargs() | kwargs))

    def fftfreq(self, n: int, d: float = 1.0):
        return self._fft_module.fftfreq(n, d=d)

    def rfftfreq(self, n: int, d: float = 1.0):
        return self._fft_module.rfftfreq(n, d=d)


def _scipy_backend(workers: int | None = None) -> FFTBackend:
    from scipy import fft as scipy_fft

    return FFTBackend(name="scipy", _fft_module=scipy_fft, workers=workers)


def _pyfftw_backend(workers: int | None = None) -> FFTBackend:
    import pyfftw.interfaces.cache as fftw_cache
    import pyfftw.interfaces.scipy_fft as fftw_fft

    fftw_cache.enable()
    return FFTBackend(name="pyfftw", _fft_module=fftw_fft, workers=workers)


def get_fft_backend(
    name: str = "scipy", workers: int | None = None, allow_fallback: bool = True
) -> FFTBackend:
    name_norm = name.lower()
    if name_norm == "scipy":
        return _scipy_backend(workers=workers)
    if name_norm == "pyfftw":
        try:
            return _pyfftw_backend(workers=workers)
        except Exception as exc:  # pragma: no cover - depends on local install/runtime
            if not allow_fallback:
                raise
            backend = _scipy_backend(workers=workers)
            return FFTBackend(
                name=backend.name,
                _fft_module=backend._fft_module,
                workers=backend.workers,
                fallback_reason=f"pyfftw unavailable: {exc}",
            )
    raise ValueError(f"Unsupported FFT backend: {name}")
