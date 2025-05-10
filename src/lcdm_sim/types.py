"""Shared typed configuration and runtime structures for lcdm_sim."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GridConfig:
    n_particles_1d: int
    n_grid_1d: int
    box_size_mpc_h: float


@dataclass(frozen=True)
class CosmologyConfig:
    h0: float
    omega_m: float
    omega_lambda: float
    sigma8: float
    n_s: float
    a_initial: float
    a_final: float


@dataclass(frozen=True)
class IntegratorConfig:
    num_steps: int
    method: str = "kdk_a"


@dataclass(frozen=True)
class OutputConfig:
    output_root: str = "outputs"
    save_density: bool = False
    save_plots: bool = True


@dataclass(frozen=True)
class PerformanceConfig:
    fft_backend: str = "scipy"
    use_numba: bool = False


@dataclass(frozen=True)
class ValidationConfig:
    enable_invariants: bool = True
    enable_reference_compare: bool = False


@dataclass(frozen=True)
class SimulationConfig:
    grid: GridConfig
    cosmology: CosmologyConfig
    integrator: IntegratorConfig
    output: OutputConfig
    performance: PerformanceConfig
    validation: ValidationConfig
    random_seed: int = 38
