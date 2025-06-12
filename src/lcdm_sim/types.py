"""Shared typed configuration and runtime structures for lcdm_sim."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GridConfig:
    n_particles_1d: int
    n_grid_1d: int
    box_size_mpc_h: float

    @property
    def particle_spacing_mpc_h(self) -> float:
        return self.box_size_mpc_h / float(self.n_particles_1d)

    @property
    def grid_spacing_mpc_h(self) -> float:
        return self.box_size_mpc_h / float(self.n_grid_1d)

    @property
    def n_particles(self) -> int:
        return int(self.n_particles_1d) ** 3


@dataclass(frozen=True)
class CosmologyConfig:
    h0: float
    omega_m: float
    omega_lambda: float
    sigma8: float
    n_s: float
    a_initial: float
    a_final: float
    omega_b: float = 0.0486
    t_cmb: float = 2.7255
    a_s: float = 2.1e-9
    k_pivot_mpc: float = 0.05


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
    fft_workers: int = 1


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


@dataclass(frozen=True)
class MeshField:
    data: np.ndarray
    box_size_mpc_h: float
    units: str = "dimensionless"

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def n_grid_1d(self) -> int:
        return int(self.data.shape[0])

    @property
    def grid_spacing_mpc_h(self) -> float:
        return self.box_size_mpc_h / float(self.n_grid_1d)


@dataclass(frozen=True)
class ParticleState:
    positions: np.ndarray
    velocities: np.ndarray
    mass: float = 1.0
    a: float = 1.0


@dataclass(frozen=True)
class AccelerationField:
    ax: np.ndarray
    ay: np.ndarray
    az: np.ndarray
    box_size_mpc_h: float
    units: str = "comoving"

    def stacked(self) -> np.ndarray:
        return np.stack((self.ax, self.ay, self.az), axis=-1)


@dataclass(frozen=True)
class Snapshot:
    step: int
    particle_state: ParticleState
    density_field: MeshField | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def a(self) -> float:
        return float(self.particle_state.a)


@dataclass(frozen=True)
class RunResult:
    config: SimulationConfig
    initial_density: MeshField
    initial_state: ParticleState
    snapshots: list[Snapshot]
    history: list[dict[str, Any]]
    step_durations_s: list[float]
    total_runtime_s: float
    run_id: str
    output_dir: str | None = None

    @property
    def output_path(self) -> Path | None:
        if self.output_dir is None:
            return None
        return Path(self.output_dir)
