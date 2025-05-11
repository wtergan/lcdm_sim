"""Typed configuration loading for lcdm_sim.

The loader supports normal YAML when PyYAML is installed. In environments without
PyYAML, it falls back to JSON parsing so JSON-compatible `.yaml` files still work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .types import (
    CosmologyConfig,
    GridConfig,
    IntegratorConfig,
    OutputConfig,
    PerformanceConfig,
    SimulationConfig,
    ValidationConfig,
)

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised by runtime environment
    yaml = None


class ConfigError(ValueError):
    """Raised when a configuration file cannot be parsed or validated."""


PathLike = str | Path


def _read_mapping(path: PathLike) -> Mapping[str, Any]:
    path_obj = Path(path)
    text = path_obj.read_text(encoding="utf-8")

    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ConfigError(
                "PyYAML is not installed, so config files must be JSON-compatible YAML. "
                f"Could not parse {path_obj}."
            ) from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Expected top-level mapping in config: {path_obj}")

    return data


def _require_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid mapping section: {key}")
    return value


def simulation_config_from_dict(data: Mapping[str, Any]) -> SimulationConfig:
    grid = _require_mapping(data, "grid")
    cosmology = _require_mapping(data, "cosmology")
    integrator = _require_mapping(data, "integrator")
    output = _require_mapping(data, "output")
    performance = _require_mapping(data, "performance")
    validation = _require_mapping(data, "validation")

    return SimulationConfig(
        grid=GridConfig(
            n_particles_1d=int(grid["n_particles_1d"]),
            n_grid_1d=int(grid["n_grid_1d"]),
            box_size_mpc_h=float(grid["box_size_mpc_h"]),
        ),
        cosmology=CosmologyConfig(
            h0=float(cosmology["h0"]),
            omega_m=float(cosmology["omega_m"]),
            omega_lambda=float(cosmology["omega_lambda"]),
            sigma8=float(cosmology["sigma8"]),
            n_s=float(cosmology["n_s"]),
            a_initial=float(cosmology["a_initial"]),
            a_final=float(cosmology["a_final"]),
            omega_b=float(cosmology.get("omega_b", 0.0486)),
            t_cmb=float(cosmology.get("t_cmb", 2.7255)),
            a_s=float(cosmology.get("a_s", 2.1e-9)),
            k_pivot_mpc=float(cosmology.get("k_pivot_mpc", 0.05)),
        ),
        integrator=IntegratorConfig(
            num_steps=int(integrator["num_steps"]),
            method=str(integrator.get("method", "kdk_a")),
        ),
        output=OutputConfig(
            output_root=str(output.get("output_root", "outputs")),
            save_density=bool(output.get("save_density", False)),
            save_plots=bool(output.get("save_plots", True)),
        ),
        performance=PerformanceConfig(
            fft_backend=str(performance.get("fft_backend", "scipy")),
            use_numba=bool(performance.get("use_numba", False)),
            fft_workers=int(performance.get("fft_workers", 1)),
        ),
        validation=ValidationConfig(
            enable_invariants=bool(validation.get("enable_invariants", True)),
            enable_reference_compare=bool(
                validation.get("enable_reference_compare", False)
            ),
        ),
        random_seed=int(data.get("random_seed", 38)),
    )


def load_simulation_config(path: PathLike) -> SimulationConfig:
    return simulation_config_from_dict(_read_mapping(path))
