"""Validation utilities for simulation outputs and run artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from .diagnostics import compare_cic_to_initial, density_stats
from .io_hdf5 import load_snapshot_hdf5
from .types import RunResult, SimulationConfig, Snapshot


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    summary: dict[str, Any]
    checks: list[dict[str, Any]]
    reference: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "summary": dict(self.summary),
            "checks": [dict(c) for c in self.checks],
            "reference": None if self.reference is None else dict(self.reference),
            "metadata": dict(self.metadata),
        }


def _check(
    name: str, passed: bool, *, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "details": {} if details is None else details,
    }


def _is_finite_array(x: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(np.asarray(x))))


def _positions_within_box(positions: np.ndarray, box_size_mpc_h: float) -> bool:
    pos = np.asarray(positions, dtype=float)
    L = float(box_size_mpc_h)
    return bool(np.all(pos >= 0.0) and np.all(pos < L))


def _snapshot_sequence_checks(snapshots: list[Snapshot]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "snapshots_present", len(snapshots) > 0, details={"count": len(snapshots)}
        )
    )
    if not snapshots:
        return checks

    steps = [int(s.step) for s in snapshots]
    a_vals = [float(s.particle_state.a) for s in snapshots]
    checks.append(
        _check(
            "snapshot_steps_monotonic",
            all(b >= a for a, b in zip(steps, steps[1:])),
            details={"steps": steps},
        )
    )
    checks.append(
        _check(
            "snapshot_a_monotonic",
            all(b >= a for a, b in zip(a_vals, a_vals[1:])),
            details={"a_values": a_vals},
        )
    )

    particle_shapes_ok = all(
        s.particle_state.positions.ndim == 2
        and s.particle_state.positions.shape[1] == 3
        for s in snapshots
    )
    checks.append(_check("snapshot_particle_shapes", particle_shapes_ok))

    particle_finite = all(
        _is_finite_array(s.particle_state.positions)
        and _is_finite_array(s.particle_state.velocities)
        for s in snapshots
    )
    checks.append(_check("snapshot_particles_finite", particle_finite))

    density_finite = True
    density_count = 0
    for s in snapshots:
        if s.density_field is not None:
            density_count += 1
            density_finite = density_finite and _is_finite_array(s.density_field.data)
    checks.append(
        _check(
            "snapshot_density_finite_if_present",
            density_finite,
            details={"density_snapshots": density_count},
        )
    )
    return checks


def _history_checks(run_result: RunResult) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if not run_result.history:
        checks.append(_check("history_present", False, details={"count": 0}))
        return checks

    steps = [int(row.get("step", -1)) for row in run_result.history]
    a_vals = [float(row.get("a", np.nan)) for row in run_result.history]
    checks.append(
        _check("history_present", True, details={"count": len(run_result.history)})
    )
    checks.append(
        _check("history_steps_monotonic", all(b >= a for a, b in zip(steps, steps[1:])))
    )
    checks.append(_check("history_a_finite", bool(np.all(np.isfinite(a_vals)))))
    return checks


def _run_result_reference_metrics(run_result: RunResult) -> dict[str, Any]:
    final_snapshot = run_result.snapshots[-1] if run_result.snapshots else None
    out: dict[str, Any] = {
        "num_snapshots": len(run_result.snapshots),
        "history_rows": len(run_result.history),
        "num_steps": int(run_result.config.integrator.num_steps),
    }
    if final_snapshot is not None:
        out["final_step"] = int(final_snapshot.step)
        out["final_a"] = float(final_snapshot.particle_state.a)
        vel = np.asarray(final_snapshot.particle_state.velocities, dtype=float)
        out["final_velocity_rms"] = (
            float(np.sqrt(np.mean(np.sum(vel * vel, axis=1)))) if vel.size else 0.0
        )
        if final_snapshot.density_field is not None:
            out["final_density_stats"] = density_stats(final_snapshot.density_field)
    return out


def _compare_run_results(reference: RunResult, candidate: RunResult) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ref_final = reference.snapshots[-1] if reference.snapshots else None
    cand_final = candidate.snapshots[-1] if candidate.snapshots else None
    if ref_final is None or cand_final is None:
        out["available"] = False
        return out

    out["available"] = True
    out["final_step_abs_diff"] = abs(int(cand_final.step) - int(ref_final.step))
    out["final_a_abs_diff"] = abs(
        float(cand_final.particle_state.a) - float(ref_final.particle_state.a)
    )

    ref_n = int(ref_final.particle_state.positions.shape[0])
    cand_n = int(cand_final.particle_state.positions.shape[0])
    out["particle_count_equal"] = cand_n == ref_n

    if ref_final.density_field is not None and cand_final.density_field is not None:
        out["final_density_compare"] = compare_cic_to_initial(
            ref_final.density_field, cand_final.density_field
        )
    return out


def run_validation_suite(
    run_result: RunResult,
    config: SimulationConfig | None = None,
    reference_artifacts: RunResult | None = None,
) -> ValidationReport:
    cfg = run_result.config if config is None else config
    checks: list[dict[str, Any]] = []

    checks.append(
        _check(
            "initial_density_finite", _is_finite_array(run_result.initial_density.data)
        )
    )
    checks.append(
        _check(
            "initial_density_shape_matches_grid",
            run_result.initial_density.data.shape
            == (cfg.grid.n_grid_1d, cfg.grid.n_grid_1d, cfg.grid.n_grid_1d),
        )
    )
    checks.append(
        _check(
            "initial_particles_finite",
            _is_finite_array(run_result.initial_state.positions)
            and _is_finite_array(run_result.initial_state.velocities),
        )
    )
    checks.append(
        _check(
            "initial_positions_within_box",
            _positions_within_box(
                run_result.initial_state.positions, cfg.grid.box_size_mpc_h
            ),
        )
    )

    checks.extend(_snapshot_sequence_checks(run_result.snapshots))

    step_durations = np.asarray(run_result.step_durations_s, dtype=float)
    checks.append(
        _check(
            "step_durations_count_matches_num_steps",
            int(step_durations.size) == int(cfg.integrator.num_steps),
            details={
                "count": int(step_durations.size),
                "expected": int(cfg.integrator.num_steps),
            },
        )
    )
    checks.append(
        _check("step_durations_nonnegative", bool(np.all(step_durations >= 0.0)))
    )
    checks.append(
        _check("step_durations_finite", bool(np.all(np.isfinite(step_durations))))
    )

    if run_result.snapshots:
        final_a = float(run_result.snapshots[-1].particle_state.a)
        checks.append(
            _check(
                "final_a_reaches_target",
                abs(final_a - float(cfg.cosmology.a_final)) <= 1e-8,
                details={"final_a": final_a, "target_a": float(cfg.cosmology.a_final)},
            )
        )

    checks.extend(_history_checks(run_result))

    reference = None
    if reference_artifacts is not None:
        reference = _compare_run_results(reference_artifacts, run_result)

    num_failed = sum(1 for c in checks if not c["passed"])
    summary = {
        "num_checks": len(checks),
        "num_failed": num_failed,
        "num_passed": len(checks) - num_failed,
        "run_id": run_result.run_id,
    }
    metadata = _run_result_reference_metrics(run_result)
    return ValidationReport(
        ok=(num_failed == 0),
        summary=summary,
        checks=checks,
        reference=reference,
        metadata=metadata,
    )


def _load_snapshots_from_run_dir(run_dir: str | Path) -> list[Snapshot]:
    run_path = Path(run_dir)
    snap_dir = run_path / "snapshots"
    files = sorted(snap_dir.glob("snapshot_*.h5"))
    return [load_snapshot_hdf5(p) for p in files]


def validate_run_directory(
    run_dir: str | Path,
    reference_run_dir: str | Path | None = None,
) -> ValidationReport:
    run_path = Path(run_dir)
    snapshots = _load_snapshots_from_run_dir(run_path)
    checks = _snapshot_sequence_checks(snapshots)
    checks.append(
        _check("run_dir_exists", run_path.exists(), details={"run_dir": str(run_path)})
    )
    checks.append(
        _check(
            "snapshot_files_present",
            len(snapshots) > 0,
            details={"count": len(snapshots)},
        )
    )

    reference = None
    if reference_run_dir is not None:
        ref_snaps = _load_snapshots_from_run_dir(reference_run_dir)
        if snapshots and ref_snaps:
            reference = {
                "final_step_abs_diff": abs(
                    int(snapshots[-1].step) - int(ref_snaps[-1].step)
                ),
                "final_a_abs_diff": abs(
                    float(snapshots[-1].a) - float(ref_snaps[-1].a)
                ),
                "snapshot_count_abs_diff": abs(len(snapshots) - len(ref_snaps)),
            }
            if (
                snapshots[-1].density_field is not None
                and ref_snaps[-1].density_field is not None
            ):
                reference["final_density_compare"] = compare_cic_to_initial(
                    ref_snaps[-1].density_field,
                    snapshots[-1].density_field,
                )
        else:
            reference = {"available": False}

    num_failed = sum(1 for c in checks if not c["passed"])
    summary = {
        "num_checks": len(checks),
        "num_failed": num_failed,
        "num_passed": len(checks) - num_failed,
        "run_dir": str(run_path),
    }
    metadata = {
        "snapshot_steps": [int(s.step) for s in snapshots],
        "final_a": None if not snapshots else float(snapshots[-1].a),
    }
    return ValidationReport(
        ok=(num_failed == 0),
        summary=summary,
        checks=checks,
        reference=reference,
        metadata=metadata,
    )


def save_validation_report_json(report: ValidationReport, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return out
