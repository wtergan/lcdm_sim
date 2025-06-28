"""Command-line interface for running, plotting, and validating PM simulations."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from .config import load_simulation_config
from .io_hdf5 import load_snapshot_hdf5
from .plotting_static import (
    plot_density_evolution,
    plot_density_projection,
    plot_density_slice,
    plot_history_summary,
    plot_particle_scatter,
)
from .simulation import run_simulation
from .validation import (
    run_validation_suite,
    save_validation_report_json,
    validate_run_directory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lcdm-sim", description="Run and inspect LCDM particle-mesh simulations"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run a simulation from a config file"
    )
    run_parser.add_argument(
        "--config", type=Path, required=True, help="Path to simulation config (.yaml)"
    )
    run_parser.add_argument(
        "--out-dir",
        type=Path,
        help="Run output directory. Defaults to <output_root>/<config-stem>.",
    )
    run_parser.add_argument(
        "--num-snapshots",
        type=int,
        help="Number of snapshots to save, including the first and final state.",
    )
    run_parser.add_argument(
        "--history-stride",
        type=int,
        default=1,
        help="Record one history row every N integration steps.",
    )
    run_parser.add_argument(
        "--no-save-snapshots",
        action="store_true",
        help="Run the simulation without writing HDF5 snapshot files.",
    )
    run_parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip the post-run validation report.",
    )
    run_parser.set_defaults(handler=_handle_run)

    plot_parser = subparsers.add_parser(
        "plot", help="Generate plots from an existing run directory"
    )
    plot_parser.add_argument("--run-dir", type=Path, required=True)
    plot_parser.add_argument(
        "--out-dir",
        type=Path,
        help="Plot output directory. Defaults to <run-dir>/plots.",
    )
    plot_parser.add_argument(
        "--max-snapshots",
        type=int,
        default=6,
        help="Maximum number of snapshots to include in summary panels.",
    )
    plot_parser.add_argument("--axis", type=int, default=2)
    plot_parser.add_argument("--slice-index", type=int)
    plot_parser.add_argument("--max-points", type=int, default=12000)
    plot_parser.set_defaults(handler=_handle_plot)

    validate_parser = subparsers.add_parser("validate", help="Validate run outputs")
    validate_parser.add_argument("--run-dir", type=Path, required=True)
    validate_parser.add_argument("--reference", type=Path)
    validate_parser.set_defaults(handler=_handle_validate)

    export_parser = subparsers.add_parser(
        "export-web-dataset",
        help="Export run outputs into a browser-emulator dataset bundle",
    )
    export_parser.add_argument("--run-dir", type=Path, required=True)
    export_parser.add_argument("--out", type=Path, required=True)
    export_parser.set_defaults(handler=_handle_export_web_dataset)

    return parser


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _default_run_dir(config_path: Path, output_root: str) -> Path:
    return Path(output_root) / config_path.stem


def _snapshot_slug(step: int, a: float) -> str:
    a_part = f"{a:.3f}".replace(".", "p")
    return f"snapshot_{int(step):04d}_a{a_part}"


def _select_evenly_spaced(paths: list[Path], max_items: int) -> list[Path]:
    if len(paths) <= max_items:
        return paths
    import numpy as np

    indices = np.linspace(0, len(paths) - 1, num=max(1, int(max_items)), dtype=int)
    return [paths[int(i)] for i in sorted(set(indices))]


def _handle_run(args: argparse.Namespace) -> int:
    cfg = load_simulation_config(args.config)
    out_dir = args.out_dir or _default_run_dir(args.config, cfg.output.output_root)
    result = run_simulation(
        cfg,
        output_dir=out_dir,
        num_snapshots=args.num_snapshots,
        history_stride=args.history_stride,
        save_snapshots=not args.no_save_snapshots,
    )

    metrics_dir = out_dir / "metrics"
    history_path = _write_json(metrics_dir / "history.json", result.history)
    summary_path = _write_json(
        metrics_dir / "run_summary.json",
        {
            "run_id": result.run_id,
            "config": asdict(cfg),
            "num_snapshots": len(result.snapshots),
            "history_rows": len(result.history),
            "total_runtime_s": result.total_runtime_s,
            "output_dir": str(out_dir),
        },
    )

    validation_path = None
    ok = True
    if not args.skip_validation:
        report = run_validation_suite(result, cfg)
        validation_path = save_validation_report_json(
            report, metrics_dir / "validation_report.json"
        )
        ok = report.ok

    print(
        "run: "
        f"ok={ok} run_id={result.run_id} "
        f"{cfg.grid.n_particles_1d}^3 particles, "
        f"{cfg.grid.n_grid_1d}^3 grid, steps={cfg.integrator.num_steps}, "
        f"snapshots={len(result.snapshots)} output={out_dir} "
        f"history={history_path} summary={summary_path}"
    )
    if validation_path is not None:
        print(f"validation report: {validation_path}")
    return 0 if ok else 1


def _handle_plot(args: argparse.Namespace) -> int:
    snapshot_dir = args.run_dir / "snapshots"
    snapshot_paths = sorted(snapshot_dir.glob("snapshot_*.h5"))
    if not snapshot_paths:
        print(f"plot: no snapshots found under {snapshot_dir}")
        return 1

    chosen_paths = _select_evenly_spaced(snapshot_paths, args.max_snapshots)
    snapshots = [load_snapshot_hdf5(path) for path in chosen_paths]
    out_dir = args.out_dir or (args.run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, str | int | float]] = []
    for snapshot in snapshots:
        slug = _snapshot_slug(snapshot.step, snapshot.a)
        particle_path = plot_particle_scatter(
            snapshot.particle_state,
            out_dir / f"{slug}_particles.png",
            max_points=args.max_points,
        )
        artifacts.append(
            {
                "kind": "particles",
                "step": snapshot.step,
                "a": snapshot.a,
                "path": str(particle_path),
            }
        )

        if snapshot.density_field is None:
            continue

        slice_path = plot_density_slice(
            snapshot.density_field,
            out_dir / f"{slug}_density_slice.png",
            axis=args.axis,
            index=args.slice_index,
            title=f"Density Slice: step {snapshot.step}, a={snapshot.a:.3f}",
        )
        projection_path = plot_density_projection(
            snapshot.density_field,
            out_dir / f"{slug}_density_projection.png",
            axis=args.axis,
        )
        artifacts.extend(
            [
                {
                    "kind": "density_slice",
                    "step": snapshot.step,
                    "a": snapshot.a,
                    "path": str(slice_path),
                },
                {
                    "kind": "density_projection",
                    "step": snapshot.step,
                    "a": snapshot.a,
                    "path": str(projection_path),
                },
            ]
        )

    if any(snapshot.density_field is not None for snapshot in snapshots):
        evolution_path = plot_density_evolution(
            snapshots,
            out_dir / "density_evolution.png",
            axis=args.axis,
            index=args.slice_index,
            max_snapshots=args.max_snapshots,
        )
        artifacts.append({"kind": "density_evolution", "path": str(evolution_path)})

    history_path = args.run_dir / "metrics" / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        history_plot_path = plot_history_summary(
            history, out_dir / "history_summary.png"
        )
        artifacts.append({"kind": "history_summary", "path": str(history_plot_path)})

    manifest_path = _write_json(
        out_dir / "plot_manifest.json",
        {
            "run_dir": str(args.run_dir),
            "snapshot_count": len(snapshot_paths),
            "plotted_snapshots": [str(path) for path in chosen_paths],
            "artifacts": artifacts,
        },
    )
    print(f"plot: wrote={len(artifacts)} out_dir={out_dir} manifest={manifest_path}")
    return 0


def _handle_validate(args: argparse.Namespace) -> int:
    report = validate_run_directory(args.run_dir, reference_run_dir=args.reference)
    metrics_dir = Path(args.run_dir) / "metrics"
    report_path = save_validation_report_json(
        report, metrics_dir / "validation_report.json"
    )
    print(
        "validation: "
        f"ok={report.ok} passed={report.summary['num_passed']} "
        f"failed={report.summary['num_failed']} report={report_path}"
    )
    return 0 if report.ok else 1


def _handle_export_web_dataset(args: argparse.Namespace) -> int:
    print(f"export-web-dataset (stub): run_dir={args.run_dir}, out={args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
