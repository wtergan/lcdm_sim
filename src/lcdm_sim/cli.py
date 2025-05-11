"""Command-line interface for lcdm_sim (early scaffold)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .config import load_simulation_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lcdm-sim", description="lcdm_sim CLI (WIP)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run a simulation from a config file"
    )
    run_parser.add_argument(
        "--config", type=Path, required=True, help="Path to simulation config (.yaml)"
    )
    run_parser.set_defaults(handler=_handle_run)

    plot_parser = subparsers.add_parser(
        "plot", help="Generate plots from an existing run directory"
    )
    plot_parser.add_argument("--run-dir", type=Path, required=True)
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


def _handle_run(args: argparse.Namespace) -> int:
    cfg = load_simulation_config(args.config)
    print(
        "run (stub): "
        f"{cfg.grid.n_particles_1d}^3 particles, "
        f"{cfg.grid.n_grid_1d}^3 grid, steps={cfg.integrator.num_steps}, "
        f"config={args.config}"
    )
    return 0


def _handle_plot(args: argparse.Namespace) -> int:
    print(f"plot (stub): run_dir={args.run_dir}")
    return 0


def _handle_validate(args: argparse.Namespace) -> int:
    print(f"validate (stub): run_dir={args.run_dir}, reference={args.reference}")
    return 0


def _handle_export_web_dataset(args: argparse.Namespace) -> int:
    print(f"export-web-dataset (stub): run_dir={args.run_dir}, out={args.out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
