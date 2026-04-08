"""Run transfer plotting jobs from a YAML spec."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from src.plot_generators.plot_runner import (
        build_tasks,
        execute_tasks,
        load_plot_spec,
    )
except ModuleNotFoundError:
    from plot_runner import build_tasks, execute_tasks, load_plot_spec


DEFAULT_SPEC_PATH = Path(__file__).resolve().parent / "configs" / "transfer.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_SPEC_PATH),
        help="Path to plotting YAML spec",
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="Checkpoint folder name override"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override output root (default from spec)",
    )
    parser.add_argument("--no-token-figs", action="store_true")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument(
        "--debug", action="store_true", help="Run sequentially for easier debugging"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_plot_spec(args.config)

    tasks = build_tasks(
        spec,
        ckpt=args.ckpt,
        make_token_figs=not args.no_token_figs,
        output_root=args.output_root,
    )

    print(f"[INFO] Loaded spec: {args.config}")
    print(f"[INFO] Starting {len(tasks)} tasks...")
    print(
        f"[INFO] Mode: {'DEBUG (sequential)' if args.debug else f'PARALLEL ({args.workers} workers)'}"
    )

    failed_plots = execute_tasks(tasks, workers=args.workers, debug=args.debug)

    print("\n" + "=" * 30)
    if failed_plots:
        print(f"[SUMMARY] {len(failed_plots)} tasks failed/missing:")
        for msg in failed_plots:
            print(f" - {msg}")
    else:
        print("[SUMMARY] All relevant plots generated successfully.")
    print("=" * 30)


if __name__ == "__main__":
    main()
