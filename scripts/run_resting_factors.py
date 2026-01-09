"""Sweep resting-state simulations over alpha/beta scaling factors.

This script loads a YAML config (default: config/resting_with_factors.yaml),
constructs an alpha/beta grid (default 11x11 between 0 and 2), and runs the
existing experiment pipeline once per grid point. Each run writes its results
into results/resting_with_factors/run_<timestamp>/alpha_<a>_beta_<b>.
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from src.experiment.config_loader import load_base_config
from src.experiment.experiment_runner import run_experiment_with_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resting-state sweep over alpha/beta scaling factors",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/resting_with_factors.yaml"),
        help="Path to base YAML config",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/resting_with_factors"),
        help="Directory where sweep results will be stored",
    )
    parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        help="Explicit list of alpha values (overrides sweep.num_steps)",
    )
    parser.add_argument(
        "--beta-values",
        type=float,
        nargs="+",
        help="Explicit list of beta values (overrides sweep.num_steps)",
    )
    parser.add_argument(
        "--alpha-min",
        type=float,
        help="Override sweep.alpha_min",
    )
    parser.add_argument(
        "--alpha-max",
        type=float,
        help="Override sweep.alpha_max",
    )
    parser.add_argument(
        "--beta-min",
        type=float,
        help="Override sweep.beta_min",
    )
    parser.add_argument(
        "--beta-max",
        type=float,
        help="Override sweep.beta_max",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        help="Override sweep.num_steps (applied to both axes)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs without executing them",
    )
    return parser.parse_args()


def linspace_values(min_val: float, max_val: float, steps: int) -> np.ndarray:
    if steps < 2:
        raise ValueError("num_steps must be >= 2 to form a grid")
    return np.linspace(float(min_val), float(max_val), int(steps))


def prepare_sweep(cfg: dict, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, int]:
    sweep_cfg = cfg.get("sweep", {})
    alpha_min = args.alpha_min if args.alpha_min is not None else sweep_cfg.get("alpha_min", 0.0)
    alpha_max = args.alpha_max if args.alpha_max is not None else sweep_cfg.get("alpha_max", 2.0)
    beta_min = args.beta_min if args.beta_min is not None else sweep_cfg.get("beta_min", 0.0)
    beta_max = args.beta_max if args.beta_max is not None else sweep_cfg.get("beta_max", 2.0)
    num_steps = args.num_steps if args.num_steps is not None else sweep_cfg.get("num_steps", 11)

    if args.alpha_values is not None:
        alpha_vals = np.asarray(args.alpha_values, dtype=float)
    else:
        alpha_vals = linspace_values(alpha_min, alpha_max, num_steps)

    if args.beta_values is not None:
        beta_vals = np.asarray(args.beta_values, dtype=float)
    else:
        beta_vals = linspace_values(beta_min, beta_max, num_steps)

    seeds_per_point = int(sweep_cfg.get("seeds_per_point", 1))
    return alpha_vals, beta_vals, seeds_per_point


def ensure_resting_state(cfg: dict) -> None:
    """Force stimulation off so we only observe resting dynamics."""
    stim_cfg = cfg.setdefault("stimulation", {})
    stim_cfg.setdefault("dc", {})["enabled"] = False
    stim_cfg.setdefault("mnist", {})["enabled"] = False


def update_scaling_block(cfg: dict, alpha: float, beta: float) -> None:
    scaling_cfg = cfg.setdefault("scaling", {})
    scaling_cfg["alpha"] = float(alpha)
    scaling_cfg["beta"] = float(beta)
    scaling_cfg.setdefault("apply_mode", "post_init")


def write_manifest(run_dir: Path, metadata: dict) -> None:
    manifest_path = run_dir / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(metadata, indent=2))


def write_summary(summary_rows: list[dict], summary_path: Path) -> None:
    if not summary_rows:
        return
    fieldnames = list(summary_rows[0].keys())
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def save_config_snapshot(cfg: dict, snapshot_path: Path) -> None:
    snapshot_path.write_text(yaml.safe_dump(cfg))


def iter_grid(alpha_vals: Iterable[float], beta_vals: Iterable[float]):
    for alpha in alpha_vals:
        for beta in beta_vals:
            yield float(alpha), float(beta)


def main() -> None:
    args = parse_args()
    base_cfg = load_base_config(args.config)
    ensure_resting_state(base_cfg)

    alpha_vals, beta_vals, seeds_per_point = prepare_sweep(base_cfg, args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = args.output_root / f"run_{timestamp}"
    sweep_root.mkdir(parents=True, exist_ok=True)

    save_config_snapshot(base_cfg, sweep_root / "sweep_config_snapshot.yaml")

    summary_rows: list[dict] = []
    base_seed = int(base_cfg["experiment"].get("seed", 0) or 0)
    run_counter = 0
    total_runs = len(alpha_vals) * len(beta_vals) * seeds_per_point

    print(f"Starting sweep with {len(alpha_vals)} alpha values, "
          f"{len(beta_vals)} beta values, seeds per point = {seeds_per_point}.")
    print(f"Total simulations: {total_runs}")

    for alpha, beta in iter_grid(alpha_vals, beta_vals):
        for seed_idx in range(seeds_per_point):
            run_counter += 1
            cfg_point = deepcopy(base_cfg)
            ensure_resting_state(cfg_point)
            update_scaling_block(cfg_point, alpha, beta)

            run_name = f"alpha_{alpha:.2f}_beta_{beta:.2f}_s{seed_idx:02d}"
            run_dir = sweep_root / run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            cfg_point["experiment"]["name"] = run_name
            cfg_point["experiment"]["alpha"] = alpha
            cfg_point["experiment"]["beta"] = beta
            cfg_point["experiment"]["seed"] = base_seed + run_counter - 1

            metadata = {
                "alpha": alpha,
                "beta": beta,
                "seed_index": seed_idx,
                "seed": cfg_point["experiment"]["seed"],
                "run_name": run_name,
                "run_dir": str(run_dir),
            }
            write_manifest(run_dir, metadata)

            print(f"[{run_counter:04d}/{total_runs:04d}] alpha={alpha:.2f}, beta={beta:.2f}, seed_idx={seed_idx}")
            if args.dry_run:
                continue

            try:
                run_experiment_with_cfg(cfg_point, run_dir)
                metadata["status"] = "ok"
            except Exception as exc:  # pragma: no cover - runtime failures
                metadata["status"] = "failed"
                metadata["error"] = repr(exc)
                print(f"Run failed for alpha={alpha:.2f}, beta={beta:.2f}: {exc}")

            summary_rows.append(metadata)

    if not args.dry_run:
        summary_path = sweep_root / "sweep_summary.csv"
        write_summary(summary_rows, summary_path)
        print(f"Sweep complete. Summary written to {summary_path}")
    else:
        print("Dry run completed (no simulations executed).")


if __name__ == "__main__":
    main()
