# scripts/generate_sweep_figures.py

from pathlib import Path
import csv
import argparse

import matplotlib.pyplot as plt

from src.analysis.util import load_run
from src.analysis.summary_metrics import compute_summary_metrics
from src.analysis.summary_figure import create_summary_figure


def load_sweep_rows(sweep_root: Path):
    csv_path = sweep_root / "sweep_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No sweep_learning_rates.csv found in {sweep_root}")

    rows = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_dir = Path(row["run_dir"])
            rows.append((run_dir, row))
    return rows


def find_latest_sweep_root(results_root: Path) -> Path:
    sweep_dirs = sorted(
        [p for p in results_root.iterdir()
         if p.is_dir() and p.name.startswith("sweep_run_")]
    )
    if not sweep_dirs:
        raise RuntimeError("No sweep_run_* directories found under results/")
    return sweep_dirs[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Generate summary figures for all runs in a sweep."
    )
    parser.add_argument(
        "--sweep-root",
        type=str,
        default=None,
        help="Path to sweep_lr_run_... directory. If omitted, the latest is used.",
    )
    args = parser.parse_args()

    results_root = Path("results")

    if args.sweep_root is not None:
        sweep_root = Path(args.sweep_root)
        if not sweep_root.exists():
            raise FileNotFoundError(f"sweep_root does not exist: {sweep_root}")
    else:
        sweep_root = find_latest_sweep_root(results_root)

    print(f"Using sweep directory: {sweep_root}")

    figures_dir = sweep_root / "figures"
    figures_dir.mkdir(exist_ok=True)

    runs = load_sweep_rows(sweep_root)

    for run_dir, row in runs:
        cfg, data, weights_data, weights_over_time = load_run(run_dir)
        metrics = compute_summary_metrics(cfg, data, weights_over_time)

        fig = create_summary_figure(cfg, data, metrics, weights_data)

        # sinnvoller Dateiname: experimentname__runname.png
        run_label = f"{run_dir.parent.name}__{run_dir.name}"
        out_path = figures_dir / f"{run_label}.png"

        print(f"Saving figure: {out_path}")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()