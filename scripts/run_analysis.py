from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.analysis.summary_metrics import compute_summary_metrics
from src.analysis.summary_figure import create_summary_figure

from src.analysis.metrics import (
    build_weight_matrix,
    normalize_weight_matrix,
)
from src.analysis.plotting import (
    plot_spike_raster_ax,
    plot_weight_matrix_ax,
)
from src.analysis.util import load_run, find_latest_run_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument(
        "--small_figure",
        help="If set, create separate figures: raster + weight matrix (no big summary).",
        type=bool,
        default=False
    )
    return p.parse_args()


def main():
    args = parse_args()
    small_figure = args.small_figure

    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        print(f"Using specified run directory: {run_dir}")
    else:
        results_root = Path("results")
        run_dir = find_latest_run_dir(results_root)
        print(f"Using latest run directory: {run_dir}")

    cfg, data, weights_data, weights_over_time = load_run(run_dir)
    metrics = compute_summary_metrics(cfg, data, weights_over_time)

    net_cfg = cfg["network"]
    N_total = (
        net_cfg["N_E"]
        + net_cfg["N_IA_1"]
        + net_cfg["N_IH"]
        + net_cfg["N_IA_2"]
    )

    # Weight-Matrix (für beide Modi nötig)
    W = build_weight_matrix(
        weights_data["sources"],
        weights_data["targets"],
        weights_data["weights"],
        N_total=N_total,
    )
    Wn = normalize_weight_matrix(W, cfg)

    if not small_figure:
        # große Summary-Figur mit allen Subplots
        fig = create_summary_figure(cfg, data, metrics, weights_data)
    else:
        # 1) separater Raster-Plot
        fig_r, ax_r = plt.subplots(figsize=(10, 5))
        plot_spike_raster_ax(ax_r, data, cfg)
        fig_r.tight_layout()

        # 2) separate Weight-Matrix in groß
        fig_w, ax_w = plt.subplots(figsize=(6, 6))
        im = plot_weight_matrix_ax(ax_w, Wn, cfg)
        cbar = fig_w.colorbar(im, ax=ax_w)
        cbar.set_label("Normalized weight")
        fig_w.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()