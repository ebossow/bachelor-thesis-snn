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
    plot_spike_raster,
    plot_pdf_cv,
    plot_pdf_R,
    plot_pdf_mean_rate,
    plot_weight_matrix,
    plot_K,
    plot_spike_raster_ax,
    plot_pdf_cv_ax,
    plot_pdf_R_ax,
    plot_pdf_mean_rate_ax,
    plot_weight_matrix_ax,
    plot_K_ax
)

from src.analysis.util import load_run, find_latest_run_dir

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--small_figure", type=bool, default=False,
                   help="If True, only plot spike raster in small figure.")
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
    #print(metrics)

    fig = create_summary_figure(cfg, data, metrics, weights_data)

    if small_figure:
        fig, ax_raster = plt.subplots(figsize=(10, 5))
        plot_spike_raster_ax(ax_raster, data, cfg)

    plt.show()
    """
    # for plotting
    syn_cfg = cfg["synapses"]
    base_Wmax = syn_cfg["base_Wmax"]
    global_lr = syn_cfg["global_lr"]
    lr_E  = float(syn_cfg["E_to_X"]["synapse_parameter"]["lambda"])
    lr_IH = float(syn_cfg["IH_to_X"]["synapse_parameter"]["eta"])
    lr_IA = float(syn_cfg["IA_to_X"]["synapse_parameter"]["eta"])
    Wmax_E  = int(syn_cfg["E_to_X"]["Wmax_factor"] * base_Wmax)
    Wmax_IH = int(syn_cfg["IH_to_X"]["Wmax_factor"] * base_Wmax)
    Wmax_IA = int(syn_cfg["IA_to_X"]["Wmax_factor"] * base_Wmax)
    N = cfg["network"]["N_E"] + cfg["network"]["N_IH"] + cfg["network"]["N_IA"]

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(
        2, 8,
        height_ratios=[2.0, 1.5],
        hspace=0.5,
        wspace=0.4,
    )

    # Zeile 0: Raster über alle 3 Spalten
    ax_raster = fig.add_subplot(gs[0, 0:6])
    ax_W = fig.add_subplot(gs[0, 6:8])

    # Zeile 1: 
    ax_cv   = fig.add_subplot(gs[1, 0:2])
    ax_R    = fig.add_subplot(gs[1, 2:4])
    ax_rate = fig.add_subplot(gs[1, 4:6])
    ax_K    = fig.add_subplot(gs[1, 6:8])

    # learning rates und Wmax
    text_lr = (
        f"E: {lr_E:.3g}, Wmax = {Wmax_E:.5g}\n"
        f"IH: {lr_IH:.3g}, Wmax = {Wmax_IH:.5g}\n"
        f"IA: {lr_IA:.3g}, Wmax = {Wmax_IA:.5g}\n"
        f"Global_LR: {global_lr:.3g}"
    )
    fig.text(
        0.5, 0.98,
        text_lr,
        ha="center",
        va="top",
        fontsize=9,
    )

    # Weight Matrix Analyse
    W = build_weight_matrix(
        weights_data["sources"],
        weights_data["targets"],
        weights_data["weights"],
        N_total=N,
    )
    Wn = normalize_weight_matrix(W, cfg)

    cv_N = metrics["cv_N"]
    R = metrics["R"]
    mean_rates_per_neuron = metrics["mean_rates_per_neuron"]
    K_post = metrics["K_post"]
    # Zeichnen
    plot_spike_raster_ax(ax_raster, data, cfg)
    plot_pdf_cv_ax(ax_cv, cv_N)
    plot_pdf_R_ax(ax_R, R)
    plot_pdf_mean_rate_ax(ax_rate, mean_rates_per_neuron)
    plot_K_ax(ax_K, K_post)

    im = plot_weight_matrix_ax(ax_W, Wn, cfg)
    cbar = fig.colorbar(im, ax=ax_W)
    cbar.set_label("Normalized weight")

    plt.tight_layout()
    plt.show()
    """

if __name__ == "__main__":
    main()