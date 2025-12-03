# src/analysis/summary_figure.py

from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from .plotting import (
    plot_spike_raster_ax,
    plot_pdf_cv_ax,
    plot_pdf_R_ax,
    plot_pdf_mean_rate_ax,
    plot_weight_matrix_ax,
    plot_K_ax,
)
from .metrics import build_weight_matrix, normalize_weight_matrix


def create_summary_figure(
    cfg: Dict[str, Any],
    data: Dict[str, Any],
    metrics: Dict[str, Any],
    weights_data: Dict[str, Any] | None = None,
) -> plt.Figure:
    """
    Erzeugt die kombinierte Analyse-Figur für einen Run und gibt die Figure zurück.
    Kein plt.show(), kein Savefig hier – das machen die Aufrufer.

    Erwartet:
      - cfg: Config
      - data: enthält spikes_E, spikes_IH, spikes_IA, ...
      - metrics: Output von compute_summary_metrics (inkl. Arrays)
      - weights_data: dict mit "sources", "targets", "weights" oder None
    """

    syn_cfg = cfg["synapses"]
    base_Wmax = syn_cfg["base_Wmax"]
    global_lr = syn_cfg["global_lr"]
    lr_E  = float(syn_cfg["E_to_X"]["synapse_parameter"]["lambda"])
    lr_IH = float(syn_cfg["IH_to_X"]["synapse_parameter"]["eta"])
    lr_IA = float(syn_cfg["IA_to_X"]["synapse_parameter"]["eta"])
    Wmax_E  = int(syn_cfg["E_to_X"]["synapse_parameter"]["Wmax"])
    Wmax_IH = int(syn_cfg["IH_to_X"]["synapse_parameter"]["Wmax"])
    Wmax_IA = int(syn_cfg["IA_to_X"]["synapse_parameter"]["Wmax"])
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

    return fig