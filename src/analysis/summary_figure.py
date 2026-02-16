# src/analysis/summary_figure.py

from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .plotting import (
    plot_spike_raster_ax,
    plot_pdf_cv_ax,
    plot_kuramoto_traces_ax,
    plot_pdf_mean_rate_ax,
    plot_weight_matrix_ax,
    plot_K_ax,
    add_weight_matrix_colorbar,
)
from .metrics import build_weight_matrix, normalize_weight_matrix


def _plot_stimulus_rate_distribution(
    ax: plt.Axes,
    stimulus_rate_traces,
) -> None:
    """Plot stimulus population PDFs with highlighted mean and std dev."""

    colors = ["#2ca02c", "#ff7f0e"]

    max_rate = 0.0
    for trace in stimulus_rate_traces:
        rates = np.asarray(trace.get("rate_Hz", []), float)
        rates = rates[np.isfinite(rates)]
        if rates.size == 0:
            continue
        max_rate = max(max_rate, float(np.nanmax(rates)))
    if max_rate <= 0.0:
        max_rate = 1.0

    bins = np.linspace(0.0, max_rate * 1.05, 60)
    bins[0] = 0.0

    max_density = 0.0
    legend_handles: list[Line2D] = []

    for idx, trace in enumerate(stimulus_rate_traces):
        rates = np.asarray(trace.get("rate_Hz", []), float)
        rates = rates[np.isfinite(rates)]
        if rates.size == 0:
            continue

        counts, edges = np.histogram(rates, bins=bins, density=True)
        counts = np.clip(counts, 1e-12, None)
        centers = 0.5 * (edges[:-1] + edges[1:])

        label = trace.get("label", f"P{idx + 1}")
        color = colors[idx % len(colors)]
        ax.step(centers, counts, where="mid", color=color, linewidth=1.8)

        count_max = float(counts.max()) if counts.size else 0.0
        max_density = max(max_density, count_max)

        mean_rate = float(np.nanmean(rates)) if rates.size else 0.0
        std_rate = float(np.nanstd(rates)) if rates.size else 0.0

        ax.axvline(mean_rate, color=color, linestyle="--", linewidth=1.1)

        legend_handles.append(Line2D([], [], color=color, linewidth=1.8, label=label))

        if std_rate > 0.0:
            base_height = count_max if count_max > 0 else (max_density if max_density > 0 else 1.0)
            y_marker = base_height * 1.15
            max_density = max(max_density, y_marker)
            ax.errorbar(
                mean_rate,
                y_marker,
                xerr=std_rate,
                fmt="o",
                color=color,
                markersize=5,
                markerfacecolor=color,
                markeredgecolor="white",
                capsize=5,
                capthick=1.0,
                linewidth=1.0,
            )

    if max_density <= 0.0:
        max_density = 1.0
    ax.set_ylim(0.0, max_density * 1.2)

    #ax.set_title("Stimulus population rate distributions")
    ax.set_xlabel("Mean firing rate (Hz)")
    ax.set_ylabel("PDF")
    ax.set_xlim(left=0.0)

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)


def create_summary_figure(
    cfg: Dict[str, Any],
    data: Dict[str, Any],
    metrics: Dict[str, Any],
    weights_data: Dict[str, Any] | None = None,
    max_raster_time_ms: float | None = None,
) -> plt.Figure:
    """Generate the combined summary plot for a single run."""

    syn_cfg = cfg["synapses"]
    base_Wmax = syn_cfg["base_Wmax"]
    global_lr = syn_cfg["global_lr"]
    lr_E = float(syn_cfg["E_to_X"]["synapse_parameter"]["lambda"])
    lr_IH = float(syn_cfg["IH_to_X"]["synapse_parameter"]["eta"])
    lr_IA = float(syn_cfg["IA_to_X"]["synapse_parameter"]["eta"])
    Wmax_E = int(syn_cfg["E_to_X"]["synapse_parameter"]["Wmax"])
    Wmax_IH = int(syn_cfg["IH_to_X"]["synapse_parameter"]["Wmax"])
    Wmax_IA = int(syn_cfg["IA_to_X"]["synapse_parameter"]["Wmax"])
    N = cfg["network"]["N_E"] + cfg["network"]["N_IH"] + cfg["network"]["N_IA"]

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 8, height_ratios=[1.3, 1], hspace=0.25, wspace=1.5)

    ax_raster = fig.add_subplot(gs[0, 0:6])
    ax_W = fig.add_subplot(gs[0, 6:8])

    ax_cv = fig.add_subplot(gs[1, 0:2])
    ax_R = fig.add_subplot(gs[1, 2:4])
    ax_rate = fig.add_subplot(gs[1, 4:6])
    ax_K = fig.add_subplot(gs[1, 6:8])

    include_text = False
    if include_text:
        text_lr = (
            f"E: {lr_E:.3g}, Wmax = {Wmax_E:.5g}\n"
            f"IH: {lr_IH:.3g}, Wmax = {Wmax_IH:.5g}\n"
            f"IA: {lr_IA:.3g}, Wmax = {Wmax_IA:.5g}\n"
            f"Global_LR: {global_lr:.3g}"
        )
        fig.text(0.5, 0.98, text_lr, ha="center", va="top", fontsize=9)

    if weights_data is None:
        raise ValueError("weights_data must be provided to plot the weight matrix")

    W = build_weight_matrix(
        weights_data["sources"],
        weights_data["targets"],
        weights_data["weights"],
        N_total=N,
    )
    Wn = normalize_weight_matrix(W, cfg)

    cv_N = metrics["cv_N"]
    R = metrics["R"]
    kuramoto_traces = metrics.get("kuramoto_traces") or []
    mean_rates_per_neuron = metrics["mean_rates_per_neuron"]
    K_post = metrics["K_post"]
    stimulus_rate_traces = metrics.get("stimulus_rate_traces") or []

    plot_spike_raster_ax(ax_raster, data, cfg, max_time_ms=max_raster_time_ms)
    ax_raster.set_box_aspect(1 / 3)
    plot_pdf_cv_ax(ax_cv, cv_N)
    if kuramoto_traces:
        plot_kuramoto_traces_ax(ax_R, kuramoto_traces)
    else:
        plot_kuramoto_traces_ax(
            ax_R,
            [
                {
                    "label": "All neurons",
                    "time_ms": np.arange(R.size),
                    "R": R,
                }
            ],
        )
    if stimulus_rate_traces:
        _plot_stimulus_rate_distribution(ax_rate, stimulus_rate_traces)
    else:
        plot_pdf_mean_rate_ax(ax_rate, mean_rates_per_neuron)
    plot_K_ax(ax_K, K_post)

    im = plot_weight_matrix_ax(ax_W, Wn, cfg)
    add_weight_matrix_colorbar(ax_W, im)

    plt.tight_layout()
    return fig
