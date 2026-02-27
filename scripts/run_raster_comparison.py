from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Iterable
import sys

import matplotlib
import numpy as np

from src.analysis.util import load_run

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import ConnectionPatch
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.analyze_factor_sweep import build_sigma_colormap
except ImportError:
    from analyze_factor_sweep import build_sigma_colormap


DEFAULT_SWEEP_ROOT = Path(
    "/Volumes/X9Pro/Backup Bachelor/260126/multiple_run_100_times_20260120_140859/run_030"
)
DEFAULT_RUN_NAMES = [
    "alpha_0.11_beta_1.89_s00",
    "alpha_0.53_beta_0.11_s00",
    "alpha_1.79_beta_0.00_s00",
]
DEFAULT_OUTPUT_DIR = Path("results/plots/raster_comparison")
DEFAULT_CRITICALITY_ANALYSIS_DIR = Path(
    "results/criticality_analysis/multiple_run_100_times_20260120_140859_analysis"
)
DEFAULT_CRITICALITY_RUN_ANALYSIS_DIR = Path(
    "results/criticality_analysis/multiple_run_100_times_20260120_140859_analysis/run_030_analysis"
)


def setup_matplotlib_style() -> None:
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a 3-row raster comparison across selected E/I configurations "
            "with stimulus interval removed (pre + post stitched)."
        )
    )
    parser.add_argument(
        "--sweep_root",
        type=Path,
        default=DEFAULT_SWEEP_ROOT,
        help="Path to the sweep run root containing alpha_... folders.",
    )
    parser.add_argument(
        "--run_names",
        nargs="+",
        default=DEFAULT_RUN_NAMES,
        help="Run folder names to compare (exactly 3 expected).",
    )
    parser.add_argument(
        "--pre_s",
        type=float,
        default=20.0,
        help="Seconds before stimulus onset to include.",
    )
    parser.add_argument(
        "--post_s",
        type=float,
        default=40.0,
        help="Seconds after stimulus offset to include.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the comparison plot.",
    )
    parser.add_argument(
        "--point_size",
        type=float,
        default=3.0,
        help="Scatter marker size for spikes.",
    )
    parser.add_argument(
        "--criticality_analysis_dir",
        type=Path,
        default=DEFAULT_CRITICALITY_ANALYSIS_DIR,
        help=(
            "Path to averaged criticality analysis folder containing averaged branching grids "
            "for the heatmaps (e.g. .../multiple_run_100_times_20260120_140859_analysis)."
        ),
    )
    parser.add_argument(
        "--criticality_run_analysis_dir",
        type=Path,
        default=DEFAULT_CRITICALITY_RUN_ANALYSIS_DIR,
        help=(
            "Path to run-specific criticality analysis folder for calculating m values "
            "shown in the raster plots (e.g. .../run_030_analysis)."
        ),
    )
    parser.add_argument(
        "--m_window",
        type=str,
        default="post_learning",
        choices=["pre_learning", "post_learning"],
        help="Criticality window from which m values are taken.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot interactively in addition to saving.",
    )
    args = parser.parse_args()

    if len(args.run_names) != 3:
        parser.error("--run_names must contain exactly 3 run names for a 3-row figure.")
    if args.pre_s <= 0 or args.post_s <= 0:
        parser.error("--pre_s and --post_s must be > 0.")

    return args


def _get_stim_times_ms(cfg: Dict[str, Any], stim_metadata: Dict[str, Any] | None) -> tuple[float, float]:
    pattern_meta = (stim_metadata or {}).get("pattern", {}) if isinstance(stim_metadata, dict) else {}
    pattern_cfg = cfg.get("stimulation", {}).get("pattern", {})

    t_on_ms = pattern_meta.get("t_on_ms", pattern_cfg.get("t_on_ms"))
    t_off_ms = pattern_meta.get("t_off_ms", pattern_cfg.get("t_off_ms"))

    if t_on_ms is None or t_off_ms is None:
        raise ValueError("Could not resolve stimulus t_on_ms / t_off_ms from metadata or config.")

    t_on_ms = float(t_on_ms)
    t_off_ms = float(t_off_ms)
    if t_off_ms <= t_on_ms:
        raise ValueError(f"Invalid stimulus window: t_on={t_on_ms}, t_off={t_off_ms}")
    return t_on_ms, t_off_ms


def _stack_spikes(data: Dict[str, Any], keys: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    times_list: list[np.ndarray] = []
    senders_list: list[np.ndarray] = []
    for key in keys:
        pop = data.get(key)
        if pop is None:
            continue
        times_list.append(np.asarray(pop["times"], dtype=float))
        senders_list.append(np.asarray(pop["senders"], dtype=int))

    if not times_list:
        return np.array([], dtype=float), np.array([], dtype=int)

    return np.concatenate(times_list), np.concatenate(senders_list)


def _slice_and_shift(
    times_ms: np.ndarray,
    senders: np.ndarray,
    t_on_ms: float,
    t_off_ms: float,
    pre_s: float,
    post_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    pre_start_ms = t_on_ms - pre_s * 1000.0
    post_end_ms = t_off_ms + post_s * 1000.0

    mask_pre = (times_ms >= pre_start_ms) & (times_ms < t_on_ms)
    mask_post = (times_ms > t_off_ms) & (times_ms <= post_end_ms)

    times_pre = (times_ms[mask_pre] - pre_start_ms) / 1000.0
    senders_pre = senders[mask_pre]

    times_post = pre_s + (times_ms[mask_post] - t_off_ms) / 1000.0
    senders_post = senders[mask_post]

    stitched_times = np.concatenate([times_pre, times_post]) if (times_pre.size + times_post.size) > 0 else np.array([], dtype=float)
    stitched_senders = np.concatenate([senders_pre, senders_post]) if (senders_pre.size + senders_post.size) > 0 else np.array([], dtype=int)

    if stitched_times.size == 0:
        return stitched_times, stitched_senders

    order = np.argsort(stitched_times)
    return stitched_times[order], stitched_senders[order]


def _load_sigma_grid(metrics_path: Path) -> dict[str, np.ndarray]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.npz not found: {metrics_path}")
    with np.load(metrics_path, allow_pickle=False) as d:
        return {
            "alphas": np.asarray(d["alphas"], dtype=float),
            "betas": np.asarray(d["betas"], dtype=float),
            "sigma_mr": np.asarray(d["sigma_mr"], dtype=float),
        }


def _lookup_sigma_for_alpha_beta(
    alpha: float,
    beta: float,
    grid: dict[str, np.ndarray],
) -> float:
    alphas = grid["alphas"]
    betas = grid["betas"]
    sigma = grid["sigma_mr"]

    alpha_idx = int(np.argmin(np.abs(alphas - alpha)))
    beta_idx = int(np.argmin(np.abs(betas - beta)))
    return float(sigma[beta_idx, alpha_idx])


def _plot_heatmap_with_markers(
    ax: plt.Axes,
    sigma_matrix: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    marker_configs: list[tuple[float, float]],
    title: str,
    show_ylabel: bool = True,
    cmap=None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> matplotlib.image.AxesImage:
    extent = [float(alphas.min()), float(alphas.max()), float(betas.min()), float(betas.max())]

    im = ax.imshow(
        sigma_matrix,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    marker_styles = ['^', 's', 'o']
    marker_labels = ['A', 'B', 'C']
    for i, (alpha, beta) in enumerate(marker_configs):
        ax.plot(alpha, beta, marker=marker_styles[i], markersize=10,
            linestyle='None', markerfacecolor='none', markeredgecolor='white', markeredgewidth=2.5,
                label=marker_labels[i])
        ax.plot(alpha, beta, marker=marker_styles[i], markersize=10,
            linestyle='None', markerfacecolor='none', markeredgecolor='black', markeredgewidth=1.2)

    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$s_{exc}$", fontsize=10)
    if show_ylabel:
        ax.set_ylabel(r"$s_{inh}$", fontsize=10)

    tick_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    ax.set_xticks(tick_values)
    ax.set_yticks(tick_values)

    return im


def _plot_single_row(
    ax: plt.Axes,
    cfg: Dict[str, Any],
    data: Dict[str, Any],
    run_name: str,
    pre_s: float,
    post_s: float,
    point_size: float,
    stim_metadata: Dict[str, Any] | None,
    m_network_pre: float,
    m_cluster_1_pre: float,
    m_network_post: float,
    m_cluster_1_post: float,
) -> None:
    t_on_ms, t_off_ms = _get_stim_times_ms(cfg, stim_metadata)

    spikes_e = data.get("spikes_E") or {"times": np.array([]), "senders": np.array([])}
    times_e = np.asarray(spikes_e["times"], dtype=float)
    senders_e = np.asarray(spikes_e["senders"], dtype=int)

    times_i, senders_i = _stack_spikes(data, ("spikes_IH", "spikes_IA"))

    plot_t_e, plot_s_e = _slice_and_shift(times_e, senders_e, t_on_ms, t_off_ms, pre_s, post_s)
    plot_t_i, plot_s_i = _slice_and_shift(times_i, senders_i, t_on_ms, t_off_ms, pre_s, post_s)

    if plot_t_e.size:
        ax.scatter(plot_t_e, plot_s_e - 1, s=point_size, c="red", alpha=0.8)
    if plot_t_i.size:
        ax.scatter(plot_t_i, plot_s_i - 1, s=point_size, c="blue", alpha=0.8)

    net_cfg = cfg.get("network", {})
    n_total = int(net_cfg.get("N_E", 0)) + int(net_cfg.get("N_IH", 0)) + int(net_cfg.get("N_IA", 0))

    alpha = float(cfg.get("experiment", {}).get("alpha", np.nan))
    beta = float(cfg.get("experiment", {}).get("beta", np.nan))
    ax.axvline(pre_s, color="black", linestyle="--", linewidth=2.4)

    ax.set_xlim(0.0, pre_s + post_s)
    ax.set_ylim(-1, n_total + 1)
    ax.set_ylabel("Neuron index")
    
    # First row: s_exc and s_inh centered over entire plot
    total_time = pre_s + post_s
    ax.text(total_time / 2, 1.12, rf"$s_{{exc}}={alpha:.2f},\ s_{{inh}}={beta:.2f}$",
            transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=13)
    
    # Second row: m values split by pre/post learning
    # Pre-learning values centered over pre area (0 to pre_s)
    ax.text(pre_s / 2, 1.05, rf"$m_{{network}}={m_network_pre:.3f},\ m_{{cluster1}}={m_cluster_1_pre:.3f}$",
            transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=11)
    
    # Post-learning values centered over post area (pre_s to pre_s + post_s)
    ax.text(pre_s + post_s / 2, 1.05, rf"$m_{{network}}={m_network_post:.3f},\ m_{{cluster1}}={m_cluster_1_post:.3f}$",
            transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=11)

    pre_start_s = (t_on_ms - pre_s * 1000.0) / 1000.0
    post_start_s = t_off_ms / 1000.0
    tick_positions = np.arange(0.0, pre_s + post_s + 1e-9, 10.0, dtype=float)
    tick_positions = tick_positions[(tick_positions >= 0.0) & (tick_positions <= pre_s + post_s)]

    tick_labels: list[str] = []
    for x in tick_positions:
        if x < pre_s:
            real_time = pre_start_s + x
        else:
            real_time = post_start_s + (x - pre_s)
        tick_labels.append(f"{real_time:.0f}")

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)


def main() -> None:
    args = _parse_args()
    setup_matplotlib_style()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_dir = args.criticality_analysis_dir
    mean_candidates = sorted(
        [
            p
            for p in analysis_dir.iterdir()
            if p.is_dir() and p.name.endswith("_mean") and not p.name.startswith(".")
        ]
    )
    if mean_candidates:
        mean_dir = mean_candidates[-1]
        print(f"Using average folder for heatmaps: {mean_dir}")
    elif (analysis_dir / "branching").exists():
        mean_dir = analysis_dir
        print(f"No *_mean folder found, using analysis folder directly for heatmaps: {mean_dir}")
    else:
        raise FileNotFoundError(
            f"Could not find a usable average folder in {analysis_dir} (expected '*_mean' or 'branching')."
        )

    branching_root_avg = mean_dir / "branching"
    metrics_network_pre_avg = branching_root_avg / "whole_population" / "pre_learning" / "metrics.npz"
    metrics_cluster_1_pre_avg = branching_root_avg / "stimulus_block_E_half_0" / "pre_learning" / "metrics.npz"
    metrics_network_post_avg = branching_root_avg / "whole_population" / "post_learning" / "metrics.npz"
    metrics_cluster_1_post_avg = branching_root_avg / "stimulus_block_E_half_0" / "post_learning" / "metrics.npz"

    network_grid_pre_avg = _load_sigma_grid(metrics_network_pre_avg)
    cluster_1_grid_pre_avg = _load_sigma_grid(metrics_cluster_1_pre_avg)
    network_grid_post_avg = _load_sigma_grid(metrics_network_post_avg)
    cluster_1_grid_post_avg = _load_sigma_grid(metrics_cluster_1_post_avg)

    branching_root_run = args.criticality_run_analysis_dir / "branching"
    metrics_network_pre_run = branching_root_run / "whole_population" / "pre_learning" / "metrics.npz"
    metrics_cluster_1_pre_run = branching_root_run / "stimulus_block_E_half_0" / "pre_learning" / "metrics.npz"
    metrics_network_post_run = branching_root_run / "whole_population" / "post_learning" / "metrics.npz"
    metrics_cluster_1_post_run = branching_root_run / "stimulus_block_E_half_0" / "post_learning" / "metrics.npz"

    network_grid_pre_run = _load_sigma_grid(metrics_network_pre_run)
    cluster_1_grid_pre_run = _load_sigma_grid(metrics_cluster_1_pre_run)
    network_grid_post_run = _load_sigma_grid(metrics_network_post_run)
    cluster_1_grid_post_run = _load_sigma_grid(metrics_cluster_1_post_run)

    print(f"Using run-specific folder for m values in raster plots: {args.criticality_run_analysis_dir}")

    marker_configs: list[tuple[float, float]] = []
    run_data: list[dict[str, Any]] = []
    for run_name in args.run_names:
        run_dir = args.sweep_root / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run folder does not exist: {run_dir}")

        cfg, data, _, _, stim_metadata = load_run(run_dir)
        alpha = float(cfg.get("experiment", {}).get("alpha", np.nan))
        beta = float(cfg.get("experiment", {}).get("beta", np.nan))
        marker_configs.append((alpha, beta))

        m_network_pre = _lookup_sigma_for_alpha_beta(alpha, beta, network_grid_pre_run)
        m_cluster_1_pre = _lookup_sigma_for_alpha_beta(alpha, beta, cluster_1_grid_pre_run)
        m_network_post = _lookup_sigma_for_alpha_beta(alpha, beta, network_grid_post_run)
        m_cluster_1_post = _lookup_sigma_for_alpha_beta(alpha, beta, cluster_1_grid_post_run)

        run_data.append(
            {
                "cfg": cfg,
                "data": data,
                "stim_metadata": stim_metadata,
                "run_name": run_name,
                "m_network_pre": m_network_pre,
                "m_cluster_1_pre": m_cluster_1_pre,
                "m_network_post": m_network_post,
                "m_cluster_1_post": m_cluster_1_post,
            }
        )

    sigma_combined = np.concatenate(
        [
            network_grid_pre_avg["sigma_mr"].ravel(),
            cluster_1_grid_pre_avg["sigma_mr"].ravel(),
            network_grid_post_avg["sigma_mr"].ravel(),
            cluster_1_grid_post_avg["sigma_mr"].ravel(),
        ]
    ).reshape(4, -1)
    sigma_cmap, sigma_limits = build_sigma_colormap(sigma_combined)
    sigma_vmin = sigma_vmax = None
    if sigma_limits is not None:
        sigma_vmin, sigma_vmax = sigma_limits

    alphas = network_grid_pre_avg["alphas"]
    betas = network_grid_pre_avg["betas"]

    fig = plt.figure(figsize=(14.94, 11.5))
    gs = GridSpec(
        4,
        8,
        figure=fig,
        height_ratios=[0.95, 1.2, 1.2, 1.2],
        width_ratios=[0.62, 1.0, 1.0, 0.14, 1.0, 1.0, 0.126, 0.126],
        hspace=0.42,
        wspace=0.06,
        left=0.05,
        right=0.95,
        top=0.90,
        bottom=0.05,
    )

    ax_heatmap_net_pre = fig.add_subplot(gs[0, 1])
    ax_heatmap_net_post = fig.add_subplot(gs[0, 2])
    ax_heatmap_cluster_pre = fig.add_subplot(gs[0, 4])
    ax_heatmap_cluster_post = fig.add_subplot(gs[0, 5])
    legend_ax = fig.add_subplot(gs[0, 0])
    sigma_cax = fig.add_subplot(gs[0, 6])
    sigma_zoom_ax = fig.add_subplot(gs[0, 7])

    top_row_shift_left = 0.030
    top_row_axes = [
        legend_ax,
        ax_heatmap_net_pre,
        ax_heatmap_net_post,
        ax_heatmap_cluster_pre,
        ax_heatmap_cluster_post,
        sigma_cax,
        sigma_zoom_ax,
    ]
    for axis in top_row_axes:
        pos = axis.get_position()
        axis.set_position([pos.x0 - top_row_shift_left, pos.y0, pos.width, pos.height])

    legend_shift_right = 0.030
    legend_pos = legend_ax.get_position()
    legend_ax.set_position([legend_pos.x0 + legend_shift_right, legend_pos.y0, legend_pos.width, legend_pos.height])

    heatmap_axes = [ax_heatmap_net_pre, ax_heatmap_net_post, ax_heatmap_cluster_pre, ax_heatmap_cluster_post]
    heatmap_titles = [
        r"$m$ - Whole Population - Pre Learning",
        r"$m$ - Whole Population - Post Learning",
        r"$m$ - Cluster 1 - Pre Learning",
        r"$m$ - Cluster 1 - Post Learning",
    ]
    sigma_matrices = [
        network_grid_pre_avg["sigma_mr"],
        network_grid_post_avg["sigma_mr"],
        cluster_1_grid_pre_avg["sigma_mr"],
        cluster_1_grid_post_avg["sigma_mr"],
    ]
    show_ylabel_flags = [True, False, False, False]

    im = None
    for ax, sigma_matrix, title, show_ylabel in zip(heatmap_axes, sigma_matrices, heatmap_titles, show_ylabel_flags):
        im = _plot_heatmap_with_markers(
            ax,
            sigma_matrix,
            alphas,
            betas,
            marker_configs,
            title,
            show_ylabel=show_ylabel,
            cmap=sigma_cmap,
            vmin=sigma_vmin,
            vmax=sigma_vmax,
        )
    for ax in (ax_heatmap_net_post, ax_heatmap_cluster_pre, ax_heatmap_cluster_post):
        ax.tick_params(axis="y", labelleft=False)

    fig.canvas.draw()

    legend_ax.axis("off")
    custom_handles = [
        Line2D([0], [0], marker='^', linestyle='None', color='black', markerfacecolor='none', markeredgewidth=1.5, markersize=9),
        Line2D([0], [0], marker='s', linestyle='None', color='black', markerfacecolor='none', markeredgewidth=1.5, markersize=9),
        Line2D([0], [0], marker='o', linestyle='None', color='black', markerfacecolor='none', markeredgewidth=1.5, markersize=9),
    ]
    legend_ax.legend(custom_handles, ['A', 'B', 'C'], loc="upper left", title="Simulation", fontsize=9, framealpha=0.9)

    sigma_cbar = fig.colorbar(im, cax=sigma_cax)
    sigma_cbar.set_label(r"$m$", fontsize=10)
    sigma_cbar.ax.yaxis.set_ticks_position("left")
    sigma_cbar.ax.yaxis.set_label_position("left")

    zoom_color_limits = (0.90, 1.0)
    zmin, zmax = zoom_color_limits
    if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
        full_vmin = float(im.norm.vmin) if im.norm.vmin is not None else float(np.nanmin(sigma_combined))
        full_vmax = float(im.norm.vmax) if im.norm.vmax is not None else float(np.nanmax(sigma_combined))
        if np.isfinite(full_vmin) and np.isfinite(full_vmax) and full_vmax > full_vmin:
            frac_min = float(np.clip((zmin - full_vmin) / (full_vmax - full_vmin), 0.0, 1.0))
            frac_max = float(np.clip((zmax - full_vmin) / (full_vmax - full_vmin), 0.0, 1.0))
        else:
            frac_min, frac_max = 0.0, 1.0
        if frac_max <= frac_min:
            frac_min, frac_max = 0.0, 1.0

        base_cmap = plt.get_cmap(sigma_cmap) if isinstance(sigma_cmap, str) else sigma_cmap
        zoom_colors = base_cmap(np.linspace(frac_min, frac_max, 256))
        zoom_cmap = ListedColormap(zoom_colors)

        sigma_zoom_sm = plt.cm.ScalarMappable(norm=Normalize(vmin=zmin, vmax=zmax), cmap=zoom_cmap)
        sigma_zoom_sm.set_array([])
        sigma_zoom_cbar = fig.colorbar(sigma_zoom_sm, cax=sigma_zoom_ax)
        sigma_zoom_cbar.ax.yaxis.set_ticks_position("right")
        sigma_zoom_cbar.ax.yaxis.set_label_position("right")

        y_main_low = full_vmin + frac_min * (full_vmax - full_vmin)
        y_main_high = full_vmin + frac_max * (full_vmax - full_vmin)
        for y_main in (y_main_low, y_main_high):
            sigma_cbar.ax.axhline(y=y_main, color="black", linewidth=1.0, alpha=0.8)
        for y_zoom in (zmin, zmax):
            sigma_zoom_cbar.ax.axhline(y=y_zoom, color="black", linewidth=1.0, alpha=0.8)

        connector_low = ConnectionPatch(
            xyA=(1.0, frac_min),
            coordsA="axes fraction",
            axesA=sigma_cbar.ax,
            xyB=(0.0, 0.0),
            coordsB="axes fraction",
            axesB=sigma_zoom_cbar.ax,
            color="black",
            linewidth=0.9,
            alpha=0.75,
            clip_on=False,
        )
        connector_high = ConnectionPatch(
            xyA=(1.0, frac_max),
            coordsA="axes fraction",
            axesA=sigma_cbar.ax,
            xyB=(0.0, 1.0),
            coordsB="axes fraction",
            axesB=sigma_zoom_cbar.ax,
            color="black",
            linewidth=0.9,
            alpha=0.75,
            clip_on=False,
        )
        fig.add_artist(connector_low)
        fig.add_artist(connector_high)
    else:
        sigma_zoom_ax.axis("off")

    for left_ax, right_ax in [
        (ax_heatmap_net_pre, ax_heatmap_net_post),
        (ax_heatmap_cluster_pre, ax_heatmap_cluster_post),
    ]:
        left_pos = left_ax.get_position()
        right_pos = right_ax.get_position()
        y_center = 0.5 * (left_pos.y0 + left_pos.y1)
        gap = max(0.0005, 0.002 * (right_pos.x0 - left_pos.x1))
        arrow_start = (left_pos.x1 + gap, y_center)
        arrow_end = (right_pos.x0 - gap, y_center)
        left_ax.annotate(
            "",
            xy=arrow_end,
            xytext=arrow_start,
            xycoords=fig.transFigure,
            textcoords=fig.transFigure,
            arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "black"},
            annotation_clip=False,
        )

    raster_axes = [
        fig.add_subplot(gs[1, :]),
        fig.add_subplot(gs[2, :]),
        fig.add_subplot(gs[3, :]),
    ]

    for ax, rd in zip(raster_axes, run_data):
        _plot_single_row(
            ax=ax,
            cfg=rd["cfg"],
            data=rd["data"],
            run_name=rd["run_name"],
            pre_s=args.pre_s,
            post_s=args.post_s,
            point_size=args.point_size,
            stim_metadata=rd["stim_metadata"],
            m_network_pre=rd["m_network_pre"],
            m_cluster_1_pre=rd["m_cluster_1_pre"],
            m_network_post=rd["m_network_post"],
            m_cluster_1_post=rd["m_cluster_1_post"],
        )

    raster_axes[-1].set_xlabel("Time (s)")

    for i, ax in enumerate(raster_axes):
        label = chr(65 + i)
        ax.text(-0.04, 1.0, label, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="right")

    spike_handles, spike_labels = raster_axes[0].get_legend_handles_labels()
    if spike_handles:
        fig.legend(spike_handles, spike_labels, loc="upper right", fontsize=10)

    fig.suptitle(
        "Model Functionality Comparison for Different E/I Ratios\n"
        "(Rasters: 20s pre + 40s post, 35s stimulus interval removed)",
        fontsize=14,
        y=0.99,
    )

    stem = f"raster_comparison_with_heatmaps_pre{int(args.pre_s)}s_post{int(args.post_s)}s"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")

    print(f"Saved: {pdf_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
