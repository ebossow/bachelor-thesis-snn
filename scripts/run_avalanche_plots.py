from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.metrics import avalanche_sizes_from_times, average_inter_event_interval
from src.analysis.util import combine_spikes, load_run

DEFAULT_SWEEP_ROOT = Path(
    "/Volumes/X9Pro/Backup Bachelor/260126/multiple_run_100_times_20260120_140859/run_030"
)
DEFAULT_RUN_NAMES = [
    "alpha_0.11_beta_1.89_s00",
    "alpha_0.53_beta_0.11_s00",
    "alpha_1.79_beta_0.00_s00",
]
DEFAULT_OUTPUT_DIR = Path("results") / "plots" / "criticality" / "avalanche"
DEFAULT_GOF_SOURCE_ANALYSIS_DIR = Path(
    "results/criticality_analysis/multiple_run_100_times_20260120_140859_analysis/run_099_analysis"
)
DEFAULT_POWERLAW_SOURCE_ANALYSIS_DIR = Path(
    "results/criticality_analysis/multiple_run_100_times_20260120_140859_analysis/"
    "multiple_run_100_times_20260120_140859_analysis_mean"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot avalanche size/duration distributions (log-log) for selected simulations "
            "in the post-learning phase."
        )
    )
    parser.add_argument(
        "--sweep_root",
        type=Path,
        default=DEFAULT_SWEEP_ROOT,
        help="Path to run root containing alpha_... simulation folders.",
    )
    parser.add_argument(
        "--run_names",
        nargs="+",
        default=DEFAULT_RUN_NAMES,
        help="Simulation folder names to plot (expected: exactly 3).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for the figure.",
    )
    parser.add_argument(
        "--gof_source_analysis_dir",
        type=Path,
        default=DEFAULT_GOF_SOURCE_ANALYSIS_DIR,
        help="Analysis folder containing run_099 avalanche metrics for GOF p-value heatmaps.",
    )
    parser.add_argument(
        "--powerlaw_source_analysis_dir",
        type=Path,
        default=DEFAULT_POWERLAW_SOURCE_ANALYSIS_DIR,
        help="Analysis folder containing mean avalanche metrics.csv files for power-law preference heatmaps.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="avalanche_distributions_analysis.pdf",
        help="Output filename for the saved figure.",
    )
    parser.add_argument(
        "--dt_ms",
        type=float,
        default=None,
        help="Avalanche bin width in ms. Default: AIEI-based per simulation.",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=1,
        help="Minimum avalanche size.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figure interactively in addition to saving.",
    )
    args = parser.parse_args()
    if len(args.run_names) != 3:
        parser.error("--run_names must contain exactly 3 entries.")
    if args.min_size < 1:
        parser.error("--min_size must be >= 1.")
    if args.dt_ms is not None and args.dt_ms <= 0:
        parser.error("--dt_ms must be > 0 when provided.")
    return args


def _get_post_window_ms(cfg: dict[str, Any], stim_metadata: dict[str, Any] | None) -> tuple[float, float]:
    pattern_meta = (stim_metadata or {}).get("pattern", {}) if isinstance(stim_metadata, dict) else {}
    pattern_cfg = cfg.get("stimulation", {}).get("pattern", {})

    t_off_ms = pattern_meta.get("t_off_ms", pattern_cfg.get("t_off_ms"))
    if t_off_ms is None:
        raise ValueError("Could not resolve stimulation.pattern.t_off_ms for post-learning window.")

    t_start_ms = float(t_off_ms)
    t_stop_ms = float(cfg.get("experiment", {}).get("simtime_ms", np.nan))
    if not np.isfinite(t_stop_ms):
        raise ValueError("experiment.simtime_ms missing or invalid.")
    if t_stop_ms <= t_start_ms:
        raise ValueError(f"Invalid post-learning window: start={t_start_ms}, stop={t_stop_ms}")
    return t_start_ms, t_stop_ms


def _empirical_distribution(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    unique_vals, counts = np.unique(values, return_counts=True)
    probs = counts.astype(float) / float(counts.sum())
    return unique_vals.astype(float), probs


def _extract_post_learning_avalanches(
    run_dir: Path,
    dt_override_ms: float | None,
    min_size: int,
) -> dict[str, Any]:
    cfg, data, _, _, stim_metadata = load_run(run_dir)
    t_start_ms, t_stop_ms = _get_post_window_ms(cfg, stim_metadata)

    spikes_whole = combine_spikes(data, ["spikes_E", "spikes_IH", "spikes_IA"])
    times_whole_ms = np.asarray(spikes_whole["times"], dtype=float)
    if times_whole_ms.size == 0:
        raise ValueError(f"No spikes available in run: {run_dir}")

    post_mask_whole = (times_whole_ms >= t_start_ms) & (times_whole_ms < t_stop_ms)
    post_times_whole = np.asarray(times_whole_ms[post_mask_whole], dtype=float)
    if post_times_whole.size == 0:
        raise ValueError(f"No post-learning spikes in run: {run_dir}")

    aiei_ms, _ = average_inter_event_interval(post_times_whole)
    if dt_override_ms is not None:
        dt_ms = float(dt_override_ms)
    else:
        dt_ms = float(aiei_ms) if np.isfinite(aiei_ms) and aiei_ms > 0 else 4.0

    sizes_whole, durations_whole_ms = avalanche_sizes_from_times(
        times_ms=post_times_whole,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
        dt_ms=dt_ms,
        min_size=min_size,
    )

    cluster_1_ids: list[int] = []
    if isinstance(stim_metadata, dict):
        dc_meta = stim_metadata.get("dc")
        if isinstance(dc_meta, dict):
            blocks = dc_meta.get("blocks")
            if isinstance(blocks, list):
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                    label = str(block.get("label", "")).lower()
                    population = str(block.get("population", "")).upper()
                    if population == "E" and "half_0" in label:
                        ids = block.get("neuron_ids")
                        if isinstance(ids, list):
                            cluster_1_ids = [int(v) for v in ids]
                            break

    if not cluster_1_ids:
        n_e = int(cfg.get("network", {}).get("N_E", 0))
        cluster_1_ids = list(range(1, (n_e // 2) + 1)) if n_e > 0 else []

    spikes_cluster_1 = combine_spikes(data, ["spikes_E"], allowed_senders=cluster_1_ids)
    times_cluster_1_ms = np.asarray(spikes_cluster_1["times"], dtype=float)
    post_mask_cluster = (times_cluster_1_ms >= t_start_ms) & (times_cluster_1_ms < t_stop_ms)
    post_times_cluster_1 = np.asarray(times_cluster_1_ms[post_mask_cluster], dtype=float)

    sizes_cluster_1, durations_cluster_1_ms = avalanche_sizes_from_times(
        times_ms=post_times_cluster_1,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
        dt_ms=dt_ms,
        min_size=min_size,
    )

    alpha = float(cfg.get("experiment", {}).get("alpha", np.nan))
    beta = float(cfg.get("experiment", {}).get("beta", np.nan))

    return {
        "run_name": run_dir.name,
        "alpha": alpha,
        "beta": beta,
        "dt_ms": dt_ms,
        "sizes_whole": sizes_whole,
        "durations_whole_ms": durations_whole_ms,
        "sizes_cluster_1": sizes_cluster_1,
        "durations_cluster_1_ms": durations_cluster_1_ms,
    }


def _load_gof_heatmap_data(gof_source_analysis_dir: Path) -> dict[str, np.ndarray]:
    whole_metrics = (
        gof_source_analysis_dir / "avalanche" / "whole_population" / "post_learning" / "metrics.npz"
    )
    cluster_metrics = (
        gof_source_analysis_dir
        / "avalanche"
        / "stimulus_block_E_half_0"
        / "post_learning"
        / "metrics.npz"
    )

    if not whole_metrics.exists():
        raise FileNotFoundError(f"Missing GOF metrics: {whole_metrics}")
    if not cluster_metrics.exists():
        raise FileNotFoundError(f"Missing GOF metrics: {cluster_metrics}")

    with np.load(whole_metrics, allow_pickle=False) as d_whole:
        alphas = np.asarray(d_whole["alphas"], dtype=float)
        betas = np.asarray(d_whole["betas"], dtype=float)
        whole_size = np.asarray(d_whole["pl_gof_p_size"], dtype=float)
        whole_duration = np.asarray(d_whole["pl_gof_p_duration"], dtype=float)

    with np.load(cluster_metrics, allow_pickle=False) as d_cluster:
        cluster_alphas = np.asarray(d_cluster["alphas"], dtype=float)
        cluster_betas = np.asarray(d_cluster["betas"], dtype=float)
        cluster_size = np.asarray(d_cluster["pl_gof_p_size"], dtype=float)
        cluster_duration = np.asarray(d_cluster["pl_gof_p_duration"], dtype=float)

    if not (np.array_equal(alphas, cluster_alphas) and np.array_equal(betas, cluster_betas)):
        raise ValueError("Alpha/Beta grids differ between whole_population and cluster_1 GOF metrics.")

    return {
        "alphas": alphas,
        "betas": betas,
        "whole_size": whole_size,
        "whole_duration": whole_duration,
        "cluster_size": cluster_size,
        "cluster_duration": cluster_duration,
    }


def _load_powerlaw_preferred_heatmap_data(
    powerlaw_source_analysis_dir: Path,
) -> dict[str, np.ndarray]:
    whole_metrics = (
        powerlaw_source_analysis_dir / "avalanche" / "whole_population" / "post_learning" / "metrics.csv"
    )
    cluster_metrics = (
        powerlaw_source_analysis_dir
        / "avalanche"
        / "stimulus_block_E_half_0"
        / "post_learning"
        / "metrics.csv"
    )

    if not whole_metrics.exists():
        raise FileNotFoundError(f"Missing power-law metrics: {whole_metrics}")
    if not cluster_metrics.exists():
        raise FileNotFoundError(f"Missing power-law metrics: {cluster_metrics}")

    def _load_csv_matrix(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        if data.size == 0:
            raise ValueError(f"Empty metrics.csv: {csv_path}")
        if data.shape == ():
            data = np.asarray([data], dtype=data.dtype)

        if "alpha" not in data.dtype.names or "beta" not in data.dtype.names or "pl_better_both" not in data.dtype.names:
            raise ValueError(f"Required columns missing in metrics.csv: {csv_path}")

        alphas = np.asarray(data["alpha"], dtype=float)
        betas = np.asarray(data["beta"], dtype=float)
        values = np.asarray(data["pl_better_both"], dtype=float)

        alpha_grid = np.unique(alphas)
        beta_grid = np.unique(betas)
        matrix = np.full((beta_grid.size, alpha_grid.size), np.nan, dtype=float)

        alpha_index = {float(v): i for i, v in enumerate(alpha_grid)}
        beta_index = {float(v): i for i, v in enumerate(beta_grid)}
        for alpha, beta, value in zip(alphas, betas, values, strict=False):
            matrix[beta_index[float(beta)], alpha_index[float(alpha)]] = value

        return alpha_grid, beta_grid, matrix

    alphas, betas, whole_preferred = _load_csv_matrix(whole_metrics)
    cluster_alphas, cluster_betas, cluster_preferred = _load_csv_matrix(cluster_metrics)

    if not (np.array_equal(alphas, cluster_alphas) and np.array_equal(betas, cluster_betas)):
        raise ValueError(
            "Alpha/Beta grids differ between whole_population and cluster_1 power-law metrics."
        )

    return {
        "alphas": alphas,
        "betas": betas,
        "whole_preferred": whole_preferred,
        "cluster_preferred": cluster_preferred,
    }


def _plot_distribution(
    ax: plt.Axes,
    values_whole: np.ndarray,
    values_cluster_1: np.ndarray,
    x_label: str,
) -> None:
    x_whole, y_whole = _empirical_distribution(values_whole)
    mask_whole = np.isfinite(x_whole) & np.isfinite(y_whole) & (x_whole > 0) & (y_whole > 0)
    x_whole = x_whole[mask_whole]
    y_whole = y_whole[mask_whole]

    x_cluster, y_cluster = _empirical_distribution(values_cluster_1)
    mask_cluster = (
        np.isfinite(x_cluster) & np.isfinite(y_cluster) & (x_cluster > 0) & (y_cluster > 0)
    )
    x_cluster = x_cluster[mask_cluster]
    y_cluster = y_cluster[mask_cluster]

    if x_whole.size == 0 and x_cluster.size == 0:
        ax.text(0.5, 0.5, "No avalanches", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(x_label)
        ax.set_ylabel("P(x)")
        return

    x_all = np.concatenate([x_whole, x_cluster]) if (x_whole.size and x_cluster.size) else (x_whole if x_whole.size else x_cluster)
    y_all = np.concatenate([y_whole, y_cluster]) if (y_whole.size and y_cluster.size) else (y_whole if y_whole.size else y_cluster)

    if x_whole.size > 0:
        ax.loglog(
            x_whole,
            y_whole,
            "o",
            markersize=3.0,
            alpha=0.85,
            color="tab:blue",
            label="Whole population",
        )
    if x_cluster.size > 0:
        ax.loglog(
            x_cluster,
            y_cluster,
            "o",
            markersize=3.0,
            alpha=0.85,
            color="tab:orange",
            label="Cluster 1",
        )

    finite_x = x_all[np.isfinite(x_all) & (x_all > 0)]
    finite_y = y_all[np.isfinite(y_all) & (y_all > 0)]
    if finite_x.size > 0 and finite_y.size > 0:
        x_min = float(np.min(finite_x))
        x_max = float(np.max(finite_x))
        if x_max > x_min:
            x_ref = np.logspace(np.log10(x_min), np.log10(x_max), 200)
            y_anchor = float(np.max(finite_y))
            y_ref = y_anchor * (x_ref / x_min) ** (-1.5)
            ax.loglog(
                x_ref,
                y_ref,
                linestyle="--",
                linewidth=1.4,
                color="black",
                alpha=0.8,
                label=r"Reference $x^{-1.5}$",
            )

    ax.set_xlabel(x_label)
    ax.set_ylabel("P(x)")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


def _plot_gof_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    title: str,
    norm: Normalize,
    cmap: str | ListedColormap = "viridis",
) -> any:
    extent = [float(alphas.min()), float(alphas.max()), float(betas.min()), float(betas.max())]
    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"$s_{exc}$")
    ax.set_ylabel(r"$s_{inh}$")
    return im


def _build_first_row_figure(
    entries: list[dict[str, Any]],
    gof_heatmap_data: dict[str, np.ndarray],
    powerlaw_preferred_data: dict[str, np.ndarray],
    output_path: Path,
    show: bool,
) -> None:
    fig = plt.figure(figsize=(16.5, 13.8), constrained_layout=False)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.35)

    top = outer[0].subgridspec(2, 3, wspace=0.08, hspace=0.22)
    axes = np.asarray([[fig.add_subplot(top[r, c]) for c in range(3)] for r in range(2)])

    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.18)
    d_grid = bottom[0, 0].subgridspec(2, 2, wspace=0.28, hspace=0.32)
    e_grid = bottom[0, 1].subgridspec(1, 2, wspace=0.28)
    ax_d_00 = fig.add_subplot(d_grid[0, 0])
    ax_d_01 = fig.add_subplot(d_grid[0, 1])
    ax_d_10 = fig.add_subplot(d_grid[1, 0])
    ax_d_11 = fig.add_subplot(d_grid[1, 1])
    ax_e_0 = fig.add_subplot(e_grid[0, 0])
    ax_e_1 = fig.add_subplot(e_grid[0, 1])

    fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.07)
    letters = ["A", "B", "C"]

    for idx, entry in enumerate(entries):
        ax_top = axes[0, idx]
        ax_bottom = axes[1, idx]

        ax_top.text(
            -0.14,
            1.16,
            letters[idx],
            transform=ax_top.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
            fontsize=18,
        )
        ax_top.set_title(
            rf"$s_{{exc}}={entry['alpha']:.2f},\ s_{{inh}}={entry['beta']:.2f}$",
            loc="center",
        )

        _plot_distribution(
            ax=ax_top,
            values_whole=np.asarray(entry["sizes_whole"], dtype=float),
            values_cluster_1=np.asarray(entry["sizes_cluster_1"], dtype=float),
            x_label="Avalanche size (spikes)",
        )
        _plot_distribution(
            ax=ax_bottom,
            values_whole=np.asarray(entry["durations_whole_ms"], dtype=float),
            values_cluster_1=np.asarray(entry["durations_cluster_1_ms"], dtype=float),
            x_label="Avalanche duration (ms)",
        )

    top_scale = 0.8
    for ax in axes.ravel():
        pos = ax.get_position()
        new_width = pos.width * top_scale
        new_height = pos.height * top_scale
        new_x0 = pos.x0 + 0.5 * (pos.width - new_width)
        new_y0 = pos.y0 + 0.5 * (pos.height - new_height)
        ax.set_position([new_x0, new_y0, new_width, new_height])

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="tab:blue", markersize=5, label="Whole population"),
        Line2D([0], [0], marker="o", linestyle="None", color="tab:orange", markersize=5, label="Cluster 1"),
        Line2D([0], [0], linestyle="--", color="black", linewidth=1.4, label=r"Reference $x^{-1.5}$"),
    ]
    top_left_pos = axes[0, 0].get_position()
    bottom_left_pos = axes[1, 0].get_position()
    legend_x = max(0.012, top_left_pos.x0 - 0.13)
    legend_y = 0.5 * (top_left_pos.y0 + bottom_left_pos.y1)
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(legend_x, legend_y),
        frameon=True,
        fontsize=9,
        #title="Curves",
        title_fontsize=9,
    )

    alphas = gof_heatmap_data["alphas"]
    betas = gof_heatmap_data["betas"]
    matrices = [
        gof_heatmap_data["whole_size"],
        gof_heatmap_data["whole_duration"],
        gof_heatmap_data["cluster_size"],
        gof_heatmap_data["cluster_duration"],
    ]
    finite_vals = np.concatenate([m[np.isfinite(m)] for m in matrices if np.isfinite(m).any()])
    if finite_vals.size > 0:
        norm = Normalize(vmin=float(max(0.0, np.nanmin(finite_vals))), vmax=float(min(1.0, np.nanmax(finite_vals))))
        if norm.vmax <= norm.vmin:
            norm = Normalize(vmin=0.0, vmax=1.0)
    else:
        norm = Normalize(vmin=0.0, vmax=1.0)

    im = _plot_gof_heatmap(
        ax=ax_d_00,
        matrix=gof_heatmap_data["whole_size"],
        alphas=alphas,
        betas=betas,
        title="GOF p-value (size)",
        norm=norm,
    )
    _plot_gof_heatmap(
        ax=ax_d_01,
        matrix=gof_heatmap_data["cluster_size"],
        alphas=alphas,
        betas=betas,
        title="GOF p-value (size)",
        norm=norm,
    )
    _plot_gof_heatmap(
        ax=ax_d_10,
        matrix=gof_heatmap_data["whole_duration"],
        alphas=alphas,
        betas=betas,
        title="GOF p-value (duration)",
        norm=norm,
    )
    _plot_gof_heatmap(
        ax=ax_d_11,
        matrix=gof_heatmap_data["cluster_duration"],
        alphas=alphas,
        betas=betas,
        title="GOF p-value (duration)",
        norm=norm,
    )

    for ax in (ax_d_01, ax_d_11):
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    ax_d_00.text(
        0.5,
        1.24,
        "Whole Population",
        transform=ax_d_00.transAxes,
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=12,
    )
    ax_d_01.text(
        0.5,
        1.24,
        "Cluster 1",
        transform=ax_d_01.transAxes,
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=12,
    )

    ax_d_00.text(
        -0.31,
        1.40,
        "D",
        transform=ax_d_00.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=18,
    )

    powerlaw_cmap = ListedColormap(["#e5e7eb", "#15803d"])
    powerlaw_cmap.set_bad(color="white")
    powerlaw_norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5], ncolors=powerlaw_cmap.N)
    im_e = _plot_gof_heatmap(
        ax=ax_e_0,
        matrix=powerlaw_preferred_data["whole_preferred"],
        alphas=powerlaw_preferred_data["alphas"],
        betas=powerlaw_preferred_data["betas"],
        title="Whole Population",
        norm=powerlaw_norm,
        cmap=powerlaw_cmap,
    )
    _plot_gof_heatmap(
        ax=ax_e_1,
        matrix=powerlaw_preferred_data["cluster_preferred"],
        alphas=powerlaw_preferred_data["alphas"],
        betas=powerlaw_preferred_data["betas"],
        title="Cluster 1",
        norm=powerlaw_norm,
        cmap=powerlaw_cmap,
    )
    e_yticks = np.arange(0.0, 2.01, 0.5)
    ax_e_0.set_yticks(e_yticks)
    ax_e_1.set_yticks(e_yticks)
    ax_e_1.set_ylabel("")
    ax_e_1.tick_params(axis="y", labelleft=False)

    target_left = axes[0, 0].get_position().x0
    d_axes = [ax_d_00, ax_d_01, ax_d_10, ax_d_11]
    e_axes = [ax_e_0, ax_e_1]
    current_left = ax_d_00.get_position().x0
    shift_x = target_left - current_left
    if abs(shift_x) > 1e-6:
        for axis in [*d_axes, *e_axes]:
            pos = axis.get_position()
            axis.set_position([pos.x0 + shift_x, pos.y0, pos.width, pos.height])

    d_positions = [ax.get_position() for ax in d_axes]
    cbar_x0 = max(pos.x1 for pos in d_positions) + 0.012
    cbar_y0 = min(pos.y0 for pos in d_positions)
    cbar_height = max(pos.y1 for pos in d_positions) - cbar_y0
    cax = fig.add_axes([cbar_x0, cbar_y0, 0.012, cbar_height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("GOF p-value")

    e_positions = [ax_e_0.get_position(), ax_e_1.get_position()]
    e_cbar_x0 = max(pos.x1 for pos in e_positions) + 0.012
    e_cbar_y0 = min(pos.y0 for pos in e_positions)
    e_cbar_height = max(pos.y1 for pos in e_positions) - e_cbar_y0
    e_cax = fig.add_axes([e_cbar_x0, e_cbar_y0, 0.012, e_cbar_height])
    e_cbar = fig.colorbar(im_e, cax=e_cax)
    #e_cbar.set_label("Power-law preferred")
    e_cbar.set_ticks([0.0, 1.0])
    e_cbar.set_ticklabels(["No", "Yes"])

    target_right = axes[0, 2].get_position().x1
    e_left = min(ax.get_position().x0 for ax in e_axes)
    current_right = e_cax.get_position().x1
    current_width = current_right - e_left
    target_width = target_right - e_left
    if current_width > 0 and abs(target_right - current_right) > 1e-6:
        scale_x = target_width / current_width
        for axis in e_axes:
            pos = axis.get_position()
            rel_x0 = pos.x0 - e_left
            rel_x1 = pos.x1 - e_left
            axis.set_position([e_left + rel_x0 * scale_x, pos.y0, (rel_x1 - rel_x0) * scale_x, pos.height])
        e_cax_pos = e_cax.get_position()
        rel_x0 = e_cax_pos.x0 - e_left
        rel_x1 = e_cax_pos.x1 - e_left
        e_cax.set_position(
            [e_left + rel_x0 * scale_x, e_cax_pos.y0, (rel_x1 - rel_x0) * scale_x, e_cax_pos.height]
        )

    e_caption_x_fig = 0.5 * (ax_e_0.get_position().x0 + e_cax.get_position().x1)
    e_caption_y_fig = max(ax_e_0.get_position().y1, ax_e_1.get_position().y1) + 0.045
    fig.text(
        e_caption_x_fig,
        e_caption_y_fig,
        "Power-Law fit better than LogNormal or Exponential",
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=12,
    )
    d_pos = ax_d_00.get_position()
    e_pos = ax_e_0.get_position()
    d_label_y_fig = d_pos.y0 + 1.40 * d_pos.height
    e_label_y_axes = (d_label_y_fig - e_pos.y0) / e_pos.height
    ax_e_0.text(
        -0.31,
        e_label_y_axes,
        "E",
        transform=ax_e_0.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=18,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_matplotlib_style()

    gof_heatmap_data = _load_gof_heatmap_data(args.gof_source_analysis_dir)
    powerlaw_preferred_data = _load_powerlaw_preferred_heatmap_data(args.powerlaw_source_analysis_dir)

    entries: list[dict[str, Any]] = []
    for run_name in args.run_names:
        run_dir = args.sweep_root / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run folder does not exist: {run_dir}")
        entry = _extract_post_learning_avalanches(
            run_dir=run_dir,
            dt_override_ms=args.dt_ms,
            min_size=args.min_size,
        )
        entries.append(entry)
        print(
            f"{run_name}: "
            f"whole(n_size={entry['sizes_whole'].size}, n_duration={entry['durations_whole_ms'].size}) | "
            f"cluster_1(n_size={entry['sizes_cluster_1'].size}, n_duration={entry['durations_cluster_1_ms'].size}), "
            f"dt_ms={entry['dt_ms']:.4f}"
        )

    output_path = args.output_dir / args.output_name
    _build_first_row_figure(
        entries=entries,
        gof_heatmap_data=gof_heatmap_data,
        powerlaw_preferred_data=powerlaw_preferred_data,
        output_path=output_path,
        show=args.show,
    )


if __name__ == "__main__":
    main()
