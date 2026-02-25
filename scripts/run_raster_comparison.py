from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Iterable

import matplotlib
import numpy as np

from src.analysis.util import load_run

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SWEEP_ROOT = Path(
    "/Volumes/X9Pro/Backup Bachelor/260126/multiple_run_100_times_20260120_140859/run_030"
)
DEFAULT_RUN_NAMES = [
    "alpha_0.11_beta_1.89_s00",
    "alpha_0.53_beta_0.11_s00",
    "alpha_1.79_beta_0.00_s00",
]
DEFAULT_OUTPUT_DIR = Path("results/plots/raster_comparison")
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
        "--criticality_run_analysis_dir",
        type=Path,
        default=DEFAULT_CRITICALITY_RUN_ANALYSIS_DIR,
        help=(
            "Path to run-specific criticality analysis folder containing branching grids "
            "(e.g. .../run_000_analysis)."
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
        ax.scatter(plot_t_e, plot_s_e - 1, s=point_size, c="red", alpha=0.8, label="E")
    if plot_t_i.size:
        ax.scatter(plot_t_i, plot_s_i - 1, s=point_size, c="blue", alpha=0.8, label="IH+IA")

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

    branching_root = args.criticality_run_analysis_dir / "branching"
    
    # Load both pre and post learning metrics
    metrics_network_pre = branching_root / "whole_population" / "pre_learning" / "metrics.npz"
    metrics_cluster_1_pre = branching_root / "stimulus_block_E_half_0" / "pre_learning" / "metrics.npz"
    metrics_network_post = branching_root / "whole_population" / "post_learning" / "metrics.npz"
    metrics_cluster_1_post = branching_root / "stimulus_block_E_half_0" / "post_learning" / "metrics.npz"

    network_grid_pre = _load_sigma_grid(metrics_network_pre)
    cluster_1_grid_pre = _load_sigma_grid(metrics_cluster_1_pre)
    network_grid_post = _load_sigma_grid(metrics_network_post)
    cluster_1_grid_post = _load_sigma_grid(metrics_cluster_1_post)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, run_name in zip(axes, args.run_names):
        run_dir = args.sweep_root / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run folder does not exist: {run_dir}")

        cfg, data, _, _, stim_metadata = load_run(run_dir)
        alpha = float(cfg.get("experiment", {}).get("alpha", np.nan))
        beta = float(cfg.get("experiment", {}).get("beta", np.nan))

        m_network_pre = _lookup_sigma_for_alpha_beta(alpha, beta, network_grid_pre)
        m_cluster_1_pre = _lookup_sigma_for_alpha_beta(alpha, beta, cluster_1_grid_pre)
        m_network_post = _lookup_sigma_for_alpha_beta(alpha, beta, network_grid_post)
        m_cluster_1_post = _lookup_sigma_for_alpha_beta(alpha, beta, cluster_1_grid_post)

        _plot_single_row(
            ax=ax,
            cfg=cfg,
            data=data,
            run_name=run_name,
            pre_s=args.pre_s,
            post_s=args.post_s,
            point_size=args.point_size,
            stim_metadata=stim_metadata,
            m_network_pre=m_network_pre,
            m_cluster_1_pre=m_cluster_1_pre,
            m_network_post=m_network_post,
            m_cluster_1_post=m_cluster_1_post,
        )

    axes[-1].set_xlabel("Time (s)")

    # Add A, B, C labels to each subplot
    for i, ax in enumerate(axes):
        label = chr(65 + i)  # 65 is ASCII for 'A'
        ax.text(-0.05, 1.0, label, transform=ax.transAxes, 
                fontsize=16, fontweight='bold', va='top', ha='right')

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.suptitle(
        "Model functionality comparison for different E/I Ratios (20s pre + 40s post, 35s stimulus interval removed)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.5)

    stem = f"raster_comparison_pre{int(args.pre_s)}s_post{int(args.post_s)}s"
    #png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    #fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)

    #print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
