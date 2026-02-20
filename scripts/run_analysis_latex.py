from pathlib import Path
import argparse
import copy
import matplotlib
import matplotlib.image as mpimg
import numpy as np

from src.analysis.summary_metrics import compute_summary_metrics
from src.analysis.summary_figure import create_summary_figure, _plot_stimulus_rate_distribution
from src.analysis.metrics import build_weight_matrix, normalize_weight_matrix
from src.analysis.plotting import (
    plot_spike_raster_ax,
    plot_weight_matrix_ax,
    plot_pdf_cv_ax,
    plot_pdf_mean_rate_ax,
    plot_K_ax,
    add_weight_matrix_colorbar,
    plot_kuramoto_traces_ax,
    plot_kuramoto_pdf_multi_ax,
    plot_weight_change_pdf_multi_ax,
)
from src.analysis.util import load_run, find_latest_run_dir

plt = None


def setup_matplotlib(save_format: str | None):
    """Configure matplotlib backend + LaTeX-friendly defaults."""

    if save_format == "pgf":
        matplotlib.use("pgf")
    else:
        for backend in ("macosx", "TkAgg", "Qt5Agg", "QtAgg"):
            try:
                matplotlib.use(backend)
                break
            except Exception:
                continue
        else:
            matplotlib.use("Agg")

    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )

    global plt
    import matplotlib.pyplot as plt_module

    plt = plt_module
    return plt_module


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate publication-ready analysis plots (PGF or PDF) for a simulation run."
        )
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to a specific run directory. Uses the latest run if omitted.",
    )
    parser.add_argument(
        "--small_figure",
        action="store_true",
        help="Create separate raster and weight-matrix figures instead of the combined summary figure.",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save the generated figures to disk instead of just displaying them.",
    )
    parser.add_argument(
        "--output_pdf",
        "--output-pdf",
        dest="output_pdf",
        action="store_true",
        help=(
            "Write individual PDF files for each summary subplot (implies no combined summary figure). "
            "Requires --generate_plots."
        ),
    )
    parser.add_argument(
        "--short_raster",
        action="store_true",
        help="Cut spike raster plots at 50,000 ms to focus on early activity.",
    )
    parser.add_argument(
        "--generate_plots",
        action="store_true",
        help="Enable generation of individual subplot PDF files when combined with --output_pdf.",
    )
    parser.add_argument(
        "--print_comparison",
        "--print-comparison",
        dest="print_comparison",
        action="store_true",
        help=(
            "Generate a comparison figure showing the original raster/weight plots "
            "versus the current run's outputs."
        ),
    )
    parser.add_argument(
        "--pre_stimulus_window_s",
        type=float,
        default=5.0,
        help="Seconds to include before stimulus onset when loading analysis data (default: 5.0).",
    )
    parser.add_argument(
        "--post_stimulus_window_s",
        type=float,
        default=20.0,
        help="Seconds to include after stimulus offset when loading analysis data (default: 20.0).",
    )
    parser.add_argument(
        "--disable_analysis_window",
        action="store_true",
        help="Disable stimulus-centered analysis window capping and load full run data.",
    )
    parser.add_argument(
        "--full_analysis_start_s",
        type=float,
        default=1550,
        help=(
            "Absolute simulation time in seconds where the auto full-analysis 30s window should start. "
            "If omitted, start of post-learning phase is used."
        ),
    )
    args = parser.parse_args()
    if args.output_pdf and not args.generate_plots:
        parser.error("--generate_plots must be specified when using --output_pdf")
    if args.pre_stimulus_window_s < 0.0:
        parser.error("--pre_stimulus_window_s must be >= 0")
    if args.post_stimulus_window_s < 0.0:
        parser.error("--post_stimulus_window_s must be >= 0")
    if args.full_analysis_start_s is not None and args.full_analysis_start_s < 0.0:
        parser.error("--full_analysis_start_s must be >= 0")
    return args


def resolve_run_directory(run_dir_arg: str | None) -> Path:
    if run_dir_arg is not None:
        run_dir = Path(run_dir_arg)
        print(f"Using specified run directory: {run_dir}")
    else:
        results_root = Path("results")
        run_dir = find_latest_run_dir(results_root)
        print(f"Using latest run directory: {run_dir}")
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return run_dir


def build_weight_matrices(cfg, weights_data):
    net_cfg = cfg["network"]
    N_total = (
        net_cfg["N_E"]
        + net_cfg["N_IA_1"]
        + net_cfg["N_IH"]
        + net_cfg["N_IA_2"]
    )
    W = build_weight_matrix(
        weights_data["sources"],
        weights_data["targets"],
        weights_data["weights"],
        N_total=N_total,
    )
    return normalize_weight_matrix(W, cfg)


def resolve_post_learning_interval_ms(cfg) -> tuple[float, float]:
    exp_cfg = cfg.get("experiment", {})
    phase_markers = exp_cfg.get("phase_markers_ms") or {}

    start_ms = phase_markers.get("main_end_ms")
    end_ms = phase_markers.get("post_end_ms")
    if start_ms is not None and end_ms is not None:
        start_ms = float(start_ms)
        end_ms = float(end_ms)
        if end_ms > start_ms:
            return start_ms, end_ms

    phase_schedule = exp_cfg.get("phase_schedule") or {}
    segments = phase_schedule.get("segments") or []
    for seg in segments:
        name = str(seg.get("name", "")).lower()
        if "post" not in name:
            continue
        seg_start = seg.get("start_ms")
        seg_end = seg.get("end_ms")
        if seg_start is None or seg_end is None:
            continue
        seg_start = float(seg_start)
        seg_end = float(seg_end)
        if seg_end > seg_start:
            return seg_start, seg_end

    stim_pattern = cfg.get("stimulation", {}).get("pattern", {})
    fallback_start = float(stim_pattern.get("t_off_ms", 0.0))
    fallback_end = float(exp_cfg.get("simtime_ms", fallback_start))
    if fallback_end <= fallback_start:
        raise ValueError("Unable to infer a valid post-learning interval from config")
    return fallback_start, fallback_end


def is_long_run(cfg) -> bool:
    min_long_duration_ms = 15.0 * 60.0 * 1000.0
    simtime_ms = float(cfg.get("experiment", {}).get("simtime_ms", 0.0))
    if simtime_ms >= min_long_duration_ms:
        return True

    min_post_duration_ms = min_long_duration_ms
    try:
        post_start_ms, post_end_ms = resolve_post_learning_interval_ms(cfg)
    except Exception:
        return False
    return (post_end_ms - post_start_ms) >= min_post_duration_ms


def _slice_spike_population(spikes, start_ms: float, end_ms: float):
    if spikes is None:
        return None
    times = np.asarray(spikes["times"])
    senders = np.asarray(spikes["senders"])
    mask = (times >= start_ms) & (times <= end_ms)
    return {
        "times": times[mask],
        "senders": senders[mask],
    }


def _slice_data_window(data, start_ms: float, end_ms: float):
    return {
        "spikes_E": _slice_spike_population(data.get("spikes_E"), start_ms, end_ms),
        "spikes_IH": _slice_spike_population(data.get("spikes_IH"), start_ms, end_ms),
        "spikes_IA": _slice_spike_population(data.get("spikes_IA"), start_ms, end_ms),
    }


def _slice_weights_window(weights_over_time, start_ms: float, end_ms: float):
    if weights_over_time is None:
        return None
    wt = np.asarray(weights_over_time["times"])
    Wt = np.asarray(weights_over_time["weights"])
    mask = (wt >= start_ms) & (wt <= end_ms)
    sliced = {
        "times": wt[mask],
        "weights": Wt[mask],
    }
    if "sources" in weights_over_time:
        sliced["sources"] = weights_over_time["sources"]
    if "targets" in weights_over_time:
        sliced["targets"] = weights_over_time["targets"]
    return sliced


def _select_named_traces(traces, include_network: bool, max_clusters: int = 2):
    network_trace = []
    cluster_traces = []
    for trace in traces or []:
        label = str(trace.get("label", "")).lower()
        if include_network and label.startswith("network") and not network_trace:
            network_trace.append(trace)
        elif label.startswith("cluster"):
            cluster_traces.append(trace)
    return network_trace + cluster_traces[:max_clusters]


def _plot_trace_lines(ax, traces, value_key: str, ylabel: str):
    if not traces:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        return

    plotted = False
    for idx, trace in enumerate(traces):
        t = np.asarray(trace.get("time_ms", []), dtype=float) * 0.001
        y = np.asarray(trace.get(value_key, []), dtype=float)
        if t.size == 0 or y.size == 0:
            continue
        label = trace.get("label", f"Trace {idx + 1}")
        normalized_label = str(label).strip().lower().replace(" ", "")
        color = None
        if normalized_label.startswith("network"):
            color = "#7f7f7f"
        elif normalized_label.startswith("cluster1"):
            color = "#ff7f0e"
        elif normalized_label.startswith("cluster2"):
            color = "#2ca02c"
        ax.plot(t, y, linewidth=2.0, label=label, color=color)
        plotted = True

    if not plotted:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.set_ylabel(ylabel)
    if len(traces) > 1:
        ax.legend(loc="upper left", fontsize=8)


def _build_synchrony_highlights(kuramoto_traces):
    highlights = []
    default_half_width_s = 0.4

    for trace in kuramoto_traces or []:
        label = str(trace.get("label", ""))
        normalized_label = label.strip().lower().replace(" ", "")
        if normalized_label.startswith("cluster1"):
            color = "#ff7f0e"
            highlight_label = "Cluster 1 Recall"
        elif normalized_label.startswith("cluster2"):
            color = "#2ca02c"
            highlight_label = "Cluster 2 Recall"
        else:
            continue

        t = np.asarray(trace.get("time_ms", []), dtype=float) * 0.001
        r = np.asarray(trace.get("R", []), dtype=float)
        finite_mask = np.isfinite(t) & np.isfinite(r)
        if not np.any(finite_mask):
            continue
        t = t[finite_mask]
        r = r[finite_mask]
        if t.size == 0 or r.size == 0:
            continue

        peak_idx = int(np.nanargmax(r))
        peak_t = float(t[peak_idx])

        half_width_s = default_half_width_s
        if t.size >= 2:
            dt = float(np.nanmedian(np.diff(t)))
            if np.isfinite(dt) and dt > 0.0:
                half_width_s = max(default_half_width_s, 2.0 * dt)

        highlights.append(
            {
                "start_s": peak_t - 0.5 * half_width_s,
                "end_s": peak_t + 0.5 * half_width_s,
                "color": color,
                "label": highlight_label,
            }
        )

    return highlights


def _build_async_irregular_highlight(kuramoto_traces):
    cluster_1 = None
    cluster_2 = None

    for trace in kuramoto_traces or []:
        label = str(trace.get("label", "")).strip().lower().replace(" ", "")
        if label.startswith("cluster1"):
            cluster_1 = trace
        elif label.startswith("cluster2"):
            cluster_2 = trace

    if cluster_1 is None or cluster_2 is None:
        return None

    t1 = np.asarray(cluster_1.get("time_ms", []), dtype=float) * 0.001
    r1 = np.asarray(cluster_1.get("R", []), dtype=float)
    t2 = np.asarray(cluster_2.get("time_ms", []), dtype=float) * 0.001
    r2 = np.asarray(cluster_2.get("R", []), dtype=float)

    n = min(t1.size, r1.size, t2.size, r2.size)
    if n < 3:
        return None

    t = t1[:n]
    both_low_score = np.maximum(r1[:n], r2[:n])
    finite_mask = np.isfinite(t) & np.isfinite(both_low_score)
    if np.count_nonzero(finite_mask) < 3:
        return None

    t = t[finite_mask]
    both_low_score = both_low_score[finite_mask]

    dt = float(np.nanmedian(np.diff(t))) if t.size >= 2 else float("nan")
    if not np.isfinite(dt) or dt <= 0.0:
        return None

    target_duration_s = 2.0
    window_len = max(1, int(round(target_duration_s / dt)))
    kernel = np.ones(window_len, dtype=float) / float(window_len)
    smoothed = np.convolve(both_low_score, kernel, mode="valid")
    if smoothed.size == 0:
        return None

    best_idx = int(np.nanargmin(smoothed))
    start_s = float(t[best_idx])
    end_s = start_s + window_len * dt

    return {
        "start_s": start_s,
        "end_s": end_s,
        "color": "#d8b4d8",
        "alpha": 0.18,
        "label": "AI State",
    }


def _resolve_stimulus_focus_window_ms(
    cfg,
    *,
    pre_ms: float = 5000.0,
    post_ms: float = 20000.0,
) -> tuple[float, float] | None:
    stim_pattern = cfg.get("stimulation", {}).get("pattern", {})
    markers = cfg.get("experiment", {}).get("phase_markers_ms", {})

    t_on_ms = stim_pattern.get("t_on_ms", markers.get("stim_on_ms"))
    t_off_ms = stim_pattern.get("t_off_ms", markers.get("stim_off_ms"))
    if t_on_ms is None or t_off_ms is None:
        return None

    sim_end_ms = float(cfg.get("experiment", {}).get("simtime_ms", 0.0))
    start_ms = max(0.0, float(t_on_ms) - float(pre_ms))
    end_ms = min(sim_end_ms, float(t_off_ms) + float(post_ms))
    if end_ms <= start_ms:
        return None
    return start_ms, end_ms


def _enable_subplot_box(ax, linewidth: float = 1.0):
    ax.set_frame_on(True)
    for side in ("left", "right", "top", "bottom"):
        spine = ax.spines.get(side)
        if spine is not None:
            spine.set_visible(True)
            spine.set_linewidth(linewidth)


def _select_weight_development_snapshots(cfg, weights_over_time):
    if weights_over_time is None:
        return []

    times = np.asarray(weights_over_time.get("times", []), dtype=float)
    if times.size == 0:
        return []

    exp_cfg = cfg.get("experiment", {})
    phase_markers = exp_cfg.get("phase_markers_ms") or {}
    pattern_cfg = cfg.get("stimulation", {}).get("pattern", {})

    target_points = [
        (pattern_cfg.get("t_on_ms", phase_markers.get("stim_on_ms")), "Pre learning"),
        (pattern_cfg.get("t_off_ms", phase_markers.get("stim_off_ms")), "After learning"),
    ]

    selected: list[tuple[int, float, str]] = []
    used_indices: set[int] = set()

    for target_ms, label in target_points:
        if target_ms is None:
            continue
        idx = int(np.argmin(np.abs(times - float(target_ms))))
        if idx in used_indices:
            continue
        used_indices.add(idx)
        selected.append((idx, float(times[idx]), label))

    if len(selected) < 2:
        fallback_indices = [0, times.size - 1]
        fallback_labels = ["Pre learning", "After learning"]
        for idx, fallback_label in zip(fallback_indices, fallback_labels):
            idx = int(idx)
            if idx in used_indices:
                continue
            used_indices.add(idx)
            selected.append((idx, float(times[idx]), fallback_label))
            if len(selected) >= 2:
                break

    selected = sorted(selected, key=lambda item: item[1])
    return selected[:2]


def create_bergoin_metrics_figure(
    cfg,
    data,
    weights_data,
    weights_over_time,
    stim_metadata,
    window_start_ms: float,
    window_end_ms: float,
):
    window_data = _slice_data_window(data, window_start_ms, window_end_ms)
    window_weights = _slice_weights_window(weights_over_time, window_start_ms, window_end_ms)

    cfg_window = copy.deepcopy(cfg)
    cfg_window.setdefault("analysis", {})["window_ms"] = {
        "start": float(window_start_ms),
        "end": float(window_end_ms),
    }
    cfg_window["experiment"]["simtime_ms"] = float(window_end_ms)
    cfg_window.setdefault("stimulation", {}).setdefault("pattern", {})["t_off_ms"] = float(window_start_ms)

    metrics_window = compute_summary_metrics(
        cfg_window,
        window_data,
        window_weights,
        stim_metadata=stim_metadata,
    )

    kuramoto_traces = _select_named_traces(
        metrics_window.get("kuramoto_traces") or [],
        include_network=True,
        max_clusters=2,
    )
    rate_traces = _select_named_traces(
        metrics_window.get("stimulus_rate_traces") or [],
        include_network=False,
        max_clusters=2,
    )

    fig = plt.figure(figsize=(14.0, 13.5))
    gs = fig.add_gridspec(
        7,
        2,
        width_ratios=(3.2, 1.35),
        height_ratios=(1.0, 0.18, 1.1, 0.18, 1.0, 1.0, 1.0),
        wspace=0.15,
        hspace=0.35,
    )

    ax_A = fig.add_subplot(gs[0, :])

    dev_gs = gs[2, :].subgridspec(
        1,
        5,
        width_ratios=(0.7, 1.0, 0.08, 1.0, 0.7),
        wspace=0.0,
    )
    matrix_axes = [fig.add_subplot(dev_gs[0, col]) for col in (1, 3)]
    arrow_axes = [fig.add_subplot(dev_gs[0, 2])]

    ax_C = fig.add_subplot(gs[4, 0])
    ax_D = fig.add_subplot(gs[4, 1])
    ax_E = fig.add_subplot(gs[5, 0])
    ax_F = fig.add_subplot(gs[5, 1])
    ax_G = fig.add_subplot(gs[6, 0])
    ax_H = fig.add_subplot(gs[6, 1])

    window_start_s = window_start_ms * 0.001
    window_end_s = window_end_ms * 0.001
    shared_ticks = np.linspace(window_start_s, window_end_s, 7)

    focus_window = _resolve_stimulus_focus_window_ms(cfg)
    if focus_window is not None:
        focus_start_ms, focus_end_ms = focus_window
        focus_data = _slice_data_window(data, focus_start_ms, focus_end_ms)
        cfg_focus = copy.deepcopy(cfg)
        cfg_focus.setdefault("analysis", {})["window_ms"] = {
            "start": float(focus_start_ms),
            "end": float(focus_end_ms),
        }
        cfg_focus["experiment"]["simtime_ms"] = float(focus_end_ms)
        plot_spike_raster_ax(ax_A, focus_data, cfg_focus)
    else:
        plot_spike_raster_ax(ax_A, window_data, cfg_window)
    ax_A.set_title("Raster (-5s/+20s around stimulus)")

    for ax_arrow in arrow_axes:
        ax_arrow.set_xticks([])
        ax_arrow.set_yticks([])
        ax_arrow.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax_arrow.text(0.5, 0.5, "\u2192", ha="center", va="center", fontsize=20)

    snapshot_series = _select_weight_development_snapshots(cfg, weights_over_time)
    sources = None
    targets = None
    if weights_over_time is not None:
        sources = weights_over_time.get("sources")
        targets = weights_over_time.get("targets")
    if (sources is None or targets is None) and weights_data is not None:
        sources = weights_data.get("sources")
        targets = weights_data.get("targets")

    im_devs = []
    if snapshot_series and sources is not None and targets is not None and weights_over_time is not None:
        all_weights = np.asarray(weights_over_time.get("weights", []), dtype=float)
        for idx_ax, ax_mat in enumerate(matrix_axes):
            if idx_ax >= len(snapshot_series):
                ax_mat.axis("off")
                continue
            snap_idx, snap_time_ms, snap_label = snapshot_series[idx_ax]
            snap_data = {
                "sources": sources,
                "targets": targets,
                "weights": all_weights[snap_idx],
            }
            Wn_snap = build_weight_matrices(cfg, snap_data)
            im_dev = plot_weight_matrix_ax(ax_mat, Wn_snap, cfg)
            im_devs.append((ax_mat, im_dev))
            ax_mat.set_title(
                f"{snap_label} ({snap_time_ms / 1000.0:.0f}s)",
                fontsize=11,
                fontweight="bold",
            )
    else:
        matrix_axes[0].text(0.5, 0.5, "no weight trajectory", ha="center", va="center")
        for ax_mat in matrix_axes[1:]:
            ax_mat.axis("off")

    for ax_mat, im_dev in im_devs:
        add_weight_matrix_colorbar(ax_mat, im_dev)
    plot_spike_raster_ax(ax_C, window_data, cfg_window)
    ax_C.set_title("Post-learning state", pad=24)

    _plot_trace_lines(ax_E, kuramoto_traces, value_key="R", ylabel="R")
    plot_kuramoto_pdf_multi_ax(ax_F, kuramoto_traces)

    _plot_trace_lines(ax_G, rate_traces, value_key="rate_Hz", ylabel="Mean firing rate (Hz)")
    if rate_traces:
        _plot_stimulus_rate_distribution(ax_H, rate_traces)
    else:
        ax_H.text(0.5, 0.5, "no data", ha="center", va="center")
        ax_H.set_axis_off()

    plot_pdf_cv_ax(ax_D, metrics_window["cv_N"])

    synchrony_highlights = _build_synchrony_highlights(kuramoto_traces)
    async_irregular_highlight = _build_async_irregular_highlight(kuramoto_traces)

    highlight_bands = list(synchrony_highlights)
    if async_irregular_highlight is not None:
        highlight_bands.append(async_irregular_highlight)

    for ax in (ax_C, ax_E, ax_G):
        for band in highlight_bands:
            ax.axvspan(
                band["start_s"],
                band["end_s"],
                color=band["color"],
                alpha=min(1.0, float(band.get("alpha", 0.16)) * 1.15),
                zorder=0,
            )

    for band in highlight_bands:
        label = band.get("label")
        if not label:
            continue
        x_mid = 0.5 * (float(band["start_s"]) + float(band["end_s"]))
        ax_C.text(
            x_mid,
            1.08,
            str(label),
            transform=ax_C.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            color=str(band.get("color", "black")),
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": str(band.get("color", "black")),
                "linewidth": 0.9,
                "alpha": 0.95,
            },
            clip_on=False,
        )

    for ax in (ax_C, ax_E, ax_G):
        ax.set_xlim(window_start_s, window_end_s)
        ax.set_xticks(shared_ticks)

    for ax in (ax_C, ax_E):
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    ax_G.set_xlabel("Time (seconds)")

    left_center_x = 0.5 * (matrix_axes[0].get_position().x0 + matrix_axes[0].get_position().x1)
    right_center_x = 0.5 * (matrix_axes[-1].get_position().x0 + matrix_axes[-1].get_position().x1)
    dev_top = max(matrix_axes[0].get_position().y1, matrix_axes[-1].get_position().y1)
    fig.text(
        0.5 * (left_center_x + right_center_x),
        dev_top + 0.02,
        "Weight Matrix Development",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

    left_anchor_x = min(
        ax_A.get_position().x0,
        ax_C.get_position().x0,
        ax_E.get_position().x0,
        ax_G.get_position().x0,
    ) - 0.04

    panel_labels = {
        ax_A: "A",
        matrix_axes[0]: "B",
        ax_C: "C",
        ax_D: "D",
        ax_E: "E",
        ax_F: "F",
        ax_G: "G",
        ax_H: "H",
    }
    for ax, label in panel_labels.items():
        bbox = ax.get_position()
        x_pos = left_anchor_x if label in {"A", "B", "C", "E", "G"} else (bbox.x0 - 0.04)
        fig.text(
            x_pos,
            bbox.y1 + 0.01,
            label,
            fontsize=18,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    boxed_axes = [
        ax_A,
        ax_C,
        ax_D,
        ax_E,
        ax_F,
        ax_G,
        ax_H,
        *matrix_axes,
    ]
    for ax in boxed_axes:
        _enable_subplot_box(ax)

    for ax in arrow_axes:
        ax.set_frame_on(False)
        for side in ("left", "right", "top", "bottom"):
            spine = ax.spines.get(side)
            if spine is not None:
                spine.set_visible(False)

    return fig


def save_single_figure_pdf(fig, output_dir: Path, file_name: str):
    from matplotlib.backends.backend_pdf import FigureCanvasPdf

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{file_name}.pdf"
    canvas = FigureCanvasPdf(fig)
    canvas.print_pdf(target)
    print(f"Saved {target}")
    plt.close(fig)


def create_statistical_metrics_figure(metrics):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    ax_cv, ax_R, ax_rate, ax_K = axes

    plot_pdf_cv_ax(ax_cv, metrics["cv_N"])
    kuramoto_traces = metrics.get("kuramoto_traces") or []
    plot_kuramoto_traces_ax(ax_R, kuramoto_traces)
    stimulus_traces = metrics.get("stimulus_rate_traces") or []
    if stimulus_traces:
        _plot_stimulus_rate_distribution(ax_rate, stimulus_traces)
    else:
        plot_pdf_mean_rate_ax(ax_rate, metrics["mean_rates_per_neuron"])
    plot_K_ax(ax_K, metrics["K_post"])

    fig.tight_layout(w_pad=1.5)
    return fig


def create_comparison_figure(
    cfg,
    data,
    normalized_W,
    max_raster_time_ms: float | None,
    original_raster_path: Path,
    original_weight_path: Path,
):
    if not original_raster_path.exists():
        raise FileNotFoundError(
            f"Original raster plot not found at {original_raster_path}"
        )
    if not original_weight_path.exists():
        raise FileNotFoundError(
            f"Original weight matrix plot not found at {original_weight_path}"
        )

    orig_raster_img = mpimg.imread(original_raster_path)
    orig_weight_img = mpimg.imread(original_weight_path)

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(6, 1),
        height_ratios=(1, 0.5),
        wspace=0.15,
        hspace=0.3,
    )

    ax_orig_raster = fig.add_subplot(gs[0, 0])
    ax_orig_raster.imshow(orig_raster_img)
    ax_orig_raster.set_title("Original Raster", fontsize=12)
    ax_orig_raster.axis("off")

    ax_orig_weight = fig.add_subplot(gs[0, 1])
    ax_orig_weight.imshow(orig_weight_img)
    ax_orig_weight.set_title("Original Weight Matrix", fontsize=12)
    ax_orig_weight.axis("off")

    ax_model_raster = fig.add_subplot(gs[1, 0])
    plot_spike_raster_ax(ax_model_raster, data, cfg, max_time_ms=max_raster_time_ms)
    ax_model_raster.set_title("Replication Raster", fontsize=12)

    ax_model_weight = fig.add_subplot(gs[1, 1])
    im = plot_weight_matrix_ax(ax_model_weight, normalized_W, cfg)
    ax_model_weight.set_title("Replication Weight Matrix", fontsize=12)
    add_weight_matrix_colorbar(ax_model_weight, im)

    for ax, label in zip(
        [ax_orig_raster, ax_orig_weight, ax_model_raster, ax_model_weight],
        ("A", "B", "C", "D"),
    ):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - 0.03,
            bbox.y1 + 0.05,
            label,
            fontsize=14,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    fig.tight_layout()
    return fig


def create_metrics_comparison_figure(metrics, original_plots_dir: Path):
    required = {
        "cv": original_plots_dir / "orig_cv.jpeg",
        "R": original_plots_dir / "orig_R.jpeg",
        "mean_rate": original_plots_dir / "orig_mean_firing_rate.jpeg",
        "K": original_plots_dir / "orig_mean_weight_change.jpeg",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing original metric plots: " + ", ".join(missing)
        )

    def _plot_mean_rate(ax):
        stimulus_traces = metrics.get("stimulus_rate_traces") or []
        if stimulus_traces:
            _plot_stimulus_rate_distribution(ax, stimulus_traces)
        else:
            plot_pdf_mean_rate_ax(ax, metrics["mean_rates_per_neuron"])

    weight_change_traces = metrics.get("weight_change_traces") or []
    def _plot_K(ax):
        plot_weight_change_pdf_multi_ax(ax, weight_change_traces[:3])

    figure_rows = [
        (
            "Coefficient of Variation",
            required["cv"],
            lambda ax: plot_pdf_cv_ax(ax, metrics["cv_N"]),
            "Replication Coefficient of Variation",
        ),
        (
            "Kuramoto Order Parameter",
            required["R"],
            lambda ax: plot_kuramoto_pdf_multi_ax(
                ax,
                (metrics.get("kuramoto_traces") or [])[:3],
            ),
            "Replication Kuramoto Order Parameter",
        ),
        (
            "Mean Firing Rate",
            required["mean_rate"],
            _plot_mean_rate,
            "Replication Mean Firing Rate",
        ),
        (
            "Mean Change Rate of Weights",
            required["K"],
            _plot_K,
            "Replication Mean Change Rate of Weights ",
        ),
    ]

    fig = plt.figure(figsize=(11.0, 14))
    gs = fig.add_gridspec(
        len(figure_rows),
        2,
        width_ratios=(1.15, 0.75),
        height_ratios=[1] * len(figure_rows),
        hspace=0.5,
        wspace=0.25,
    )

    axes_for_labels = []
    for row_idx, (title, img_path, plot_fn, replication_title) in enumerate(figure_rows):
        ax_orig = fig.add_subplot(gs[row_idx, 0])
        ax_orig.imshow(mpimg.imread(img_path))
        ax_orig.set_title(f"Original {title}", fontsize=12)
        ax_orig.axis("off")
        axes_for_labels.append(ax_orig)

        ax_model = fig.add_subplot(gs[row_idx, 1])
        plot_fn(ax_model)
        ax_model.set_title(replication_title, fontsize=12)
        axes_for_labels.append(ax_model)

    for idx, ax in enumerate(axes_for_labels):
        label = chr(ord("A") + idx)
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - 0.035,
            bbox.y1 + 0.02,
            label,
            fontsize=15,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    fig.tight_layout()
    return fig


def create_summary_component_figures(
    cfg,
    data,
    metrics,
    normalized_W,
    max_raster_time_ms: float | None = None,
):
    figures = []

    fig_r, ax_r = plt.subplots(figsize=(20, 2.8))
    plot_spike_raster_ax(ax_r, data, cfg, max_time_ms=max_raster_time_ms)
    fig_r.tight_layout()
    figures.append(("spike_raster", fig_r))

    fig_cv, ax_cv = plt.subplots(figsize=(4.5, 3.5))
    plot_pdf_cv_ax(ax_cv, metrics["cv_N"])
    fig_cv.tight_layout()
    figures.append(("pdf_cv", fig_cv))

    fig_R, ax_R = plt.subplots(figsize=(4.5, 3.5))
    plot_kuramoto_traces_ax(ax_R, metrics.get("kuramoto_traces") or [])
    fig_R.tight_layout()
    figures.append(("pdf_R", fig_R))

    fig_rate, ax_rate = plt.subplots(figsize=(4.5, 3.5))
    stimulus_traces = metrics.get("stimulus_rate_traces") or []
    if stimulus_traces:
        _plot_stimulus_rate_distribution(ax_rate, stimulus_traces)
    else:
        plot_pdf_mean_rate_ax(ax_rate, metrics["mean_rates_per_neuron"])
    fig_rate.tight_layout()
    figures.append(("pdf_mean_rate", fig_rate))

    fig_K, ax_K = plt.subplots(figsize=(4.5, 3.5))
    plot_K_ax(ax_K, metrics["K_post"])
    fig_K.tight_layout()
    figures.append(("pdf_mean_weight_change", fig_K))

    fig_W, ax_W = plt.subplots(figsize=(6, 6))
    im = plot_weight_matrix_ax(ax_W, normalized_W, cfg)
    add_weight_matrix_colorbar(ax_W, im)
    fig_W.tight_layout()
    figures.append(("weight_matrix", fig_W))

    return figures


def save_figures(figures, output_dir: Path, file_format: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figures:
        target = output_dir / f"{name}.{file_format}"
        fig.savefig(target, format=file_format, bbox_inches="tight")
        print(f"Saved {target}")
        plt.close(fig)


def display_figures(figures):
    plt.show()


def main():
    args = parse_args()
    if args.output_pdf:
        args.save_plots = True
    raster_limit = 58000.0 if args.short_raster else None
    save_format = None
    if args.save_plots:
        save_format = "pdf" if args.output_pdf else "pgf"
    setup_matplotlib(save_format)
    run_dir = resolve_run_directory(args.run_dir)

    cfg_full, data_full, weights_data_full, weights_over_time_full, stim_metadata_full = load_run(run_dir)
    auto_full_analysis = is_long_run(cfg_full)

    if args.disable_analysis_window:
        cfg, data, weights_data, weights_over_time, stim_metadata = (
            cfg_full,
            data_full,
            weights_data_full,
            weights_over_time_full,
            stim_metadata_full,
        )
    else:
        cfg, data, weights_data, weights_over_time, stim_metadata = load_run(
            run_dir,
            pre_stimulus_window_ms=args.pre_stimulus_window_s * 1000.0,
            post_stimulus_window_ms=args.post_stimulus_window_s * 1000.0,
        )

    if not args.disable_analysis_window and "analysis" in cfg:
        window = cfg["analysis"].get("window_ms") or {}
        if window:
            print(
                "Applied analysis window "
                f"[{window['start']:.1f}, {window['end']:.1f}] ms"
            )
    metrics = compute_summary_metrics(
        cfg,
        data,
        weights_over_time,
        stim_metadata=stim_metadata,
    )
    normalized_W = build_weight_matrices(cfg, weights_data)

    figures = []
    full_analysis_fig = None
    if auto_full_analysis:
        post_start_ms, post_end_ms = resolve_post_learning_interval_ms(cfg_full)
        if args.full_analysis_start_s is not None:
            window_start_ms = float(args.full_analysis_start_s) * 1000.0
            if window_start_ms < post_start_ms or window_start_ms >= post_end_ms:
                raise ValueError(
                    "--full_analysis_start_s must lie inside inferred post-learning phase "
                    f"[{post_start_ms/1000.0:.3f}, {post_end_ms/1000.0:.3f}] s"
                )
        else:
            window_start_ms = post_start_ms
        window_end_ms = min(post_end_ms, window_start_ms + 30000.0)
        if window_end_ms > window_start_ms:
            print(
                "Auto full-analysis window "
                f"[{window_start_ms:.1f}, {window_end_ms:.1f}] ms "
                f"within post phase [{post_start_ms:.1f}, {post_end_ms:.1f}] ms"
            )
            full_analysis_fig = create_bergoin_metrics_figure(
                cfg_full,
                data_full,
                weights_data_full,
                weights_over_time_full,
                stim_metadata_full,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
            )

    if args.output_pdf:
        figures = create_summary_component_figures(
            cfg,
            data,
            metrics,
            normalized_W,
            max_raster_time_ms=raster_limit,
        )
        summary_fig = create_summary_figure(
            cfg,
            data,
            metrics,
            weights_data,
            max_raster_time_ms=raster_limit,
        )
        figures.insert(0, ("summary", summary_fig))
        statistical_fig = create_statistical_metrics_figure(metrics)
        figures.insert(1, ("statistical_metric", statistical_fig))
    elif not args.small_figure:
        fig = create_summary_figure(
            cfg,
            data,
            metrics,
            weights_data,
            max_raster_time_ms=raster_limit,
        )
        figures.append(("summary", fig))
    else:
        fig_r, ax_r = plt.subplots(figsize=(10, 2.5))
        plot_spike_raster_ax(ax_r, data, cfg, max_time_ms=raster_limit)
        fig_r.tight_layout()
        figures.append(("spike_raster", fig_r))

        fig_w, ax_w = plt.subplots(figsize=(6, 6))
        im = plot_weight_matrix_ax(ax_w, normalized_W, cfg)
        add_weight_matrix_colorbar(ax_w, im)
        fig_w.tight_layout()
        figures.append(("weight_matrix", fig_w))

    if args.print_comparison:
        original_plots_dir = Path("results") / "plots" / "original_plots"
        comparison_fig = create_comparison_figure(
            cfg,
            data,
            normalized_W,
            max_raster_time_ms=raster_limit,
            original_raster_path=original_plots_dir / "orig_raster.jpeg",
            original_weight_path=original_plots_dir / "orig_W.jpeg",
        )
        figures.append(("comparison", comparison_fig))
        comparison_metrics_fig = create_metrics_comparison_figure(
            metrics,
            original_plots_dir,
        )
        figures.append(("comparison_metrics", comparison_metrics_fig))

    if args.save_plots:
        output_dir = Path("results") / "plots" / run_dir.name
        if save_format is None:
            raise RuntimeError("save_format must be defined when saving plots")
        save_figures(figures, output_dir, save_format)
    else:
        display_figures(figures)

    if full_analysis_fig is not None:
        output_dir = Path("results") / "plots" / run_dir.name
        save_single_figure_pdf(full_analysis_fig, output_dir, "full_analysis")


if __name__ == "__main__":
    main()
