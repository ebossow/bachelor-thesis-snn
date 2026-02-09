from pathlib import Path
import argparse
import matplotlib
import matplotlib.image as mpimg

from src.analysis.summary_metrics import compute_summary_metrics
from src.analysis.summary_figure import create_summary_figure, _plot_stimulus_rate_distribution
from src.analysis.metrics import build_weight_matrix, normalize_weight_matrix
from src.analysis.plotting import (
    plot_spike_raster_ax,
    plot_weight_matrix_ax,
    plot_pdf_cv_ax,
    plot_pdf_R_ax,
    plot_pdf_mean_rate_ax,
    plot_K_ax,
    add_weight_matrix_colorbar,
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
    args = parser.parse_args()
    if args.output_pdf and not args.generate_plots:
        parser.error("--generate_plots must be specified when using --output_pdf")
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


def create_statistical_metrics_figure(metrics):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    ax_cv, ax_R, ax_rate, ax_K = axes

    plot_pdf_cv_ax(ax_cv, metrics["cv_N"])
    plot_pdf_R_ax(ax_R, metrics["R"])
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
    plot_pdf_R_ax(ax_R, metrics["R"])
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

    cfg, data, weights_data, weights_over_time, stim_metadata = load_run(run_dir)
    metrics = compute_summary_metrics(
        cfg,
        data,
        weights_over_time,
        stim_metadata=stim_metadata,
    )
    normalized_W = build_weight_matrices(cfg, weights_data)

    figures = []
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

    if args.save_plots:
        output_dir = Path("results") / "plots" / run_dir.name
        if save_format is None:
            raise RuntimeError("save_format must be defined when saving plots")
        save_figures(figures, output_dir, save_format)
    else:
        display_figures(figures)


if __name__ == "__main__":
    main()
