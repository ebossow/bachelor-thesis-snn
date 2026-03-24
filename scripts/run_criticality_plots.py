from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import ConnectionPatch
import numpy as np

# Ensure project root is importable when running as `python scripts/...`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scripts.analyze_factor_sweep import build_sigma_colormap, plot_heatmap
except ImportError:
    from analyze_factor_sweep import build_sigma_colormap, plot_heatmap


CRITICALITY_ANALYSIS_ROOT = Path("results") / "criticality_analysis"
CRITICALITY_PLOTS_ROOT = Path("results") / "plots" / "criticality"


def setup_matplotlib_style() -> None:
    matplotlib.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.family": "serif",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )


def _window_display_name(window_name: str) -> str:
    normalized = window_name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized == "pre_learning":
        return "Pre Learning"
    if normalized == "post_learning":
        return "Post Learning"
    return window_name.replace("_", " ").title()


def _population_display_name(population: str) -> str:
    norm = population.strip().lower()
    if "half_0" in norm:
        return "Cluster 1"
    if "half_1" in norm:
        return "Cluster 2"
    return population.replace("_", " ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-ready criticality comparison plots from existing "
            "average analysis heatmaps."
        )
    )
    parser.add_argument(
        "--analysis_dir",
        type=str,
        default=None,
        help=(
            "Path to a specific criticality analysis folder under results/criticality_analysis. "
            "If omitted, the newest folder is used."
        ),
    )
    parser.add_argument(
        "--population",
        type=str,
        default="stimulus_block_E_half_0",
        help="Population folder to compare (default: stimulus_block_E_half_0).",
    )
    parser.add_argument(
        "--pre_window",
        type=str,
        default="pre_learning",
        help="Name of the pre-learning window folder (default: pre_learning).",
    )
    parser.add_argument(
        "--post_window",
        type=str,
        default="post_learning",
        help="Name of the post-learning window folder (default: post_learning).",
    )
    return parser.parse_args()


def find_latest_analysis_dir(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Criticality analysis root not found: {root}")
    candidates = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not candidates:
        raise FileNotFoundError(f"No analysis folders found under: {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_analysis_dir(analysis_dir_arg: str | None) -> Path:
    if analysis_dir_arg is None:
        analysis_dir = find_latest_analysis_dir(CRITICALITY_ANALYSIS_ROOT)
        print(f"Using latest analysis directory: {analysis_dir}")
        return analysis_dir

    analysis_dir = Path(analysis_dir_arg)
    if not analysis_dir.exists():
        raise FileNotFoundError(f"Specified analysis directory not found: {analysis_dir}")
    print(f"Using specified analysis directory: {analysis_dir}")
    return analysis_dir


def resolve_mean_folder(analysis_dir: Path) -> Path:
    mean_candidates = sorted(
        [
            p
            for p in analysis_dir.iterdir()
            if p.is_dir() and p.name.endswith("_mean") and not p.name.startswith(".")
        ]
    )
    if mean_candidates:
        mean_dir = mean_candidates[-1]
        print(f"Using average folder: {mean_dir}")
        return mean_dir

    if (analysis_dir / "branching").exists():
        print(f"No *_mean folder found, using analysis folder directly: {analysis_dir}")
        return analysis_dir

    raise FileNotFoundError(
        f"Could not find a usable average folder in {analysis_dir} (expected '*_mean' or 'branching')."
    )


def _load_branching_metrics_npz(window_dir: Path) -> dict[str, np.ndarray]:
    metrics_path = window_dir / "metrics.npz"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    with np.load(metrics_path, allow_pickle=False) as payload:
        return {
            "alphas": np.asarray(payload["alphas"], dtype=float),
            "betas": np.asarray(payload["betas"], dtype=float),
            "sigma_mr": np.asarray(payload["sigma_mr"], dtype=float),
            "tau_mr_ms": np.asarray(payload["tau_mr_ms"], dtype=float),
        }


def _render_window_heatmaps(
    matrix: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    population: str,
    window_name: str,
    output_dir: Path,
    sigma_cmap_override=None,
    sigma_limits_override: tuple[float, float] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    window_label = _window_display_name(window_name)
    population_label = _population_display_name(population)

    if sigma_cmap_override is None:
        sigma_cmap, sigma_limits = build_sigma_colormap(matrix)
    else:
        sigma_cmap = sigma_cmap_override
        sigma_limits = sigma_limits_override
    sigma_path = output_dir / f"heatmap_sigma_{window_name}.pdf"
    plot_heatmap(
        matrix,
        alphas,
        betas,
        title=rf"Branching Ratio $m$ - {window_label} - {population_label}",
        out_path=sigma_path,
        cmap=sigma_cmap,
        color_limits=sigma_limits,
        zoom_color_limits=(0.90, 1.0),
        main_cbar_label="",
        zoom_cbar_label="",
        x_label=r"$s_{exc}$",
        y_label=r"$s_{inh}$",
    )
    print(f"Saved {sigma_path}")


def _render_window_tau_heatmap(
    matrix: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    population: str,
    window_name: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    window_label = _window_display_name(window_name)
    population_label = _population_display_name(population)
    tau_path = output_dir / f"heatmap_tau_mr_ms_{window_name}.pdf"
    plot_heatmap(
        matrix,
        alphas,
        betas,
        title=rf"Autocorrelation Time $\tau_{{MR}}$ (ms) - {window_label} - {population_label}",
        out_path=tau_path,
        main_cbar_label="",
        main_cbar_tick_format="%.2f",
        x_label=r"$s_{exc}$",
        y_label=r"$s_{inh}$",
    )
    print(f"Saved {tau_path}")


def _render_sigma_comparison_shared_colorbars(
    pre_matrix: np.ndarray,
    post_matrix: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    population: str,
    pre_window: str,
    post_window: str,
    output_dir: Path,
    zoom_color_limits: tuple[float, float] = (0.90, 1.0),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    population_label = _population_display_name(population)
    pre_label = _window_display_name(pre_window)
    post_label = _window_display_name(post_window)

    combined_for_scale = np.concatenate([pre_matrix.ravel(), post_matrix.ravel()]).reshape(2, -1)
    sigma_cmap, sigma_limits = build_sigma_colormap(combined_for_scale)

    vmin = vmax = None
    if sigma_limits is not None:
        vmin, vmax = sigma_limits

    fig, (ax_pre, ax_post) = plt.subplots(1, 2, figsize=(11.2, 5.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.80, bottom=0.12, top=0.86, wspace=0.16)

    extent = [float(alphas.min()), float(alphas.max()), float(betas.min()), float(betas.max())]
    im_pre = ax_pre.imshow(
        pre_matrix,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap=sigma_cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax_post.imshow(
        post_matrix,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap=sigma_cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax_pre.set_title(rf"$m$ - {pre_label}")
    ax_post.set_title(rf"$m$ - {post_label}")
    ax_pre.set_xlabel(r"$s_{exc}$")
    ax_post.set_xlabel(r"$s_{exc}$")
    ax_pre.set_ylabel(r"$s_{inh}$")
    ax_post.set_ylabel("")
    tick_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    for ax in (ax_pre, ax_post):
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)
    ax_post.tick_params(axis="y", labelleft=False)
    fig.suptitle(rf"Branching Ratio $m$ - {population_label}")

    pre_pos = ax_pre.get_position()
    post_pos = ax_post.get_position()
    arrow_y = 0.5 * (pre_pos.y0 + pre_pos.y1)
    arrow_start = (pre_pos.x1 + 0.01, arrow_y)
    arrow_end = (post_pos.x0 - 0.01, arrow_y)
    ax_pre.annotate(
        "",
        xy=arrow_end,
        xytext=arrow_start,
        xycoords=fig.transFigure,
        textcoords=fig.transFigure,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.4,
            "color": "black",
        },
        annotation_clip=False,
    )

    fig.canvas.draw()
    right_ax_pos = ax_post.get_position()
    cbar_width = 0.030
    cbar_gap = 0.020
    main_x = right_ax_pos.x1 + 0.045
    zoom_x = main_x + cbar_width + cbar_gap

    main_cax = fig.add_axes([main_x, right_ax_pos.y0, cbar_width, right_ax_pos.height])
    main_cbar = fig.colorbar(im_pre, cax=main_cax)
    main_cbar.ax.yaxis.set_ticks_position("left")
    main_cbar.ax.yaxis.set_label_position("left")

    zmin, zmax = zoom_color_limits
    if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
        full_vmin = float(im_pre.norm.vmin) if im_pre.norm.vmin is not None else float(np.nanmin(combined_for_scale))
        full_vmax = float(im_pre.norm.vmax) if im_pre.norm.vmax is not None else float(np.nanmax(combined_for_scale))
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

        zoom_ax = fig.add_axes([zoom_x, right_ax_pos.y0, cbar_width, right_ax_pos.height])
        zoom_sm = plt.cm.ScalarMappable(norm=Normalize(vmin=zmin, vmax=zmax), cmap=zoom_cmap)
        zoom_sm.set_array([])
        zoom_cbar = fig.colorbar(zoom_sm, cax=zoom_ax)
        zoom_cbar.ax.yaxis.set_ticks_position("right")
        zoom_cbar.ax.yaxis.set_label_position("right")

        y_main_low = full_vmin + frac_min * (full_vmax - full_vmin)
        y_main_high = full_vmin + frac_max * (full_vmax - full_vmin)
        for y_main in (y_main_low, y_main_high):
            main_cbar.ax.axhline(y=y_main, color="black", linewidth=1.0, alpha=0.8)
        for y_zoom in (zmin, zmax):
            zoom_cbar.ax.axhline(y=y_zoom, color="black", linewidth=1.0, alpha=0.8)

        connector_low = ConnectionPatch(
            xyA=(1.0, frac_min),
            coordsA="axes fraction",
            axesA=main_cbar.ax,
            xyB=(0.0, 0.0),
            coordsB="axes fraction",
            axesB=zoom_cbar.ax,
            color="black",
            linewidth=0.9,
            alpha=0.75,
            clip_on=False,
        )
        connector_high = ConnectionPatch(
            xyA=(1.0, frac_max),
            coordsA="axes fraction",
            axesA=main_cbar.ax,
            xyB=(0.0, 1.0),
            coordsB="axes fraction",
            axesB=zoom_cbar.ax,
            color="black",
            linewidth=0.9,
            alpha=0.75,
            clip_on=False,
        )
        fig.add_artist(connector_low)
        fig.add_artist(connector_high)

    out_path = output_dir / "heatmap_sigma_pre_vs_post.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _finite_limits_from_two(a: np.ndarray, b: np.ndarray) -> tuple[float, float] | None:
    merged = np.concatenate([a.ravel(), b.ravel()])
    finite = merged[np.isfinite(merged)]
    if finite.size == 0:
        return None
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return lo, hi


def _render_branching_combined_comparison(
    pre_sigma: np.ndarray,
    post_sigma: np.ndarray,
    pre_tau: np.ndarray,
    post_tau: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    population: str,
    pre_window: str,
    post_window: str,
    output_dir: Path,
    zoom_color_limits: tuple[float, float] = (0.90, 1.0),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    population_label = _population_display_name(population)
    pre_label = _window_display_name(pre_window)
    post_label = _window_display_name(post_window)

    sigma_combined = np.concatenate([pre_sigma.ravel(), post_sigma.ravel()]).reshape(2, -1)
    sigma_cmap, sigma_limits = build_sigma_colormap(sigma_combined)
    sigma_vmin = sigma_vmax = None
    if sigma_limits is not None:
        sigma_vmin, sigma_vmax = sigma_limits

    tau_limits = _finite_limits_from_two(pre_tau, post_tau)
    tau_vmin = tau_vmax = None
    if tau_limits is not None:
        tau_vmin, tau_vmax = tau_limits

    fig, axs = plt.subplots(2, 2, figsize=(11.5, 9.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.80, bottom=0.10, top=0.90, wspace=0.16, hspace=0.28)
    ax_sigma_pre, ax_sigma_post = axs[0]
    ax_tau_pre, ax_tau_post = axs[1]

    extent = [float(alphas.min()), float(alphas.max()), float(betas.min()), float(betas.max())]
    im_sigma = ax_sigma_pre.imshow(
        pre_sigma,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap=sigma_cmap,
        vmin=sigma_vmin,
        vmax=sigma_vmax,
    )
    ax_sigma_post.imshow(
        post_sigma,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap=sigma_cmap,
        vmin=sigma_vmin,
        vmax=sigma_vmax,
    )

    im_tau = ax_tau_pre.imshow(
        pre_tau,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap="viridis",
        vmin=tau_vmin,
        vmax=tau_vmax,
    )
    ax_tau_post.imshow(
        post_tau,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap="viridis",
        vmin=tau_vmin,
        vmax=tau_vmax,
    )

    ax_sigma_pre.set_title(rf"$m$ - {pre_label}")
    ax_sigma_post.set_title(rf"$m$ - {post_label}")
    ax_tau_pre.set_title(rf"$\tau_{{MR}}$ - {pre_label}")
    ax_tau_post.set_title(rf"$\tau_{{MR}}$ - {post_label}")

    tick_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    for ax in (ax_sigma_pre, ax_sigma_post, ax_tau_pre, ax_tau_post):
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)
        ax.set_xlabel(r"$s_{exc}$")

    ax_sigma_pre.set_ylabel(r"$s_{inh}$")
    ax_tau_pre.set_ylabel(r"$s_{inh}$")
    ax_sigma_post.set_ylabel("")
    ax_tau_post.set_ylabel("")
    ax_sigma_post.tick_params(axis="y", labelleft=False)
    ax_tau_post.tick_params(axis="y", labelleft=False)

    fig.suptitle(rf"Branching Ratio $m$ and Autocorrelation Time $\tau_{{MR}}$ Comparison - {population_label}")

    for left_ax, right_ax in ((ax_sigma_pre, ax_sigma_post), (ax_tau_pre, ax_tau_post)):
        left_pos = left_ax.get_position()
        right_pos = right_ax.get_position()
        arrow_y = 0.5 * (left_pos.y0 + left_pos.y1)
        arrow_start = (left_pos.x1 + 0.01, arrow_y)
        arrow_end = (right_pos.x0 - 0.01, arrow_y)
        left_ax.annotate(
            "",
            xy=arrow_end,
            xytext=arrow_start,
            xycoords=fig.transFigure,
            textcoords=fig.transFigure,
            arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "black"},
            annotation_clip=False,
        )

    fig.canvas.draw()
    sigma_right = ax_sigma_post.get_position()
    tau_right = ax_tau_post.get_position()
    cbar_width = 0.030
    cbar_gap = 0.020
    main_x = sigma_right.x1 + 0.045
    zoom_x = main_x + cbar_width + cbar_gap

    sigma_cax = fig.add_axes([main_x, sigma_right.y0, cbar_width, sigma_right.height])
    sigma_cbar = fig.colorbar(im_sigma, cax=sigma_cax)
    sigma_cbar.set_label(r"$m$")
    sigma_cbar.ax.yaxis.set_ticks_position("left")
    sigma_cbar.ax.yaxis.set_label_position("left")

    zmin, zmax = zoom_color_limits
    if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
        full_vmin = float(im_sigma.norm.vmin) if im_sigma.norm.vmin is not None else float(np.nanmin(sigma_combined))
        full_vmax = float(im_sigma.norm.vmax) if im_sigma.norm.vmax is not None else float(np.nanmax(sigma_combined))
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

        sigma_zoom_ax = fig.add_axes([zoom_x, sigma_right.y0, cbar_width, sigma_right.height])
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

    tau_cax = fig.add_axes([main_x, tau_right.y0, cbar_width, tau_right.height])
    tau_cbar = fig.colorbar(im_tau, cax=tau_cax)
    tau_cbar.set_label(r"$\tau_{MR}$ (ms)")
    tau_cbar.ax.yaxis.set_ticks_position("left")
    tau_cbar.ax.yaxis.set_label_position("left")

    pop_slug = population_label.lower().replace(" ", "_")
    out_path = output_dir / f"branching_ratio_comparison_{pop_slug}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _render_branching_whole_vs_cluster_comparison(
    whole_pre_sigma: np.ndarray,
    whole_post_sigma: np.ndarray,
    whole_pre_tau: np.ndarray,
    whole_post_tau: np.ndarray,
    cluster_pre_sigma: np.ndarray,
    cluster_post_sigma: np.ndarray,
    cluster_pre_tau: np.ndarray,
    cluster_post_tau: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    pre_window: str,
    post_window: str,
    output_dir: Path,
    zoom_color_limits: tuple[float, float] = (0.90, 1.0),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pre_label = _window_display_name(pre_window)
    post_label = _window_display_name(post_window)

    title_fontsize = 14
    axis_label_fontsize = 18
    tick_label_fontsize = 11
    suptitle_fontsize = 16
    panel_label_fontsize = 16
    cbar_label_fontsize = 13
    cbar_tick_fontsize = 11

    sigma_combined = np.concatenate(
        [
            whole_pre_sigma.ravel(),
            whole_post_sigma.ravel(),
            cluster_pre_sigma.ravel(),
            cluster_post_sigma.ravel(),
        ]
    ).reshape(4, -1)
    sigma_cmap, sigma_limits = build_sigma_colormap(sigma_combined)
    sigma_vmin = sigma_vmax = None
    if sigma_limits is not None:
        sigma_vmin, sigma_vmax = sigma_limits

    tau_combined = np.concatenate(
        [
            whole_pre_tau.ravel(),
            whole_post_tau.ravel(),
            cluster_pre_tau.ravel(),
            cluster_post_tau.ravel(),
        ]
    )
    tau_finite = tau_combined[np.isfinite(tau_combined)]
    tau_vmin = tau_vmax = None
    if tau_finite.size > 0:
        tau_vmin = float(np.min(tau_finite))
        tau_vmax = float(np.max(tau_finite))
        if not np.isfinite(tau_vmin) or not np.isfinite(tau_vmax) or tau_vmax <= tau_vmin:
            tau_vmin = tau_vmax = None

    fig, axs = plt.subplots(2, 4, figsize=(20.0, 9.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.06, right=0.86, bottom=0.10, top=0.90, wspace=0.14, hspace=0.28)

    extent = [float(alphas.min()), float(alphas.max()), float(betas.min()), float(betas.max())]

    im_sigma = axs[0, 0].imshow(
        whole_pre_sigma,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap=sigma_cmap,
        vmin=sigma_vmin,
        vmax=sigma_vmax,
    )
    axs[0, 1].imshow(
        whole_post_sigma,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap=sigma_cmap,
        vmin=sigma_vmin,
        vmax=sigma_vmax,
    )
    axs[0, 2].imshow(
        cluster_pre_sigma,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap=sigma_cmap,
        vmin=sigma_vmin,
        vmax=sigma_vmax,
    )
    axs[0, 3].imshow(
        cluster_post_sigma,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap=sigma_cmap,
        vmin=sigma_vmin,
        vmax=sigma_vmax,
    )

    im_tau = axs[1, 0].imshow(
        whole_pre_tau,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap="viridis",
        vmin=tau_vmin,
        vmax=tau_vmax,
    )
    axs[1, 1].imshow(
        whole_post_tau,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap="viridis",
        vmin=tau_vmin,
        vmax=tau_vmax,
    )
    axs[1, 2].imshow(
        cluster_pre_tau,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap="viridis",
        vmin=tau_vmin,
        vmax=tau_vmax,
    )
    axs[1, 3].imshow(
        cluster_post_tau,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap="viridis",
        vmin=tau_vmin,
        vmax=tau_vmax,
    )

    axs[0, 0].set_title(rf"$m$ - Whole Population - {pre_label}", fontsize=title_fontsize)
    axs[0, 1].set_title(rf"$m$ - Whole Population - {post_label}", fontsize=title_fontsize)
    axs[0, 2].set_title(rf"$m$ - Cluster 1 - {pre_label}", fontsize=title_fontsize)
    axs[0, 3].set_title(rf"$m$ - Cluster 1 - {post_label}", fontsize=title_fontsize)
    axs[1, 0].set_title(rf"$\tau_{{MR}}$ - Whole Population - {pre_label}", fontsize=title_fontsize)
    axs[1, 1].set_title(rf"$\tau_{{MR}}$ - Whole Population - {post_label}", fontsize=title_fontsize)
    axs[1, 2].set_title(rf"$\tau_{{MR}}$ - Cluster 1 - {pre_label}", fontsize=title_fontsize)
    axs[1, 3].set_title(rf"$\tau_{{MR}}$ - Cluster 1 - {post_label}", fontsize=title_fontsize)

    tick_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    for ax in axs.ravel():
        ax.set_xticks(tick_values)
        ax.set_yticks(tick_values)
        ax.set_xlabel(r"$s_{exc}$", fontsize=axis_label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_label_fontsize)

    axs[0, 0].set_ylabel(r"$s_{inh}$", fontsize=axis_label_fontsize)
    axs[1, 0].set_ylabel(r"$s_{inh}$", fontsize=axis_label_fontsize)
    for col in (1, 2, 3):
        axs[0, col].set_ylabel("")
        axs[1, col].set_ylabel("")
        axs[0, col].tick_params(axis="y", labelleft=False)
        axs[1, col].tick_params(axis="y", labelleft=False)

    fig.suptitle(
        rf"Branching Ratio $m$ and Autocorrelation Time $\tau_{{MR}}$ Comparison - Whole Population vs Cluster 1",
        fontsize=suptitle_fontsize,
    )

    panel_labels = [
        ("A", axs[0, 0]),
        ("B", axs[0, 2]),
        ("C", axs[1, 0]),
        ("D", axs[1, 2]),
    ]
    for label, ax in panel_labels:
        pos = ax.get_position()
        fig.text(
            pos.x0 - 0.010,
            pos.y1 + 0.012,
            label,
            va="bottom",
            ha="left",
            fontsize=panel_label_fontsize,
            fontweight="bold",
        )

    for row in (0, 1):
        for left_col, right_col in ((0, 1), (2, 3)):
            left_ax = axs[row, left_col]
            right_ax = axs[row, right_col]
            left_pos = left_ax.get_position()
            right_pos = right_ax.get_position()
            arrow_y = 0.5 * (left_pos.y0 + left_pos.y1)
            arrow_start = (left_pos.x1 + 0.008, arrow_y)
            arrow_end = (right_pos.x0 - 0.008, arrow_y)
            left_ax.annotate(
                "",
                xy=arrow_end,
                xytext=arrow_start,
                xycoords=fig.transFigure,
                textcoords=fig.transFigure,
                arrowprops={"arrowstyle": "->", "linewidth": 1.4, "color": "black"},
                annotation_clip=False,
            )

    fig.canvas.draw()
    sigma_right = axs[0, 3].get_position()
    tau_right = axs[1, 3].get_position()
    cbar_width = 0.018
    cbar_gap = 0.014
    main_x = sigma_right.x1 + 0.030
    zoom_x = main_x + cbar_width + cbar_gap

    sigma_cax = fig.add_axes([main_x, sigma_right.y0, cbar_width, sigma_right.height])
    sigma_cbar = fig.colorbar(im_sigma, cax=sigma_cax)
    sigma_cbar.set_label(r"$m$", fontsize=cbar_label_fontsize)
    sigma_cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
    sigma_cbar.ax.yaxis.set_ticks_position("left")
    sigma_cbar.ax.yaxis.set_label_position("left")

    zmin, zmax = zoom_color_limits
    if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
        full_vmin = float(im_sigma.norm.vmin) if im_sigma.norm.vmin is not None else float(np.nanmin(sigma_combined))
        full_vmax = float(im_sigma.norm.vmax) if im_sigma.norm.vmax is not None else float(np.nanmax(sigma_combined))
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

        sigma_zoom_ax = fig.add_axes([zoom_x, sigma_right.y0, cbar_width, sigma_right.height])
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

    tau_cax = fig.add_axes([main_x, tau_right.y0, cbar_width, tau_right.height])
    tau_cbar = fig.colorbar(im_tau, cax=tau_cax)
    tau_cbar.set_label(r"$\tau_{MR}$ (ms)", fontsize=cbar_label_fontsize)
    tau_cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
    tau_cbar.ax.yaxis.set_ticks_position("left")
    tau_cbar.ax.yaxis.set_label_position("left")

    out_path = output_dir / "branching_ratio_whole_vs_cluster.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def create_branching_pre_post_plots(
    mean_dir: Path,
    output_root: Path,
    population: str,
    pre_window: str,
    post_window: str,
) -> None:
    branching_source = mean_dir / "branching" / population
    pre_dir = branching_source / pre_window
    post_dir = branching_source / post_window

    if not pre_dir.exists():
        raise FileNotFoundError(f"Missing pre window directory: {pre_dir}")
    if not post_dir.exists():
        raise FileNotFoundError(f"Missing post window directory: {post_dir}")

    branching_output = output_root / "branching" / population
    (output_root / "avalanche").mkdir(parents=True, exist_ok=True)

    pre_metrics = _load_branching_metrics_npz(pre_dir)
    post_metrics = _load_branching_metrics_npz(post_dir)

    if not np.allclose(pre_metrics["alphas"], post_metrics["alphas"]):
        raise ValueError("Alpha grids differ between pre and post windows")
    if not np.allclose(pre_metrics["betas"], post_metrics["betas"]):
        raise ValueError("Beta grids differ between pre and post windows")

    alphas = pre_metrics["alphas"]
    betas = pre_metrics["betas"]
    _render_branching_combined_comparison(
        pre_sigma=pre_metrics["sigma_mr"],
        post_sigma=post_metrics["sigma_mr"],
        pre_tau=pre_metrics["tau_mr_ms"],
        post_tau=post_metrics["tau_mr_ms"],
        alphas=alphas,
        betas=betas,
        population=population,
        pre_window=pre_window,
        post_window=post_window,
        output_dir=branching_output,
        zoom_color_limits=(0.90, 1.0),
    )

    whole_population = "whole_population"
    cluster_population = "stimulus_block_E_half_0"
    whole_pre_dir = mean_dir / "branching" / whole_population / pre_window
    whole_post_dir = mean_dir / "branching" / whole_population / post_window
    cluster_pre_dir = mean_dir / "branching" / cluster_population / pre_window
    cluster_post_dir = mean_dir / "branching" / cluster_population / post_window

    if not whole_pre_dir.exists():
        raise FileNotFoundError(f"Missing whole-population pre window directory: {whole_pre_dir}")
    if not whole_post_dir.exists():
        raise FileNotFoundError(f"Missing whole-population post window directory: {whole_post_dir}")
    if not cluster_pre_dir.exists():
        raise FileNotFoundError(f"Missing cluster-1 pre window directory: {cluster_pre_dir}")
    if not cluster_post_dir.exists():
        raise FileNotFoundError(f"Missing cluster-1 post window directory: {cluster_post_dir}")

    whole_pre_metrics = _load_branching_metrics_npz(whole_pre_dir)
    whole_post_metrics = _load_branching_metrics_npz(whole_post_dir)
    cluster_pre_metrics = _load_branching_metrics_npz(cluster_pre_dir)
    cluster_post_metrics = _load_branching_metrics_npz(cluster_post_dir)

    for label, lhs, rhs in (
        ("whole_population", whole_pre_metrics, whole_post_metrics),
        ("cluster_1", cluster_pre_metrics, cluster_post_metrics),
    ):
        if not np.allclose(lhs["alphas"], rhs["alphas"]):
            raise ValueError(f"Alpha grids differ between pre and post windows for {label}")
        if not np.allclose(lhs["betas"], rhs["betas"]):
            raise ValueError(f"Beta grids differ between pre and post windows for {label}")

    if not np.allclose(whole_pre_metrics["alphas"], cluster_pre_metrics["alphas"]):
        raise ValueError("Alpha grids differ between whole population and cluster 1")
    if not np.allclose(whole_pre_metrics["betas"], cluster_pre_metrics["betas"]):
        raise ValueError("Beta grids differ between whole population and cluster 1")

    _render_branching_whole_vs_cluster_comparison(
        whole_pre_sigma=whole_pre_metrics["sigma_mr"],
        whole_post_sigma=whole_post_metrics["sigma_mr"],
        whole_pre_tau=whole_pre_metrics["tau_mr_ms"],
        whole_post_tau=whole_post_metrics["tau_mr_ms"],
        cluster_pre_sigma=cluster_pre_metrics["sigma_mr"],
        cluster_post_sigma=cluster_post_metrics["sigma_mr"],
        cluster_pre_tau=cluster_pre_metrics["tau_mr_ms"],
        cluster_post_tau=cluster_post_metrics["tau_mr_ms"],
        alphas=whole_pre_metrics["alphas"],
        betas=whole_pre_metrics["betas"],
        pre_window=pre_window,
        post_window=post_window,
        output_dir=output_root / "branching",
        zoom_color_limits=(0.90, 1.0),
    )


def main() -> None:
    args = parse_args()
    setup_matplotlib_style()

    analysis_dir = resolve_analysis_dir(args.analysis_dir)
    mean_dir = resolve_mean_folder(analysis_dir)

    create_branching_pre_post_plots(
        mean_dir=mean_dir,
        output_root=CRITICALITY_PLOTS_ROOT,
        population=args.population,
        pre_window=args.pre_window,
        post_window=args.post_window,
    )


if __name__ == "__main__":
    main()