"""Aggregate MR estimator and avalanche-based DCC across a resting sweep."""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import numpy as np
import yaml

try:
    import powerlaw  # type: ignore

    POWERLAW_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    powerlaw = None
    POWERLAW_AVAILABLE = False

from src.analysis.metrics import (
    avalanche_sizes_from_times,
    average_inter_event_interval,
    binned_spike_counts,
    branching_ratio_mr_estimator,
)
from src.analysis.util import combine_spikes, load_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze resting sweep results")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        help="Path to results/resting_with_factors/run_* directory",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/resting_with_factors"),
        help="Root folder containing run_* directories",
    )
    parser.add_argument(
        "--t-start-ms",
        type=float,
        default=0.0,
        help="Start of analysis window (ms)",
    )
    parser.add_argument(
        "--t-stop-ms",
        type=float,
        default=None,
        help="End of analysis window (ms); default = simtime",
    )
    parser.add_argument(
        "--mr-dt-ms",
        type=float,
        default=1,
        help="Bin width for MR estimator (ms)",
    )
    parser.add_argument(
        "--avalanche-dt-ms",
        type=float,
        default=None,
        help="Bin width for avalanche segmentation (ms)",
    )
    parser.add_argument(
        "--min-avalanche-size",
        type=int,
        default=1,
        help="Minimum avalanche size to consider",
    )
    parser.add_argument(
        "--disable-plots",
        action="store_true",
        help="Skip heatmap generation",
    )
    parser.add_argument(
        "--mr-fit-start-ms",
        type=float,
        default=10.0,
        help="Lower lag bound (ms) for MR fit; <=0 keeps the full range",
    )
    parser.add_argument(
        "--mr-fit-stop-ms",
        type=float,
        default=60.0,
        help="Upper lag bound (ms) for MR fit; <=0 keeps the full range",
    )
    parser.add_argument(
        "--mr-min-fit-points",
        type=int,
        default=3,
        help="Minimum r_k samples required after applying the lag window",
    )
    parser.add_argument(
        "--mr-fit-use-offset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use exponential-with-offset MR model (use --no-mr-fit-use-offset for plain exp)",
    )

    args = parser.parse_args()
    if args.mr_fit_start_ms is not None and args.mr_fit_start_ms <= 0.0:
        args.mr_fit_start_ms = None
    if args.mr_fit_stop_ms is not None and args.mr_fit_stop_ms <= 0.0:
        args.mr_fit_stop_ms = None
    if (
        args.mr_fit_start_ms is not None
        and args.mr_fit_stop_ms is not None
        and args.mr_fit_stop_ms <= args.mr_fit_start_ms
    ):
        parser.error("--mr-fit-stop-ms must be greater than --mr-fit-start-ms")
    args.mr_min_fit_points = max(2, int(args.mr_min_fit_points))
    return args


def find_latest_sweep_dir(root: Path) -> Path:
    candidates: list[Path] = []
    if not root.exists():
        raise FileNotFoundError(f"results root not found: {root}")
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.startswith("run_"):
            candidates.append(entry)
    if not candidates:
        raise RuntimeError(f"No run_* directories in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_sweep_summary(sweep_dir: Path) -> list[dict[str, Any]]:
    summary_path = sweep_dir / "sweep_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    rows: list[dict[str, Any]] = []
    with summary_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def load_config_snapshot(sweep_dir: Path) -> dict[str, Any] | None:
    snapshot = sweep_dir / "sweep_config_snapshot.yaml"
    if not snapshot.exists():
        return None
    with snapshot.open("r") as fh:
        return yaml.safe_load(fh)


def build_grid_coords(rows: Iterable[dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    alpha_set = sorted({float(row["alpha"]) for row in rows})
    beta_set = sorted({float(row["beta"]) for row in rows})
    return np.asarray(alpha_set, float), np.asarray(beta_set, float)


def fit_powerlaw_exponent(values: np.ndarray, discrete: bool = True) -> float:
    if not POWERLAW_AVAILABLE:
        return float("nan")
    vals = np.asarray(values, float)
    vals = vals[vals > 0]
    if vals.size < 5:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = powerlaw.Fit(vals, verbose=False, discrete=discrete)  # type: ignore[arg-type]
    return float(fit.power_law.alpha)


def fit_gamma_exponent(sizes: np.ndarray, durations: np.ndarray) -> float:
    size_arr = np.asarray(sizes, float)
    duration_arr = np.asarray(durations, float)
    mask = (size_arr > 0) & (duration_arr > 0)
    if mask.sum() < 2:
        return float("nan")

    # Match Fontenele et al. by regressing mean size per duration bin instead of raw points.
    filtered_sizes = size_arr[mask]
    filtered_durations = duration_arr[mask]
    duration_unique, inverse = np.unique(filtered_durations, return_inverse=True)
    mean_sizes = np.zeros(duration_unique.shape, float)
    counts = np.zeros(duration_unique.shape, float)
    np.add.at(mean_sizes, inverse, filtered_sizes)
    np.add.at(counts, inverse, 1.0)
    valid = counts > 0
    if np.count_nonzero(valid) < 2:
        return float("nan")
    mean_sizes[valid] /= counts[valid]

    log_d = np.log(duration_unique[valid])
    log_s = np.log(mean_sizes[valid])
    if np.allclose(log_d, log_d[0]):
        return float("nan")
    slope, _ = np.polyfit(log_d, log_s, 1)
    return float(slope)


def compute_gamma_pred(tau_size: float, tau_duration: float) -> float:
    if not np.isfinite(tau_size) or not np.isfinite(tau_duration):
        return float("nan")
    if abs(tau_duration - 1.0) < 1e-9:
        return float("nan")
    return float((tau_size - 1.0) / (tau_duration - 1.0))


def compute_dcc(tau_size: float, tau_duration: float, gamma_fitted: float) -> tuple[float, float]:
    gamma_pred = compute_gamma_pred(tau_size, tau_duration)
    if not np.isfinite(gamma_pred) or not np.isfinite(gamma_fitted):
        return float("nan"), gamma_pred
    return float(abs(gamma_pred - gamma_fitted)), gamma_pred


def resolve_run_path(row: dict[str, Any], sweep_dir: Path) -> Path:
    run_name = row.get("run_name")
    if run_name:
        candidate = sweep_dir / run_name
        if candidate.exists():
            return candidate
    run_dir = row.get("run_dir")
    if run_dir:
        path = Path(run_dir)
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot locate run folder for row: {row}")


def analyze_run(
    run_path: Path,
    t_start_ms: float,
    t_stop_override: float | None,
    dt_mr_ms: float,
    dt_avalanche_override_ms: float | None,
    min_avalanche_size: int,
    fit_lag_ms_min: float | None,
    fit_lag_ms_max: float | None,
    fit_use_offset: bool,
    min_fit_points: int,
) -> dict[str, float]:
    cfg, data, _, _ = load_run(run_path)
    simtime_ms = float(cfg["experiment"]["simtime_ms"])
    t_stop_ms = float(t_stop_override) if t_stop_override is not None else simtime_ms
    spikes = combine_spikes(data, ["spikes_E", "spikes_IH", "spikes_IA"])
    times = np.asarray(spikes["times"], float)
    mask = (times >= t_start_ms) & (times < t_stop_ms)
    times = times[mask]

    result = {
        "sigma_mr": float("nan"),
        "tau_mr_ms": float("nan"),
        "tau_size": float("nan"),
        "tau_duration": float("nan"),
        "gamma_fitted": float("nan"),
        "gamma_pred": float("nan"),
        "dcc": float("nan"),
        "aiei_ms": float("nan"),
        "dt_avalanche_ms": float("nan"),
        "n_sizes": float(0),
        "n_durations": float(0),
        "n_counts": float(0),
    }
    if times.size == 0:
        return result

    counts, _ = binned_spike_counts(times, t_start_ms, t_stop_ms, dt_mr_ms)
    result["n_counts"] = float(counts.size)
    if counts.size >= 3:
        sigma_mr, tau_ms = branching_ratio_mr_estimator(
            counts,
            dt_mr_ms,
            fit_lag_ms_min=fit_lag_ms_min,
            fit_lag_ms_max=fit_lag_ms_max,
            fit_use_offset=fit_use_offset,
            min_fit_points=min_fit_points,
        )
        result["sigma_mr"] = sigma_mr
        result["tau_mr_ms"] = tau_ms

    # adaptive avalanche bin from AIEI if no override present
    aiei_ms, _ = average_inter_event_interval(times_ms=times)
    result["aiei_ms"] = aiei_ms
    if dt_avalanche_override_ms is not None and dt_avalanche_override_ms > 0:
        dt_avalanche_ms = float(dt_avalanche_override_ms)
    else:
        dt_avalanche_ms = float(aiei_ms) if np.isfinite(aiei_ms) and aiei_ms > 0 else float(dt_mr_ms)
    result["dt_avalanche_ms"] = dt_avalanche_ms

    sizes, durations = avalanche_sizes_from_times(
        times_ms=times,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
        dt_ms=dt_avalanche_ms,
        min_size=min_avalanche_size,
    )
    result["n_sizes"] = float(sizes.size)
    result["n_durations"] = float(durations.size)

    tau_size = fit_powerlaw_exponent(sizes)
    tau_duration = fit_powerlaw_exponent(durations)
    result["tau_size"] = tau_size
    result["tau_duration"] = tau_duration
    gamma_fitted = fit_gamma_exponent(sizes, durations)
    result["gamma_fitted"] = gamma_fitted
    dcc, gamma_pred = compute_dcc(tau_size, tau_duration, gamma_fitted)
    result["gamma_pred"] = gamma_pred
    result["dcc"] = dcc
    return result


def save_table(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_npz(output_path: Path, payload: dict[str, Any]) -> None:
    np.savez_compressed(output_path, **payload)


def build_sigma_colormap(
    matrix: np.ndarray,
    center: float = 1.0,
    margin: float = 0.01,
    base_cmap_name: str = "viridis",
) -> tuple[LinearSegmentedColormap, tuple[float, float] | None]:
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return plt.get_cmap(base_cmap_name), None
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return plt.get_cmap(base_cmap_name), (vmin, vmax)

    base = plt.get_cmap(base_cmap_name)
    low = max(vmin, center - margin)
    high = min(vmax, center + margin)
    if low >= high:
        return base, (vmin, vmax)

    def norm(val: float) -> float:
        return float(np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0))

    positions: list[float] = []
    colors: list[str | tuple[float, float, float, float]] = []
    control_points = [
        (0, "#2027ef"),  # deep blue for strongly subcritical
        (norm(low), "#df3ec4"),  # teal near lower band edge
        (norm(center), "#ff0000"),  # vivid red at sigma≈1
        (norm(high), "#fffa6b"),  # pastel yellow exiting band
        (1.0, "#2aff4a"),  # warm orange for supercritical tail
    ]
    for pos, color in control_points:
        if positions and abs(pos - positions[-1]) < 1e-3:
            positions[-1] = pos
            colors[-1] = color
        else:
            positions.append(pos)
            colors.append(color)

    cmap = LinearSegmentedColormap.from_list("sigma_custom", list(zip(positions, colors)))
    return cmap, (vmin, vmax)


def plot_heatmap(
    matrix: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    title: str,
    out_path: Path,
    cmap: str | ListedColormap | LinearSegmentedColormap = "viridis",
    color_limits: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    extent = [alphas.min(), alphas.max(), betas.min(), betas.max()]
    vmin = vmax = None
    if color_limits is not None:
        vmin, vmax = color_limits
    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("alpha (E→E scaling)")
    ax.set_ylabel("beta (I→E scaling)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    sweep_dir = args.sweep_dir or find_latest_sweep_dir(args.results_root)
    summary_rows = load_sweep_summary(sweep_dir)
    cfg_snapshot = load_config_snapshot(sweep_dir) or {}

    dt_mr_ms = args.mr_dt_ms or float(
        cfg_snapshot.get("analysis", {}).get("mr_dt_ms", cfg_snapshot.get("analysis", {}).get("spike_bin_ms", 10.0))
    )
    dt_avalanche_override_ms = args.avalanche_dt_ms

    alphas, betas = build_grid_coords(summary_rows)
    alpha_index = {float(val): idx for idx, val in enumerate(alphas)}
    beta_index = {float(val): idx for idx, val in enumerate(betas)}

    sigma_grid = np.full((betas.size, alphas.size), np.nan)
    dcc_grid = np.full_like(sigma_grid, np.nan)
    tau_size_grid = np.full_like(sigma_grid, np.nan)
    tau_duration_grid = np.full_like(sigma_grid, np.nan)
    gamma_fit_grid = np.full_like(sigma_grid, np.nan)
    gamma_pred_grid = np.full_like(sigma_grid, np.nan)
    aiei_grid = np.full_like(sigma_grid, np.nan)
    dt_avalanche_grid = np.full_like(sigma_grid, np.nan)

    analysis_rows: list[dict[str, Any]] = []
    out_root = sweep_dir / "analysis"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing sweep in {sweep_dir}")
    if dt_avalanche_override_ms is not None:
        avalanche_dt_msg = f"{dt_avalanche_override_ms} ms (fixed)"
    else:
        avalanche_dt_msg = "AIEI-derived per run"
    print(f"MR dt = {dt_mr_ms} ms | Avalanche dt = {avalanche_dt_msg}")
    if not POWERLAW_AVAILABLE:
        print("Warning: powerlaw package not installed; DCC exponents will be NaN")

    for row in summary_rows:
        alpha = float(row["alpha"])
        beta = float(row["beta"])
        run_path = resolve_run_path(row, sweep_dir)
        metrics = analyze_run(
            run_path=run_path,
            t_start_ms=float(args.t_start_ms),
            t_stop_override=args.t_stop_ms,
            dt_mr_ms=dt_mr_ms,
            dt_avalanche_override_ms=dt_avalanche_override_ms,
            min_avalanche_size=int(args.min_avalanche_size),
            fit_lag_ms_min=args.mr_fit_start_ms,
            fit_lag_ms_max=args.mr_fit_stop_ms,
            fit_use_offset=bool(args.mr_fit_use_offset),
            min_fit_points=int(args.mr_min_fit_points),
        )

        analysis_row = {
            "alpha": alpha,
            "beta": beta,
            "sigma_mr": metrics["sigma_mr"],
            "tau_mr_ms": metrics["tau_mr_ms"],
            "tau_size": metrics["tau_size"],
            "tau_duration": metrics["tau_duration"],
            "gamma_fitted": metrics["gamma_fitted"],
            "gamma_pred": metrics["gamma_pred"],
            "dcc": metrics["dcc"],
            "aiei_ms": metrics["aiei_ms"],
            "dt_avalanche_ms": metrics["dt_avalanche_ms"],
            "n_counts": metrics["n_counts"],
            "n_sizes": metrics["n_sizes"],
            "n_durations": metrics["n_durations"],
            "run_name": row.get("run_name"),
        }
        analysis_rows.append(analysis_row)

        bi = beta_index[beta]
        ai = alpha_index[alpha]
        sigma_grid[bi, ai] = metrics["sigma_mr"]
        dcc_grid[bi, ai] = metrics["dcc"]
        tau_size_grid[bi, ai] = metrics["tau_size"]
        tau_duration_grid[bi, ai] = metrics["tau_duration"]
        gamma_fit_grid[bi, ai] = metrics["gamma_fitted"]
        gamma_pred_grid[bi, ai] = metrics["gamma_pred"]
        aiei_grid[bi, ai] = metrics["aiei_ms"]
        dt_avalanche_grid[bi, ai] = metrics["dt_avalanche_ms"]
        print(
            f"alpha={alpha:.2f}, beta={beta:.2f} -> sigma={metrics['sigma_mr']:.3f}, "
            f"tau_mr={metrics['tau_mr_ms']:.1f} ms, gamma_fit={metrics['gamma_fitted']:.3f}, "
            f"DCC={metrics['dcc']:.3f}"
        )

    save_table(analysis_rows, out_root / "metrics.csv")
    save_npz(
        out_root / "metrics.npz",
        {
            "alphas": alphas,
            "betas": betas,
            "sigma_mr": sigma_grid,
            "tau_size": tau_size_grid,
            "tau_duration": tau_duration_grid,
            "gamma_fitted": gamma_fit_grid,
            "gamma_pred": gamma_pred_grid,
            "dcc": dcc_grid,
            "aiei_ms": aiei_grid,
            "dt_avalanche_ms": dt_avalanche_grid,
            "dt_mr_ms": dt_mr_ms,
        },
    )

    if not args.disable_plots:
        sigma_cmap, sigma_limits = build_sigma_colormap(sigma_grid)
        plot_heatmap(
            sigma_grid,
            alphas,
            betas,
            "MR branching ratio",
            out_root / "heatmap_sigma.png",
            cmap=sigma_cmap,
            color_limits=sigma_limits,
        )
        plot_heatmap(dcc_grid, alphas, betas, "Distance to criticality (DCC)", out_root / "heatmap_dcc.png")

    print(f"Analysis artifacts saved to {out_root}")


if __name__ == "__main__":
    main()
