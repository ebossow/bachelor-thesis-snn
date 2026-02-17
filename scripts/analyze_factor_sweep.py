"""Aggregate MR estimator and avalanche-based DCC across a resting sweep."""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
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
from src.analysis.util import combine_spikes, load_run, load_stimulation_metadata


GRID_METRIC_KEYS = [
    "sigma_mr",
    "tau_mr_ms",
    "tau_size",
    "tau_duration",
    "gamma_fitted",
    "gamma_pred",
    "dcc",
    "aiei_ms",
    "dt_avalanche_ms",
    "n_counts",
    "n_sizes",
    "n_durations",
    # New powerlaw diagnostic fields:
    "pl_xmin_size",
    "pl_ntail_size",
    "pl_R_size_lognormal",
    "pl_p_size_lognormal",
    "pl_R_size_exponential",
    "pl_p_size_exponential",
    "pl_xmin_duration",
    "pl_ntail_duration",
    "pl_R_duration_lognormal",
    "pl_p_duration_lognormal",
    "pl_R_duration_exponential",
    "pl_p_duration_exponential",
]

BRANCHING_METRIC_KEYS = [
    "sigma_mr",
    "tau_mr_ms",
    "n_counts",
]

AVALANCHE_METRIC_KEYS = [
    "tau_size",
    "tau_duration",
    "gamma_fitted",
    "gamma_pred",
    "dcc",
    "aiei_ms",
    "dt_avalanche_ms",
    "n_sizes",
    "n_durations",
    "pl_xmin_size",
    "pl_ntail_size",
    "pl_R_size_lognormal",
    "pl_p_size_lognormal",
    "pl_R_size_exponential",
    "pl_p_size_exponential",
    "pl_xmin_duration",
    "pl_ntail_duration",
    "pl_R_duration_lognormal",
    "pl_p_duration_lognormal",
    "pl_R_duration_exponential",
    "pl_p_duration_exponential",
]

CRITICALITY_RESULTS_ROOT = Path("results/criticality_analysis")
TAU_MR_MIN_MS = 0.0
TAU_MR_MAX_MS = 500.0

MNIST_PATTERN_CURRENT_THRESHOLD_P_A = 1e-9


def build_criticality_output_root(source_name: str, suffix: str = "analysis") -> Path:
    return CRITICALITY_RESULTS_ROOT / f"{source_name}_{suffix}"


def sanitize_tau_mr_ms(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    if value < TAU_MR_MIN_MS or value > TAU_MR_MAX_MS:
        return float("nan")
    return float(value)


def slugify_label(label: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_-]+", "_", label.strip())
    slug = slug.strip("_")
    return slug or "population"


def base_population_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "Whole Population",
            "folder": "whole_population",
            "spike_keys": ("spikes_E", "spikes_IH", "spikes_IA"),
            "filter": None,
        },
        {
            "name": "Excitatory Only",
            "folder": "e_only",
            "spike_keys": ("spikes_E",),
            "filter": None,
        },
    ]


def build_population_specs(stim_metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    specs = base_population_specs()
    specs.extend(build_stimulus_population_specs(stim_metadata))
    return specs


def build_stimulus_population_specs(stim_metadata: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not stim_metadata:
        return []
    specs: list[dict[str, Any]] = []
    specs.extend(_build_dc_block_specs(stim_metadata))
    specs.extend(_build_mnist_pattern_specs(stim_metadata))
    return specs


def _build_dc_block_specs(stim_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    dc_meta = stim_metadata.get("dc")
    if not isinstance(dc_meta, dict):
        return []
    specs: list[dict[str, Any]] = []
    for block in dc_meta.get("blocks", []):
        if not isinstance(block, dict):
            continue
        if block.get("population") != "E":
            continue
        label = str(block.get("label") or f"dc_block_{block.get('pattern_index', 0)}")
        folder = slugify_label(f"stimulus_block_{label}")
        filter_info: dict[str, Any] = {
            "kind": "dc_block",
        }
        if block.get("label") is not None:
            filter_info["label"] = block["label"]
        if block.get("pattern_index") is not None:
            filter_info["pattern_index"] = block["pattern_index"]
        specs.append(
            {
                "name": f"Stimulus Block {label}",
                "folder": folder,
                "spike_keys": ("spikes_E",),
                "filter": filter_info,
            }
        )
    return specs


def _build_mnist_pattern_specs(stim_metadata: dict[str, Any]) -> list[dict[str, Any]]:
    mnist_meta = stim_metadata.get("mnist")
    if not isinstance(mnist_meta, dict):
        return []
    e_meta = mnist_meta.get("E") or {}
    pattern_currents = e_meta.get("pattern_currents_pA") or []
    if not pattern_currents:
        return []
    specs: list[dict[str, Any]] = []
    pattern_labels = mnist_meta.get("pattern_labels") or []
    for idx, currents in enumerate(pattern_currents):
        # Skip patterns without any activated neurons above threshold
        if not any(float(val) > MNIST_PATTERN_CURRENT_THRESHOLD_P_A for val in currents):
            continue
        label = pattern_labels[idx] if idx < len(pattern_labels) else idx
        folder = slugify_label(f"stimulus_pattern_{label}_{idx}")
        specs.append(
            {
                "name": f"Stimulus Pattern {label}",
                "folder": folder,
                "spike_keys": ("spikes_E",),
                "filter": {
                    "kind": "mnist_pattern",
                    "pattern_index": idx,
                    "label": label,
                    "current_threshold_pA": MNIST_PATTERN_CURRENT_THRESHOLD_P_A,
                },
            }
        )
    return specs


def resolve_allowed_senders(
    pop_spec: dict[str, Any],
    stim_metadata: dict[str, Any] | None,
    run_path: Path,
) -> list[int] | None:
    filter_info = pop_spec.get("filter")
    if not filter_info:
        return None
    if stim_metadata is None:
        raise RuntimeError(
            f"Population '{pop_spec['name']}' requires stimulation metadata, but {run_path} does not contain stimulation_metadata.json"
        )

    kind = filter_info.get("kind")
    if kind == "dc_block":
        dc_meta = stim_metadata.get("dc") or {}
        blocks = dc_meta.get("blocks", [])
        label = filter_info.get("label")
        pattern_idx = filter_info.get("pattern_index")
        for block in blocks:
            if block.get("population") != "E":
                continue
            if label is not None and block.get("label") != label:
                continue
            if label is None and pattern_idx is not None and block.get("pattern_index") != pattern_idx:
                continue
            neuron_ids = block.get("neuron_ids") or []
            return [int(nid) for nid in neuron_ids]
        print(
            f"Warning: could not find DC block '{label or pattern_idx}' in stimulation metadata for {run_path}"
        )
        return []

    if kind == "mnist_pattern":
        mnist_meta = stim_metadata.get("mnist") or {}
        e_meta = mnist_meta.get("E") or {}
        neuron_ids = e_meta.get("neuron_ids") or []
        pattern_idx = int(filter_info.get("pattern_index", -1))
        pattern_currents = e_meta.get("pattern_currents_pA") or []
        if pattern_idx < 0 or pattern_idx >= len(pattern_currents):
            print(
                f"Warning: MNIST pattern index {pattern_idx} missing in stimulation metadata for {run_path}"
            )
            return []
        threshold = float(filter_info.get("current_threshold_pA", 0.0))
        currents = pattern_currents[pattern_idx]
        allowed = [
            int(nid)
            for nid, curr in zip(neuron_ids, currents)
            if float(curr) > threshold
        ]
        if not allowed:
            label = filter_info.get("label", pattern_idx)
            print(
                f"Warning: MNIST pattern '{label}' has no neurons above threshold in {run_path}"
            )
        return allowed

    if kind == "explicit_ids":
        neuron_ids = filter_info.get("neuron_ids") or []
        return [int(nid) for nid in neuron_ids]

    raise ValueError(f"Unknown population filter kind: {kind}")

def determine_analysis_windows(cfg_snapshot: dict[str, Any], args: argparse.Namespace) -> list[dict[str, float | str]]:
    exp_cfg = cfg_snapshot.get("experiment", {})
    if "simtime_ms" not in exp_cfg and args.t_stop_ms is None:
        raise ValueError("experiment.simtime_ms missing; provide --t-stop-ms")
    simtime_ms = float(exp_cfg.get("simtime_ms", args.t_stop_ms))

    global_start = float(args.t_start_ms)
    global_stop = float(args.t_stop_ms) if args.t_stop_ms is not None else simtime_ms
    if global_stop <= global_start:
        raise ValueError("Analysis window has non-positive duration; adjust --t-start-ms/--t-stop-ms")

    if not args.dual_phase:
        return [
            {
                "name": "overall",
                "start_ms": global_start,
                "stop_ms": global_stop,
            }
        ]

    pattern_cfg = cfg_snapshot.get("stimulation", {}).get("pattern", {})
    stim_on = args.stim_t_on_ms if args.stim_t_on_ms is not None else pattern_cfg.get("t_on_ms")
    stim_off = args.stim_t_off_ms if args.stim_t_off_ms is not None else pattern_cfg.get("t_off_ms")
    if stim_on is None or stim_off is None:
        raise ValueError(
            "--dual-phase requires stimulation.pattern.t_on_ms/t_off_ms or explicit overrides"
        )

    stim_on = float(stim_on)
    stim_off = float(stim_off)

    pre_start = global_start
    pre_stop = min(stim_on, global_stop)
    post_start = max(stim_off, global_start)
    post_stop = global_stop

    if pre_stop <= pre_start:
        raise ValueError(
            "Pre-learning window empty (check stimulation onset vs analysis start/stop)"
        )
    if post_stop <= post_start:
        raise ValueError(
            "Post-learning window empty (check stimulation offset vs analysis stop)"
        )

    return [
        {
            "name": args.phase_pre_name,
            "start_ms": pre_start,
            "stop_ms": pre_stop,
        },
        {
            "name": args.phase_post_name,
            "start_ms": post_start,
            "stop_ms": post_stop,
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze resting sweep results")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        help="Path to results/resting_with_factors/run_* directory",
    )
    parser.add_argument(
        "--multi-run-dir",
        type=Path,
        help="Path to results/resting_with_factors/multiple_run_* directory",
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
        default=4,
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
        "--pl-min-samples",
        type=int,
        default=200,
        help="Minimum number of avalanches required to attempt a power-law fit",
    )
    parser.add_argument(
        "--pl-min-tail-samples",
        type=int,
        default=50,
        help="Minimum number of samples in the fitted power-law tail (>= xmin)",
    )
    parser.add_argument(
        "--pl-require-not-worse",
        action="store_true",
        help="If set, mark power-law exponents as NaN when power law is significantly worse than lognormal or exponential (based on likelihood-ratio test)",
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
    parser.add_argument(
        "--dual-phase",
        action="store_true",
        help=(
            "Compute metrics separately for resting (pre-stimulation) and post-learning "
            "windows using stimulation.pattern t_on/t_off"
        ),
    )
    parser.add_argument(
        "--stim-t-on-ms",
        type=float,
        help="Override stimulation.pattern.t_on_ms when defining dual-phase windows",
    )
    parser.add_argument(
        "--stim-t-off-ms",
        type=float,
        help="Override stimulation.pattern.t_off_ms when defining dual-phase windows",
    )
    parser.add_argument(
        "--phase-pre-name",
        type=str,
        default="pre_learning",
        help="Name for the pre-learning window when --dual-phase is enabled",
    )
    parser.add_argument(
        "--phase-post-name",
        type=str,
        default="post_learning",
        help="Name for the post-learning window when --dual-phase is enabled",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Number of worker processes used to analyze individual runs; default = CPU count"
        ),
    )
    parser.add_argument(
        "--compute-branching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable MR branching-ratio analysis (use --no-compute-branching to disable)",
    )
    parser.add_argument(
        "--compute-avalanche",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable avalanche/power-law/DCC analysis (use --no-compute-avalanche to disable)",
    )
    parser.add_argument(
        "--log-timing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print per-stage timing summary (useful for profiling large sweeps)",
    )
    parser.add_argument(
        "--individual-run-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In multi-run mode, save full artifacts (CSV/plots) for each individual run; "
            "default is NPZ-only per run while summary keeps CSV/plots"
        ),
    )
    parser.add_argument(
        "--summary-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In multi-run mode, skip per-run recomputation and rebuild only the aggregated "
            "summary from existing per-run NPZ artifacts"
        ),
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
    args.pl_min_samples = max(10, int(args.pl_min_samples))
    args.pl_min_tail_samples = max(10, int(args.pl_min_tail_samples))
    if args.max_workers is not None and args.max_workers < 1:
        parser.error("--max-workers must be >= 1")
    if not args.compute_branching and not args.compute_avalanche:
        parser.error("At least one analysis path must be enabled: --compute-branching and/or --compute-avalanche")
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



def fit_powerlaw_diagnostics(values: np.ndarray, discrete: bool = True) -> dict[str, float]:
    """Fit a power-law and return exponent plus basic diagnostics.

    Returns NaNs when powerlaw is unavailable or the input is too small.

    Diagnostics:
      - xmin: lower cutoff selected by the fitter
      - ntail: number of samples >= xmin
      - R/p: likelihood-ratio comparison vs lognormal and exponential
    """
    out = {
        "alpha": float("nan"),
        "xmin": float("nan"),
        "ntail": float("nan"),
        "R_lognormal": float("nan"),
        "p_lognormal": float("nan"),
        "R_exponential": float("nan"),
        "p_exponential": float("nan"),
    }
    if not POWERLAW_AVAILABLE:
        return out

    vals = np.asarray(values, float)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > 0]
    if vals.size < 5:
        return out

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = powerlaw.Fit(vals, verbose=False, discrete=discrete)  # type: ignore[arg-type]

    try:
        out["alpha"] = float(fit.power_law.alpha)
    except Exception:
        out["alpha"] = float("nan")

    try:
        out["xmin"] = float(fit.xmin)
        out["ntail"] = float(np.count_nonzero(vals >= out["xmin"]))
    except Exception:
        out["xmin"] = float("nan")
        out["ntail"] = float("nan")

    # Likelihood-ratio tests (R>0 => power law preferred; p is significance)
    try:
        R, p = fit.distribution_compare("power_law", "lognormal", normalized_ratio=True)
        out["R_lognormal"] = float(R)
        out["p_lognormal"] = float(p)
    except Exception:
        pass

    try:
        R, p = fit.distribution_compare("power_law", "exponential", normalized_ratio=True)
        out["R_exponential"] = float(R)
        out["p_exponential"] = float(p)
    except Exception:
        pass

    return out


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
    if abs(tau_size - 1.0) < 1e-9:
        return float("nan")
    return float((tau_duration - 1.0) / (tau_size - 1.0))


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
    population_spec: dict[str, Any],
    t_start_ms: float,
    t_stop_override: float | None,
    dt_mr_ms: float,
    dt_avalanche_override_ms: float | None,
    min_avalanche_size: int,
    fit_lag_ms_min: float | None,
    fit_lag_ms_max: float | None,
    fit_use_offset: bool,
    min_fit_points: int,
    pl_min_samples: int,
    pl_min_tail_samples: int,
    pl_require_not_worse: bool,
    compute_branching: bool,
    compute_avalanche: bool,
) -> dict[str, float]:
    cfg, data, _, _, stim_metadata = load_run(run_path)
    simtime_ms = float(cfg["experiment"]["simtime_ms"])
    t_stop_ms = float(t_stop_override) if t_stop_override is not None else simtime_ms
    spike_keys = population_spec["spike_keys"]
    allowed_senders = resolve_allowed_senders(population_spec, stim_metadata, run_path)
    spikes = combine_spikes(data, spike_keys, allowed_senders=allowed_senders)
    times = np.asarray(spikes["times"], float)
    mask = (times >= t_start_ms) & (times < t_stop_ms)
    times = times[mask]
    return compute_metrics_from_times(
        times=times,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
        dt_mr_ms=dt_mr_ms,
        dt_avalanche_override_ms=dt_avalanche_override_ms,
        min_avalanche_size=min_avalanche_size,
        fit_lag_ms_min=fit_lag_ms_min,
        fit_lag_ms_max=fit_lag_ms_max,
        fit_use_offset=fit_use_offset,
        min_fit_points=min_fit_points,
        pl_min_samples=pl_min_samples,
        pl_min_tail_samples=pl_min_tail_samples,
        pl_require_not_worse=pl_require_not_worse,
        compute_branching=compute_branching,
        compute_avalanche=compute_avalanche,
    )


def compute_metrics_from_times(
    times: np.ndarray,
    t_start_ms: float,
    t_stop_ms: float,
    dt_mr_ms: float,
    dt_avalanche_override_ms: float | None,
    min_avalanche_size: int,
    fit_lag_ms_min: float | None,
    fit_lag_ms_max: float | None,
    fit_use_offset: bool,
    min_fit_points: int,
    pl_min_samples: int,
    pl_min_tail_samples: int,
    pl_require_not_worse: bool,
    compute_branching: bool,
    compute_avalanche: bool,
) -> dict[str, float]:
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
        "pl_xmin_size": float("nan"),
        "pl_ntail_size": float("nan"),
        "pl_R_size_lognormal": float("nan"),
        "pl_p_size_lognormal": float("nan"),
        "pl_R_size_exponential": float("nan"),
        "pl_p_size_exponential": float("nan"),
        "pl_xmin_duration": float("nan"),
        "pl_ntail_duration": float("nan"),
        "pl_R_duration_lognormal": float("nan"),
        "pl_p_duration_lognormal": float("nan"),
        "pl_R_duration_exponential": float("nan"),
        "pl_p_duration_exponential": float("nan"),
    }
    if times.size == 0:
        return result

    if compute_branching:
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
            result["tau_mr_ms"] = sanitize_tau_mr_ms(tau_ms)

    sizes = np.asarray([], dtype=float)
    durations = np.asarray([], dtype=float)
    if compute_avalanche:
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

    # Power-law fits are only meaningful with enough events and enough tail samples.
    tau_size = float("nan")
    tau_duration = float("nan")

    if compute_avalanche and sizes.size >= pl_min_samples:
        diag_s = fit_powerlaw_diagnostics(sizes)
        result["pl_xmin_size"] = diag_s["xmin"]
        result["pl_ntail_size"] = diag_s["ntail"]
        result["pl_R_size_lognormal"] = diag_s["R_lognormal"]
        result["pl_p_size_lognormal"] = diag_s["p_lognormal"]
        result["pl_R_size_exponential"] = diag_s["R_exponential"]
        result["pl_p_size_exponential"] = diag_s["p_exponential"]

        # accept exponent only if tail is large enough
        if np.isfinite(diag_s["alpha"]) and np.isfinite(diag_s["ntail"]) and diag_s["ntail"] >= pl_min_tail_samples:
            tau_size = float(diag_s["alpha"])

            # optionally reject when power law is significantly worse than alternatives
            if pl_require_not_worse:
                # If lognormal is preferred (R<0) with significance, reject.
                if np.isfinite(diag_s["R_lognormal"]) and np.isfinite(diag_s["p_lognormal"]) and diag_s["R_lognormal"] < 0 and diag_s["p_lognormal"] < 0.1:
                    tau_size = float("nan")
                # If exponential is preferred (R<0) with significance, reject.
                if np.isfinite(diag_s["R_exponential"]) and np.isfinite(diag_s["p_exponential"]) and diag_s["R_exponential"] < 0 and diag_s["p_exponential"] < 0.1:
                    tau_size = float("nan")

    if compute_avalanche and durations.size >= pl_min_samples:
        diag_t = fit_powerlaw_diagnostics(durations)
        result["pl_xmin_duration"] = diag_t["xmin"]
        result["pl_ntail_duration"] = diag_t["ntail"]
        result["pl_R_duration_lognormal"] = diag_t["R_lognormal"]
        result["pl_p_duration_lognormal"] = diag_t["p_lognormal"]
        result["pl_R_duration_exponential"] = diag_t["R_exponential"]
        result["pl_p_duration_exponential"] = diag_t["p_exponential"]

        if np.isfinite(diag_t["alpha"]) and np.isfinite(diag_t["ntail"]) and diag_t["ntail"] >= pl_min_tail_samples:
            tau_duration = float(diag_t["alpha"])

            if pl_require_not_worse:
                if np.isfinite(diag_t["R_lognormal"]) and np.isfinite(diag_t["p_lognormal"]) and diag_t["R_lognormal"] < 0 and diag_t["p_lognormal"] < 0.1:
                    tau_duration = float("nan")
                if np.isfinite(diag_t["R_exponential"]) and np.isfinite(diag_t["p_exponential"]) and diag_t["R_exponential"] < 0 and diag_t["p_exponential"] < 0.1:
                    tau_duration = float("nan")

    result["tau_size"] = tau_size
    result["tau_duration"] = tau_duration
    if compute_avalanche:
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


def write_mr_dt_marker(out_dir: Path, dt_mr_ms: float) -> None:
    marker_path = out_dir / "mr_dt_ms.txt"
    marker_path.write_text(f"{dt_mr_ms:.6f}\n")


def build_sigma_colormap(
    matrix: np.ndarray,
    center: float = 1.0,
    margin: float = 0.01,
    precritical_margin: float = 0.05,
    nearcritical_margin: float = 0.10,
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
    precritical_low = max(vmin, center - precritical_margin)
    nearcritical_low = max(vmin, center - nearcritical_margin)
    if low >= high:
        return base, (vmin, vmax)

    def norm(val: float) -> float:
        return float(np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0))

    positions: list[float] = []
    colors: list[str | tuple[float, float, float, float]] = []
    control_points = [
        (0.0, "#2027ef"),  # deep blue for strongly subcritical
        (norm(nearcritical_low), "#3f63ff"),  # start of near-critical ramp around sigma≈0.9
        (norm(center - 0.08), "#556dff"),  # sigma≈0.92
        (norm(center - 0.06), "#6b61ff"),  # sigma≈0.94
        (norm(precritical_low), "#8f49ff"),  # purple around sigma≈0.95
        (norm(center - 0.04), "#ad3ef2"),  # sigma≈0.96
        (norm(center - 0.03), "#bf39e5"),  # sigma≈0.97
        (norm(center - 0.02), "#cf35d8"),  # sigma≈0.98
        (norm(center - 0.015), "#d83ccf"),  # sigma≈0.985
        (norm(low), "#df3ec4"),  # magenta just below critical band
        (norm(center), "#ff0000"),  # vivid red at sigma≈1
        (norm(high), "#fffa6b"),  # yellow just above critical band
        (1.0, "#2aff4a"),  # green for supercritical tail
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
    zoom_color_limits: tuple[float, float] | None = None,
    main_cbar_label: str | None = None,
    main_cbar_tick_format: str | None = None,
) -> None:
    has_zoom_bar = zoom_color_limits is not None
    fig_size = (7.8, 5) if has_zoom_bar else (6, 5)
    fig, ax = plt.subplots(figsize=fig_size)
    if has_zoom_bar:
        fig.subplots_adjust(left=0.11, right=0.74, bottom=0.11, top=0.90)
    extent = [alphas.min(), alphas.max(), betas.min(), betas.max()]
    vmin = vmax = None
    if color_limits is not None:
        vmin, vmax = color_limits
    im = ax.imshow(
        matrix,
        origin="lower",
        aspect="equal",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("alpha (E→E scaling)")
    ax.set_ylabel("beta (I→E scaling)")
    ax.set_title(title)
    if has_zoom_bar:
        fig.canvas.draw()
        ax_pos = ax.get_position()
        cbar_width = 0.034
        cbar_gap = 0.03
        main_x = ax_pos.x1 + 0.1
        zoom_x = main_x + cbar_width + cbar_gap
        main_cax = fig.add_axes([main_x, ax_pos.y0, cbar_width, ax_pos.height])
        main_cbar = fig.colorbar(im, cax=main_cax)
    else:
        main_cbar = fig.colorbar(im, ax=ax)
    if main_cbar_label is not None:
        main_cbar.set_label(main_cbar_label)
    elif has_zoom_bar:
        main_cbar.set_label("Full range")
    if main_cbar_tick_format is not None:
        main_cbar.formatter = ticker.FormatStrFormatter(main_cbar_tick_format)
        main_cbar.update_ticks()
    if has_zoom_bar:
        main_cbar.ax.yaxis.set_ticks_position("left")
        main_cbar.ax.yaxis.set_label_position("left")

    if zoom_color_limits is not None:
        zmin, zmax = zoom_color_limits
        if np.isfinite(zmin) and np.isfinite(zmax) and zmax > zmin:
            full_vmin = float(im.norm.vmin) if im.norm.vmin is not None else float(np.nanmin(matrix))
            full_vmax = float(im.norm.vmax) if im.norm.vmax is not None else float(np.nanmax(matrix))
            if np.isfinite(full_vmin) and np.isfinite(full_vmax) and full_vmax > full_vmin:
                frac_min = float(np.clip((zmin - full_vmin) / (full_vmax - full_vmin), 0.0, 1.0))
                frac_max = float(np.clip((zmax - full_vmin) / (full_vmax - full_vmin), 0.0, 1.0))
            else:
                frac_min, frac_max = 0.0, 1.0
            if frac_max <= frac_min:
                frac_min, frac_max = 0.0, 1.0

            base_cmap = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
            zoom_colors = base_cmap(np.linspace(frac_min, frac_max, 256))
            zoom_cmap = ListedColormap(zoom_colors)

            zoom_ax = fig.add_axes([zoom_x, ax_pos.y0, cbar_width, ax_pos.height])
            zoom_sm = plt.cm.ScalarMappable(norm=Normalize(vmin=zmin, vmax=zmax), cmap=zoom_cmap)
            zoom_sm.set_array([])
            zoom_cbar = fig.colorbar(zoom_sm, cax=zoom_ax)
            zoom_cbar.set_label(f"Zoom [{zmin:.2f}, {zmax:.2f}]")
            zoom_cbar.ax.yaxis.set_ticks_position("right")
            zoom_cbar.ax.yaxis.set_label_position("right")

    if not has_zoom_bar:
        fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def build_average_rows(
    alphas: np.ndarray,
    betas: np.ndarray,
    grids: dict[str, np.ndarray],
    run_label: str,
    window_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bi, beta in enumerate(betas):
        for ai, alpha in enumerate(alphas):
            row = {
                "alpha": float(alpha),
                "beta": float(beta),
                "run_name": run_label,
                "window": window_name,
            }
            for key in GRID_METRIC_KEYS:
                row[key] = float(grids[key][bi, ai])
            rows.append(row)
    return rows


def select_metric_rows(rows: list[dict[str, Any]], metric_keys: list[str]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    base_keys = ("alpha", "beta", "run_name", "window")
    for row in rows:
        out_row: dict[str, Any] = {}
        for key in base_keys:
            out_row[key] = row.get(key)
        for key in metric_keys:
            out_row[key] = row.get(key, float("nan"))
        selected.append(out_row)
    return selected


def write_analysis_outputs(
    window_result: dict[str, Any],
    alphas: np.ndarray,
    betas: np.ndarray,
    dt_mr_ms: float,
    analysis_root: Path,
    relative_out_dir: Path,
    disable_plots: bool,
    save_csv: bool,
    compute_branching: bool,
    compute_avalanche: bool,
) -> None:
    def build_payload(metric_keys: list[str]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "alphas": alphas,
            "betas": betas,
            "dt_mr_ms": dt_mr_ms,
            "window_name": window_result["name"],
            "window_start_ms": window_result["start_ms"],
            "window_stop_ms": window_result["stop_ms"],
        }
        for key in metric_keys:
            payload[key] = window_result["grids"][key]
        return payload

    if compute_branching:
        branching_out = analysis_root / "branching" / relative_out_dir
        branching_out.mkdir(parents=True, exist_ok=True)
        write_mr_dt_marker(branching_out, dt_mr_ms)
        branching_rows = select_metric_rows(window_result["analysis_rows"], BRANCHING_METRIC_KEYS)
        if save_csv:
            save_table(branching_rows, branching_out / "metrics.csv")
        save_npz(branching_out / "metrics.npz", build_payload(BRANCHING_METRIC_KEYS))

        if not disable_plots:
            sigma_grid = window_result["grids"]["sigma_mr"]
            sigma_cmap, sigma_limits = build_sigma_colormap(sigma_grid)
            plot_heatmap(
                sigma_grid,
                alphas,
                betas,
                f"MR branching ratio ({window_result['name']})",
                branching_out / "heatmap_sigma.png",
                cmap=sigma_cmap,
                color_limits=sigma_limits,
                zoom_color_limits=(0.85, 1.0),
            )
            tau_grid = window_result["grids"]["tau_mr_ms"]
            plot_heatmap(
                tau_grid,
                alphas,
                betas,
                f"MR autocorrelation time tau_mr (ms) [{window_result['name']}]",
                branching_out / "heatmap_tau_mr_ms.png",
                main_cbar_label="tau_mr (ms)",
                main_cbar_tick_format="%.2f",
            )

    if compute_avalanche:
        avalanche_out = analysis_root / "avalanche" / relative_out_dir
        avalanche_out.mkdir(parents=True, exist_ok=True)
        write_mr_dt_marker(avalanche_out, dt_mr_ms)
        avalanche_rows = select_metric_rows(window_result["analysis_rows"], AVALANCHE_METRIC_KEYS)
        if save_csv:
            save_table(avalanche_rows, avalanche_out / "metrics.csv")
        save_npz(avalanche_out / "metrics.npz", build_payload(AVALANCHE_METRIC_KEYS))

        if not disable_plots:
            dcc_grid = window_result["grids"]["dcc"]
            plot_heatmap(
                dcc_grid,
                alphas,
                betas,
                f"Distance to criticality (DCC) [{window_result['name']}]",
                avalanche_out / "heatmap_dcc.png",
            )


def analyze_window(
    summary_rows: list[dict[str, Any]],
    sweep_dir: Path,
    alpha_index: dict[float, int],
    beta_index: dict[float, int],
    alphas: np.ndarray,
    betas: np.ndarray,
    window: dict[str, float | str],
    args: argparse.Namespace,
    dt_mr_ms: float,
    dt_avalanche_override_ms: float | None,
    population_spec: dict[str, Any],
    worker_count: int,
) -> dict[str, Any]:
    window_name = str(window["name"])
    start_ms = float(window["start_ms"])
    stop_ms = float(window["stop_ms"])

    grids: dict[str, np.ndarray] = {
        key: np.full((betas.size, alphas.size), np.nan)
        for key in GRID_METRIC_KEYS
    }
    num_rows = len(summary_rows)
    worker_count = max(1, min(worker_count, num_rows)) if num_rows else 1
    analysis_rows_data: list[dict[str, Any] | None] = [None] * num_rows

    population_label = population_spec["name"]
    print(
        f"Window '{window_name}' [{population_label}]: analyzing spikes between {start_ms} ms and {stop_ms} ms"
    )
    if num_rows == 0:
        return {
            "name": window_name,
            "start_ms": start_ms,
            "stop_ms": stop_ms,
            "grids": grids,
            "analysis_rows": [],
        }

    min_avalanche_size = int(args.min_avalanche_size)
    fit_lag_ms_min = args.mr_fit_start_ms
    fit_lag_ms_max = args.mr_fit_stop_ms
    fit_use_offset = bool(args.mr_fit_use_offset)
    min_fit_points = int(args.mr_min_fit_points)
    pl_min_samples = int(args.pl_min_samples)
    pl_min_tail_samples = int(args.pl_min_tail_samples)
    pl_require_not_worse = bool(args.pl_require_not_worse)
    compute_branching = bool(args.compute_branching)
    compute_avalanche = bool(args.compute_avalanche)

    def record_result(
        row_index: int,
        alpha: float,
        beta: float,
        run_name: str | None,
        metrics: dict[str, float],
    ) -> None:
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
            "pl_xmin_size": metrics["pl_xmin_size"],
            "pl_ntail_size": metrics["pl_ntail_size"],
            "pl_R_size_lognormal": metrics["pl_R_size_lognormal"],
            "pl_p_size_lognormal": metrics["pl_p_size_lognormal"],
            "pl_R_size_exponential": metrics["pl_R_size_exponential"],
            "pl_p_size_exponential": metrics["pl_p_size_exponential"],
            "pl_xmin_duration": metrics["pl_xmin_duration"],
            "pl_ntail_duration": metrics["pl_ntail_duration"],
            "pl_R_duration_lognormal": metrics["pl_R_duration_lognormal"],
            "pl_p_duration_lognormal": metrics["pl_p_duration_lognormal"],
            "pl_R_duration_exponential": metrics["pl_R_duration_exponential"],
            "pl_p_duration_exponential": metrics["pl_p_duration_exponential"],
            "run_name": run_name,
            "window": window_name,
        }
        analysis_rows_data[row_index] = analysis_row

        bi = beta_index[beta]
        ai = alpha_index[alpha]
        for key in GRID_METRIC_KEYS:
            grids[key][bi, ai] = metrics[key]
        print(
            f"[{window_name} | {population_label}] alpha={alpha:.2f}, beta={beta:.2f} -> "
            f"sigma={metrics['sigma_mr']:.3f}, tau_mr={metrics['tau_mr_ms']:.1f} ms, "
            f"gamma_fit={metrics['gamma_fitted']:.3f}, DCC={metrics['dcc']:.3f}"
        )

    if worker_count == 1:
        for row_index, row in enumerate(summary_rows):
            row_idx, alpha, beta, run_name, metrics = _compute_row_metrics(
                row_index,
                row,
                sweep_dir,
                population_spec,
                start_ms,
                stop_ms,
                dt_mr_ms,
                dt_avalanche_override_ms,
                min_avalanche_size,
                fit_lag_ms_min,
                fit_lag_ms_max,
                fit_use_offset,
                min_fit_points,
                pl_min_samples,
                pl_min_tail_samples,
                pl_require_not_worse,
                compute_branching,
                compute_avalanche,
            )
            record_result(row_idx, alpha, beta, run_name, metrics)
    else:
        future_to_row: dict[Any, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for row_index, row in enumerate(summary_rows):
                future = executor.submit(
                    _compute_row_metrics,
                    row_index,
                    row,
                    sweep_dir,
                    population_spec,
                    start_ms,
                    stop_ms,
                    dt_mr_ms,
                    dt_avalanche_override_ms,
                    min_avalanche_size,
                    fit_lag_ms_min,
                    fit_lag_ms_max,
                    fit_use_offset,
                    min_fit_points,
                    pl_min_samples,
                    pl_min_tail_samples,
                    pl_require_not_worse,
                    compute_branching,
                    compute_avalanche,
                )
                future_to_row[future] = row

            for future in as_completed(future_to_row):
                try:
                    row_idx, alpha, beta, run_name, metrics = future.result()
                except Exception as exc:  # pragma: no cover - bubble up with context
                    row = future_to_row[future]
                    raise RuntimeError(
                        f"Failed to analyze alpha={row.get('alpha')} beta={row.get('beta')} "
                        f"(run={row.get('run_name')})"
                    ) from exc
                record_result(row_idx, alpha, beta, run_name, metrics)

    analysis_rows = [row for row in analysis_rows_data if row is not None]
    return {
        "name": window_name,
        "start_ms": start_ms,
        "stop_ms": stop_ms,
        "grids": grids,
        "analysis_rows": analysis_rows,
    }


def analyze_sweep_dir(
    sweep_dir: Path,
    args: argparse.Namespace,
    output_root_override: Path | None = None,
    save_csv: bool = True,
    disable_plots_override: bool | None = None,
) -> dict[str, Any]:
    analysis_t0 = time.perf_counter()
    summary_rows = load_sweep_summary(sweep_dir)
    cfg_snapshot = load_config_snapshot(sweep_dir) or {}

    dt_mr_ms = args.mr_dt_ms or float(
        cfg_snapshot.get("analysis", {}).get(
            "mr_dt_ms", cfg_snapshot.get("analysis", {}).get("spike_bin_ms", 10.0)
        )
    )
    dt_avalanche_override_ms = args.avalanche_dt_ms
    windows = determine_analysis_windows(cfg_snapshot, args)

    alphas, betas = build_grid_coords(summary_rows)
    alpha_index = {float(val): idx for idx, val in enumerate(alphas)}
    beta_index = {float(val): idx for idx, val in enumerate(betas)}

    analysis_root = output_root_override or build_criticality_output_root(sweep_dir.name, "analysis")
    analysis_root.mkdir(parents=True, exist_ok=True)

    sample_metadata = None
    for row in summary_rows:
        run_path = resolve_run_path(row, sweep_dir)
        sample_metadata = load_stimulation_metadata(run_path)
        if sample_metadata:
            break

    population_specs = build_population_specs(sample_metadata)

    preferred_workers = args.max_workers if args.max_workers is not None else (os.cpu_count() or 1)
    worker_count = max(1, preferred_workers)
    if summary_rows:
        worker_count = min(worker_count, len(summary_rows))
    if worker_count > 1:
        print(f"Run-level analysis will use {worker_count} worker processes")
    else:
        print("Run-level analysis will execute sequentially (worker count = 1)")

    print(f"Analyzing sweep in {sweep_dir}")
    print(f"Saving analysis to {analysis_root}")
    if dt_avalanche_override_ms is not None:
        avalanche_dt_msg = f"{dt_avalanche_override_ms} ms (fixed)"
    else:
        avalanche_dt_msg = "AIEI-derived per run"
    print(f"MR dt = {dt_mr_ms} ms | Avalanche dt = {avalanche_dt_msg}")
    if not args.compute_branching:
        print("Branching analysis disabled (--no-compute-branching)")
    if not args.compute_avalanche:
        print("Avalanche/DCC analysis disabled (--no-compute-avalanche)")
    if args.compute_avalanche and not POWERLAW_AVAILABLE:
        print("Warning: powerlaw package not installed; DCC exponents will be NaN")

    population_results: dict[str, dict[str, Any]] = {}
    population_order: list[str] = []
    window_order = [str(w["name"]) for w in windows]

    for pop_spec in population_specs:
        pop_folder = pop_spec["folder"]
        population_order.append(pop_folder)
        window_results: dict[str, dict[str, Any]] = {}
        for window in windows:
            window_name = str(window["name"])
            window_results[window_name] = {
                "name": window_name,
                "start_ms": float(window["start_ms"]),
                "stop_ms": float(window["stop_ms"]),
                "grids": {
                    key: np.full((betas.size, alphas.size), np.nan)
                    for key in GRID_METRIC_KEYS
                },
                "analysis_rows_data": [None] * len(summary_rows),
            }
        population_results[pop_folder] = {
            "name": pop_spec["name"],
            "windows": window_results,
        }

    if not summary_rows:
        for pop_folder in population_order:
            for window_name in window_order:
                window_result = population_results[pop_folder]["windows"][window_name]
                window_result["analysis_rows"] = []
                del window_result["analysis_rows_data"]
        if args.log_timing:
            print("Timing summary: empty sweep (no rows)")
        return {
            "sweep_dir": sweep_dir,
            "alphas": alphas,
            "betas": betas,
            "dt_mr_ms": dt_mr_ms,
            "population_results": population_results,
            "population_order": population_order,
            "window_order": window_order,
        }

    min_avalanche_size = int(args.min_avalanche_size)
    fit_lag_ms_min = args.mr_fit_start_ms
    fit_lag_ms_max = args.mr_fit_stop_ms
    fit_use_offset = bool(args.mr_fit_use_offset)
    min_fit_points = int(args.mr_min_fit_points)
    pl_min_samples = int(args.pl_min_samples)
    pl_min_tail_samples = int(args.pl_min_tail_samples)
    pl_require_not_worse = bool(args.pl_require_not_worse)
    compute_branching = bool(args.compute_branching)
    compute_avalanche = bool(args.compute_avalanche)
    log_timing = bool(args.log_timing)
    disable_plots_effective = bool(args.disable_plots) if disable_plots_override is None else bool(disable_plots_override)

    timing_totals = {
        "load_run_s": 0.0,
        "build_population_spikes_s": 0.0,
        "compute_windows_s": 0.0,
        "total_row_s": 0.0,
    }
    max_row_total_s = 0.0

    def record_row_result(
        row_idx: int,
        alpha: float,
        beta: float,
        run_name: str | None,
        per_population: dict[str, dict[str, dict[str, float]]],
        timing: dict[str, float],
    ) -> None:
        bi = beta_index[beta]
        ai = alpha_index[alpha]
        nonlocal max_row_total_s
        for key in timing_totals:
            timing_totals[key] += float(timing.get(key, 0.0))
        max_row_total_s = max(max_row_total_s, float(timing.get("total_row_s", 0.0)))
        for pop_folder in population_order:
            pop_windows = population_results[pop_folder]["windows"]
            row_windows = per_population[pop_folder]
            for window_name in window_order:
                metrics = row_windows[window_name]
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
                    "pl_xmin_size": metrics["pl_xmin_size"],
                    "pl_ntail_size": metrics["pl_ntail_size"],
                    "pl_R_size_lognormal": metrics["pl_R_size_lognormal"],
                    "pl_p_size_lognormal": metrics["pl_p_size_lognormal"],
                    "pl_R_size_exponential": metrics["pl_R_size_exponential"],
                    "pl_p_size_exponential": metrics["pl_p_size_exponential"],
                    "pl_xmin_duration": metrics["pl_xmin_duration"],
                    "pl_ntail_duration": metrics["pl_ntail_duration"],
                    "pl_R_duration_lognormal": metrics["pl_R_duration_lognormal"],
                    "pl_p_duration_lognormal": metrics["pl_p_duration_lognormal"],
                    "pl_R_duration_exponential": metrics["pl_R_duration_exponential"],
                    "pl_p_duration_exponential": metrics["pl_p_duration_exponential"],
                    "run_name": run_name,
                    "window": window_name,
                }
                pop_windows[window_name]["analysis_rows_data"][row_idx] = analysis_row
                for key in GRID_METRIC_KEYS:
                    pop_windows[window_name]["grids"][key][bi, ai] = metrics[key]

    if worker_count == 1:
        for row_index, row in enumerate(summary_rows):
            row_idx, alpha, beta, run_name, per_population, timing = _compute_row_all_metrics(
                row_index,
                row,
                sweep_dir,
                population_specs,
                windows,
                dt_mr_ms,
                dt_avalanche_override_ms,
                min_avalanche_size,
                fit_lag_ms_min,
                fit_lag_ms_max,
                fit_use_offset,
                min_fit_points,
                pl_min_samples,
                pl_min_tail_samples,
                pl_require_not_worse,
                compute_branching,
                compute_avalanche,
            )
            record_row_result(row_idx, alpha, beta, run_name, per_population, timing)
    else:
        future_to_row: dict[Any, dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for row_index, row in enumerate(summary_rows):
                future = executor.submit(
                    _compute_row_all_metrics,
                    row_index,
                    row,
                    sweep_dir,
                    population_specs,
                    windows,
                    dt_mr_ms,
                    dt_avalanche_override_ms,
                    min_avalanche_size,
                    fit_lag_ms_min,
                    fit_lag_ms_max,
                    fit_use_offset,
                    min_fit_points,
                    pl_min_samples,
                    pl_min_tail_samples,
                    pl_require_not_worse,
                    compute_branching,
                    compute_avalanche,
                )
                future_to_row[future] = row

            for future in as_completed(future_to_row):
                try:
                    row_idx, alpha, beta, run_name, per_population, timing = future.result()
                except Exception as exc:  # pragma: no cover - bubble up with context
                    row = future_to_row[future]
                    raise RuntimeError(
                        f"Failed to analyze alpha={row.get('alpha')} beta={row.get('beta')} "
                        f"(run={row.get('run_name')})"
                    ) from exc
                record_row_result(row_idx, alpha, beta, run_name, per_population, timing)

    save_t0 = time.perf_counter()
    for pop_folder in population_order:
        print(f"--- Population: {population_results[pop_folder]['name']} [{pop_folder}] ---")
        for window_name in window_order:
            window_result = population_results[pop_folder]["windows"][window_name]
            analysis_rows_data = window_result.pop("analysis_rows_data")
            window_result["analysis_rows"] = [
                row for row in analysis_rows_data if row is not None
            ]

            relative_out_dir = Path(pop_folder)
            if len(windows) > 1:
                relative_out_dir = relative_out_dir / window_name
            write_analysis_outputs(
                window_result,
                alphas,
                betas,
                dt_mr_ms,
                analysis_root,
                relative_out_dir,
                disable_plots_effective,
                save_csv,
                compute_branching,
                compute_avalanche,
            )
            if compute_branching:
                print(f"Branching artifacts saved to {analysis_root / 'branching' / relative_out_dir}")
            if compute_avalanche:
                print(f"Avalanche artifacts saved to {analysis_root / 'avalanche' / relative_out_dir}")
    save_total_s = time.perf_counter() - save_t0

    if log_timing:
        n_rows = float(len(summary_rows))
        print("Timing summary:")
        print(
            f"  load_run total={timing_totals['load_run_s']:.2f}s avg={timing_totals['load_run_s']/n_rows:.4f}s/row"
        )
        print(
            "  build_population_spikes "
            f"total={timing_totals['build_population_spikes_s']:.2f}s "
            f"avg={timing_totals['build_population_spikes_s']/n_rows:.4f}s/row"
        )
        print(
            f"  compute_windows total={timing_totals['compute_windows_s']:.2f}s avg={timing_totals['compute_windows_s']/n_rows:.4f}s/row"
        )
        print(
            f"  total_row_worker total={timing_totals['total_row_s']:.2f}s avg={timing_totals['total_row_s']/n_rows:.4f}s/row max={max_row_total_s:.4f}s"
        )
        print(f"  save_outputs total={save_total_s:.2f}s")
        print(f"  end_to_end total={time.perf_counter() - analysis_t0:.2f}s")

    return {
        "sweep_dir": sweep_dir,
        "alphas": alphas,
        "betas": betas,
        "dt_mr_ms": dt_mr_ms,
        "population_results": population_results,
        "population_order": population_order,
        "window_order": window_order,
    }


def _compute_row_all_metrics(
    row_index: int,
    row: dict[str, Any],
    sweep_dir: Path,
    population_specs: list[dict[str, Any]],
    windows: list[dict[str, float | str]],
    dt_mr_ms: float,
    dt_avalanche_override_ms: float | None,
    min_avalanche_size: int,
    fit_lag_ms_min: float | None,
    fit_lag_ms_max: float | None,
    fit_use_offset: bool,
    min_fit_points: int,
    pl_min_samples: int,
    pl_min_tail_samples: int,
    pl_require_not_worse: bool,
    compute_branching: bool,
    compute_avalanche: bool,
) -> tuple[int, float, float, str | None, dict[str, dict[str, dict[str, float]]], dict[str, float]]:
    row_t0 = time.perf_counter()
    run_path = resolve_run_path(row, sweep_dir)
    t0 = time.perf_counter()
    cfg, data, _, _, stim_metadata = load_run(run_path)
    load_run_s = time.perf_counter() - t0
    simtime_ms = float(cfg["experiment"]["simtime_ms"])

    t1 = time.perf_counter()
    times_by_population: dict[str, np.ndarray] = {}
    for pop_spec in population_specs:
        pop_folder = str(pop_spec["folder"])
        spike_keys = pop_spec["spike_keys"]
        allowed_senders = resolve_allowed_senders(pop_spec, stim_metadata, run_path)
        spikes = combine_spikes(data, spike_keys, allowed_senders=allowed_senders)
        times = np.asarray(spikes["times"], float)
        if times.size > 1:
            times = np.sort(times)
        times_by_population[pop_folder] = times
    build_population_spikes_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    metrics_by_population: dict[str, dict[str, dict[str, float]]] = {}
    for pop_spec in population_specs:
        pop_folder = str(pop_spec["folder"])
        full_times = times_by_population[pop_folder]
        window_metrics: dict[str, dict[str, float]] = {}
        for window in windows:
            window_name = str(window["name"])
            start_ms = float(window["start_ms"])
            stop_ms = float(window["stop_ms"]) if window.get("stop_ms") is not None else simtime_ms

            if full_times.size:
                lo = int(np.searchsorted(full_times, start_ms, side="left"))
                hi = int(np.searchsorted(full_times, stop_ms, side="left"))
                window_times = full_times[lo:hi]
            else:
                window_times = full_times

            window_metrics[window_name] = compute_metrics_from_times(
                times=window_times,
                t_start_ms=start_ms,
                t_stop_ms=stop_ms,
                dt_mr_ms=dt_mr_ms,
                dt_avalanche_override_ms=dt_avalanche_override_ms,
                min_avalanche_size=min_avalanche_size,
                fit_lag_ms_min=fit_lag_ms_min,
                fit_lag_ms_max=fit_lag_ms_max,
                fit_use_offset=fit_use_offset,
                min_fit_points=min_fit_points,
                pl_min_samples=pl_min_samples,
                pl_min_tail_samples=pl_min_tail_samples,
                pl_require_not_worse=pl_require_not_worse,
                compute_branching=compute_branching,
                compute_avalanche=compute_avalanche,
            )
        metrics_by_population[pop_folder] = window_metrics
    compute_windows_s = time.perf_counter() - t2

    alpha = float(row["alpha"])
    beta = float(row["beta"])
    run_name = row.get("run_name")
    timing = {
        "load_run_s": load_run_s,
        "build_population_spikes_s": build_population_spikes_s,
        "compute_windows_s": compute_windows_s,
        "total_row_s": time.perf_counter() - row_t0,
    }
    return row_index, alpha, beta, run_name, metrics_by_population, timing


def _compute_row_metrics(
    row_index: int,
    row: dict[str, Any],
    sweep_dir: Path,
    population_spec: dict[str, Any],
    start_ms: float,
    stop_ms: float,
    dt_mr_ms: float,
    dt_avalanche_override_ms: float | None,
    min_avalanche_size: int,
    fit_lag_ms_min: float | None,
    fit_lag_ms_max: float | None,
    fit_use_offset: bool,
    min_fit_points: int,
    pl_min_samples: int,
    pl_min_tail_samples: int,
    pl_require_not_worse: bool,
    compute_branching: bool,
    compute_avalanche: bool,
) -> tuple[int, float, float, str | None, dict[str, float]]:
    run_path = resolve_run_path(row, sweep_dir)
    metrics = analyze_run(
        run_path=run_path,
        population_spec=population_spec,
        t_start_ms=start_ms,
        t_stop_override=stop_ms,
        dt_mr_ms=dt_mr_ms,
        dt_avalanche_override_ms=dt_avalanche_override_ms,
        min_avalanche_size=min_avalanche_size,
        fit_lag_ms_min=fit_lag_ms_min,
        fit_lag_ms_max=fit_lag_ms_max,
        fit_use_offset=fit_use_offset,
        min_fit_points=min_fit_points,
        pl_min_samples=pl_min_samples,
        pl_min_tail_samples=pl_min_tail_samples,
        pl_require_not_worse=pl_require_not_worse,
        compute_branching=compute_branching,
        compute_avalanche=compute_avalanche,
    )
    alpha = float(row["alpha"])
    beta = float(row["beta"])
    run_name = row.get("run_name")
    return row_index, alpha, beta, run_name, metrics


def aggregate_multi_run_results(
    results: list[dict[str, Any]],
    multi_root: Path,
) -> dict[str, Any]:
    if not results:
        raise RuntimeError("No successful sweep analyses to aggregate")

    base = results[0]
    alphas = base["alphas"]
    betas = base["betas"]
    dt_mr_ms = base["dt_mr_ms"]
    window_order = base["window_order"]
    population_order = base["population_order"]

    for res in results[1:]:
        if not np.allclose(res["alphas"], alphas):
            raise ValueError("Alpha grids differ across runs; cannot average")
        if not np.allclose(res["betas"], betas):
            raise ValueError("Beta grids differ across runs; cannot average")
        if not np.isclose(res["dt_mr_ms"], dt_mr_ms):
            raise ValueError("MR dt differs across runs; cannot average")
        if res["window_order"] != window_order:
            raise ValueError("Window layouts differ across runs; cannot average")
        if res["population_order"] != population_order:
            raise ValueError("Population layouts differ across runs; cannot average")

    aggregated_population: dict[str, dict[str, Any]] = {}
    for pop_name in population_order:
        base_pop_entry = base["population_results"][pop_name]
        base_pop_windows = base_pop_entry["windows"]
        aggregated_windows: dict[str, dict[str, Any]] = {}
        for window_name in window_order:
            base_window = base_pop_windows[window_name]
            aggregated_grids: dict[str, np.ndarray] = {}
            for key in GRID_METRIC_KEYS:
                stack = np.stack(
                    [
                        res["population_results"][pop_name]["windows"][window_name]["grids"][key]
                        for res in results
                    ],
                    axis=0,
                )
                aggregated_grids[key] = np.nanmean(stack, axis=0)

            analysis_rows = build_average_rows(
                alphas,
                betas,
                aggregated_grids,
                run_label=f"mean_of_{len(results)}_runs",
                window_name=window_name,
            )
            aggregated_windows[window_name] = {
                "name": window_name,
                "start_ms": base_window["start_ms"],
                "stop_ms": base_window["stop_ms"],
                "grids": aggregated_grids,
                "analysis_rows": analysis_rows,
            }

        aggregated_population[pop_name] = {
            "name": base_pop_entry.get("name", pop_name),
            "windows": aggregated_windows,
        }

    return {
        "sweep_dir": multi_root,
        "alphas": alphas,
        "betas": betas,
        "dt_mr_ms": dt_mr_ms,
        "population_results": aggregated_population,
        "population_order": population_order,
        "window_order": window_order,
    }


def _npz_scalar(payload: Any, key: str, default: Any = None) -> Any:
    if key not in payload:
        return default
    value = payload[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def load_cached_run_analysis_result(run_dir: Path, run_output_root: Path) -> dict[str, Any]:
    branches: list[tuple[str, Path]] = [
        ("branching", run_output_root / "branching"),
        ("avalanche", run_output_root / "avalanche"),
    ]

    records: list[dict[str, Any]] = []
    for branch_name, branch_root in branches:
        if not branch_root.exists():
            continue
        for metrics_path in sorted(branch_root.rglob("metrics.npz")):
            rel = metrics_path.relative_to(branch_root)
            parts = rel.parts
            if len(parts) < 2:
                continue
            population = parts[0]
            with np.load(metrics_path, allow_pickle=False) as payload:
                alphas = np.asarray(payload["alphas"], float)
                betas = np.asarray(payload["betas"], float)
                dt_mr_ms = float(_npz_scalar(payload, "dt_mr_ms", np.nan))
                window_name = str(_npz_scalar(payload, "window_name", "overall"))
                window_start_ms = float(_npz_scalar(payload, "window_start_ms", np.nan))
                window_stop_ms = float(_npz_scalar(payload, "window_stop_ms", np.nan))
                grids: dict[str, np.ndarray] = {}
                shape = (betas.size, alphas.size)
                for key in GRID_METRIC_KEYS:
                    if key in payload:
                        grids[key] = np.asarray(payload[key], float)
                    else:
                        grids[key] = np.full(shape, np.nan)
                if "tau_mr_ms" in grids:
                    tau_grid = np.asarray(grids["tau_mr_ms"], float)
                    invalid_mask = (tau_grid < TAU_MR_MIN_MS) | (tau_grid > TAU_MR_MAX_MS)
                    tau_grid = tau_grid.copy()
                    tau_grid[invalid_mask] = np.nan
                    grids["tau_mr_ms"] = tau_grid
            records.append(
                {
                    "branch": branch_name,
                    "population": population,
                    "window": window_name,
                    "window_start_ms": window_start_ms,
                    "window_stop_ms": window_stop_ms,
                    "alphas": alphas,
                    "betas": betas,
                    "dt_mr_ms": dt_mr_ms,
                    "grids": grids,
                }
            )

    if not records:
        raise RuntimeError(
            f"No cached metrics.npz files found for {run_dir.name} under {run_output_root}"
        )

    alphas_ref = records[0]["alphas"]
    betas_ref = records[0]["betas"]
    dt_ref = records[0]["dt_mr_ms"]
    for rec in records[1:]:
        if not np.allclose(rec["alphas"], alphas_ref):
            raise ValueError(f"Cached alpha grid mismatch in {run_output_root}")
        if not np.allclose(rec["betas"], betas_ref):
            raise ValueError(f"Cached beta grid mismatch in {run_output_root}")
        if not np.isclose(rec["dt_mr_ms"], dt_ref):
            raise ValueError(f"Cached MR dt mismatch in {run_output_root}")

    population_results: dict[str, dict[str, Any]] = {}
    for rec in records:
        pop_name = rec["population"]
        window_name = rec["window"]
        pop_entry = population_results.setdefault(
            pop_name,
            {
                "name": pop_name,
                "windows": {},
            },
        )
        window_entry = pop_entry["windows"].setdefault(
            window_name,
            {
                "name": window_name,
                "start_ms": rec["window_start_ms"],
                "stop_ms": rec["window_stop_ms"],
                "grids": {
                    key: np.full((betas_ref.size, alphas_ref.size), np.nan)
                    for key in GRID_METRIC_KEYS
                },
                "analysis_rows": [],
            },
        )
        for key in GRID_METRIC_KEYS:
            if key in rec["grids"]:
                existing = window_entry["grids"][key]
                incoming = rec["grids"][key]
                if np.all(np.isnan(existing)):
                    window_entry["grids"][key] = incoming

    population_order = sorted(population_results.keys())
    window_name_set = set()
    for pop_entry in population_results.values():
        window_name_set.update(pop_entry["windows"].keys())
    window_order = sorted(window_name_set)

    return {
        "sweep_dir": run_dir,
        "alphas": alphas_ref,
        "betas": betas_ref,
        "dt_mr_ms": float(dt_ref),
        "population_results": population_results,
        "population_order": population_order,
        "window_order": window_order,
    }


def analyze_multi_run_parent(multi_root: Path, args: argparse.Namespace) -> None:
    if not multi_root.exists():
        raise FileNotFoundError(f"multi-run folder not found: {multi_root}")

    run_dirs = sorted(
        [p for p in multi_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    )
    if not run_dirs:
        raise RuntimeError(f"No run_* directories found inside {multi_root}")

    multi_analysis_root = build_criticality_output_root(multi_root.name, "analysis")
    multi_analysis_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving multi-run analysis bundle to {multi_analysis_root}")
    per_run_full_output = bool(args.individual_run_output)
    summary_only = bool(args.summary_only)
    if per_run_full_output:
        print("Individual run output enabled: saving NPZ/CSV and optional plots per run")
    else:
        print("Individual run output disabled: saving NPZ-only per run (no per-run plots/CSV)")
    if summary_only:
        print("Summary-only mode enabled: skipping per-run recomputation")

    results: list[dict[str, Any]] = []
    if summary_only:
        for run_dir in run_dirs:
            run_output_root = multi_analysis_root / f"{run_dir.name}_analysis"
            try:
                result = load_cached_run_analysis_result(run_dir, run_output_root)
                results.append(result)
            except Exception as exc:
                print(f"Skipping cached {run_dir}: {exc}")
    else:
        for run_dir in run_dirs:
            try:
                run_output_root = multi_analysis_root / f"{run_dir.name}_analysis"
                result = analyze_sweep_dir(
                    run_dir,
                    args,
                    output_root_override=run_output_root,
                    save_csv=per_run_full_output,
                    disable_plots_override=(not per_run_full_output),
                )
                results.append(result)
            except Exception as exc:
                print(f"Skipping {run_dir}: {exc}")

    if not results:
        raise RuntimeError("All run analyses failed; nothing to aggregate")

    aggregate_result = aggregate_multi_run_results(results, multi_root)
    aggregate_root = multi_analysis_root / f"{multi_root.name}_analysis_mean"
    aggregate_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving averaged analysis to {aggregate_root}")

    window_order = aggregate_result["window_order"]
    population_order = aggregate_result["population_order"]
    for pop_name in population_order:
        pop_entry = aggregate_result["population_results"][pop_name]
        pop_label = pop_entry.get("name", pop_name)
        for window_name in window_order:
            window_result = pop_entry["windows"][window_name]
            relative_out_dir = Path(pop_name)
            if len(window_order) == 1:
                pass
            else:
                relative_out_dir = relative_out_dir / window_name
            write_analysis_outputs(
                window_result,
                aggregate_result["alphas"],
                aggregate_result["betas"],
                aggregate_result["dt_mr_ms"],
                aggregate_root,
                relative_out_dir,
                args.disable_plots,
                True,
                bool(args.compute_branching),
                bool(args.compute_avalanche),
            )
            if args.compute_branching:
                print(
                    f"Averaged branching saved to {aggregate_root / 'branching' / relative_out_dir} ({pop_label})"
                )
            if args.compute_avalanche:
                print(
                    f"Averaged avalanche saved to {aggregate_root / 'avalanche' / relative_out_dir} ({pop_label})"
                )


def main() -> None:
    args = parse_args()
    if args.multi_run_dir and args.sweep_dir:
        raise ValueError("--multi-run-dir and --sweep-dir are mutually exclusive")
    if args.summary_only and not args.multi_run_dir:
        raise ValueError("--summary-only is only supported with --multi-run-dir")

    if args.multi_run_dir:
        analyze_multi_run_parent(args.multi_run_dir, args)
        return

    sweep_dir = args.sweep_dir or find_latest_sweep_dir(args.results_root)
    analyze_sweep_dir(sweep_dir, args)


if __name__ == "__main__":
    main()
