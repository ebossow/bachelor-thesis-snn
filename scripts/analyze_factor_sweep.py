"""Aggregate MR estimator and avalanche-based DCC across a resting sweep."""

from __future__ import annotations

import argparse
import csv
import os
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
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

MNIST_PATTERN_CURRENT_THRESHOLD_P_A = 1e-9


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

    # Power-law fits are only meaningful with enough events and enough tail samples.
    tau_size = float("nan")
    tau_duration = float("nan")

    if sizes.size >= pl_min_samples:
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

    if durations.size >= pl_min_samples:
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


def write_analysis_outputs(
    window_result: dict[str, Any],
    alphas: np.ndarray,
    betas: np.ndarray,
    dt_mr_ms: float,
    out_root: Path,
    disable_plots: bool,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    save_table(window_result["analysis_rows"], out_root / "metrics.csv")
    payload: dict[str, Any] = {
        "alphas": alphas,
        "betas": betas,
        "dt_mr_ms": dt_mr_ms,
        "window_name": window_result["name"],
        "window_start_ms": window_result["start_ms"],
        "window_stop_ms": window_result["stop_ms"],
    }
    payload.update(window_result["grids"])
    save_npz(out_root / "metrics.npz", payload)

    if disable_plots:
        return

    sigma_grid = window_result["grids"]["sigma_mr"]
    dcc_grid = window_result["grids"]["dcc"]
    sigma_cmap, sigma_limits = build_sigma_colormap(sigma_grid)
    plot_heatmap(
        sigma_grid,
        alphas,
        betas,
        f"MR branching ratio ({window_result['name']})",
        out_root / "heatmap_sigma.png",
        cmap=sigma_cmap,
        color_limits=sigma_limits,
    )
    plot_heatmap(
        dcc_grid,
        alphas,
        betas,
        f"Distance to criticality (DCC) [{window_result['name']}]",
        out_root / "heatmap_dcc.png",
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


def analyze_sweep_dir(sweep_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
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

    analysis_base = sweep_dir / "analysis"
    analysis_base.mkdir(parents=True, exist_ok=True)

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
    if dt_avalanche_override_ms is not None:
        avalanche_dt_msg = f"{dt_avalanche_override_ms} ms (fixed)"
    else:
        avalanche_dt_msg = "AIEI-derived per run"
    print(f"MR dt = {dt_mr_ms} ms | Avalanche dt = {avalanche_dt_msg}")
    if not POWERLAW_AVAILABLE:
        print("Warning: powerlaw package not installed; DCC exponents will be NaN")

    population_results: dict[str, dict[str, Any]] = {}
    population_order: list[str] = []
    for pop_spec in population_specs:
        pop_folder = pop_spec["folder"]
        population_order.append(pop_folder)
        population_root = analysis_base / pop_folder
        population_root.mkdir(parents=True, exist_ok=True)

        print(f"--- Population: {pop_spec['name']} [{pop_folder}] ---")
        window_results: dict[str, dict[str, Any]] = {}
        for window in windows:
            window_result = analyze_window(
                summary_rows=summary_rows,
                sweep_dir=sweep_dir,
                alpha_index=alpha_index,
                beta_index=beta_index,
                alphas=alphas,
                betas=betas,
                window=window,
                args=args,
                dt_mr_ms=dt_mr_ms,
                dt_avalanche_override_ms=dt_avalanche_override_ms,
                population_spec=pop_spec,
                worker_count=worker_count,
            )
            window_results[window_result["name"]] = window_result

            if len(windows) == 1:
                out_dir = population_root
            else:
                out_dir = population_root / window_result["name"]
            write_analysis_outputs(
                window_result,
                alphas,
                betas,
                dt_mr_ms,
                out_dir,
                args.disable_plots,
            )
            print(f"Analysis artifacts saved to {out_dir}")

        population_results[pop_folder] = {
            "name": pop_spec["name"],
            "windows": window_results,
        }

    return {
        "sweep_dir": sweep_dir,
        "alphas": alphas,
        "betas": betas,
        "dt_mr_ms": dt_mr_ms,
        "population_results": population_results,
        "population_order": population_order,
        "window_order": [str(w["name"]) for w in windows],
    }


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


def analyze_multi_run_parent(multi_root: Path, args: argparse.Namespace) -> None:
    if not multi_root.exists():
        raise FileNotFoundError(f"multi-run folder not found: {multi_root}")

    run_dirs = sorted(
        [p for p in multi_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    )
    if not run_dirs:
        raise RuntimeError(f"No run_* directories found inside {multi_root}")

    results: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        try:
            result = analyze_sweep_dir(run_dir, args)
            results.append(result)
        except Exception as exc:
            print(f"Skipping {run_dir}: {exc}")

    if not results:
        raise RuntimeError("All run analyses failed; nothing to aggregate")

    aggregate_result = aggregate_multi_run_results(results, multi_root)
    aggregate_root = multi_root / "analysis_mean"
    aggregate_root.mkdir(parents=True, exist_ok=True)

    window_order = aggregate_result["window_order"]
    population_order = aggregate_result["population_order"]
    for pop_name in population_order:
        pop_entry = aggregate_result["population_results"][pop_name]
        pop_label = pop_entry.get("name", pop_name)
        pop_root = aggregate_root / pop_name
        pop_root.mkdir(parents=True, exist_ok=True)
        for window_name in window_order:
            window_result = pop_entry["windows"][window_name]
            if len(window_order) == 1:
                out_dir = pop_root
            else:
                out_dir = pop_root / window_name
            write_analysis_outputs(
                window_result,
                aggregate_result["alphas"],
                aggregate_result["betas"],
                aggregate_result["dt_mr_ms"],
                out_dir,
                args.disable_plots,
            )
            print(f"Averaged analysis saved to {out_dir} ({pop_label})")


def main() -> None:
    args = parse_args()
    if args.multi_run_dir and args.sweep_dir:
        raise ValueError("--multi-run-dir and --sweep-dir are mutually exclusive")

    if args.multi_run_dir:
        analyze_multi_run_parent(args.multi_run_dir, args)
        return

    sweep_dir = args.sweep_dir or find_latest_sweep_dir(args.results_root)
    analyze_sweep_dir(sweep_dir, args)


if __name__ == "__main__":
    main()
