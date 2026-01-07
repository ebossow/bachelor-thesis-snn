# scripts/run_branching_summary.py

from pathlib import Path
import argparse
import math
import numpy as np

from src.analysis.util import load_run, find_latest_run_dir, combine_spikes
from src.analysis.metrics import (
    average_inter_event_interval,
    empty_bin_fraction,
    branching_ratios_binned_global,
    branching_ratio_mr_estimator,
    MRESTIMATOR_AVAILABLE,
)


def _collect_synaptic_delays_ms(cfg) -> list[float]:
    """Return sorted unique synaptic delays (ms) found in config."""
    delays: list[float] = []
    syn_cfg = cfg.get("synapses", {}) if isinstance(cfg, dict) else {}
    for entry in syn_cfg.values():
        if isinstance(entry, dict) and "delay_ms" in entry:
            try:
                delays.append(float(entry["delay_ms"]))
            except (TypeError, ValueError):
                continue
    if not delays:
        return []
    unique: list[float] = []
    for value in sorted(delays):
        if not any(math.isclose(value, seen, rel_tol=1e-9, abs_tol=1e-9) for seen in unique):
            unique.append(value)
    return unique


def _plot_mr_regression(coeffs, fit_result, title: str | None = None) -> None:
    """Render r_k vs. lag along with the MR exponential fit."""
    import matplotlib.pyplot as plt  # local import to avoid hard dependency

    steps = np.asarray(coeffs.steps)
    rk = np.asarray(coeffs.coefficients)
    fit_vals = fit_result.fitfunc(steps, *fit_result.popt)
    time_ms = steps * coeffs.dt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(time_ms, rk, "o", label=r"$r_k$ (data)")
    ax.plot(time_ms, fit_vals, "-", label="MR fit")
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel(r"Correlation $r_k$")
    ax.set_title(title or "MR regression")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def parse_args():
    p = argparse.ArgumentParser(
        description="Branching-Ratio-Zusammenfassung als Markdown-Tabelle."
    )
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Pfad zu einem spezifischen run_*-Ordner. Wenn nicht gesetzt: letzter Run.",
    )
    p.add_argument(
        "--t_start_ms",
        type=float,
        default=None,
        help="Start des Auswertefensters (ms). Default: Stimulus-Off (t_off_ms).",
    )
    p.add_argument(
        "--t_stop_ms",
        type=float,
        default=None,
        help="Ende des Auswertefensters (ms). Default: simtime_ms.",
    )
    p.add_argument(
        "--dt_factors",
        type=str,
        default="0.5,1.0,2.0",
        help="Kommagetrennte Faktoren für dt = factor * AIEI, z.B. '0.5,1.0,2.0'.",
    )
    p.add_argument(
        "--dt_min_ms",
        type=float,
        default=2.0,
        help="Minimale Binbreite dt in ms (Hard-Lower-Bound).",
    )
    p.add_argument(
        "--plot_regression",
        action="store_true",
        help="Wenn gesetzt, zeige MR-Autokorrelations-/Regressionsplot pro dt-Bin.",
    )
    p.add_argument(
        "--mr_fit_start_ms",
        type=float,
        default=10.0,
        help="Unterer Lag-Grenzwert (ms) für den MR-Fit; <=0 nutzt alle Lags.",
    )
    p.add_argument(
        "--mr_fit_stop_ms",
        type=float,
        default=60.0,
        help="Oberer Lag-Grenzwert (ms) für den MR-Fit; <=0 nutzt alle Lags.",
    )
    p.add_argument(
        "--mr_min_fit_points",
        type=int,
        default=3,
        help="Minimale Anzahl r_k-Stützstellen nach dem Lag-Fenster.",
    )
    p.add_argument(
        "--mr_fit_use_offset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Nutze exp+offset-Modell (mit --no-mr_fit_use_offset für plain exp).",
    )

    args = p.parse_args()

    if args.mr_fit_start_ms is not None and args.mr_fit_start_ms <= 0.0:
        args.mr_fit_start_ms = None
    if args.mr_fit_stop_ms is not None and args.mr_fit_stop_ms <= 0.0:
        args.mr_fit_stop_ms = None
    if (
        args.mr_fit_start_ms is not None
        and args.mr_fit_stop_ms is not None
        and args.mr_fit_stop_ms <= args.mr_fit_start_ms
    ):
        p.error("--mr_fit_stop_ms must be greater than --mr_fit_start_ms")
    args.mr_min_fit_points = max(2, int(args.mr_min_fit_points))

    return args


def main():
    args = parse_args()

    # Run-Verzeichnis bestimmen
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        results_root = Path("results")
        run_dir = find_latest_run_dir(results_root)

    cfg, data, weights_data, weights_over_time = load_run(run_dir)

    simtime_ms = float(cfg["experiment"]["simtime_ms"])
    t_off_ms = float(cfg["stimulation"]["pattern"].get("t_off_ms", 0.0))

    t_start_ms = float(args.t_start_ms) if args.t_start_ms is not None else t_off_ms
    t_stop_ms = float(args.t_stop_ms) if args.t_stop_ms is not None else simtime_ms

    # dt-Faktoren parsen
    dt_factors = []
    for tok in args.dt_factors.split(","):
        tok = tok.strip()
        if not tok:
            continue
        dt_factors.append(float(tok))
    dt_factors = np.array(dt_factors, dtype=float)

    synaptic_delays_ms = _collect_synaptic_delays_ms(cfg)

    # Spikes aller Populationen kombinieren
    spikes_all = combine_spikes(data, ["spikes_E", "spikes_IH", "spikes_IA"])
    times_full = spikes_all["times"]
    senders_full = spikes_all["senders"]

    # Auf Analysefenster beschränken
    mask = (times_full >= t_start_ms) & (times_full <= t_stop_ms)
    times = times_full[mask]
    senders = senders_full[mask]

    # N_total bestimmen
    net_cfg = cfg["network"]
    N_total = int(
        net_cfg["N_E"]
        + net_cfg["N_IH"]
        + net_cfg.get("N_IA_1", 0)
        + net_cfg.get("N_IA_2", 0)
    )

    # AIEI
    aiei_ms, isi_ms = average_inter_event_interval(
        times_ms=times,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
    )

    if not np.isfinite(aiei_ms) or aiei_ms <= 0.0:
        print("AIEI ist nicht definiert (zu wenige Spikes im Fenster).")
        return

    # Zeile für Info oberhalb der Tabelle
    print(f"# Branching-Summary")
    print()
    print(f"- Run dir         : `{run_dir}`")
    print(f"- Time window (ms): ({t_start_ms:.1f}, {t_stop_ms:.1f})")
    print(f"- N_total         : {N_total}")
    print(f"- AIEI (ms)       : {aiei_ms:.3f}")
    print(f"- dt_factors      : {', '.join(f'{f:.2f}' for f in dt_factors)}")
    if synaptic_delays_ms:
        print(
            f"- Synaptic delays  : {', '.join(f'{d:.3f}' for d in synaptic_delays_ms)} ms"
        )
    if not MRESTIMATOR_AVAILABLE:
        print("- MR estimator     : unavailable (install 'mrestimator')")
    elif args.plot_regression:
        print("- Plot regression  : enabled")
    if MRESTIMATOR_AVAILABLE:
        window_desc = "full range"
        if args.mr_fit_start_ms is not None or args.mr_fit_stop_ms is not None:
            lo = args.mr_fit_start_ms if args.mr_fit_start_ms is not None else 0.0
            hi = args.mr_fit_stop_ms if args.mr_fit_stop_ms is not None else float("inf")
            window_desc = f"[{lo:.1f} ms, {hi if np.isfinite(hi) else np.inf:.1f} ms]"
        offset_flag = "with offset" if args.mr_fit_use_offset else "no offset"
        print(f"- MR fit window   : {window_desc} ({offset_flag})")
        print(f"- MR min points   : {args.mr_min_fit_points}")
    print()

    # Markdown-Tabelle Header
    print("| dt/AIEI | dt (ms) | p0(dt) | sigma_global | sigma_MR | tau_MR (ms) |")
    print("|--------|---------|--------|--------------|----------|-------------|")

    dt_values_ms: list[float] = []

    for f in dt_factors:
        dt_raw = f * aiei_ms
        dt_ms = max(dt_raw, args.dt_min_ms)
        dt_values_ms.append(dt_ms)

    for delay_ms in synaptic_delays_ms:
        dt_ms = max(delay_ms, args.dt_min_ms)
        dt_values_ms.append(dt_ms)

    final_dt_values: list[float] = []
    for value in dt_values_ms:
        if not any(math.isclose(value, existing, rel_tol=1e-9, abs_tol=1e-9) for existing in final_dt_values):
            final_dt_values.append(value)

    # Für jeden dt: p0, sigma_global, sigma_MR
    for dt_ms in final_dt_values:
        # p0(dt)
        p0 = empty_bin_fraction(
            times_ms=times,
            t_start_ms=t_start_ms,
            t_stop_ms=t_stop_ms,
            dt_ms=dt_ms,
        )

        # Branching-Ratio (neuronwise, globaler Mittelwert)
        sigma_binned, sigma_binned_aval, counts = branching_ratios_binned_global(
            times_ms=times,
            t_start_ms=t_start_ms,
            t_stop_ms=t_stop_ms,
            dt_ms=dt_ms,
            min_aval_len=2,
        )

        want_details = args.plot_regression and MRESTIMATOR_AVAILABLE
        result = branching_ratio_mr_estimator(
            spike_counts=counts,
            dt_ms=dt_ms,
            fit_lag_ms_min=args.mr_fit_start_ms,
            fit_lag_ms_max=args.mr_fit_stop_ms,
            fit_use_offset=args.mr_fit_use_offset,
            min_fit_points=args.mr_min_fit_points,
            return_details=want_details,
        )

        if want_details:
            sigma_mr, tau_mr, coeffs, fit_result = result
        else:
            sigma_mr, tau_mr = result
            coeffs = fit_result = None

        dt_over_aiei = dt_ms / aiei_ms

        # Tabellenzeile
        print(
            f"| {dt_over_aiei:6.3f} | {dt_ms:7.3f} | {p0:6.3f} | {sigma_binned:12.3f} | {sigma_mr:8.3f} | {tau_mr:11.3f} |"
        )

        if want_details and coeffs is not None and fit_result is not None:
            title = f"MR regression dt={dt_ms:.3f} ms ({run_dir.name})"
            try:
                _plot_mr_regression(coeffs, fit_result, title)
            except Exception as exc:  # pragma: no cover - plotting issues
                print(f"[WARN] Could not plot MR regression for dt={dt_ms:.3f}: {exc}")


if __name__ == "__main__":
    main()