# scripts/run_branching_summary.py

from pathlib import Path
import argparse
import math
import numpy as np

from src.analysis.util import load_run, find_latest_run_dir, combine_spikes
from src.analysis.metrics import (
    average_inter_event_interval,
    binned_spike_counts,
    empty_bin_fraction,
    branching_ratios_binned_global,
    branching_ratio_mr_estimator,
    MRESTIMATOR_AVAILABLE,
)


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


def _parse_manual_dt_values(raw: str | None) -> list[float] | None:
    """Parse --mr_dt_ms input into a clean list of positive dt values in ms."""

    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    if cleaned.startswith("[") and cleaned.endswith("]"):
        cleaned = cleaned[1:-1]

    values: list[float] = []
    for token in cleaned.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = float(token)
        except ValueError as exc:  # pragma: no cover - argument parsing guard
            raise ValueError(
                f"Invalid number '{token}' for --mr_dt_ms (comma-separated floats required)."
            ) from exc
        if value > 0.0:
            values.append(value)

    return values or None


def _plot_mr_timecourse(
    t_centers_ms: np.ndarray,
    sigma_mr: np.ndarray,
    tau_mr_ms: np.ndarray,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    axes[0].plot(t_centers_ms, sigma_mr, "-o", ms=3, lw=1.25)
    axes[0].set_ylabel(r"$m_{MR}(t)$")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_centers_ms, tau_mr_ms, "-o", ms=3, lw=1.25, color="tab:orange")
    axes[1].set_ylabel(r"$\tau_{MR}(t)$ [ms]")
    axes[1].set_xlabel("Time (ms)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def _compute_mr_timecourse(
    times_ms: np.ndarray,
    *,
    t_start_ms: float,
    t_stop_ms: float,
    dt_ms: float,
    window_ms: float,
    step_ms: float,
    fit_lag_ms_min: float | None,
    fit_lag_ms_max: float | None,
    fit_use_offset: bool,
    min_fit_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts, _ = binned_spike_counts(
        times_ms=times_ms,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
        dt_ms=dt_ms,
    )

    if counts.size < 3:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    window_bins = max(3, int(round(window_ms / dt_ms)))
    step_bins = max(1, int(round(step_ms / dt_ms)))
    if window_bins > counts.size:
        return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

    centers: list[float] = []
    sigmas: list[float] = []
    taus: list[float] = []

    last_start = counts.size - window_bins
    for start in range(0, last_start + 1, step_bins):
        stop = start + window_bins
        segment = counts[start:stop]
        sigma_mr, tau_mr = branching_ratio_mr_estimator(
            spike_counts=segment,
            dt_ms=dt_ms,
            fit_lag_ms_min=fit_lag_ms_min,
            fit_lag_ms_max=fit_lag_ms_max,
            fit_use_offset=fit_use_offset,
            min_fit_points=min_fit_points,
        )
        center_ms = t_start_ms + (start + 0.5 * window_bins) * dt_ms
        centers.append(center_ms)
        sigmas.append(sigma_mr)
        taus.append(tau_mr)

    return np.asarray(centers, dtype=float), np.asarray(sigmas, dtype=float), np.asarray(taus, dtype=float)


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
        default=1.0,
        help="Minimale Binbreite dt in ms (Hard-Lower-Bound).",
    )
    p.add_argument(
        "--mr_dt_ms",
        type=str,
        default="1.0",
        help=(
            "Feste dt-Werte (ms) für MR-Regression/Tabelle; akzeptiert eine Zahl oder "
            "Liste wie '1,2,3'. Werte <=0 deaktivieren den manuellen Eintrag."
        ),
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
        default=False,
        help="Nutze exp+offset-Modell (mit --no-mr_fit_use_offset für plain exp).",
    )
    p.add_argument(
        "--mr_timecourse",
        action="store_true",
        help="Berechne zeitaufgelöstes MR m(t) mit gleitendem Fenster.",
    )
    p.add_argument(
        "--mr_timecourse_dt_ms",
        type=float,
        default=None,
        help="Binbreite dt (ms) für m(t). Default: erster Wert aus --mr_dt_ms oder AIEI-basiert.",
    )
    p.add_argument(
        "--mr_timecourse_window_ms",
        type=float,
        default=10000.0,
        help="Fensterbreite (ms) für gleitendes m(t).",
    )
    p.add_argument(
        "--mr_timecourse_step_ms",
        type=float,
        default=1000.0,
        help="Schrittweite (ms) für gleitendes m(t).",
    )
    p.add_argument(
        "--mr_timecourse_plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zeige Plot von m(t) und tau_MR(t) (mit --no-mr_timecourse_plot deaktivieren).",
    )
    p.add_argument(
        "--mr_timecourse_csv",
        type=str,
        default=None,
        help="Optionaler Ausgabepfad für CSV mit Spalten time_center_ms,sigma_mr,tau_mr_ms.",
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

    if args.mr_timecourse_dt_ms is not None and args.mr_timecourse_dt_ms <= 0.0:
        p.error("--mr_timecourse_dt_ms must be > 0")
    if args.mr_timecourse_window_ms <= 0.0:
        p.error("--mr_timecourse_window_ms must be > 0")
    if args.mr_timecourse_step_ms <= 0.0:
        p.error("--mr_timecourse_step_ms must be > 0")

    args.mr_min_fit_points = max(2, int(args.mr_min_fit_points))

    try:
        args.mr_dt_ms_values = _parse_manual_dt_values(args.mr_dt_ms)
    except ValueError as exc:
        p.error(str(exc))

    return args


def main():
    args = parse_args()

    # Run-Verzeichnis bestimmen
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    else:
        results_root = Path("results")
        run_dir = find_latest_run_dir(results_root)

    cfg, data, weights_data, weights_over_time, _ = load_run(run_dir)

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
    if args.mr_dt_ms_values:
        manual_dt_desc = ", ".join(
            f"{max(dt, args.dt_min_ms):.3f}" for dt in args.mr_dt_ms_values
        )
        print(f"- dt_manual (ms)  : {manual_dt_desc}")
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
    if args.mr_timecourse:
        print("- MR timecourse   : enabled")
        print(f"- MR tc window ms : {args.mr_timecourse_window_ms:.1f}")
        print(f"- MR tc step ms   : {args.mr_timecourse_step_ms:.1f}")
    print()

    # Markdown-Tabelle Header
    print("| dt/AIEI | dt (ms) | p0(dt) | sigma_global | sigma_MR | tau_MR (ms) |")
    print("|--------|---------|--------|--------------|----------|-------------|")

    dt_values_ms: list[float] = []

    for f in dt_factors:
        dt_raw = f * aiei_ms
        dt_ms = max(dt_raw, args.dt_min_ms)
        dt_values_ms.append(dt_ms)

    if args.mr_dt_ms_values:
        for dt_ms in args.mr_dt_ms_values:
            dt_values_ms.append(max(dt_ms, args.dt_min_ms))

    final_dt_values: list[float] = []
    for value in dt_values_ms:
        if not any(
            math.isclose(value, existing, rel_tol=1e-9, abs_tol=1e-9)
            for existing in final_dt_values
        ):
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

    if args.mr_timecourse:
        if not MRESTIMATOR_AVAILABLE:
            print("\n[WARN] Skipping MR timecourse: mrestimator is not available.")
            return

        if args.mr_timecourse_dt_ms is not None:
            dt_timecourse_ms = float(args.mr_timecourse_dt_ms)
        elif args.mr_dt_ms_values and len(args.mr_dt_ms_values) > 0:
            dt_timecourse_ms = max(float(args.mr_dt_ms_values[0]), float(args.dt_min_ms))
        else:
            dt_timecourse_ms = max(float(aiei_ms), float(args.dt_min_ms))

        centers_ms, sigma_tc, tau_tc = _compute_mr_timecourse(
            times_ms=times,
            t_start_ms=t_start_ms,
            t_stop_ms=t_stop_ms,
            dt_ms=dt_timecourse_ms,
            window_ms=float(args.mr_timecourse_window_ms),
            step_ms=float(args.mr_timecourse_step_ms),
            fit_lag_ms_min=args.mr_fit_start_ms,
            fit_lag_ms_max=args.mr_fit_stop_ms,
            fit_use_offset=bool(args.mr_fit_use_offset),
            min_fit_points=int(args.mr_min_fit_points),
        )

        if centers_ms.size == 0:
            print(
                "\n[WARN] MR timecourse is empty. Increase t-window, reduce dt, or reduce "
                "--mr_timecourse_window_ms."
            )
        else:
            finite_mask = np.isfinite(sigma_tc)
            n_valid = int(np.sum(finite_mask))
            print(
                f"\nMR timecourse: {centers_ms.size} windows, {n_valid} valid m estimates "
                f"(dt={dt_timecourse_ms:.3f} ms)."
            )
            if n_valid > 0:
                print(
                    f"m(t): mean={np.nanmean(sigma_tc):.4f}, std={np.nanstd(sigma_tc):.4f}, "
                    f"min={np.nanmin(sigma_tc):.4f}, max={np.nanmax(sigma_tc):.4f}"
                )

            if args.mr_timecourse_csv:
                csv_path = Path(args.mr_timecourse_csv)
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                export_mat = np.column_stack((centers_ms, sigma_tc, tau_tc))
                np.savetxt(
                    csv_path,
                    export_mat,
                    delimiter=",",
                    header="time_center_ms,sigma_mr,tau_mr_ms",
                    comments="",
                )
                print(f"Saved MR timecourse CSV: {csv_path}")

            if args.mr_timecourse_plot:
                try:
                    title = (
                        f"MR timecourse ({run_dir.name}) | dt={dt_timecourse_ms:.3f} ms, "
                        f"window={args.mr_timecourse_window_ms:.0f} ms"
                    )
                    _plot_mr_timecourse(centers_ms, sigma_tc, tau_tc, title=title)
                except Exception as exc:  # pragma: no cover - plotting issues
                    print(f"[WARN] Could not plot MR timecourse: {exc}")


if __name__ == "__main__":
    main()