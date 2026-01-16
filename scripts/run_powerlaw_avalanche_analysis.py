# scripts/run_powerlaw_avalanche_analysis.py

from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import powerlaw

from src.analysis.util import load_run, find_latest_run_dir, combine_spikes
from src.analysis.metrics import avalanche_sizes_from_times


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Pfad zu run_*-Ordner. Wenn nicht gesetzt: letzter Run unter results/.",
    )
    p.add_argument(
        "--t_start_ms",
        type=float,
        default=None,
        help="Start des Auswertefensters (ms). Default: Stimulus-Off oder 0.",
    )
    p.add_argument(
        "--t_stop_ms",
        type=float,
        default=None,
        help="Ende des Auswertefensters (ms). Default: simtime_ms.",
    )
    p.add_argument(
        "--dt_list",
        type=float,
        nargs="*",
        default=[2.0, 3.0, 7.0, 8.0, 9.0, 10.0, 15.0, 16.0, 17.0, 18.0],
        help="Liste von dt_ms für Avalanche-Binning.",
    )
    p.add_argument(
        "--min_size",
        type=int,
        default=1,
        help="Minimale Avalanche-Size (in Spikes), die in den Fit eingeht.",
    )
    p.add_argument(
        "--min_avalanches",
        type=int,
        default=50,
        help="Minimale Anzahl Avalanches für einen Fit.",
    )
    p.add_argument(
        "--no_plots",
        action="store_true",
        help="Wenn gesetzt, keine Plots erzeugen (nur Zahlen).",
    )
    return p.parse_args()


def fit_powerlaw_to_sizes(
    sizes: np.ndarray,
    discrete: bool = True,
) -> dict:
    """
    Wrapper für powerlaw.Fit auf Avalanche-Sizes.

    Gibt ein Dict mit den wichtigsten Fitparametern zurück.
    """
    fit = powerlaw.Fit(sizes, discrete=discrete)  # xmin wird automatisch geschätzt

    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin
    D = fit.power_law.D  # KS-Distanz

    # Vergleich mit alternativen Verteilungen
    R_exp, p_exp = fit.distribution_compare("power_law", "exponential")
    R_logn, p_logn = fit.distribution_compare("power_law", "lognormal")

    return {
        "fit": fit,          # Original-Objekt für Plots
        "alpha": alpha,
        "xmin": xmin,
        "D": D,
        "R_exp": R_exp,
        "p_exp": p_exp,
        "R_logn": R_logn,
        "p_logn": p_logn,
    }


def main():
    args = parse_args()

    # Run-Verzeichnis bestimmen
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        print(f"Using specified run directory: {run_dir}")
    else:
        results_root = Path("results")
        run_dir = find_latest_run_dir(results_root)
        print(f"Using latest run directory: {run_dir}")

    cfg, data, weights_data, weights_over_time, _ = load_run(run_dir)

    simtime_ms = float(cfg["experiment"]["simtime_ms"])

    # t_off_ms, falls vorhanden (z.B. Stimulus-Off); sonst 0
    stim_cfg = cfg.get("stimulation", {})
    pattern_cfg = stim_cfg.get("pattern", {})
    if stim_cfg["dc"]["enabled"]:
        t_off_ms = float(pattern_cfg.get("t_off_ms", 0.0))
    else:
        t_off_ms = 0.0

    t_start_ms = float(args.t_start_ms) if args.t_start_ms is not None else t_off_ms
    t_stop_ms = float(args.t_stop_ms) if args.t_stop_ms is not None else simtime_ms

    # Spikes aller Populationen kombinieren
    spikes_all = combine_spikes(data, ["spikes_E", "spikes_IH", "spikes_IA"])
    times_full = spikes_all["times"]

    print("\n=== Power-law analysis of avalanche sizes ===")
    print(f"Run dir    : {run_dir}")
    print(f"Time window: [{t_start_ms} ms, {t_stop_ms} ms]")

    # Histogram-/Fit-Ergebnisse pro dt sammeln
    results_per_dt = []

    for dt_ms in args.dt_list:
        dt_ms = float(dt_ms)

        sizes, durations_ms = avalanche_sizes_from_times(
            times_ms=times_full,
            t_start_ms=t_start_ms,
            t_stop_ms=t_stop_ms,
            dt_ms=dt_ms,
            min_size=args.min_size,
        )

        n_aval = sizes.size
        print(f"\n--- dt = {dt_ms:.1f} ms ---")
        print(f"Avalanches (size >= {args.min_size}): {n_aval}")

        if n_aval < args.min_avalanches:
            print(f"  Zu wenige Avalanches (< {args.min_avalanches}) -> kein Fit.")
            continue

        # powerlaw-Fit
        fit_info = fit_powerlaw_to_sizes(sizes, discrete=True)

        print(f"  alpha           = {fit_info['alpha']:.3f}")
        print(f"  xmin            = {fit_info['xmin']:.3g}")
        print(f"  KS D            = {fit_info['D']:.3f}")
        print(f"  compare PL vs exp: R = {fit_info['R_exp']:.3f}, p = {fit_info['p_exp']:.3f}")
        print(f"  compare PL vs logn: R = {fit_info['R_logn']:.3f}, p = {fit_info['p_logn']:.3f}")

        results_per_dt.append((dt_ms, sizes, fit_info))

        if not args.no_plots:
            # CCDF + Fit in log-log Darstellung
            fit = fit_info["fit"]

            fig, ax = plt.subplots(figsize=(6, 5))
            fit.plot_ccdf(ax=ax, label="data")
            fit.power_law.plot_ccdf(ax=ax, linestyle="--", label="power-law fit")

            ax.set_xlabel("Avalanche size (spikes)")
            ax.set_ylabel("CCDF")
            ax.set_title(f"Avalanche sizes, dt = {dt_ms:.1f} ms")
            ax.legend()
            ax.set_xscale("log")
            ax.set_yscale("log")

            plt.tight_layout()
            plt.show()

    # Optional: zusammenfassende Tabelle (z.B. für Copy&Paste)
    if results_per_dt:
        print("\nSummary per dt:")
        print("dt_ms\talpha\txmin\tD\tR_exp\tp_exp\tR_logn\tp_logn")
        for dt_ms, sizes, fit_info in results_per_dt:
            print(
                f"{dt_ms:.1f}\t"
                f"{fit_info['alpha']:.3f}\t"
                f"{fit_info['xmin']:.3g}\t"
                f"{fit_info['D']:.3f}\t"
                f"{fit_info['R_exp']:.3f}\t{fit_info['p_exp']:.3f}\t"
                f"{fit_info['R_logn']:.3f}\t{fit_info['p_logn']:.3f}"
            )
    else:
        print("\nNo dt produced enough avalanches for fitting.")


if __name__ == "__main__":
    main()