# scripts/run_branching_summary.py

from pathlib import Path
import argparse
import numpy as np

from src.analysis.util import load_run, find_latest_run_dir, combine_spikes
from src.analysis.metrics import (
    average_inter_event_interval,
    empty_bin_fraction,
    branching_ratios_binned_global,
)


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
    return p.parse_args()


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
    print()

    # Markdown-Tabelle Header
    print("| dt/AIEI | dt (ms) | p0(dt) | sigma_global |")
    print("|--------|---------|--------|--------------|")

    # Für jeden Faktor: dt, p0, sigma_global
    for f in dt_factors:
        dt_raw = f * aiei_ms
        dt_ms = max(dt_raw, args.dt_min_ms)  # Hard-Lower-Bound

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

        dt_over_aiei = dt_ms / aiei_ms

        # Tabellenzeile
        print(
            f"| {dt_over_aiei:6.3f} | {dt_ms:7.3f} | {p0:6.3f} | {sigma_binned:12.3f} |"
        )


if __name__ == "__main__":
    main()