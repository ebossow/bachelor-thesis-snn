from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import powerlaw

from src.analysis.util import load_run, find_latest_run_dir, combine_spikes
from src.analysis.metrics import (
    average_inter_event_interval,
    empty_bin_fraction_scan,
    branching_ratio_neuronwise,
    avalanche_sizes_from_times,
    instantaneous_rates,
    branching_ratios_binned_global,
)


def parse_args():
    p = argparse.ArgumentParser()
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
        help="Start des Auswertefensters (ms).",
    )
    p.add_argument(
        "--t_stop_ms",
        type=float,
        default=None,
        help="Ende des Auswertefensters (ms). Default: simtime_ms.",
    )
    p.add_argument(
        "--dt_ms",
        type=float,
        default=2.0,
        help="Binbreite für σ(dt) und p0(dt) in ms.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Run-Verzeichnis auswählen
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        print(f"Using specified run directory: {run_dir}")
    else:
        results_root = Path("results")
        run_dir = find_latest_run_dir(results_root)
        print(f"Using latest run directory: {run_dir}")

    # Run laden
    cfg, data, weights_data, weights_over_time = load_run(run_dir)

    simtime_ms = float(cfg["experiment"]["simtime_ms"])
    if cfg["stimulation"]["dc"]["enabled"]:
        t_off_ms = float(cfg["stimulation"]["pattern"].get("t_off_ms", simtime_ms))
    else:
        t_off_ms = 0.0
    t_start_ms = float(args.t_start_ms) if args.t_start_ms is not None else t_off_ms
    t_stop_ms = float(args.t_stop_ms) if args.t_stop_ms is not None else simtime_ms
    dt_sigma = float(args.dt_ms)

    # Spikes aller Populationen kombinieren (E, IH, IA)
    spikes_all = combine_spikes(data, ["spikes_E", "spikes_IH", "spikes_IA"])
    times_full = spikes_all["times"]
    senders_full = spikes_all["senders"]

    # Auf das Auswertefenster beschneiden
    mask = (times_full >= t_start_ms) & (times_full <= t_stop_ms)
    times = times_full[mask]
    senders = senders_full[mask]

    # Gesamtzahl Neuronen
    net_cfg = cfg["network"]
    N_total = int(
        net_cfg["N_E"]
        + net_cfg["N_IH"]
        + net_cfg.get("N_IA_1", 0)
        + net_cfg.get("N_IA_2", 0)
    )

    rates, t_bins, mean_rate_pop, mean_rates_per_neuron = instantaneous_rates(
        times=times,          # bereits auf t_start_ms..t_stop_ms gefiltert
        senders=senders,
        N_population=N_total,
        t_start=t_start_ms,
        t_stop=t_stop_ms,
        bin_size_ms=50.0,
    )
    print("mean_rate_population (Hz):", mean_rate_pop)

    def suggested_dt_ms(x: float, N: int, mean_rate_hz: float = 1.45, dt_min_ms: float = 2.0) -> float:
        if mean_rate_hz <= 0:
            return dt_min_ms
        dt = x * 1000.0 / (N * mean_rate_hz)
        return max(dt_min_ms, dt)

    for x in [0.5, 1.0, 2.0]:
        dt = suggested_dt_ms(x, N_total, mean_rate_pop, dt_min_ms=2.0)
        print(f"x={x}: dt ≈ {dt:.2f} ms")

    print("\n=== Criticality analysis ===")
    print(f"Run dir    : {run_dir}")
    print(f"Time window: [{t_start_ms} ms, {t_stop_ms} ms]")
    print(f"N_total    : {N_total}")
    print(f"dt_sigma   : {dt_sigma} ms")

    # ------------------------------------------------------------------
    # 1) Average Inter-Event Interval (AIEI)
    # ------------------------------------------------------------------
    aiei_ms, isi_ms = average_inter_event_interval(
        times_ms=times,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
    )
    print(f"\nAIEI (ms): {aiei_ms:.3f} (based on {isi_ms.size} intervals)")

    # ------------------------------------------------------------------
    # 2) p0(dt) Scan (optional)
    # ------------------------------------------------------------------
    dt_list = np.array([2.0, 3.0, 7.0, 8.0, 9.0, 10.0])
    p0_list = empty_bin_fraction_scan(
        times_ms=times,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
        dt_list_ms=dt_list,
    )
    print("\nEmpty-bin fraction p0(dt):")
    for dt, p0 in zip(dt_list, p0_list):
        print(f"dt = {dt:4.1f} ms -> p0 = {p0:6.3f}")

    # ------------------------------------------------------------------
    # 3) σ(dt) neuronweise + global + avalanche wise
    # ------------------------------------------------------------------
    sigma_global, sigma_per_neuron = branching_ratio_neuronwise(
        times=times,
        senders=senders,
        N_population=N_total,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
        dt_ms=dt_sigma,
    )

    # Robust auswerten, egal ob array oder (array, scalar) zurückkommt
    if isinstance(sigma_per_neuron, tuple):
        sigma_j = np.asarray(sigma_per_neuron[0], float)
        sigma_mean = float(sigma_per_neuron[1])
    else:
        sigma_j = np.asarray(sigma_per_neuron, float)
        sigma_mean = float(np.nanmean(sigma_j))

    print(f"\nNeuron-wise branching σ_j(dt={dt_sigma:.1f} ms):")
    print(f"  mean σ = {sigma_mean:.3f}")
    print(f"  #neurons with valid σ_j: {np.isfinite(sigma_j).sum()} / {sigma_j.size}")

    # --- Globale und Avalanch Branching Ratio ---

    sigma_binned, sigma_binned_aval, counts = branching_ratios_binned_global(
        times_ms=times,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
        dt_ms=dt_sigma,
        min_aval_len=2,
    )

    print(f"\nBinned/global branching (dt={dt_sigma:.1f} ms):")
    print(f"  σ_global_counts      = {sigma_binned:.3f}")
    print(f"  σ_from_avalanches    = {sigma_binned_aval:.3f}")
    print(f"  #bins with spikes    = {(counts > 0).sum()} / {counts.size}")

    # ------------------------------------------------------------------
    # 4) Einfache Plots: ISI-Histogramm und Verteilung von σ_j
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Histogramm der ISIs
    ax_isi = axes[0]
    if isi_ms.size > 0:
        ax_isi.hist(isi_ms, bins=50, density=True)
    ax_isi.set_xlabel("Inter-event interval (ms)")
    ax_isi.set_ylabel("PDF")
    ax_isi.set_title("Global ISI distribution")

    # (b) Histogramm der σ_j
    ax_sig = axes[1]
    finite_sigma = sigma_j[np.isfinite(sigma_j)]
    if finite_sigma.size > 0:
        ax_sig.hist(finite_sigma, bins=40, density=True)
    ax_sig.axvline(sigma_mean, color="red", linestyle="--", label=f"mean σ = {sigma_mean:.3f}")
    ax_sig.set_xlabel("σ_j")
    ax_sig.set_ylabel("PDF")
    ax_sig.set_title(f"Neuron-wise branching (dt = {dt_sigma:.1f} ms)")
    ax_sig.legend()

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 5) Avalanche-Size-Distribution
    # ------------------------------------------------------------------
    sizes, durations_ms = avalanche_sizes_from_times(
        times_ms=times,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
        dt_ms=dt_sigma,   # oder ein anderer dt für Avalanches, z.B. derselbe wie für p0
        min_size=1,
    )

    print(f"\nAvalanches found: {sizes.size}")
    if sizes.size > 0:
        print(f"  mean size     : {sizes.mean():.3f}")
        print(f"  median size   : {np.median(sizes):.3f}")
        print(f"  mean duration : {durations_ms.mean():.3f} ms")

        # einfache lineare Histogramme; später kannst du log-log machen
        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))

        axes2[0].hist(sizes, bins=40, density=True)
        axes2[0].set_xlabel("Avalanche size (spikes)")
        axes2[0].set_ylabel("PDF")
        axes2[0].set_title("Avalanche size distribution")

        axes2[1].hist(durations_ms, bins=40, density=True)
        axes2[1].set_xlabel("Avalanche duration (ms)")
        axes2[1].set_ylabel("PDF")
        axes2[1].set_title("Avalanche duration distribution")

        plt.tight_layout()
        plt.show()
    else:
        print("No avalanches in this window (with given dt).")



if __name__ == "__main__":
    main()