from pathlib import Path
import numpy as np
import yaml
import matplotlib.pyplot as plt

from src.analysis.metrics import (
    instantaneous_rates,
    population_rate,
    cv_isi,
    kuramoto_order_parameter,
    build_weight_matrix,
    normalize_weight_matrix,
    mean_weight_change
)
from src.analysis.plotting import (
    plot_spike_raster,
    plot_pdf_cv,
    plot_pdf_R,
    plot_pdf_mean_rate,
    plot_weight_matrix,
    plot_K,
    plot_spike_raster_ax,
    plot_pdf_cv_ax,
    plot_pdf_R_ax,
    plot_pdf_mean_rate_ax,
    plot_weight_matrix_ax,
    plot_K_ax
)

def load_run(run_dir: Path):
    with (run_dir / "config_resolved.yaml").open("r") as f:
        cfg = yaml.safe_load(f)

    def load_spikes(name: str):
        fpath = run_dir / f"{name}.npz"
        if not fpath.exists():
            return None
        d = np.load(fpath)
        return {"times": d["times"], "senders": d["senders"]}

    data = {
        "spikes_E":  load_spikes("spikes_E"),
        "spikes_IH": load_spikes("spikes_IH"),
        "spikes_IA": load_spikes("spikes_IA"),
    }

    weights_file = run_dir / "weights_final.npz"
    weights_data = None
    if weights_file.exists():
        d = np.load(weights_file, allow_pickle=True)
        weights_data = {
            "sources": d["sources"],
            "targets": d["targets"],
            "weights": d["weights"],
        }

    wtraj_file = run_dir / "weights_trajectory.npz"
    weights_over_time = None
    if wtraj_file.exists():
        d = np.load(wtraj_file)
        weights_over_time = {
            "times": d["times"],
            "weights": d["weights"],  # shape (n_snap, M)
        }

    return cfg, data, weights_data, weights_over_time

def find_latest_run_dir(results_root: Path) -> Path:
    """
    Suche den neuesten run_*-Ordner unterhalb von results_root.
    Erwartete Struktur:
      results/
        <experiment_name>/
          run_YYYYMMDD_HHMMSS/
    """
    run_dirs: list[Path] = []

    if not results_root.exists():
        raise FileNotFoundError(f"results root does not exist: {results_root}")

    for exp_dir in results_root.iterdir():
        if not exp_dir.is_dir():
            continue
        for run_dir in exp_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                run_dirs.append(run_dir)

    if not run_dirs:
        raise RuntimeError(f"No run_* directories found under {results_root}")

    # nimm den mit der neuesten Modifikationszeit
    latest = max(run_dirs, key=lambda p: p.stat().st_mtime)
    return latest

def combine_spikes(data, keys):
    times_list = []
    senders_list = []

    for key in keys:
        sd = data.get(key)
        if sd is None:
            continue
        times_list.append(sd["times"])
        senders_list.append(sd["senders"])

    if not times_list:
        return {"times": np.array([]), "senders": np.array([])}

    times = np.concatenate(times_list)
    senders = np.concatenate(senders_list)
    return {"times": times, "senders": senders}

def main():
    results_root = Path("results")
    run_dir = find_latest_run_dir(results_root)
    print(f"Using latest run directory: {run_dir}")
    cfg, data, weights_data, weights_over_time = load_run(run_dir)

    simtime_ms = cfg["experiment"]["simtime_ms"]

    stim_dc = cfg["stimulation"]["dc"]
    t_analysis_start = stim_dc["t_off_ms"]  # z.B. 40000.0 ms
    t_analysis_stop = simtime_ms           # z.B. 60000.0 ms

    # all spikes
    spikes_N = combine_spikes(data, ["spikes_E", "spikes_IH", "spikes_IA"])
    N = cfg["network"]["N_E"] + cfg["network"]["N_IH"] + cfg["network"]["N_IA"]
    # nur Spikes nach Stimulus-Off für Metriken
    times_all = spikes_N["times"]
    senders_all = spikes_N["senders"]

    mask_post = (times_all >= t_analysis_start) & (times_all < t_analysis_stop)
    times_post = times_all[mask_post]
    senders_post = senders_all[mask_post]


    # Firing Rates
    rates_N, t_bins, mean_rate_population, mean_rates_per_neuron = instantaneous_rates(
        times_post, 
        senders_post,
        N_population=N,
        t_start=t_analysis_start,
        t_stop=t_analysis_stop,
        bin_size_ms=50.0,
    )
    pop_rate_N = population_rate(rates_N)

    #CV ISI
    cv_N = cv_isi(times_post, senders_post, N_population=N)

    # Kuramoto (z.B. E-Population)
    R, Phi = kuramoto_order_parameter(
        times_post, 
        senders_post,
        N,
        t_eval=t_bins,
    )

    # Weight Trajectory Analyse
    W = build_weight_matrix(
        weights_data["sources"],
        weights_data["targets"],
        weights_data["weights"],
        N_total=N,
    )
    Wn = normalize_weight_matrix(W, cfg)

    if weights_over_time is not None:
        t_mid, K = mean_weight_change(
            weight_times=weights_over_time["times"],
            weights=weights_over_time["weights"],
            N_total=N,
        )   
    
    # nur Intervalle nach Stimulus-Off
    mask_K = (t_mid >= t_analysis_start) & (t_mid < t_analysis_stop)
    K_post = K[mask_K]

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(
        2, 8,
        height_ratios=[2.0, 1.5],
        hspace=0.5,
        wspace=0.4,
    )

    # Zeile 0: Raster über alle 3 Spalten
    ax_raster = fig.add_subplot(gs[0, 0:6])
    ax_W = fig.add_subplot(gs[0, 6:8])

    # Zeile 1: 
    ax_cv   = fig.add_subplot(gs[1, 0:2])
    ax_R    = fig.add_subplot(gs[1, 2:4])
    ax_rate = fig.add_subplot(gs[1, 4:6])
    ax_K    = fig.add_subplot(gs[1, 6:8])

    # Zeichnen
    plot_spike_raster_ax(ax_raster, data, cfg)
    plot_pdf_cv_ax(ax_cv, cv_N)
    plot_pdf_R_ax(ax_R, R)
    plot_pdf_mean_rate_ax(ax_rate, mean_rates_per_neuron)
    plot_K_ax(ax_K, K_post)

    im = plot_weight_matrix_ax(ax_W, Wn, cfg)
    cbar = fig.colorbar(im, ax=ax_W)
    cbar.set_label("Normalized weight")

    plt.tight_layout()
    plt.show()

    # hier dann plotting-Funktionen aufrufen
    # plots
    #plot_spike_raster(data, cfg)
    #plot_weight_matrix(Wn, cfg)
    #plot_pdf_cv(cv_N)
    #plot_pdf_R(R)
    #plot_pdf_mean_rate(mean_rates_per_neuron)

if __name__ == "__main__":
    main()