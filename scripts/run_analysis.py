from pathlib import Path
import numpy as np
import yaml

from src.analysis.metrics import (instantaneous_rates, population_rate, cv_isi)
from src.analysis.plotting import plot_spike_raster

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
    return cfg, data

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

def main():
    results_root = Path("results")
    run_dir = find_latest_run_dir(results_root)
    print(f"Using latest run directory: {run_dir}")
    cfg, data = load_run(run_dir)

    spikes_E = data["spikes_E"]
    N_E = cfg["network"]["N_E"]

    rates_E, t_bins, mean_E = instantaneous_rates(
        spikes_E["times"],
        spikes_E["senders"],
        N_population=N_E,
        t_start=0.0,
        t_stop=cfg["experiment"]["simtime_ms"],
        bin_size_ms=50.0,
    )
    nu_E = population_rate(rates_E)
    cv_E = cv_isi(spikes_E["times"], spikes_E["senders"], N_population=N_E)
    #R, Phi = kuramoto_order_parameter(
    #    spikes_E["times"],
    #    spikes_E["senders"],
    #    N_population=N_E,
    #    t_eval=t_bins,
    #)
    print(nu_E)
    print(cv_E)
    # hier dann plotting-Funktionen aufrufen
    plot_spike_raster(data, cfg)

if __name__ == "__main__":
    main()