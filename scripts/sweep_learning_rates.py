# scripts/sweep_learning_rates.py

from pathlib import Path
import itertools
import csv
import copy
import yaml
from datetime import datetime

from src.experiment.config_loader import load_base_config
from src.experiment.experiment_runner import run_experiment_with_cfg  # musst du evtl. noch bauen
from src.analysis.summary_metrics import compute_summary_metrics
from src.analysis.util import load_run  # lädt cfg, data, weights_over_time


def load_search_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def main():
    search_cfg = load_search_config(Path("config/search_learning_rates.yaml"))

    base_cfg_path = Path(search_cfg["base_config"])
    base_cfg = load_base_config(base_cfg_path)

    lambda_values = search_cfg["search"]["lambda_E"]["values"]
    eta_IH_values    = search_cfg["search"]["eta_IH"]["values"]
    eta_IA_values    = search_cfg["search"]["eta_IA"]["values"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = Path("results") / f"sweep_lr_run_{timestamp}"
    sweep_root.mkdir(parents=True, exist_ok=True)

    results = []

    for lam, eta_IH, eta_IA in itertools.product(lambda_values, eta_IH_values, eta_IA_values):
        cfg = copy.deepcopy(base_cfg)
        # Lernraten im Config-Dict überschreiben
        cfg["synapses"]["E_to_X"]["lambda"] = float(lam)
        cfg["synapses"]["IH_to_X"]["eta"]   = float(eta_IH)
        cfg["synapses"]["IA_to_X"]["eta"]   = float(eta_IA)

        # Optional: Experimentname anpassen
        cfg["experiment"]["name"] = f"sweep_lam{float(lam):g}_etaIH{float(eta_IH):g}_etaIA{float(eta_IA):g}"

        # Experiment laufen lassen -> run_dir zurückbekommen
        run_dir = run_experiment_with_cfg(cfg, sweep_root)

        # Run laden und Metriken berechnen
        cfg_run, data, weights_data, weights_over_time, _ = load_run(run_dir)
        metrics = compute_summary_metrics(cfg_run, data, weights_over_time)

        row = {
            "run_dir": str(run_dir),
            "lambda_E": lam,
            "eta_IH": eta_IH,
            "eta_IA": eta_IA,
            "mean_rate_Hz": metrics["mean_rate_Hz"],
            "mean_cv": metrics["mean_cv"],
            "mean_R_all": metrics["mean_R_all"],
            #"mean_R_E": metrics["mean_R_E"], --- IGNORE ---
            "mean_K": metrics["mean_K"],
        }
        results.append(row)
        print(row)

    # Ergebnisse als CSV speichern
    if results:
        # Ergebnisse als CSV speichern – JETZT in den Sweep-Ordner
        out_csv = sweep_root / "sweep_learning_rates.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        print(f"Wrote sweep results to {out_csv}")
    else:
        print("No sweep results (grid was empty?).")


if __name__ == "__main__":
    main()