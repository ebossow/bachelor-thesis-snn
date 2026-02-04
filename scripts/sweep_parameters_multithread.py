# scripts/sweep_parameters_multithread.py

from pathlib import Path
import itertools
import csv
import copy
import yaml
from datetime import datetime
from typing import Dict, Any, List

import concurrent.futures as cf

from src.experiment.config_loader import load_base_config
from src.experiment.experiment_runner import run_experiment_with_cfg
from src.analysis.summary_metrics import compute_summary_metrics
from src.analysis.util import load_run  # lädt cfg, data, weights_data, weights_over_time


def load_search_config(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def set_by_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    """
    Setze cfg[...] an einem Pfad wie 'synapses.E_to_X.lambda' auf value.
    """
    keys = path.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def format_value_for_name(v: Any) -> str:
    """
    Wert für Experimentnamen sinnvoll formattieren.
    Floats -> g-Format, sonst str().
    """
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def run_single_config(
    param_defs: List[Dict[str, Any]],
    param_values: Dict[str, Any],
    base_cfg: Dict[str, Any],
    result_path: Path,
) -> Dict[str, Any]:
    """
    Eine Parameterkombination ausführen:
    - Config kopieren & Parameter setzen
    - Experiment laufen lassen
    - Run laden & Metriken berechnen
    - Ergebniszeile (Dict) zurückgeben
    """
    cfg = copy.deepcopy(base_cfg)

    # Parameter setzen
    for pdef in param_defs:
        name = pdef["name"]
        path = pdef["path"]
        val  = param_values[name]
        set_by_path(cfg, path, val)

    # Experimentname bauen
    parts = [f"{name}{format_value_for_name(param_values[name])}"
             for name in param_values.keys()]
    exp_name = "sweep_" + "_".join(parts)
    cfg["experiment"]["name"] = exp_name

    # Experiment laufen lassen -> run_dir (unter result_path)
    run_dir = run_experiment_with_cfg(cfg, result_path, multithreaded=True)

    # Run laden und Metriken berechnen
    cfg_run, data, weights_data, weights_over_time, stim_metadata = load_run(run_dir)
    metrics = compute_summary_metrics(
        cfg_run,
        data,
        weights_over_time,
        stim_metadata=stim_metadata,
    )

    row = {
        "run_dir": str(run_dir),
    }
    # Parameter in row schreiben
    for name, value in param_values.items():
        row[name] = value

    # Metriken anhängen
    row["mean_rate_Hz"] = metrics["mean_rate_Hz"]
    row["mean_cv"]      = metrics["mean_cv"]
    row["mean_R_all"]   = metrics["mean_R_all"]
    row["mean_K"]       = metrics["mean_K"]

    return row


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generic multi-parameter sweep for SNN experiments."
    )
    parser.add_argument(
        "--search-config",
        type=str,
        default="config/search_learning_rates.yaml",
        help="Path to search config YAML (default: config/search_learning_rates.yaml)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Number of parallel worker processes.",
    )
    args = parser.parse_args()

    search_cfg = load_search_config(Path(args.search_config))

    base_cfg_path = Path(search_cfg["base_config"])
    base_cfg = load_base_config(base_cfg_path)

    # Parameterdefinitionen laden
    param_defs: List[Dict[str, Any]] = search_cfg["parameters"]

    # Liste der Namen und Werte-Listen für kartesisches Produkt
    param_names = [p["name"] for p in param_defs]
    value_lists = [p["values"] for p in param_defs]

    # --------- eigener Ordner für diesen Sweep-Durchlauf ---------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = Path("results") / f"sweep_run_{timestamp}"
    sweep_root.mkdir(parents=True, exist_ok=True)
    # -------------------------------------------------------------

    # Parametergitter aufbauen: Liste von Dicts {name: value}
    param_grid: List[Dict[str, Any]] = []
    for combo in itertools.product(*value_lists):
        param_grid.append(dict(zip(param_names, combo)))

    results: List[Dict[str, Any]] = []

    max_workers = args.max_workers

    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_single_config,
                param_defs,
                param_values,
                base_cfg,
                sweep_root,
            )
            for param_values in param_grid
        ]

        for fut in cf.as_completed(futures):
            row = fut.result()
            results.append(row)
            print(row)

    if results:
        out_csv = sweep_root / "sweep_results.csv"

        # Feldnamen: run_dir + alle Parameter + Metriken
        fieldnames = ["run_dir"] + param_names + [
            "mean_rate_Hz", "mean_cv", "mean_R_all", "mean_K"
        ]

        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Wrote sweep results to {out_csv}")
    else:
        print("No sweep results (grid was empty?).")


if __name__ == "__main__":
    main()