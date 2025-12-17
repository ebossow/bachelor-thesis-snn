# scripts/run_experiment.py
import argparse
from pathlib import Path

from src.experiment.config_loader import load_base_config
from src.experiment.experiment_runner import run_experiment_with_cfg

from src.analysis.plotting import plot_spike_raster

from src.analysis.metrics import instantaneous_rates, population_rate, cv_isi
from datetime import datetime

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/base.yaml"))
    p.add_argument("--base_Wmax", type=float, default=None)
    p.add_argument("--base_LR", type=float, default=None)
    p.add_argument("--lambda_exc", type=float, default=None)
    p.add_argument("--eta_ih", type=float, default=None)
    p.add_argument("--eta_ia", type=float, default=None)
    p.add_argument("--global_lr", type=float, default=None)
    p.add_argument("--long_run", type=bool, default=False)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    # 1) Config laden
    cfg_path = Path(args.config) if args.config is not None else Path("config/base.yaml")
    cfg = load_base_config(cfg_path)

    # load arguments from cli
    syn_cfg = cfg["synapses"]
    if args.base_Wmax is not None:
        syn_cfg["base_Wmax"] = args.base_Wmax
    if args.base_LR is not None:
        syn_cfg["base_LR"] = args.base_LR
    if args.lambda_exc is not None:
        syn_cfg["E_to_X"]["synapse_parameter"]["lambda"] = args.lambda_exc
    if args.eta_ih is not None:
        syn_cfg["IH_to_X"]["synapse_parameter"]["eta"] = args.eta_ih
    if args.eta_ia is not None:
        syn_cfg["IA_to_X"]["synapse_parameter"]["eta"] = args.eta_ia
    if args.global_lr is not None:
        syn_cfg["global_lr"] = args.global_lr
    if args.long_run:
        syn_cfg["long_run"] = True
    elif args.long_run == False:
        syn_cfg["long_run"] = False

    # 2) Experiment ausführen

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_root = Path("results") / f"base_experiment" / f"run_{timestamp}_{cfg['experiment']['name']}"
    base_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_experiment_with_cfg(cfg, base_root)
    print(f"Experiment finished. Results saved in {run_dir}")

if __name__ == "__main__":
    main()