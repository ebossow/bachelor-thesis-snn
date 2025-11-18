# scripts/run_experiment.py
import argparse
from pathlib import Path

from src.experiment.config_loader import load_base_config
from src.experiment.io import make_run_dir, save_run
from src.setup.kernel import init_kernel
from src.setup.network import build_populations, connect_synapses
from src.setup.stimulation import setup_stimulation
from src.setup.recording import setup_recording
from src.setup.run_simulation import run_simulation


from src.analysis.plotting import plot_spike_raster

from src.analysis.metrics import instantaneous_rates, population_rate, cv_isi

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("config/base.yaml"))
    p.add_argument("--base_Wmax", type=float, default=None)
    p.add_argument("--lambda_exc", type=float, default=None)
    p.add_argument("--eta_ih", type=float, default=None)
    p.add_argument("--eta_ia", type=float, default=None)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    # 1) Config laden
    cfg_path = Path("config/base.yaml")
    cfg = load_base_config(cfg_path)

    # load arguments from cli
    syn_cfg = cfg["synapses"]
    if args.base_Wmax is not None:
        syn_cfg["base_Wmax"] = args.base_Wmax
    if args.lambda_exc is not None:
        syn_cfg["E_to_X"]["synapse_parameter"]["lambda"] = args.lambda_exc
    if args.eta_ih is not None:
        syn_cfg["IH_to_X"]["synapse_parameter"]["eta"] = args.eta_ih
    if args.eta_ia is not None:
        syn_cfg["IA_to_X"]["synapse_parameter"]["eta"] = args.eta_ia

    # 2) Kernel initialisieren
    init_kernel(cfg["experiment"])

    # 3) Netzwerk aufbauen
    pops = build_populations(
        network_cfg=cfg["network"],
        noise_cfg=cfg["noise"],
        neuron_model_cfg=cfg["neuron_model"],
        excitability_cfg=cfg["neuron_excitability"],
    )

    connect_synapses(
        populations=pops,
        synapse_cfg=cfg["synapses"],
    )

    # 4) Stimulation + Recording
    stim_devices = setup_stimulation(
        populations=pops,
        stim_cfg=cfg["stimulation"],
    )

    rec_devices = setup_recording(
        populations=pops,
        analysis_cfg=cfg["analysis"],
    )

    # 5) Simulation
    data = run_simulation(
        simtime_ms=cfg["experiment"]["simtime_ms"],
        recording_devices=rec_devices,
        populations=pops,
        synapse_cfg=cfg["synapses"],
        record_weight_trajectory = True
    )

    run_root = Path("results")
    run_dir = make_run_dir(run_root, cfg["experiment"]["name"])
    save_run(cfg, data, run_dir, pops)

if __name__ == "__main__":
    main()