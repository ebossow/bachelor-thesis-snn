# src/experiment/experiment_runner.py

from pathlib import Path
from typing import Dict, Any

from src.setup.kernel import init_kernel
from src.setup.network import build_populations, connect_synapses
from src.setup.stimulation import setup_stimulation
from src.setup.recording import setup_recording
from src.setup.run_simulation import run_simulation
from src.experiment.io import make_run_dir, save_run
from src.setup.scaling import apply_post_init_scaling


def run_experiment_with_cfg(cfg: Dict[str, Any], result_path, multithreaded=False) -> Path:
    """
    Führe einen kompletten Run mit der gegebenen Config aus
    und gib den run_dir zurück.
    """
    init_kernel(cfg["experiment"])

    pops = build_populations(
        network_cfg=cfg["network"],
        noise_cfg=cfg["noise"],
        neuron_model_cfg=cfg["neuron_model"],
        excitability_cfg=cfg["neuron_excitability"],
    )
    connect_synapses(pops, cfg["synapses"])
    apply_post_init_scaling(pops, cfg.get("scaling"))
    stim_devs = setup_stimulation(pops, cfg["stimulation"])
    rec_devs = setup_recording(pops, cfg["analysis"])

    snapshot_times = [0, cfg["stimulation"]["pattern"]["t_on_ms"], cfg["stimulation"]["pattern"]["t_off_ms"]]

    data = run_simulation(
        simtime_ms=cfg["experiment"]["simtime_ms"],
        recording_devices=rec_devs,
        populations=pops,
        synapse_cfg=cfg["synapses"],
        record_weight_trajectory=True,  # für K(t)
        snapshot_times_ms=snapshot_times,
    )

    #run_root = Path("results")
    if multithreaded:
        run_dir = make_run_dir(result_path, cfg["experiment"]["name"])
        save_run(cfg, data, run_dir, pops)
        return run_dir
    else:
        #run_dir = make_run_dir(result_path, cfg["experiment"]["name"])
        save_run(cfg, data, result_path, pops)
        return result_path
    