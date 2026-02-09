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


def _resolve_snapshot_times(cfg: Dict[str, Any]) -> list[float]:
    exp_cfg = cfg.get("experiment", {})
    total_ms = float(exp_cfg.get("simtime_ms", 0.0))
    times: set[float] = {0.0, total_ms}

    pattern_cfg = cfg.get("stimulation", {}).get("pattern", {})
    for key in ("t_on_ms", "t_off_ms"):
        val = pattern_cfg.get(key)
        if val is not None:
            times.add(float(val))

    phase_markers = exp_cfg.get("phase_markers_ms") or {}
    if isinstance(phase_markers, dict):
        for val in phase_markers.values():
            if isinstance(val, (int, float)):
                times.add(float(val))

    syn_cfg = cfg.get("synapses", {})
    weight_decay_cfg = syn_cfg.get("weight_decay", {}) if isinstance(syn_cfg, dict) else {}
    decay_enabled = bool(weight_decay_cfg.get("enabled", False))
    post_interval_ms = 100.0
    if pattern_cfg.get("t_off_ms") is not None:
        t_off = float(pattern_cfg["t_off_ms"])
        if t_off < total_ms:
            current = max(t_off, 0.0)
            while current <= total_ms:
                times.add(round(current, 6))
                current += post_interval_ms

    snapshot_times = sorted(t for t in times if 0.0 <= t <= total_ms)
    return snapshot_times


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
    stim_devs, stim_metadata = setup_stimulation(pops, cfg["stimulation"])
    rec_devs = setup_recording(pops, cfg["analysis"])

    snapshot_times = _resolve_snapshot_times(cfg)

    data = run_simulation(
        simtime_ms=cfg["experiment"]["simtime_ms"],
        recording_devices=rec_devs,
        populations=pops,
        synapse_cfg=cfg["synapses"],
        record_weight_trajectory=True,  # für K(t)
        snapshot_times_ms=snapshot_times,
        phase_schedule=cfg["experiment"].get("phase_schedule"),
    )

    #run_root = Path("results")
    if multithreaded:
        run_dir = make_run_dir(result_path, cfg["experiment"]["name"])
        save_run(cfg, data, run_dir, pops, stim_metadata=stim_metadata)
        return run_dir
    else:
        #run_dir = make_run_dir(result_path, cfg["experiment"]["name"])
        save_run(cfg, data, result_path, pops, stim_metadata=stim_metadata)
        return result_path
    