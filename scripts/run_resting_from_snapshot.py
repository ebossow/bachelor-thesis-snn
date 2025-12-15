from pathlib import Path
import argparse
import copy
import nest

from src.experiment.config_loader import load_base_config
from src.setup.network import build_populations, connect_synapses
from src.setup.run_simulation import run_simulation
from src.setup.ei_scaling import load_scaled_snapshot_from_run, apply_snapshot_weights
from src.setup.recording import setup_recording
from src.experiment.io import make_run_dir, save_run


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_run_dir", type=str, required=True,
                   help="run_*-Ordner des Trainingslaufs (mit weights_trajectory).")
    p.add_argument("--snapshot_time_ms", type=float, default=40000.0)
    p.add_argument("--scale_E", type=float, default=1.0)
    p.add_argument("--scale_IH", type=float, default=1.0)
    p.add_argument("--scale_IA", type=float, default=1.0)
    p.add_argument("--rest_simtime_ms", type=float, default=20000.0)
    p.add_argument("--base_cfg", type=str, default="config/base.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    train_run_dir = Path(args.train_run_dir)

    # 1) Snapshot laden und skalieren
    sources, targets, weights_scaled, cfg_train = load_scaled_snapshot_from_run(
        run_dir=train_run_dir,
        snapshot_time_ms=args.snapshot_time_ms,
        scale_E=args.scale_E,
        scale_IH=args.scale_IH,
        scale_IA=args.scale_IA,
    )

    # 2) Basis-Config laden und als "resting"-Config anpassen
    cfg_rest = copy.deepcopy(cfg_train)
    cfg_rest["experiment"]["name"] = (
        cfg_train["experiment"]["name"]
        + f"_rest_scaled_E{args.scale_E}_IH{args.scale_IH}_IA{args.scale_IA}"
    )
    cfg_rest["experiment"]["simtime_ms"] = float(args.rest_simtime_ms)

    # Stimulation aus
    if "stimulation" in cfg_rest:
        if "dc" in cfg_rest["stimulation"]:
            cfg_rest["stimulation"]["dc"]["enabled"] = False
        if "mnist" in cfg_rest["stimulation"]:
            cfg_rest["stimulation"]["mnist"]["enabled"] = False

    # Plastizität einfrieren: global_lr = 0, evtl. lambda/eta = 0
    syn_cfg = cfg_rest["synapses"]
    syn_cfg["global_lr"] = 0.0
    syn_cfg["E_to_X"]["synapse_parameter"]["lambda"] = 0.0
    syn_cfg["IH_to_X"]["synapse_parameter"]["eta"] = 0.0
    syn_cfg["IA_to_X"]["synapse_parameter"]["eta"] = 0.0
    # Decay aus
    syn_cfg["weight_decay"]["enabled"] = False

    # 3) NEST reset + Netzwerk aufbauen
    nest.ResetKernel()

    pops = build_populations(
        network_cfg=cfg_rest["network"],
        noise_cfg=cfg_rest["noise"],
        neuron_model_cfg=cfg_rest["neuron_model"],
        excitability_cfg=cfg_rest["neuron_excitability"],
    )
    connect_synapses(pops, syn_cfg)

    # 4) Snapshot-Gewichte einspielen (E/I skaliert)
    apply_snapshot_weights(
        populations=pops,
        synapse_cfg=syn_cfg,
        sources_snap=sources,
        targets_snap=targets,
        weights_snap=weights_scaled,
    )

    # 5) Recorder aufsetzen + Simulation
    rec_devs = setup_recording(pops, cfg_rest["analysis"])

    data = run_simulation(
        simtime_ms=cfg_rest["experiment"]["simtime_ms"],
        recording_devices=rec_devs,
        populations=pops,
        synapse_cfg=cfg_rest["synapses"],
        record_weight_trajectory=False,
    )

    # 6) Ergebnisordner + Speichern
    results_root = Path("results/criticality")
    results_root.mkdir(parents=True, exist_ok=True)

    run_dir = make_run_dir(results_root, cfg_rest["experiment"]["name"])

    save_run(cfg_rest, data, run_dir, pops)

    print(f"Saved resting run to {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()