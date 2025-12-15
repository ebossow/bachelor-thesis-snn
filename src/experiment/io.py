# src/experiment/io.py

from pathlib import Path
import datetime as dt
import yaml
import numpy as np
import nest
from typing import Dict, Any


def make_run_dir(root: Path, experiment_name: str) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / experiment_name / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run(cfg: Dict[str, Any],
             data: Dict[str, Any],
             run_dir: Path,
             pops) -> None:
    # Config
    with (run_dir / "config_resolved.yaml").open("w") as f:
        yaml.safe_dump(cfg, f)

    # Spikes pro Population
    for key in ("spikes_E", "spikes_IH", "spikes_IA"):
        if key in data and data[key] is not None:
            arr = data[key]
            np.savez_compressed(
                run_dir / f"{key}.npz",
                times=arr["times"],
                senders=arr["senders"],
            )

    meta = {
        "timestamp": dt.datetime.now().isoformat(),
        "experiment_name": cfg["experiment"]["name"],
        "simtime_ms": cfg["experiment"]["simtime_ms"],
    }
    with (run_dir / "metadata.yaml").open("w") as f:
        yaml.safe_dump(meta, f)

    all_neurons = pops["E"] + pops["IH"] + pops["IA"]
    conns = nest.GetConnections(all_neurons, all_neurons)
    sources = np.array(list(conns.sources()), dtype=int)      # pre-synaptic gids (1D array)
    targets = np.array(list(conns.targets()), dtype=int)      # post-synaptic gids
    weights = np.array(conns.get("weight"), dtype=float)      # list/array of weights

    np.savez_compressed(
        run_dir / "weights_final.npz",
        sources=sources,
        targets=targets,
        weights=np.array(weights, float),
    )

    # Gewichtstrajektorie (optional)
    wtraj = data.get("weights_trajectory", None)
    if wtraj is not None:
        payload: Dict[str, Any] = {
            "times": wtraj["times"],
            "weights": wtraj["weights"],
        }
        if "sources" in wtraj:
            payload["sources"] = wtraj["sources"]
        if "targets" in wtraj:
            payload["targets"] = wtraj["targets"]
        np.savez_compressed(run_dir / "weights_trajectory.npz", **payload)
    