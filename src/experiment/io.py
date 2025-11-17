# src/experiment/io.py

from pathlib import Path
import datetime as dt
import yaml
import numpy as np
from typing import Dict, Any


def make_run_dir(root: Path, experiment_name: str) -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / experiment_name / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_run(cfg: Dict[str, Any],
             data: Dict[str, Any],
             run_dir: Path) -> None:
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