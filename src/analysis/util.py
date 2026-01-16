import json
import numpy as np
import yaml
from pathlib import Path

def combine_spikes(data, keys, allowed_senders=None):
    times_list = []
    senders_list = []

    allowed_array = None
    if allowed_senders is not None:
        allowed_unique = sorted({int(s) for s in allowed_senders})
        allowed_array = np.asarray(allowed_unique, dtype=int)
        if allowed_array.size == 0:
            return {"times": np.array([]), "senders": np.array([])}

    for key in keys:
        sd = data.get(key)
        if sd is None:
            continue
        times = sd["times"]
        senders = sd["senders"]
        if allowed_array is not None:
            mask = np.isin(senders, allowed_array)
            if not np.any(mask):
                continue
            times = times[mask]
            senders = senders[mask]
        times_list.append(times)
        senders_list.append(senders)

    if not times_list:
        return {"times": np.array([]), "senders": np.array([])}

    times = np.concatenate(times_list)
    senders = np.concatenate(senders_list)
    return {"times": times, "senders": senders}

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

    weights_file = run_dir / "weights_final.npz"
    weights_data = None
    if weights_file.exists():
        d = np.load(weights_file, allow_pickle=True)
        weights_data = {
            "sources": d["sources"],
            "targets": d["targets"],
            "weights": d["weights"],
        }

    wtraj_file = run_dir / "weights_trajectory.npz"
    weights_over_time = None
    if wtraj_file.exists():
        d = np.load(wtraj_file)
        weights_over_time = {
            "times": d["times"],
            "weights": d["weights"],  # shape (n_snap, M)
        }
        if "sources" in d.files:
            weights_over_time["sources"] = d["sources"]
        if "targets" in d.files:
            weights_over_time["targets"] = d["targets"]

    stim_metadata = load_stimulation_metadata(run_dir)

    return cfg, data, weights_data, weights_over_time, stim_metadata


def load_stimulation_metadata(run_dir: Path):
    path = run_dir / "stimulation_metadata.json"
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)

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