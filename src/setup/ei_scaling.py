import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import nest

from src.analysis.util import load_run

def load_scaled_snapshot_from_run(
    run_dir: Path,
    snapshot_time_ms: float,
    scale_E: float = 1.0,
    scale_IH: float = 1.0,
    scale_IA: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Hole Gewichts-Snapshot bei snapshot_time_ms aus einem Trainings-Run
    und skaliere E/I getrennt.

    Rückgabe:
        sources_scaled, targets_scaled, weights_scaled, cfg
    """
    cfg, data, weights_data, weights_over_time, _ = load_run(run_dir)

    if weights_over_time is None:
        raise ValueError("Run has no weights_trajectory – snapshot_time_ms nicht verfügbar.")

    snap_times = np.asarray(weights_over_time["times"], float)
    snap_weights = np.asarray(weights_over_time["weights"], float)  # (n_snap, M)
    # Index des Snapshots, der snapshot_time_ms am nächsten ist
    idx = int(np.argmin(np.abs(snap_times - snapshot_time_ms)))
    w_vec = snap_weights[idx]  # (M,)

    # Optional: Vergleich mit initialem Snapshot
    w0 = snap_weights[0]
    print(
        "Initial snapshot stats  :",
        "mean =", w0.mean(), "std =", w0.std(),
        "min =", w0.min(), "max =", w0.max()
    )
    print(
        f"Chosen snapshot stats @t={snap_times[idx]:.1f} ms:",
        "mean =", w_vec.mean(), "std =", w_vec.std(),
        "min =", w_vec.min(), "max =", w_vec.max()
    )

    print("Snapshot stats (not scaled):",
      "mean =", w_vec.mean(),
      "std =", w_vec.std(),
      "min =", w_vec.min(),
      "max =", w_vec.max())

    if "sources" not in weights_over_time or "targets" not in weights_over_time:
        raise ValueError(
            "weights_trajectory.npz does not contain sources/targets. "
            "Regenerate the training run with the updated simulation pipeline to store connection indices."
        )

    sources = np.asarray(weights_over_time["sources"], int)
    targets = np.asarray(weights_over_time["targets"], int)

    net_cfg = cfg["network"]
    N_E     = int(net_cfg["N_E"])
    N_IA_1  = int(net_cfg.get("N_IA_1", 0))
    N_IH    = int(net_cfg["N_IH"])
    N_IA_2  = int(net_cfg.get("N_IA_2", 0))

    # GID-Ranges: [E | IA_1 | IH | IA_2]
    gid_E_start   = 1
    gid_E_end     = N_E

    gid_IA1_start = gid_E_end + 1
    gid_IA1_end   = gid_IA1_start + N_IA_1 - 1

    gid_IH_start  = gid_IA1_end + 1
    gid_IH_end    = gid_IH_start + N_IH - 1

    gid_IA2_start = gid_IH_end + 1
    gid_IA2_end   = gid_IA2_start + N_IA_2 - 1

    weights_scaled = w_vec.copy()

    # Skalierung nach Präsyn-Typ
    for k, src in enumerate(sources):
        if gid_E_start <= src <= gid_E_end:
            weights_scaled[k] *= scale_E
        elif gid_IH_start <= src <= gid_IH_end:
            weights_scaled[k] *= scale_IH
        elif gid_IA1_start <= src <= gid_IA1_end or gid_IA2_start <= src <= gid_IA2_end:
            weights_scaled[k] *= scale_IA
        # sonst nichts (falls irgendwann weitere Populationsarten hinzukommen)

    print("Snapshot stats (scaled):",
      "mean =", weights_scaled.mean(),
      "std =", weights_scaled.std(),
      "min =", weights_scaled.min(),
      "max =", weights_scaled.max())
    return sources, targets, weights_scaled, cfg


def apply_snapshot_weights(
    populations: Dict[str, Any],
    synapse_cfg: Dict[str, Any],
    sources_snap: np.ndarray,
    targets_snap: np.ndarray,
    weights_snap: np.ndarray,
) -> None:
    """
    Setze Gewichte im aktuellen NEST-Netzwerk entsprechend
    eines Snapshot-Vektors (sources/targets/weights).
    Annahme: GID-Layout und Konnektivität sind identisch.
    """

    # Map (src_gid, tgt_gid) -> weight
    weights_dict = {
        (int(s), int(t)): float(w)
        for s, t, w in zip(sources_snap, targets_snap, weights_snap)
    }

    E  = populations["E"]
    IH = populations["IH"]
    IA = populations["IA"]

    model_E  = synapse_cfg["E_to_X"]["copy_model_name"]
    model_IH = synapse_cfg["IH_to_X"]["copy_model_name"]
    model_IA = synapse_cfg["IA_to_X"]["copy_model_name"]

    for model, src_pop in [(model_E, E), (model_IH, IH), (model_IA, IA)]:
        conns = nest.GetConnections(source=src_pop, synapse_model=model)
        if len(conns) == 0:
            continue

        src_gids = np.array(conns.get("source"), int)
        tgt_gids = np.array(conns.get("target"), int)

        new_w = np.zeros(len(conns), float)
        for i, (s, t) in enumerate(zip(src_gids, tgt_gids)):
            key = (int(s), int(t))
            if key in weights_dict:
                new_w[i] = weights_dict[key]
            else:
                # Fallback: wenn Verbindung im Snapshot nicht existiert, 0 oder altes Gewicht lassen
                new_w[i] = conns.get("weight")[i]
        conns.set({"weight": new_w})

    all_neurons = E + IH + IA
    conns = nest.GetConnections(all_neurons, all_neurons)
    w_now = np.array(conns.get("weight"), float)
    print("After apply_snapshot_weights:",
        "mean =", w_now.mean(),
        "std =", w_now.std(),
        "min =", w_now.min(),
        "max =", w_now.max())