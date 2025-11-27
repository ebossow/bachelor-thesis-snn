# src/setup/run_simulation.py

from typing import Dict, Any

import nest
import numpy as np
import time


def _apply_weight_decay_clipped(conns,
                                decay_factor: float,
                                clip_min: float,
                                clip_max: float) -> None:
    """
    Multiply all weights in a ConnectionCollection by decay_factor
    and clip to [clip_min, clip_max].
    """
    weights = np.array(conns.get("weight"), dtype=float)
    weights *= decay_factor
    np.clip(weights, clip_min, clip_max, out=weights)
    conns.set({"weight": weights})


def run_simulation(
    simtime_ms: float,
    recording_devices: Dict[str, Any],
    populations: Dict[str, Any],
    synapse_cfg: Dict[str, Any],
    record_weight_trajectory: bool = False,
) -> Dict[str, Any]:
    """
    NEST-Simulation in Chunks ausführen und Rohdaten + optional Gewichtstrajektorie holen.

    - Wenn synapse_cfg["weight_decay"]["enabled"] = True:
        Simulation in Schritten von chunk_ms, nach jedem n-ten Chunk Weight-Decay.
    - Sonst: eine einfache Simulate(simtime_ms).

    Wenn record_weight_trajectory=True, werden Gewichte von E-, IH- und IA-Synapsen
    regelmäßig gesampelt und in data["weights_trajectory"] zurückgegeben.
    """
    sim_start = time.time()

    decay_cfg = synapse_cfg.get("weight_decay", {})
    decay_enabled = decay_cfg.get("enabled", False)

    # Populationen
    E  = populations["E"]
    IH = populations["IH"]
    IA = populations["IA"]

    # Synapsenmodell-Namen (müssen zu connect_synapses passen)
    model_E  = synapse_cfg["E_to_X"]["copy_model_name"]   # z.B. "stdp_ex_asym"
    model_IH = synapse_cfg["IH_to_X"]["copy_model_name"]  # z.B. "stdp_inh_sym"
    model_IA = synapse_cfg["IA_to_X"]["copy_model_name"]  # z.B. "stdp_inh_sym_anti"

    # ConnectionCollections pro Synapsetyp (brauchen wir für Decay und für Trajektorie)
    conn_E  = nest.GetConnections(source=E,  synapse_model=model_E)
    conn_IH = nest.GetConnections(source=IH, synapse_model=model_IH)
    conn_IA = nest.GetConnections(source=IA, synapse_model=model_IA)

    # Wmax-Werte aus Config (beachte: inh negativ)
    base_Wmax = float(synapse_cfg["base_Wmax"])
    Wmax_exc      = float(synapse_cfg["E_to_X"]["Wmax_factor"])  * base_Wmax
    Wmax_inh_hebb = float(synapse_cfg["IH_to_X"]["Wmax_factor"]) * base_Wmax
    Wmax_inh_anti = float(synapse_cfg["IA_to_X"]["Wmax_factor"]) * base_Wmax

    # Gewichtstrajektorie (optional)
    weight_times: list[float] = []
    weight_snapshots: list[np.ndarray] = []

    def _snapshot_weights(current_time_ms: float) -> None:
        if not record_weight_trajectory:
            return
        wE  = np.array(conn_E.get("weight"),  dtype=float)
        wIH = np.array(conn_IH.get("weight"), dtype=float)
        wIA = np.array(conn_IA.get("weight"), dtype=float)
        w_all = np.concatenate([wE, wIH, wIA])
        weight_times.append(current_time_ms)
        weight_snapshots.append(w_all)

    current_time = 0.0
    _snapshot_weights(current_time)  # initialer Snapshot bei t = 0

    if decay_enabled:
        every_n = int(decay_cfg.get("every_n_chunks", 1))
        decay_factor = float(decay_cfg.get("decay_factor", 1.0))
        chunk_ms = float(decay_cfg.get("chunk_ms", simtime_ms))

        if chunk_ms <= 0.0:
            raise ValueError("chunk_ms must be > 0 when decay is enabled")

        n_full_chunks = int(simtime_ms // chunk_ms)
        remainder = simtime_ms - n_full_chunks * chunk_ms

        for i in range(n_full_chunks):
            nest.Simulate(chunk_ms)
            current_time += chunk_ms

            # ggf. Decay nur alle every_n Chunks
            if (i + 1) % every_n == 0:
                factor = decay_factor ** every_n  # effektiver Faktor für diesen Block
                _apply_weight_decay_clipped(conn_E,  factor, 0.0, Wmax_exc)
                #_apply_weight_decay_clipped(conn_IH, factor, Wmax_inh_hebb, 0.0)
                #_apply_weight_decay_clipped(conn_IA, factor, Wmax_inh_anti, 0.0)

            _snapshot_weights(current_time)

        if remainder > 0.0:
            nest.Simulate(remainder)
            current_time += remainder
            _snapshot_weights(current_time)

    else:
        # kein Decay: eine einzige Simulate
        nest.Simulate(simtime_ms)
        current_time = simtime_ms
        _snapshot_weights(current_time)

    # Eventdaten einsammeln (wie bisher)
    data: Dict[str, Any] = {}

    for name, dev in recording_devices.items():
        events = nest.GetStatus(dev, "events")[0]
        data[name] = {
            "times": np.array(events["times"]),
            "senders": np.array(events["senders"]),
        }

    # Gewichtstrajektorie anhängen, falls aufgezeichnet
    if record_weight_trajectory and weight_snapshots:
        data["weights_trajectory"] = {
            "times": np.array(weight_times, dtype=float),                # (n_snap,)
            "weights": np.stack(weight_snapshots, axis=0).astype(float)  # (n_snap, M)
        }

    sim_end = time.time()
    print(f"Simulation completed in {sim_end - sim_start:.2f} seconds.")

    return data