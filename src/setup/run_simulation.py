# src/setup/run_simulation.py

from typing import Dict, Any

import nest
import numpy as np


def _apply_weight_decay_clipped(conns,
                                decay_factor: float,
                                clip_min: float,
                                clip_max: float) -> None:
    """
    Multiply all weights in a ConnectionCollection by decay_factor
    and clip to [clip_min, clip_max].
    """
    # weights aligned with connections
    weights = np.array(conns.get("weight"), dtype=float)
    weights *= decay_factor
    np.clip(weights, clip_min, clip_max, out=weights)
    conns.set({"weight": weights})


def run_simulation(
    simtime_ms: float,
    recording_devices: Dict[str, Any],
    populations: Dict[str, Any],
    synapse_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    NEST-Simulation in Chunks ausführen und am Ende Rohdaten aus den Recordern holen.

    - Wenn synapse_cfg["decay"]["enabled"] = True:
        Simulation in Schritten von chunk_ms, nach jedem Chunk Weight-Decay.
    - Sonst: eine einfache Simulate(simtime_ms).

    Parameters
    ----------
    simtime_ms : float
        Gesamtsimulationszeit in ms.
    recording_devices : dict
        Spike-Recorder etc., wie von setup.recording.setup_recording geliefert.
    populations : dict
        Neuronenpopulationen, z.B. {"E": NodeCollection, "IH": ..., "IA": ...}.
    synapse_cfg : dict
        config["synapses"]-Block (enthält auch "decay").
    """

    decay_cfg = synapse_cfg.get("weight_decay", {})
    decay_enabled = decay_cfg.get("enabled", False)

    if decay_enabled:
        decay_factor = float(decay_cfg.get("decay_factor", 1.0))
        chunk_ms = float(decay_cfg.get("chunk_ms", simtime_ms))

        if chunk_ms <= 0.0:
            raise ValueError("chunk_ms must be > 0 when decay is enabled")

        # Populationen
        E  = populations["E"]
        IH = populations["IH"]
        IA = populations["IA"]

        # Synapsenmodell-Namen: hier sollten die Modelle stehen,
        # die du in connect_synapses() tatsächlich benutzt.
        # Typisch: copy_model_name aus der Config.
        model_E  = synapse_cfg["E_to_X"]["copy_model_name"]   # z.B. "stdp_ex_asym"
        model_IH = synapse_cfg["IH_to_X"]["copy_model_name"]  # z.B. "stdp_inh_sym"
        model_IA = synapse_cfg["IA_to_X"]["copy_model_name"]  # z.B. "stdp_inh_sym_anti"

        # ConnectionCollections pro Synapsetyp (wie im Notebook)
        conn_E  = nest.GetConnections(source=E,  synapse_model=model_E)
        conn_IH = nest.GetConnections(source=IH, synapse_model=model_IH)
        conn_IA = nest.GetConnections(source=IA, synapse_model=model_IA)

        # Wmax-Werte aus Config (beachte: inh negativ)
        Wmax_exc      = float(synapse_cfg["E_to_X"]["Wmax_factor"]) * synapse_cfg["base_Wmax"]
        Wmax_inh_hebb = float(synapse_cfg["IH_to_X"]["Wmax_factor"]) * synapse_cfg["base_Wmax"]
        Wmax_inh_anti = float(synapse_cfg["IA_to_X"]["Wmax_factor"]) * synapse_cfg["base_Wmax"]

        # Schritte
        n_full_chunks = int(simtime_ms // chunk_ms)
        remainder = simtime_ms - n_full_chunks * chunk_ms

        for _ in range(n_full_chunks):
            nest.Simulate(chunk_ms)

            # excitatory: [0, Wmax_exc]
            _apply_weight_decay_clipped(
                conn_E,
                decay_factor=decay_factor,
                clip_min=0.0,
                clip_max=Wmax_exc,
            )

            # inhibitory hebbian: [Wmax_inh_hebb, 0]
            _apply_weight_decay_clipped(
                conn_IH,
                decay_factor=decay_factor,
                clip_min=Wmax_inh_hebb,
                clip_max=0.0,
            )

            # inhibitory anti-hebbian: [Wmax_inh_anti, 0]
            _apply_weight_decay_clipped(
                conn_IA,
                decay_factor=decay_factor,
                clip_min=Wmax_inh_anti,
                clip_max=0.0,
            )

        if remainder > 0.0:
            nest.Simulate(remainder)
    else:
        # kein Decay: eine einzige Simulate
        nest.Simulate(simtime_ms)

    # Eventdaten einsammeln (wie bisher)
    data: Dict[str, Any] = {}

    for name, dev in recording_devices.items():
        events = nest.GetStatus(dev, "events")[0]
        data[name] = {
            "times": np.array(events["times"]),
            "senders": np.array(events["senders"]),
        }

    return data