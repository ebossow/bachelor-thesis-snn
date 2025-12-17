# src/setup/recording.py

from typing import Dict, Any

import nest


def setup_recording(populations: Dict[str, Any],
                    analysis_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Spike-Detektoren usw. für alle Populationen anlegen.
    """

    devices: Dict[str, Any] = {}

    for name, gids in populations.items():
        if len(gids) == 0:
            continue
        sd = nest.Create("spike_recorder")
        nest.Connect(gids, sd)
        devices[f"spikes_{name}"] = sd

    # Weitere Recorder (Membranpotenziale, Gewichte) kannst du hier ergänzen.
    return devices