# src/setup/run_simulation.py

from typing import Dict, Any

import nest
import numpy as np


def run_simulation(simtime_ms: float,
                   recording_devices: Dict[str, Any]) -> Dict[str, Any]:
    """
    NEST-Simulation ausführen und Rohdaten aus den Recordern holen.
    Hier erstmal einfache Einmal-Simulation ohne Chunks/Decay.
    """

    nest.Simulate(simtime_ms)

    data: Dict[str, Any] = {}

    for name, dev in recording_devices.items():
        events = nest.GetStatus(dev, "events")[0]
        data[name] = {
            "times": np.array(events["times"]),
            "senders": np.array(events["senders"]),
        }

    return data