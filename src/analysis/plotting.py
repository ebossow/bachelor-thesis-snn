# src/analysis/plotting.py

from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt


def plot_spike_raster(data: Dict[str, Any],
                      cfg: Dict[str, Any]) -> None:
    """
    Rasterplot für alle Populationen.
    Erwartet data["spikes_E"], data["spikes_IH"], data["spikes_IA"]
    mit Feldern "times" und "senders".
    """

    N_E  = cfg["network"]["N_E"]
    N_IH = cfg["network"]["N_IH"]
    N_IA = cfg["network"]["N_IA"]
    N_total = N_E + N_IH + N_IA

    times_E = data["spikes_E"]["times"]
    senders_E = data["spikes_E"]["senders"]
    times_I = np.concatenate((data["spikes_IH"]["times"], data["spikes_IA"]["times"]))
    senders_I = np.concatenate((data["spikes_IH"]["senders"], data["spikes_IA"]["senders"]))

    plt.figure(figsize=(10, 5))
    # excitatory in red
    plt.scatter(times_E, senders_E, s=4, c='red', label='Excitatory (E)')
    # inhibitory in blue (shifted IDs so that they appear above)
    plt.scatter(times_I, senders_I, s=4, c='blue', label='Inhibitory (IH + IA)')
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron ID")
    #plt.title("Spikes: Excitatory (red) vs Inhibitory (blue)")
    plt.legend(loc='upper left')
    #if learning_rates is not None:
    #    lr_text = ", ".join([f"{key}: {value:.3f}" for key, value in learning_rates.items()])
    #    plt.suptitle(f"Learning Rates - {lr_text}", y=0.95, fontsize=10)
    plt.ylim(0, N_E + N_IH + N_IA + 5)
    plt.tight_layout()

    #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #filename = f"./plots/spike_raster_{timestamp}.jpg"
    #plt.savefig(filename, dpi=150)
    plt.show()