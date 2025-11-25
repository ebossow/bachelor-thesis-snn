# src/setup/stimulation.py

from typing import Dict, Any

import nest
import numpy as np


def setup_stimulation(populations: Dict[str, Any],
                      stim_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    DC-Stimulation und Muster (pulse/gap/split) für die Populationen anlegen.
    """
    #TODO: setup function where number of splits can change
    if stim_cfg["dc"]["enabled"]:
        t_on = stim_cfg["dc"]["t_on_ms"]
        t_off = stim_cfg["dc"]["t_off_ms"]
        rnd_mean_pA = stim_cfg["dc"]["rnd_mean_pA"]
        rnd_std_pA = stim_cfg["dc"]["rnd_std_pA"]

        pulse_ms = stim_cfg["pattern"]["pulse_ms"]
        gap_ms = stim_cfg["pattern"]["gap_ms"]
        split = stim_cfg["pattern"]["split"]
        if stim_cfg["pattern"]["use_seed"]:
            rng_seed = stim_cfg["pattern"]["rng_seed"]
        else:
            rng_seed = None

        cycle_ms = pulse_ms + gap_ms

        pop_E = populations["E"]
        pop_I = populations["IH"] + populations["IA"]

        e_A, e_B = _split_half(pop_E)
        i_A, i_B = _split_half(pop_I)

        rng = np.random.default_rng(rng_seed)

        created = {"E": [], "I": []}
        t = float(t_on)

        # safety
        if t_off <= t_on or pulse_ms <= 0 or gap_ms < 0:
            return created
        

        while t < t_off:
            # verbleibende Zeit; ggf. letzten Puls einkürzen
            remaining = t_off - t
            if remaining <= 0:
                break
            this_pulse = min(pulse_ms, remaining)
            
            choice_E = choice_I = rng.integers(0, split)

            targets = {
                "E":  e_A if choice_E  == 0 else e_B,
                "I": i_A if choice_I == 0 else i_B
            }

            # DC-Generatoren für diesen Puls erzeugen und verbinden
            for key, tgt in targets.items():
                if len(tgt) == 0:
                    continue  # falls ungerade Größe und eine Hälfte leer wäre
                rnd_norm_pA = np.random.normal(rnd_mean_pA, rnd_std_pA)
                dc = nest.Create("dc_generator", params={"amplitude": rnd_norm_pA, "start": t, "stop": t + this_pulse})
                nest.Connect(dc, tgt)
                created[key].append(dc)

            # zum nächsten Zyklus springen (Puls + Pause)
            t += cycle_ms

        return created
    return None


def _split_half(pop):
    n = len(pop)
    return pop[: n // 2], pop[n // 2 :]