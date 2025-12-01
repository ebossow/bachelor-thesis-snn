# src/setup/stimulation.py

from typing import Dict, Any

import nest
import numpy as np
import matplotlib.pyplot as plt

def setup_stimulation(populations: Dict[str, Any],
                      stim_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stimulation für die Populationen anlegen.

    - Wenn mnist.enabled = True:
        * MNIST-Strom auf E-Population
        * optional DC-Block-Stimulus weiterhin auf Inhibitoren (IH+IA)
    - Wenn mnist.enabled = False und dc.enabled = True:
        * alter DC-Block-Stimulus auf E und I (wie bisher)

    Rückgabe: Dict mit optionalen Einträgen "mnist" und "dc".
    """
    created: Dict[str, Any] = {}

    mnist_cfg = stim_cfg.get("mnist", {})
    mnist_enabled = bool(mnist_cfg.get("enabled", False))

    dc_cfg = stim_cfg.get("dc", {})
    pattern_cfg = stim_cfg.get("pattern", {})
    dc_enabled = bool(dc_cfg.get("enabled", False))

    # Gemeinsame Pulsfolge für alle Stimuli, wenn gebraucht
    pulses = []
    if mnist_enabled or dc_enabled:
        pulses = _build_random_pulses(pattern_cfg)

    # 1) MNIST-Stimulation auf E
    if mnist_enabled:
        created["mnist"] = _setup_mnist_stimulation(
            populations=populations,
            pattern_cfg=pattern_cfg,
            pulses=pulses,
            mnist_cfg=mnist_cfg
        )

    # 2) DC-Block-Stimulation
    if dc_enabled:
        # Wenn MNIST aktiv ist: DC nur auf Inhibitoren (IH+IA)
        only_inhibitory = mnist_enabled

        created["dc"] = _setup_dc_block_stimulation(
            populations=populations,
            dc_cfg=dc_cfg,
            pattern_cfg=pattern_cfg,
            pulses=pulses,
            only_inhibitory=only_inhibitory
        )

    return created


# --------------------------------------------------------------------------
# DC-Block-Stimulation (vorheriger Code, leicht verallgemeinert)
# --------------------------------------------------------------------------

def _setup_dc_block_stimulation(populations: Dict[str, Any],
                                dc_cfg: Dict[str, Any],
                                pattern_cfg: Dict[str, Any],
                                pulses: list[tuple[float, float, int]],
                                only_inhibitory: bool = False) -> Dict[str, Any]:
    """
    DC-Stimulation mit Puls/Gap/Random-Amplitude.

    - Wenn only_inhibitory = False:
        E + (IH+IA) wie bisher.
    - Wenn only_inhibitory = True:
        nur IH+IA werden stimuliert (E bleibt frei, z.B. für MNIST).
    """
    
    rnd_mean_pA = float(dc_cfg["rnd_mean_pA"])
    rnd_std_pA = float(dc_cfg["rnd_std_pA"])
    if pattern_cfg.get("use_seed", False):
        rng_seed = int(pattern_cfg.get("rng_seed", 0))
    else:
        rng_seed = None

    pop_E = populations["E"] if not only_inhibitory else None
    pop_I = populations["IH"] + populations["IA"]

    if pop_E is not None:
        e_A, e_B = _split_half(pop_E)
    else:
        e_A, e_B = None, None

    i_A, i_B = _split_half(pop_I)

    created = {"E": [], "I": []}

    if not pulses:
        return created

    for (t_start, t_end, pattern_idx) in pulses:
        targets = {}

        # E bekommt denselben Pattern-Index wie I, falls E hier noch mit DC laufen soll
        if pop_E is not None:
            targets["E"] = e_A if pattern_idx == 0 else e_B
        targets["I"] = i_A if pattern_idx == 0 else i_B

        for key, tgt in targets.items():
            if tgt is None or len(tgt) == 0:
                continue
            rnd_norm_pA = float(np.random.normal(rnd_mean_pA, rnd_std_pA))
            dc = nest.Create(
                "dc_generator",
                params={"amplitude": rnd_norm_pA, "start": t_start, "stop": t_end},
            )
            nest.Connect(dc, tgt)
            created[key].append(dc)

    return created


def _split_half(pop):
    n = len(pop)
    return pop[: n // 2], pop[n // 2 :]


# --------------------------------------------------------------------------
# MNIST-Stimulation auf E
# --------------------------------------------------------------------------

def _setup_mnist_stimulation(populations: Dict[str, Any],
                             pattern_cfg: Dict[str, Any],
                             pulses: list[tuple[float, float, int]],
                             mnist_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    MNIST-basierte Stimulation:

    - Nimmt 2 Bilder (Paths in mnist_cfg["image_paths"]) mit shape (H,W),
      H=W=image_size, Werte [0,255].
    - Mapped die (image_size^2) Pixel auf die ersten N_pix E-Neuronen.
    - Erzeugt pro E-Neuron einen step_current_generator mit:
        Bild1-Puls, Gap, Bild2-Puls.
    """
    image_size = int(mnist_cfg["image_size"])
    n_pixels = image_size * image_size

    pixel_I_max = float(mnist_cfg["pixel_current_pA"])

    if pattern_cfg.get("use_seed", False):
        rng_seed = int(pattern_cfg.get("rng_seed", 0))
    else:
        rng_seed = None


    # Bilder laden und zu Vektoren normieren
    images = np.load(mnist_cfg["image_path"])   # (70000, 28, 28)
    labels = np.load(mnist_cfg["label_path"])   # (70000,)

    # Index eines '0'-Beispiels
    idx0 = np.where(labels == 0)[0][0]
    # Index eines '1'-Beispiels
    idx1 = np.where(labels == 1)[0][0]

    img_0 = images[idx0]   # (28, 28), '0'
    img_1 = images[idx1]   # (28, 28), '1'

    img_0 = img_0.reshape(-1) / 255.0  # in [0,1]
    img_1 = img_1.reshape(-1) / 255.0  # in [0,1]
    images = [img_0, img_1]
    # images[k][j] = normierter Pixelwert des j-ten Pixels im k-ten Bild
    print("Loaded MNIST images with shapes:", img_0.shape, img_1.shape)

    #fig, axes = plt.subplots(1, 2)
    #axes[0].imshow(img_0.reshape(image_size, image_size), cmap="gray")
    #axes[1].imshow(img_1.reshape(image_size, image_size), cmap="gray")
    #plt.show()


    pop_E = populations["E"]
    N_E = len(pop_E)
    n_target = min(N_E, n_pixels)

    created = {"E": []}

    if not pulses:
        return created

    # Für jedes der n_target E-Neuronen einen SCG mit kompletter Zeitreihe
    for j in range(n_target):
        amplitude_times: list[float] = []
        amplitude_values: list[float] = []

        for (t_start, t_end, pattern_idx) in pulses:
            x = images[pattern_idx][j]  # Pixel in [0,1]
            I = x * pixel_I_max

            amplitude_times.extend([t_start, t_end])
            amplitude_values.extend([I, 0.0])

        if not amplitude_times:
            continue

        scg = nest.Create(
            "step_current_generator",
            params={
                "amplitude_times": amplitude_times,
                "amplitude_values": amplitude_values,
            },
        )
        nest.Connect(scg, pop_E[j])
        created["E"].append(scg)

    return created


# --------------------------------------------------------------------------
# Pulse Sequenz erstellen
# --------------------------------------------------------------------------
def _build_random_pulses(pattern_cfg: Dict[str, Any]) -> list[tuple[float, float, int]]:
    """
    Erzeuge eine Liste von Pulsen mit zufälligen Pattern-Indizes.

    Rückgabe: Liste von (t_start, t_end, pattern_idx),
    wobei pattern_idx in [0, split) liegt.
    """
    t_on = float(pattern_cfg["t_on_ms"])
    t_off = float(pattern_cfg["t_off_ms"])
    pulse_ms = float(pattern_cfg["pulse_ms"])
    gap_ms = float(pattern_cfg["gap_ms"])
    split = int(pattern_cfg["split"])

    if pattern_cfg.get("use_seed", False):
        rng_seed = int(pattern_cfg.get("rng_seed", 0))
    else:
        rng_seed = None

    rng = np.random.default_rng(rng_seed)

    pulses: list[tuple[float, float, int]] = []

    # safety
    if t_off <= t_on or pulse_ms <= 0.0 or gap_ms < 0.0:
        return pulses

    cycle_ms = pulse_ms + gap_ms
    t = float(t_on)

    while t < t_off:
        remaining = t_off - t
        if remaining <= 0.0:
            break
        this_pulse = min(pulse_ms, remaining)

        pattern_idx = int(rng.integers(0, split))  # 0..split-1
        pulses.append((t, t + this_pulse, pattern_idx))

        t += cycle_ms

    return pulses