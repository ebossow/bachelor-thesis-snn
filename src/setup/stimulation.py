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

    - E: 28x28-Pixel (784) -> erste N_E Excitatory-Neurone (1 Pixel / Neuron)
    - IH+IA: downsampled 10x10-Pixel (100) -> erste 100 inhibitorische Neurone
             (IH + IA zusammen). Beide nutzen dieselben Puls-/Pattern-Indizes.
    """

    image_size = int(mnist_cfg["image_size"])  # 28
    n_pixels_E = image_size * image_size       # 784

    pixel_I_max_E   = float(mnist_cfg["pixel_current_pA"])
    pixel_I_max_inh = float(mnist_cfg.get("pixel_current_pA_inh",
                                          pixel_I_max_E))  # optional separater Wert

    # MNIST-Bilder und Labels laden
    images_all = np.load(mnist_cfg["image_path"])   # (70000, 28, 28)
    labels     = np.load(mnist_cfg["label_path"])   # (70000,)

    idx0 = np.where(labels == 0)[0][0]
    idx1 = np.where(labels == 1)[0][0]

    img0_28 = images_all[idx0].astype(float)  # (28, 28)
    img1_28 = images_all[idx1].astype(float)  # (28, 28)

    # --- Bilder für E (28x28 -> 784, normiert 0..1) ---
    img0_flat = img0_28.reshape(-1) / 255.0
    img1_flat = img1_28.reshape(-1) / 255.0
    images_E = [img0_flat, img1_flat]  # images_E[pattern_idx][j]

    # --- Bilder für Inhibitoren (28x28 -> 10x10 -> 100, normiert 0..1) ---
    img0_10 = _downsample_mnist_img(img0_28) / 255.0  # (10, 10)
    img1_10 = _downsample_mnist_img(img1_28) / 255.0  # (10, 10)

    img0_10_flat = img0_10.reshape(-1)  # (100,)
    img1_10_flat = img1_10.reshape(-1)  # (100,)
    images_I = [img0_10_flat, img1_10_flat]  # images_I[pattern_idx][k]

    print("MNIST E-images shape:", img0_flat.shape, img1_flat.shape)
    print("MNIST I-images shape:", img0_10_flat.shape, img1_10_flat.shape)

    pop_E = populations["E"]
    pop_I = populations["IH"] + populations["IA"]

    N_E  = len(pop_E)
    N_I  = len(pop_I)

    n_target_E = min(N_E, n_pixels_E)
    n_pixels_I = 10 * 10
    n_target_I = min(N_I, n_pixels_I)

    created = {"E": [], "I": []}

    if not pulses:
        return created

    # -----------------------------
    # 1) E-Population stimulieren
    # -----------------------------
    for j in range(n_target_E):
        amplitude_times: list[float] = []
        amplitude_values: list[float] = []

        for (t_start, t_end, pattern_idx) in pulses:
            x = images_E[pattern_idx][j]  # in [0,1]
            I = x * pixel_I_max_E

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

    # -----------------------------
    # 2) Inhibitoren (IH+IA) stimulieren: Hälfte A ↔ Bild0, Hälfte B ↔ Bild1
    # -----------------------------
    pop_I = populations["IH"] + populations["IA"]
    N_I   = len(pop_I)

    # halbiere Inhibitoren
    half_I = N_I // 2
    pop_I_A = pop_I[:half_I]      # Hälfte A (Soll-Bild: img0_10_flat)
    pop_I_B = pop_I[half_I:]      # Hälfte B (Soll-Bild: img1_10_flat)

    n_pixels_I = 10 * 10
    n_target_A = min(len(pop_I_A), n_pixels_I)
    n_target_B = min(len(pop_I_B), n_pixels_I)

    # Hälfte A: kodiert Bild 0
    for k in range(n_target_A):
        amplitude_times: list[float] = []
        amplitude_values: list[float] = []

        for (t_start, t_end, pattern_idx) in pulses:
            if pattern_idx == 0:
                # IMG1 aktiv: Hälfte A bekommt Bild0
                x_A = images_I[0][k]          # Pixel von Bild0 in [0,1]
                I_A = x_A * pixel_I_max_inh
            else:
                # IMG2 aktiv: Hälfte A bekommt nichts
                I_A = 0.0

            amplitude_times.extend([t_start, t_end])
            amplitude_values.extend([I_A, 0.0])

        if not amplitude_times:
            continue

        scg = nest.Create(
            "step_current_generator",
            params={
                "amplitude_times": amplitude_times,
                "amplitude_values": amplitude_values,
            },
        )
        nest.Connect(scg, pop_I_A[k])
        created["I"].append(scg)

    # Hälfte B: kodiert Bild 1
    for k in range(n_target_B):
        amplitude_times: list[float] = []
        amplitude_values: list[float] = []

        for (t_start, t_end, pattern_idx) in pulses:
            if pattern_idx == 1:
                # IMG2 aktiv: Hälfte B bekommt Bild1
                x_B = images_I[1][k]          # Pixel von Bild1 in [0,1]
                I_B = x_B * pixel_I_max_inh
            else:
                # IMG1 aktiv: Hälfte B bekommt nichts
                I_B = 0.0

            amplitude_times.extend([t_start, t_end])
            amplitude_values.extend([I_B, 0.0])

        if not amplitude_times:
            continue

        scg = nest.Create(
            "step_current_generator",
            params={
                "amplitude_times": amplitude_times,
                "amplitude_values": amplitude_values,
            },
        )
        nest.Connect(scg, pop_I_B[k])
        created["I"].append(scg)

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


def _downsample_mnist_img(img_28: np.ndarray) -> np.ndarray:
    """
    Einfache 28x28 -> 10x10 Downsample-Funktion via Subsampling
    (gleichmäßig verteilte Sample-Punkte).
    img_28: shape (28, 28)
    Rückgabe: shape (10, 10)
    """
    img_28 = np.asarray(img_28)
    if img_28.shape != (28, 28):
        raise ValueError(f"Expected (28, 28) image, got {img_28.shape}")

    ys = np.linspace(0, 27, 10).astype(int)
    xs = np.linspace(0, 27, 10).astype(int)
    img_10 = img_28[np.ix_(ys, xs)]
    return img_10