# src/analysis/metrics.py

from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import numpy.typing as npt


def _build_sender_index(senders: npt.NDArray[np.int_],
                        N_population: int) -> Dict[int, int]:
    """
    Map NEST-GIDs auf lokale Indizes 0..N_population-1.
    Neuronen, die nie spiken, bekommen keinen Eintrag, sind aber
    als leere Zeilen in den Ergebnissen enthalten.
    """
    unique_ids = np.unique(senders)
    if unique_ids.size > N_population:
        raise ValueError(
            f"Mehr verschiedene Sender-IDs ({unique_ids.size}) als N_population={N_population}"
        )

    # Wir gehen davon aus, dass GIDs 1..N_population oder ähnlich sind.
    # Mapping: sortierte GIDs -> 0..k-1
    mapping: Dict[int, int] = {}
    for local_idx, gid in enumerate(unique_ids):
        mapping[int(gid)] = int(local_idx)
    return mapping


def instantaneous_rates(
    times: npt.NDArray[np.floating],
    senders: npt.NDArray[np.int_],
    N_population: int,
    t_start: float,
    t_stop: float,
    bin_size_ms: float,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    v_j(t) nach Gleichung (12), diskretisiert in Bins der Größe bin_size_ms.

    times:   Spike-Zeiten in ms
    senders: NEST-GIDs (oder IDs) der Neuronen
    N_population: Anzahl Neuronen in dieser Population
    t_start, t_stop: Auswertebereich in ms
    bin_size_ms: T; z.B. 50.0 für 0.05 s

    Returns
    -------
    rates:  shape (N_population, n_bins)
    t_centers: shape (n_bins,) – mittlere Zeitpunkte der Bins
    """
    times = np.asarray(times, float)
    senders = np.asarray(senders, int)

    mask = (times >= t_start) & (times < t_stop)
    times = times[mask]
    senders = senders[mask]

    n_bins = int(np.floor((t_stop - t_start) / bin_size_ms))
    if n_bins <= 0:
        raise ValueError("Ungültiger Zeitbereich oder Bin-Breite.")

    # e.g Bin 0: ([t_edges[0], t_edges[1])
    t_edges = t_start + np.arange(n_bins + 1) * bin_size_ms
    t_centers = (t_edges[:-1] + t_edges[1:]) / 2.0

    counts = np.zeros((N_population, n_bins), dtype=float)

    if times.size == 0:
        # keine Spikes: alles 0
        rates = counts  # bleib 0
        mean_rates_per_neuron = np.zeros(N_population, dtype=float)
        mean_rate_population = 0.0
        return rates, t_centers, mean_rate_population, mean_rates_per_neuron

    sender_index = _build_sender_index(senders, N_population)

    # Bin-Index für jeden Spike
    bin_idx = np.floor((times - t_start) / bin_size_ms).astype(int)
    valid_mask = (bin_idx >= 0) & (bin_idx < n_bins)
    bin_idx = bin_idx[valid_mask]
    senders = senders[valid_mask]

    for t_bin, gid in zip(bin_idx, senders):
        j = sender_index.get(int(gid), None)
        if j is None:
            # Neuron ohne Mapping (z.B. nie gespikt) wird ignoriert
            continue
        counts[j, t_bin] += 1.0

    # von Spike-Counts auf Rate (Hz)
    T_bin_s = bin_size_ms / 1000.0
    rates = counts / T_bin_s  # shape (N, n_bins)

    # mittlere Feuerrate pro Neuron über gesamtes Zeitfenster
    total_time_s = n_bins * T_bin_s
    spike_counts_per_neuron = counts.sum(axis=1)          # shape (N,)
    mean_rates_per_neuron = spike_counts_per_neuron / total_time_s  # Hz

    # globaler Mittelwert über alle Neuronen
    mean_rate_population = mean_rates_per_neuron.mean()

    return rates, t_centers, mean_rate_population, mean_rates_per_neuron

def population_rate(
    rates: npt.NDArray[np.floating],
    neuron_indices: npt.NDArray[np.int_] | None = None,
) -> npt.NDArray[np.floating]:
    """
    v_p(t) = (1/N_p) sum_j v_j(t) für eine Population.

    rates: shape (N_total, n_bins)
    neuron_indices: Indizes der Neuronen, die zur Population gehören.
                    Wenn None: alle Zeilen.
    """
    if neuron_indices is None:
        sub = rates
    else:
        sub = rates[np.asarray(neuron_indices, int)]
    return sub.mean(axis=0)


def cv_isi(
    times: npt.NDArray[np.floating],
    senders: npt.NDArray[np.int_],
    N_population: int,
    min_spikes: int = 2,
) -> npt.NDArray[np.floating]:
    """
    Coefficient of Variation der ISIs pro Neuron.

    Gibt ein Array der Länge N_population zurück.
    Neuronen mit weniger als min_spikes Spikes bekommen np.nan.
    """

    times = np.asarray(times, float)
    senders = np.asarray(senders, int)

    cv = np.full(N_population, np.nan, dtype=float)

    if times.size == 0:
        return cv

    # nach Neuron gruppieren
    sender_index = _build_sender_index(senders, N_population)

    # sortiert nach Zeit durchgehen
    order = np.argsort(times)
    times_sorted = times[order]
    senders_sorted = senders[order]

    # Spikelisten pro lokalem Index sammeln
    spikes_per_neuron: Dict[int, list[float]] = {j: [] for j in range(N_population)}
    for t, gid in zip(times_sorted, senders_sorted):
        j = sender_index.get(int(gid), None)
        if j is None:
            continue
        spikes_per_neuron[j].append(float(t))

    for j in range(N_population):
        ts = np.asarray(spikes_per_neuron[j], float)
        if ts.size < min_spikes:
            continue
        isi = np.diff(ts)
        if isi.size == 0:
            continue
        mu = isi.mean()
        sigma = isi.std(ddof=0)
        if mu > 0:
            cv[j] = sigma / mu

    return cv


def kuramoto_order_parameter(
    times: npt.NDArray[np.floating],
    senders: npt.NDArray[np.int_],
    N_population: int,
    t_eval: npt.NDArray[np.floating],
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Kuramoto-Order-Parameter Z(t) = R(t) e^{i Φ(t)}.

    times, senders: Spikes einer Population (z.B. alle E-Neuronen)
    N_population:   Anzahl Neuronen
    t_eval:         Zeitpunkte, an denen Z(t) ausgewertet wird (ms)

    Returns
    -------
    R:  |Z(t)|, shape (len(t_eval),)
    Phi: arg(Z(t)), shape (len(t_eval),)
    """
    times = np.asarray(times, float)
    senders = np.asarray(senders, int)
    t_eval = np.asarray(t_eval, float)

    R = np.zeros_like(t_eval, dtype=float)
    Phi = np.zeros_like(t_eval, dtype=float)

    if times.size == 0 or N_population == 0:
        return R, Phi

    sender_index = _build_sender_index(senders, N_population)

    # Spikezeiten pro Neuron sammeln
    spikes_per_neuron: Dict[int, npt.NDArray[np.floating]] = {j: np.array([], float) for j in range(N_population)}
    order = np.argsort(times)
    times_sorted = times[order]
    senders_sorted = senders[order]

    for t, gid in zip(times_sorted, senders_sorted):
        j = sender_index.get(int(gid), None)
        if j is None:
            continue
        spikes_per_neuron[j] = np.append(spikes_per_neuron[j], float(t))

    N = N_population

    for k, t in enumerate(t_eval):
        # Summe über j: e^{i θ_j(t)}
        z_sum = 0.0 + 0.0j
        count_valid = 0

        for j in range(N):
            ts = spikes_per_neuron[j]
            if ts.size < 2:
                continue
            # Position des ersten Spikes > t
            idx = np.searchsorted(ts, t, side="right")
            n = idx - 1
            if n < 0 or n >= ts.size - 1:
                continue  # t vor erstem oder nach letztem Intervall

            t_n = ts[n]
            t_np1 = ts[n + 1]
            if t_np1 <= t_n:
                continue

            theta = 2.0 * np.pi * (t - t_n) / (t_np1 - t_n)
            z_sum += np.exp(1j * theta)
            count_valid += 1

        if count_valid > 0:
            Z_t = z_sum / count_valid
        else:
            Z_t = 0.0 + 0.0j

        R[k] = np.abs(Z_t)
        Phi[k] = np.angle(Z_t)

    return R, Phi


def build_weight_matrix(
    sources: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    N_total: int,
) -> np.ndarray:
    """
    Build dense weight matrix W[post_idx, pre_idx] from connection lists.
    Assumes gids are 1..N_total in order.
    """
    W = np.zeros((N_total, N_total), dtype=float)

    sources = np.asarray(sources, int)
    targets = np.asarray(targets, int)
    weights = np.asarray(weights, float)

    for src, tgt, w in zip(sources, targets, weights):
        j_pre = src - 1   # 0-based
        i_post = tgt - 1
        if 0 <= j_pre < N_total and 0 <= i_post < N_total:
            W[i_post, j_pre] = w
    return W

def normalize_weight_matrix(
    W: np.ndarray,
    cfg: Dict[str, Any],
) -> np.ndarray:
    """
    Normalize weights to [-1,1] using per-projection Wmax from cfg.
    Assumes population ordering: [E | IH | IA] both for pre and post.
    """

    N_E  = cfg["network"]["N_E"]
    N_IH = cfg["network"]["N_IH"]
    N_IA = cfg["network"]["N_IA"]
    N_total = N_E + N_IH + N_IA

    if W.shape != (N_total, N_total):
        raise ValueError("Weight matrix shape does not match network size")

    Wn = np.zeros_like(W, dtype=float)

    # index ranges
    E_slice  = slice(0, N_E)
    IH_slice = slice(N_E, N_E + N_IH)
    IA_slice = slice(N_E + N_IH, N_total)

    syn_cfg = cfg["synapses"]

    # E -> all (E, IH, IA)
    Wmax_exc = float(syn_cfg["E_to_X"]["Wmax_factor"]) * float(syn_cfg["base_Wmax"])
    if Wmax_exc <= 0:
        raise ValueError("E_to_X Wmax must be > 0")
    Wn[:, E_slice] = W[:, E_slice] / Wmax_exc  # columns: presyn E

    # IH -> all
    Wmax_ih = float(syn_cfg["IH_to_X"]["Wmax_factor"]) * float(syn_cfg["base_Wmax"])  # negative
    Wn[:, IH_slice] = W[:, IH_slice] / abs(Wmax_ih)

    # IA -> all
    Wmax_ia = float(syn_cfg["IA_to_X"]["Wmax_factor"]) * float(syn_cfg["base_Wmax"])  # negative
    Wn[:, IA_slice] = W[:, IA_slice] / abs(Wmax_ia)

    # clip to [-1,1] for safety
    np.clip(Wn, -1.0, 1.0, out=Wn)
    return Wn