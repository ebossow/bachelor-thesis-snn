# src/analysis/metrics.py

from __future__ import annotations

from typing import Dict, Any, Tuple

import logging
import warnings
import numpy as np
import numpy.typing as npt

try:  # Optional dependency for MR estimator
    import mrestimator as _mre
except ImportError:  # pragma: no cover - best effort fallback
    _mre = None

try:  # Optional: only available when SciPy present
    from scipy.optimize import OptimizeWarning as _OptimizeWarning
except Exception:  # pragma: no cover - SciPy might be absent
    _OptimizeWarning = None


logger = logging.getLogger(__name__)
MRESTIMATOR_AVAILABLE = _mre is not None


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

    #print("len spikes: ", times.size, " total time (s): ", total_time_s, " N_population: ", N_population, "mean_pop_rate: ", times.size / (total_time_s * N_population))

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
        #print(np.min(isi))
        if isi.size == 0:
            continue
        mu = isi.mean()
        #print(mu)
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
            Z_t = z_sum / N
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
    Assumes population ordering: [E | IA_1 | IH | IA_2] (global GIDs).
    Normierung erfolgt spaltenweise (nach Präsyn-Population).
    """
    N_E     = cfg["network"]["N_E"]
    N_IH    = cfg["network"]["N_IH"]
    N_IA_1  = cfg["network"]["N_IA_1"]
    N_IA_2  = cfg["network"]["N_IA_2"]
    N_IA    = N_IA_1 + N_IA_2

    # globale Gesamtzahl
    N_total = N_E + N_IA_1 + N_IH + N_IA_2

    if W.shape != (N_total, N_total):
        raise ValueError(
            f"Weight matrix shape {W.shape} does not match network size {N_total}."
        )

    Wn = np.zeros_like(W, dtype=float)

    # Indexbereiche im GID-Ordering: [E | IA_1 | IH | IA_2]
    E_slice     = slice(0, N_E)
    IA1_slice   = slice(N_E, N_E + N_IA_1)
    IH_slice    = slice(N_E + N_IA_1, N_E + N_IA_1 + N_IH)
    IA2_slice   = slice(N_E + N_IA_1 + N_IH, N_total)

    syn_cfg   = cfg["synapses"]
    base_Wmax = float(syn_cfg["base_Wmax"])

    # E -> X (exzitatorisch)
    Wmax_exc = float(syn_cfg["E_to_X"]["Wmax_factor"]) * base_Wmax
    if Wmax_exc < 0.0:
        raise ValueError("E_to_X Wmax must be > 0 for excitatory normalization.")
    Wn[:, E_slice] = W[:, E_slice] / Wmax_exc

    # IH -> X (inhibitorisch, Hebb)
    Wmax_ih = float(syn_cfg["IH_to_X"]["Wmax_factor"]) * base_Wmax  # < 0
    if Wmax_ih > 0:
        raise ValueError("IH_to_X Wmax must be < 0 for inhibitory normalization.")
    Wn[:, IH_slice] = W[:, IH_slice] / abs(Wmax_ih)

    # IA_1/IA_2 -> X (inhibitorisch, anti-Hebb), beide über IA_to_X konfiguriert
    Wmax_ia = float(syn_cfg["IA_to_X"]["Wmax_factor"]) * base_Wmax  # < 0
    if Wmax_ia > 0:
        raise ValueError("IA_to_X Wmax must be < 0 for inhibitory normalization.")
    Wn[:, IA1_slice] = W[:, IA1_slice] / abs(Wmax_ia)
    Wn[:, IA2_slice] = W[:, IA2_slice] / abs(Wmax_ia)

    # Sicherheit: in [-1,1] clampen
    np.clip(Wn, -1.0, 1.0, out=Wn)
    return Wn

def mean_weight_change(
    weight_times: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
    N_total: int,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    K(t) nach Gleichung (16) für ein all-to-all Netzwerk ohne Autapses.

    K(t_k) = 1 / [N (N-1)] * sum_{i != j} (w_ij(t_{k+1}) - w_ij(t_k)) / Δt_k

    Parameters
    ----------
    weight_times : (n_snap,) in ms
        Zeitpunkte der Gewichtssnapshots.
    weights : (n_snap, M)
        Gewichte pro Snapshot, flach über alle Synapsen (z.B. [E, IH, IA] concat).
        Für all-to-all ohne Autapses gilt M = N*(N-1).
    N_total : int
        Anzahl Neuronen N.

    Returns
    -------
    t_mid : (n_snap-1,) in ms
        Mittlere Zeitpunkte der Intervalle.
    K : (n_snap-1,)
        Mittlere Änderungsrate der Gewichte (Gewichtseinheiten pro Sekunde).
    """
    weight_times = np.asarray(weight_times, float)
    weights = np.asarray(weights, float)

    if weight_times.ndim != 1:
        raise ValueError("weight_times must be 1D")
    if weights.ndim != 2:
        raise ValueError("weights must be 2D (n_snap, M)")
    if weight_times.shape[0] != weights.shape[0]:
        raise ValueError("Mismatch between number of time points and snapshots")

    n_snap, M = weights.shape
    if n_snap < 2:
        raise ValueError("Need at least two snapshots to compute K(t)")

    expected_M = N_total * (N_total - 1)
    if M != expected_M:
        # wenn du sicher weißt, dass immer all-to-all, kannst du hier auch hart fehlschlagen:
        # raise ValueError(...)
        print(f"Warning: weights.shape[1] = {M}, but N(N-1) = {expected_M}")

    # Δt in Sekunden
    dt = np.diff(weight_times) / 1000.0  # ms -> s  -> shape (n_snap-1,)

    # Δw = w(t_{k+1}) - w(t_k)
    dW = np.diff(weights, axis=0)        # (n_snap-1, M)

    # Summe über alle Synapsen
    sum_dW = dW.sum(axis=1)              # (n_snap-1,)

    norm = float(N_total * (N_total - 1))
    K = sum_dW / (norm * dt)             # (n_snap-1,)

    # mittlere Zeitpunkte
    t_mid = 0.5 * (weight_times[:-1] + weight_times[1:])

    return t_mid, K

def branching_ratio_neuronwise(
    times: npt.NDArray[np.floating],
    senders: npt.NDArray[np.int_],
    N_population: int,
    dt_ms: float,
    delay_offset_ms: float = 0.5,
    t_start_ms: float | None = None,
    t_stop_ms: float | None = None,
) -> Tuple[float, npt.NDArray[np.floating]]:
    """
    Neuron-spezifische Branching-Schätzung.

    Für jeden Spike bei Zeit t_i (Sender g_i):
      - Betrachte das Fenster [t_i + delay_offset_ms, t_i + delay_offset_ms + dt_ms).
      - Zähle die Anzahl *verschiedener* Neuronen, die in diesem Fenster spiken.
    Die globale Branching Ratio ist der Mittelwert der Kinderzahl pro Spike.

    Parameters
    ----------
    times          : Spikezeiten in ms, shape (n_spikes,)
    senders        : GIDs der spikenden Neuronen, shape (n_spikes,)
    N_population   : Gesamtanzahl der betrachteten Neuronen
    dt_ms          : Fensterbreite hinter jedem Spike (ms)
    delay_offset_ms: Offset vom Spikezeitpunkt zum Start des Fensters (ms),
                     z.B. ~ synaptische Verzögerung.
    t_start_ms     : optional, nur Spikes mit t in [t_start_ms, t_stop_ms) als Eltern
    t_stop_ms      : optional, s.o.

    Returns
    -------
    sigma_global   : mittlere Kinderzahl pro Spike (Skalar),
                     np.nan falls keine gültigen Eltern-Spikes
    sigma_per_neuron : Array der Länge N_population,
                       mittlere Kinderzahl pro Spike pro Neuron;
                       np.nan, wenn Neuron keine Eltern-Spikes hatte.
    """
    times = np.asarray(times, float)
    senders = np.asarray(senders, int)

    if dt_ms <= 0.0:
        raise ValueError("dt_ms must be > 0")

    if times.size == 0:
        return float("nan"), np.full(N_population, np.nan, dtype=float)

    # optionales Zeitfenster für Eltern-Spikes
    if t_start_ms is not None:
        times_mask = times >= t_start_ms
    else:
        times_mask = np.ones_like(times, dtype=bool)
    if t_stop_ms is not None:
        times_mask &= (times < t_stop_ms)

    # sortieren
    order = np.argsort(times)
    times_sorted = times[order]
    senders_sorted = senders[order]

    times_mask_sorted = times_mask[order]

    # Mapping GID -> lokaler Index 0..N-1
    sender_index = _build_sender_index(senders_sorted, N_population)

    n_spikes = times_sorted.size
    children_counts = np.zeros(n_spikes, dtype=float)

    # Akkumulatoren pro Neuron
    sum_children_per_neuron = np.zeros(N_population, dtype=float)
    n_parent_spikes_per_neuron = np.zeros(N_population, dtype=float)

    # Zwei Zeiger für das Slide-Fenster über times_sorted
    j_start = 0
    j_end = 0

    for i in range(n_spikes):
        t_i = times_sorted[i]
        gid_i = senders_sorted[i]

        # nur Spikes in gewünschtem Zeitfenster als Eltern zählen
        if not times_mask_sorted[i]:
            continue

        # Fenstergrenzen
        t_win_start = t_i + delay_offset_ms
        t_win_end = t_win_start + dt_ms

        # j_start: erster Spike mit time >= t_win_start
        while j_start < n_spikes and times_sorted[j_start] < t_win_start:
            j_start += 1

        # j_end: erster Spike mit time >= t_win_end
        if j_end < j_start:
            j_end = j_start
        while j_end < n_spikes and times_sorted[j_end] < t_win_end:
            j_end += 1

        # Slice der Kinder-Spikes (nur nach dem Eltern-Spike)
        if j_start >= j_end:
            n_children = 0.0
        else:
            # Sender im Fenster; mehrere Spikes desselben Neurons -> nur 1 Kind
            child_senders = senders_sorted[j_start:j_end]
            if child_senders.size == 0:
                n_children = 0.0
            else:
                n_children = float(np.unique(child_senders).size)

        children_counts[i] = n_children

        # Beitrag zu Neuron-Statistik
        j_local = sender_index.get(int(gid_i), None)
        if j_local is not None and 0 <= j_local < N_population:
            sum_children_per_neuron[j_local] += n_children
            n_parent_spikes_per_neuron[j_local] += 1.0

    # globale Branching Ratio: Mittelwert über alle Eltern-Spikes mit Fenster
    valid_spikes = n_parent_spikes_per_neuron.sum() > 0
    if np.any(children_counts > 0) or valid_spikes:
        # Mittel über alle Eltern-Spikes, die im Zeitfenster liegen:
        # times_mask_sorted markiert Eltern-Spikes, die wir berücksichtigen
        parent_mask = times_mask_sorted
        if np.any(parent_mask):
            sigma_global = float(np.mean(children_counts[parent_mask]))
        else:
            sigma_global = float("nan")
    else:
        sigma_global = float("nan")

    # sigma_j pro Neuron
    sigma_per_neuron = np.full(N_population, np.nan, dtype=float)
    nonzero = n_parent_spikes_per_neuron > 0
    sigma_per_neuron[nonzero] = (
        sum_children_per_neuron[nonzero] / n_parent_spikes_per_neuron[nonzero]
    )

    return sigma_global, sigma_per_neuron


def empty_bin_fraction(
    times_ms: npt.NDArray[np.floating],
    t_start_ms: float,
    t_stop_ms: float,
    dt_ms: float,
) -> float:
    """
    Anteil leerer Zeitbins p0 für gegebenes dt_ms.

    times_ms : Spikezeiten (ms, alle Neuronen)
    t_start_ms, t_stop_ms : Auswertefenster
    dt_ms : Binbreite

    p0 = (#Bins mit 0 Events) / (Gesamtzahl Bins)
    """
    times = np.asarray(times_ms, float)
    mask = (times >= t_start_ms) & (times < t_stop_ms)
    times = times[mask]

    if dt_ms <= 0.0 or t_stop_ms <= t_start_ms:
        raise ValueError("Invalid dt_ms or time window")

    n_bins = int(np.floor((t_stop_ms - t_start_ms) / dt_ms))
    if n_bins <= 0:
        return float("nan")

    edges = t_start_ms + np.arange(n_bins + 1) * dt_ms
    counts, _ = np.histogram(times, bins=edges)

    if n_bins == 0:
        return float("nan")
    p0 = float(np.mean(counts == 0))
    return p0


def empty_bin_fraction_scan(
    times_ms: npt.NDArray[np.floating],
    t_start_ms: float,
    t_stop_ms: float,
    dt_list_ms: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """
    p0(dt) für mehrere dt-Werte.

    dt_list_ms : 1D-Array von dt-Werten (ms)
    Returns: p0_array mit gleicher Länge.
    """
    dt_list = np.asarray(dt_list_ms, float)
    p0_list = np.zeros_like(dt_list, dtype=float)

    for i, dt in enumerate(dt_list):
        p0_list[i] = empty_bin_fraction(times_ms, t_start_ms, t_stop_ms, dt)

    return p0_list


def binned_active_neurons(
    times_ms: npt.NDArray[np.floating],
    senders: npt.NDArray[np.int_],
    N_population: int,
    t_start_ms: float,
    t_stop_ms: float,
    dt_ms: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Anzahl aktiver Neuronen pro Zeit-Bin.

    "Aktiv" = mindestens ein Spike im Bin.

    Returns
    -------
    N_t       : shape (n_bins,) – #aktive Neuronen pro Bin
    t_centers : shape (n_bins,) – mittlere Zeitpunkte der Bins
    """
    times = np.asarray(times_ms, float)
    senders = np.asarray(senders, int)

    # Wir nutzen deine bestehende Binning-Logik aus instantaneous_rates.
    rates, t_centers, _, _ = instantaneous_rates(
        times,
        senders,
        N_population=N_population,
        t_start=t_start_ms,
        t_stop=t_stop_ms,
        bin_size_ms=dt_ms,
    )
    # rates: (N_population, n_bins)
    # aktiv, wenn rate > 0 ⇒ es gab mindestens einen Spike.
    active = rates > 0.0
    N_t = active.sum(axis=0).astype(float)  # (n_bins,)

    return N_t, t_centers


def average_inter_event_interval(
    times_ms: npt.NDArray[np.floating],
    t_start_ms: float | None = None,
    t_stop_ms: float | None = None,
) -> tuple[float, npt.NDArray[np.floating]]:
    """
    Globales Average Inter-Event Interval (AIEI) über alle Spikes.

    times_ms : Spikezeiten in ms (alle Neuronen kombiniert).
    t_start_ms, t_stop_ms : optionales Auswertefenster.

    Returns
    -------
    aiei_ms : float (np.nan, falls <2 Spikes im Fenster)
    isi_ms  : Array der einzelnen Inter-Event-Intervalle in ms
    """
    t = np.asarray(times_ms, float)
    if t_start_ms is not None:
        t = t[t >= t_start_ms]
    if t_stop_ms is not None:
        t = t[t < t_stop_ms]

    t = np.sort(t)
    if t.size < 2:
        return float("nan"), np.array([], dtype=float)

    isi = np.diff(t)  # ms
    aiei = float(isi.mean()) if isi.size > 0 else float("nan")
    return aiei, isi


def avalanche_sizes_from_times(
    times_ms: npt.NDArray[np.floating],
    t_start_ms: float,
    t_stop_ms: float,
    dt_ms: float,
    min_size: int = 1,
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Berechne Avalanche-Size- und -Dauer-Verteilung nach zeit-bin-basierter Definition.

    Definition:
      - Zeitfenster [t_start_ms, t_stop_ms) wird in Bins der Breite dt_ms eingeteilt.
      - Ein Bin ist 'aktiv', wenn er mindestens einen Spike enthält.
      - Eine Avalanche ist ein maximaler zusammenhängender Block aktiver Bins,
        getrennt durch mindestens einen leeren Bin.
      - Size S = Summe der Spike-Counts in den Bins dieser Avalanche.
      - Duration D = (Anzahl Bins der Avalanche) * dt_ms.

    Parameters
    ----------
    times_ms : array-like
        Spikezeiten in ms (alle Neuronen kombiniert).
    t_start_ms : float
        Startzeit des Auswertefensters in ms (inklusive).
    t_stop_ms : float
        Endzeit des Auswertefensters in ms (exklusiv).
    dt_ms : float
        Binbreite in ms.
    min_size : int
        Nur Avalanches mit S >= min_size werden zurückgegeben.

    Returns
    -------
    sizes : (n_avalanches,) array
        Avalanche-Größen (Anzahl Spikes).
    durations_ms : (n_avalanches,) array
        Avalanche-Dauern in ms.
    """
    t = np.asarray(times_ms, float)

    # Auf das Fenster beschränken
    mask = (t >= t_start_ms) & (t < t_stop_ms)
    t = t[mask]

    if t.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if dt_ms <= 0.0:
        raise ValueError("dt_ms must be > 0.")

    # Bins
    total_T = t_stop_ms - t_start_ms
    n_bins = int(np.floor(total_T / dt_ms))
    if n_bins <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    edges = t_start_ms + np.arange(n_bins + 1) * dt_ms

    # Spike-Counts pro Bin
    counts, _ = np.histogram(t, bins=edges)
    counts = counts.astype(int)  # shape (n_bins,)

    active = counts > 0

    sizes: list[float] = []
    durations: list[float] = []

    in_avalanche = False
    start_bin = 0

    for n in range(n_bins):
        if active[n]:
            if not in_avalanche:
                # Start einer neuen Avalanche
                in_avalanche = True
                start_bin = n
        else:
            if in_avalanche:
                # Ende der Avalanche bei n-1
                end_bin = n - 1
                s = counts[start_bin : end_bin + 1].sum()
                if s >= min_size:
                    sizes.append(float(s))
                    dur_ms = (end_bin - start_bin + 1) * dt_ms
                    durations.append(float(dur_ms))
                in_avalanche = False

    # Falls Avalanche bis zum letzten Bin läuft
    if in_avalanche:
        end_bin = n_bins - 1
        s = counts[start_bin : end_bin + 1].sum()
        if s >= min_size:
            sizes.append(float(s))
            dur_ms = (end_bin - start_bin + 1) * dt_ms
            durations.append(float(dur_ms))

    if not sizes:
        return np.array([], dtype=float), np.array([], dtype=float)

    return np.asarray(sizes, float), np.asarray(durations, float)

### ----- Branching Ratio Varianten -----

def binned_spike_counts(
    times_ms: npt.NDArray[np.floating],
    t_start_ms: float,
    t_stop_ms: float,
    dt_ms: float,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.floating]]:
    """
    Gesamt-Spikecounts pro Zeitfenster (alle Neuronen zusammen).

    times_ms  : Spikezeiten (ms, alle Neuronen kombiniert)
    t_start_ms, t_stop_ms : Auswertefenster [t_start, t_stop)
    dt_ms     : Binbreite

    Returns
    -------
    counts    : (n_bins,) int – #Spikes pro Bin
    edges_ms  : (n_bins+1,) float – Bin-Grenzen in ms
    """
    t = np.asarray(times_ms, float)
    if dt_ms <= 0.0 or t_stop_ms <= t_start_ms:
        raise ValueError("Invalid dt_ms or time window")

    mask = (t >= t_start_ms) & (t < t_stop_ms)
    t = t[mask]

    total_T = t_stop_ms - t_start_ms
    n_bins = int(np.floor(total_T / dt_ms))
    if n_bins <= 0:
        return np.zeros(0, dtype=int), np.array([t_start_ms, t_stop_ms], dtype=float)

    edges = t_start_ms + np.arange(n_bins + 1) * dt_ms
    counts, _ = np.histogram(t, bins=edges)
    counts = counts.astype(int)
    return counts, edges


def branching_ratio_from_counts(
    spike_counts: npt.NDArray[np.int_],
) -> float:
    """
    Globale Branching-Ratio aus Bin-Counts.

    spike_counts : (n_bins,) – #Spikes pro Zeitfenster

    σ = sum( N_{t+1} ) / sum( N_t ) über alle Bins mit N_t > 0
    """
    counts = np.asarray(spike_counts, float)
    if counts.size < 2:
        return float("nan")

    parents = counts[:-1]
    children = counts[1:]

    mask = parents > 0.0
    if not np.any(mask):
        return float("nan")

    parents = parents[mask]
    children = children[mask]

    denom = parents.sum()
    if denom <= 0.0:
        return float("nan")

    sigma = float(children.sum() / denom)
    return sigma


def _resolve_mr_fit_function(fitfunc):
    if _mre is None:
        return None

    if fitfunc is None:
        return _mre.f_exponential
    if callable(fitfunc):
        return fitfunc

    key = str(fitfunc).lower()
    mapping = {
        "e": _mre.f_exponential,
        "exp": _mre.f_exponential,
        "exponential": _mre.f_exponential,
        "exp_offset": _mre.f_exponential_offset,
        "exp_off": _mre.f_exponential_offset,
        "exp_offs": _mre.f_exponential_offset,
        "exponential_offset": _mre.f_exponential_offset,
        "complex": _mre.f_complex,
        "c": _mre.f_complex,
    }
    return mapping.get(key, _mre.f_exponential)


def _restrict_coefficients_to_lag_window(
    coeffs,
    min_lag_ms: float | None,
    max_lag_ms: float | None,
    min_points: int,
):
    """Return coefficients restricted to a lag window (inclusive)."""
    if coeffs is None or (min_lag_ms is None and max_lag_ms is None):
        return coeffs

    steps = np.asarray(coeffs.steps, dtype=int)
    if steps.size == 0:
        return None

    dt_ms = float(getattr(coeffs, "dt", 1.0))
    lags_ms = steps.astype(float) * dt_ms

    mask = np.ones_like(lags_ms, dtype=bool)
    if min_lag_ms is not None:
        mask &= lags_ms >= float(min_lag_ms)
    if max_lag_ms is not None:
        mask &= lags_ms <= float(max_lag_ms)

    if not np.any(mask):
        return None

    if mask.sum() < max(1, int(min_points)):
        return None

    coeff_vals = np.asarray(coeffs.coefficients, dtype=float)[mask]
    steps_sel = steps[mask]

    replacements: dict[str, Any] = {
        "coefficients": coeff_vals,
        "steps": steps_sel,
    }

    stderrs = getattr(coeffs, "stderrs", None)
    if stderrs is not None:
        stderrs_arr = np.asarray(stderrs, dtype=float)
        if stderrs_arr.shape == lags_ms.shape:
            replacements["stderrs"] = stderrs_arr[mask]

    # Drop correlation histories that no longer align with the filtered steps
    replacements["bootstrapcrs"] = None
    replacements["trialcrs"] = None

    return coeffs._replace(**replacements)


def branching_ratio_mr_estimator(
    spike_counts: npt.NDArray[np.int_],
    dt_ms: float,
    *,
    method: str = "stationarymean",
    steps: tuple[int, int] | npt.NDArray[np.int_] | None = None,
    fitfunc: str | None | Any = "exp",
    num_bootstrap: int = 0,
    known_mean: float | None = None,
    max_step_bins: int | None = 50,
    fit_lag_ms_min: float | None = None,
    fit_lag_ms_max: float | None = None,
    fit_use_offset: bool | None = None,
    min_fit_points: int = 3,
    raise_on_error: bool = False,
    return_details: bool = False,
) -> tuple[float, float] | tuple[float, float, Any, Any]:
    """Estimate branching ratio via MR estimator (Wilting & Priesemann, 2018).

    Parameters
    ----------
    spike_counts : array-like
        Spike counts per time bin. Can be 1D (single trial) or 2D
        ``(num_trials, num_bins)``.
    dt_ms : float
        Bin width in milliseconds (passed to :mod:`mrestimator`).
    method : str, optional
        ``'stationarymean'`` or ``'trialseparated'`` (aliases ``'sm'``/``'ts'``).
    steps : tuple or array, optional
        Step configuration forwarded to :func:`mrestimator.coefficients`.
        ``None`` defaults to the library heuristics.
        max_step_bins : int or None, optional
            Upper bound for the automatically chosen ``steps`` range when
            ``steps`` is ``None``. ``None`` keeps the library default.
    fitfunc : str or callable, optional
        Which built-in MR fit to use; defaults to the exponential model.
    num_bootstrap : int, optional
        Number of bootstrap replicas for coefficient estimation.
    known_mean : float, optional
        Forwarded as ``knownmean`` to :func:`mrestimator.coefficients`.
    fit_lag_ms_min : float, optional
        Lower lag bound (ms) for the fit; ``None`` keeps the full range.
    fit_lag_ms_max : float, optional
        Upper lag bound (ms) for the fit; ``None`` keeps the full range.
    fit_use_offset : bool, optional
        Force the exponential-with-offset fit (``True``) or plain exponential
        (``False``); ``None`` leaves ``fitfunc`` unchanged.
    min_fit_points : int, optional
        Minimum number of r_k samples required after windowing.
    raise_on_error : bool, optional
        Raise encountered exceptions instead of returning ``nan``.
    return_details : bool, optional
        If ``True``, additionally return the :class:`mrestimator.CoefficientResult`
        and :class:`mrestimator.fit.FitResult` objects (``coeffs``, ``fit``).

    Returns
    -------
    sigma_mr : float
        Branching parameter ``m`` estimated by MR.
    tau_ms : float
        Autocorrelation time in milliseconds.
    fit_result : optional
        Returned when ``return_details=True``.
    """

    if not MRESTIMATOR_AVAILABLE:
        msg = "mrestimator is not installed; cannot compute MR estimator."
        if raise_on_error:
            raise ImportError(msg)
        logger.warning(msg)
        if return_details:
            return float("nan"), float("nan"), None, None
        return float("nan"), float("nan")

    dt_ms = float(dt_ms)
    if dt_ms <= 0.0:
        raise ValueError("dt_ms must be > 0")

    data = np.asarray(spike_counts, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    elif data.ndim != 2:
        raise ValueError("spike_counts must be 1D or 2D array-like")

    if data.shape[1] < 3:
        logger.warning("Not enough bins (%d) for MR estimator", data.shape[1])
        if return_details:
            return float("nan"), float("nan"), None, None
        return float("nan"), float("nan")

    method_key = method.lower()
    if method_key in ("sm", "stationarymean"):
        method_arg = "stationarymean"
    elif method_key in ("ts", "trialseparated"):
        method_arg = "trialseparated"
    else:
        raise ValueError("method must be 'stationarymean' or 'trialseparated'")

    if (
        fit_lag_ms_min is not None
        and fit_lag_ms_max is not None
        and float(fit_lag_ms_max) <= float(fit_lag_ms_min)
    ):
        raise ValueError("fit_lag_ms_max must be greater than fit_lag_ms_min")

    steps_arg: tuple[int, int] | npt.NDArray[np.int_] | None
    if steps is not None:
        steps_arg = steps
    elif max_step_bins is None:
        steps_arg = (None, None)
    else:
        upper = max(1, min(int(max_step_bins), data.shape[1] // 2))
        steps_arg = (1, upper)

    coeffs_to_fit = None
    try:
        coeffs = _mre.coefficients(
            data,
            method=method_arg,
            steps=steps_arg,
            dt=dt_ms,
            dtunit="ms",
            knownmean=known_mean,
            numboot=int(num_bootstrap),
        )

        fit_function = _resolve_mr_fit_function(fitfunc)
        if fit_use_offset is True and _mre is not None:
            fit_function = _mre.f_exponential_offset
        elif fit_use_offset is False and _mre is not None:
            fit_function = _mre.f_exponential

        coeffs_to_fit = _restrict_coefficients_to_lag_window(
            coeffs,
            min_lag_ms=fit_lag_ms_min,
            max_lag_ms=fit_lag_ms_max,
            min_points=min_fit_points,
        )

        if coeffs_to_fit is None:
            msg = (
                "MR estimator: no coefficients remain after applying lag window;"
                " relax fit_lag_ms_min/max or min_fit_points"
            )
            if raise_on_error:
                raise ValueError(msg)
            logger.warning(msg)
            if return_details:
                return float("nan"), float("nan"), None, None
            return float("nan"), float("nan")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if _OptimizeWarning is not None:
                warnings.simplefilter("ignore", _OptimizeWarning)
            fit_result = _mre.fit(coeffs_to_fit, fitfunc=fit_function)
        sigma_mr = float(fit_result.mre)
        tau_ms = float(fit_result.tau) if fit_result.tau is not None else float("nan")
    except Exception as exc:  # pragma: no cover - depends on external package
        if raise_on_error:
            raise
        logger.warning("MR estimator failed: %s", exc)
        if return_details:
            return float("nan"), float("nan"), None, None
        return float("nan"), float("nan")

    if return_details:
        return sigma_mr, tau_ms, coeffs_to_fit, fit_result
    return sigma_mr, tau_ms

def avalanche_segments_from_counts(
    spike_counts: npt.NDArray[np.int_],
    min_len: int = 2,
) -> list[npt.NDArray[np.int_]]:
    """
    Zerlegt die Bin-Counts in Avalanche-Segmente (zusammenhängende aktive Blöcke).

    spike_counts : (n_bins,) – #Spikes pro Bin
    min_len      : minimale Anzahl Bins für eine Avalanche

    Returns
    -------
    avalanches : Liste von 1D-Arrays (Counts pro Bin innerhalb einer Avalanche)
    """
    counts = np.asarray(spike_counts, int)
    if counts.size == 0:
        return []

    padded = np.concatenate(([0], counts, [0]))
    zero_idx = np.where(padded == 0)[0]

    avalanches: list[npt.NDArray[np.int_]] = []
    for i in range(len(zero_idx) - 1):
        lower = zero_idx[i]
        upper = zero_idx[i + 1]
        if upper - lower <= 1:
            continue
        seg = padded[lower + 1 : upper]
        if seg.size >= min_len:
            avalanches.append(seg)

    return avalanches

def branching_ratio_from_avalanches(
    spike_counts: npt.NDArray[np.int_],
    min_len: int = 2,
) -> float:
    """
    Branching-Ratio aus einzelnen Avalanches.

    Idee:
      - Zerlege die Zeitserie der Bin-Counts in Avalanches.
      - Berechne pro Avalanche die einfache Branching-Ratio
        (wie in branching_ratio_from_counts).
      - Gib den Mittelwert über alle Avalanches zurück.

    Returns
    -------
    sigma_aval : float (np.nan, falls keine gültigen Avalanches)
    """
    #print("spike counts", spike_counts)
    #print("spike_counts_len: ", len(spike_counts))
    #print("max_spike_count: ", np.max(spike_counts))
    avalanches = avalanche_segments_from_counts(spike_counts, min_len=min_len)
    #print("avalanches: ", avalanches)   
    if not avalanches:
        return float("nan")

    sigmas: list[float] = []
    #print(f"# Avalanches: {len(avalanches)}")
    for seg in avalanches:
        sigma_seg = branching_ratio_from_counts(seg)
        if np.isfinite(sigma_seg):
            sigmas.append(sigma_seg)

    if not sigmas:
        return float("nan")

    return float(np.mean(sigmas))

def branching_ratios_binned_global(
    times_ms: npt.NDArray[np.floating],
    t_start_ms: float,
    t_stop_ms: float,
    dt_ms: float,
    min_aval_len: int = 2,
) -> tuple[float, float, npt.NDArray[np.int_]]:
    """
    Komfortfunktion:

      - binnt times_ms in Bins der Breite dt_ms
      - berechnet
          σ_global = branching_ratio_from_counts(counts)
          σ_aval   = branching_ratio_from_avalanches(counts)

    Returns
    -------
    sigma_global : float
    sigma_aval   : float
    counts       : (n_bins,) int – verwendete Spike-Counts
    """
    counts, _ = binned_spike_counts(times_ms, t_start_ms, t_stop_ms, dt_ms)
    sigma_global = branching_ratio_from_counts(counts)
    sigma_aval   = branching_ratio_from_avalanches(counts, min_len=min_aval_len)
    return sigma_global, sigma_aval, counts