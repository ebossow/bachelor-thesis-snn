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
    ν_j(t) nach Gleichung (12), diskretisiert in Bins der Größe bin_size_ms.

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

    t_edges = t_start + np.arange(n_bins + 1) * bin_size_ms
    t_centers = (t_edges[:-1] + t_edges[1:]) / 2.0

    rates = np.zeros((N_population, n_bins), dtype=float)

    if times.size == 0:
        return rates, t_centers

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
        rates[j, t_bin] += 1.0

    # von Spike-Counts auf Rate (Hz): ms -> s ⇒ * 1000 / T
    T = bin_size_ms / 1000.0  # s
    rates /= T
    rate_mean = rates.mean()

    return rates, t_centers, rate_mean


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