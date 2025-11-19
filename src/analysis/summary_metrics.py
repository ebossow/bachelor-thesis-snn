# src/analysis/summary.py

from typing import Dict, Any, Tuple
import numpy as np

from .metrics import (
    instantaneous_rates,
    population_rate,
    cv_isi,
    kuramoto_order_parameter,
    mean_weight_change,
)
from .util import combine_spikes  # falls du so eine helper-Funktion hast


def compute_summary_metrics(
    cfg: Dict[str, Any],
    data: Dict[str, Any],
    weights_over_time: Dict[str, Any] | None,
) -> Dict[str, float]:
    """
    Berechnet alle skalaren Metriken für einen Run.
    Kein Plotten, nur Zahlen.
    """

    N_E  = cfg["network"]["N_E"]
    N_IH = cfg["network"]["N_IH"]
    N_IA = cfg["network"]["N_IA"]
    N    = N_E + N_IH + N_IA

    simtime_ms = cfg["experiment"]["simtime_ms"]
    t_off_ms   = cfg["stimulation"]["dc"]["t_off_ms"]

    # --- Spikes: Gesamt und E-only, jeweils nur nach Stimulus ---
    # Gesamt
    spikes_N = combine_spikes(data, ["spikes_E", "spikes_IH", "spikes_IA"])
    mask_post = (spikes_N["times"] >= t_off_ms) & (spikes_N["times"] <= simtime_ms)
    times_post = spikes_N["times"][mask_post]
    senders_post = spikes_N["senders"][mask_post]

    # E-only für Subpopulations-R
    spikes_E = data["spikes_E"]
    mask_post_E = (spikes_E["times"] >= t_off_ms) & (spikes_E["times"] <= simtime_ms)
    times_E_post = spikes_E["times"][mask_post_E]
    senders_E_post = spikes_E["senders"][mask_post_E]

    # --- Instantaneous rates / mean firing rate ---
    rates_N, t_bins, mean_rate_pop, mean_rates_per_neuron = instantaneous_rates(
        times_post,
        senders_post,
        N_population=N,
        t_start=t_off_ms,
        t_stop=simtime_ms,
        bin_size_ms=50.0,
    )
    # „mean firing rate (mean)“
    mean_rate = float(mean_rates_per_neuron.mean())

    # --- CV ---
    cv_N = cv_isi(times_post, senders_post, N_population=N)
    mean_cv = float(np.nanmean(cv_N))

    # --- Kuramoto R gesamt ---
    R_all, Phi_all = kuramoto_order_parameter(
        times_post,
        senders_post,
        N_population=N,
        t_eval=t_bins,
    )
    mean_R_all = float(np.nanmean(R_all))

    # --- Kuramoto R für Subpopulation (z.B. E) ---
    #TODO: this is not right - subpopoulation is a cluster not E or I
    R_E, Phi_E = kuramoto_order_parameter(
        times_E_post,
        senders_E_post,
        N_population=N_E,
        t_eval=t_bins,
    )
    mean_R_E = float(np.nanmean(R_E))

    # --- K(t): mean change rate of weights ---
    mean_K = float("nan")
    if weights_over_time is not None:
        wt = weights_over_time["times"]
        Wt = weights_over_time["weights"]
        t_mid, K = mean_weight_change(wt, Wt, N_total=N)
        # nur nach Stimulus-Off
        mask_K = (t_mid >= t_off_ms) & (t_mid <= simtime_ms)
        K_post = K[mask_K]
        if K_post.size > 0:
            mean_K = float(np.nanmean(K_post))

    return {
        "mean_rate_Hz": mean_rate,
        "mean_cv": mean_cv,
        "mean_R_all": mean_R_all,
        #"mean_R_E": mean_R_E,
        "mean_K": mean_K,
        "K_post": K_post,
        "cv_N": cv_N,
        "mean_rates_per_neuron": mean_rates_per_neuron,
        "R": R_all,
    }