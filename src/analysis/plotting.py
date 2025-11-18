# src/analysis/plotting.py

from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Spike raster
# ---------------------------------------------------------------------------

def plot_spike_raster_ax(ax: plt.Axes,
                         data: Dict[str, Any],
                         cfg: Dict[str, Any]) -> None:
    """
    Rasterplot für alle Populationen auf gegebener Axes.
    Erwartet data["spikes_E"], data["spikes_IH"], data["spikes_IA"]
    mit Feldern "times" und "senders".
    """
    N_E  = cfg["network"]["N_E"]
    N_IH = cfg["network"]["N_IH"]
    N_IA = cfg["network"]["N_IA"]
    N_total = N_E + N_IH + N_IA

    spikes_E = data.get("spikes_E")
    spikes_IH = data.get("spikes_IH")
    spikes_IA = data.get("spikes_IA")

    if spikes_E is not None:
        times_E = np.asarray(spikes_E["times"])
        senders_E = np.asarray(spikes_E["senders"])
        ax.scatter(times_E, senders_E - 1, s=4, c="red", label="E")

    if spikes_IH is not None and spikes_IA is not None:
        times_I = np.concatenate(
            (np.asarray(spikes_IH["times"]), np.asarray(spikes_IA["times"]))
        )
        senders_I = np.concatenate(
            (np.asarray(spikes_IH["senders"]), np.asarray(spikes_IA["senders"]))
        )
        ax.scatter(times_I, senders_I - 1, s=4, c="blue", label="IH+IA")
    elif spikes_IH is not None:
        times_I = np.asarray(spikes_IH["times"])
        senders_I = np.asarray(spikes_IH["senders"])
        ax.scatter(times_I, senders_I - 1, s=4, c="blue", label="IH")
    elif spikes_IA is not None:
        times_I = np.asarray(spikes_IA["times"])
        senders_I = np.asarray(spikes_IA["senders"])
        ax.scatter(times_I, senders_I - 1, s=4, c="blue", label="IA")

    ax.set_xlim(0, cfg["experiment"]["simtime_ms"])
    ax.set_ylim(-1, N_total + 1)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("neuron index")
    ax.legend(loc="upper right")
    ax.set_title("Spike raster")


def plot_spike_raster(data: Dict[str, Any],
                      cfg: Dict[str, Any]) -> None:
    """
    Wrapper: eigener Plot für das Raster.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_spike_raster_ax(ax, data, cfg)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# PDF / Histogramme
# ---------------------------------------------------------------------------

def _plot_pdf_ax(ax: plt.Axes,
                 values: np.ndarray,
                 xlabel: str,
                 title: str,
                 bins: int = 50) -> None:
    """
    Helper: plot a normalized histogram (PDF) of `values` on given Axes.
    """
    vals = np.asarray(values, float)
    vals = vals[np.isfinite(vals)]  # drop NaN/inf

    if vals.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.hist(vals, bins=bins, density=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PDF")
    ax.set_title(title)


def plot_pdf_cv_ax(ax: plt.Axes, cv: np.ndarray, bins: int = 50) -> None:
    """
    PDF of CV_i across neurons on given Axes.
    """
    _plot_pdf_ax(ax, cv, xlabel="CV_i", title="Distribution of CV_i", bins=bins)


def plot_pdf_R_ax(ax: plt.Axes, R: np.ndarray, bins: int = 50) -> None:
    """
    PDF of Kuramoto order parameter R(t) across time on given Axes.
    """
    _plot_pdf_ax(ax, R, xlabel="R", title="Distribution of R(t)", bins=bins)


def plot_pdf_mean_rate_ax(ax: plt.Axes, mean_rates: np.ndarray, bins: int = 50) -> None:
    """
    PDF of mean firing rates (Hz) across neurons on given Axes.
    """
    _plot_pdf_ax(
        ax,
        mean_rates,
        xlabel="Mean firing rate (Hz)",
        title="Distribution of mean firing rates",
        bins=bins,
    )


# Wrapper, falls du sie weiterhin einzeln aufrufen willst:

def _plot_pdf(values, xlabel: str, title: str, bins: int = 50):
    fig, ax = plt.subplots()
    _plot_pdf_ax(ax, values, xlabel, title, bins=bins)
    plt.tight_layout()
    plt.show()


def plot_pdf_cv(cv: np.ndarray, bins: int = 50):
    fig, ax = plt.subplots()
    plot_pdf_cv_ax(ax, cv, bins=bins)
    plt.tight_layout()
    plt.show()


def plot_pdf_R(R: np.ndarray, bins: int = 50):
    fig, ax = plt.subplots()
    plot_pdf_R_ax(ax, R, bins=bins)
    plt.tight_layout()
    plt.show()


def plot_pdf_mean_rate(mean_rates: np.ndarray, bins: int = 50):
    fig, ax = plt.subplots()
    plot_pdf_mean_rate_ax(ax, mean_rates, bins=bins)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Weight matrix
# ---------------------------------------------------------------------------

def plot_weight_matrix_ax(
    ax: plt.Axes,
    Wn: np.ndarray,
    cfg: Dict[str, Any],
):
    """
    Plot normalized weight matrix Wn[post, pre] with colorbar in [-1,1]
    on a given Axes.
    Assumes populations in order [E | IH | IA].
    """
    N_E  = cfg["network"]["N_E"]
    N_IH = cfg["network"]["N_IH"]
    N_IA = cfg["network"]["N_IA"]
    N_total = N_E + N_IH + N_IA

    if Wn.shape != (N_total, N_total):
        raise ValueError("Wn shape does not match network size")

    im = ax.imshow(
        Wn,
        origin="lower",
        vmin=-1.0,
        vmax=1.0,
        cmap="bwr",   # blue-white-red
        interpolation="nearest",
    )

    ax.set_xlabel("Pre-synaptic neuron index")
    ax.set_ylabel("Post-synaptic neuron index")

    # Population boundaries
    for x in [N_E, N_E + N_IH]:
        ax.axvline(x - 0.5, color="k", linewidth=1)
        ax.axhline(x - 0.5, color="k", linewidth=1)

    ax.set_title("Normalized weight matrix")

    return im  # Caller can attach a colorbar


def plot_weight_matrix(
    Wn: np.ndarray,
    cfg: Dict[str, Any],
) -> None:
    """
    Wrapper: eigener Plot für die Weight-Matrix.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    im = plot_weight_matrix_ax(ax, Wn, cfg)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized weight")
    plt.tight_layout()
    plt.show()