# src/analysis/plotting.py

from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ---------------------------------------------------------------------------
# Spike raster
# ---------------------------------------------------------------------------

def plot_spike_raster_ax(ax: plt.Axes,
                         data: Dict[str, Any],
                         cfg: Dict[str, Any],
                         max_time_ms: float | None = None) -> None:
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

    ms_to_s = 0.001

    if spikes_E is not None:
        times_E = np.asarray(spikes_E["times"], dtype=float) * ms_to_s
        senders_E = np.asarray(spikes_E["senders"])
        ax.scatter(times_E, senders_E - 1, s=1, c="red", label="E")

    if spikes_IH is not None and spikes_IA is not None:
        times_I = np.concatenate(
            (
                np.asarray(spikes_IH["times"], dtype=float),
                np.asarray(spikes_IA["times"], dtype=float),
            )
        ) * ms_to_s
        senders_I = np.concatenate(
            (np.asarray(spikes_IH["senders"]), np.asarray(spikes_IA["senders"]))
        )
        ax.scatter(times_I, senders_I - 1, s=1, c="blue", label="IH+IA")
    elif spikes_IH is not None:
        times_I = np.asarray(spikes_IH["times"], dtype=float) * ms_to_s
        senders_I = np.asarray(spikes_IH["senders"])
        ax.scatter(times_I, senders_I - 1, s=1, c="blue", label="IH")
    elif spikes_IA is not None:
        times_I = np.asarray(spikes_IA["times"], dtype=float) * ms_to_s
        senders_I = np.asarray(spikes_IA["senders"])
        ax.scatter(times_I, senders_I - 1, s=1, c="blue", label="IA")

    sim_duration_ms = cfg["experiment"]["simtime_ms"]
    if max_time_ms is not None:
        sim_duration_ms = min(sim_duration_ms, max_time_ms)
    sim_duration_s = sim_duration_ms * ms_to_s
    ax.set_xlim(0, sim_duration_s)
    x_ticks = [tick for tick in (0, 5, 40) if tick <= sim_duration_s + 1e-9]
    ax.set_xticks(x_ticks)
    ax.set_ylim(-1, N_total + 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron index")
    ax.legend(loc="upper left")
    ax.set_title("Spike raster")


def plot_spike_raster(data: Dict[str, Any],
                      cfg: Dict[str, Any],
                      max_time_ms: float | None = None) -> None:
    """
    Wrapper: eigener Plot für das Raster.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_spike_raster_ax(ax, data, cfg, max_time_ms=max_time_ms)
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

    ax.hist(
        vals,
        bins=bins,
        density=True,
        histtype="step",
        color="gray",
        linewidth=1.5,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("PDF")
    ax.tick_params(axis="y", left=False, labelleft=False)


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
    values = np.asarray(mean_rates, float)
    values = values[np.isfinite(values)]  # NaNs rausfiltern
    mu = float(values.mean())
    #print(f"Mean firing rate over neurons: {mu:.3f} Hz")
    # vertikale Linie für den Mittelwert
    ax.axvline(mu, color="red", linestyle="--", label=f"mean = {mu:.2f} Hz")
    ax.legend()


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

    ax.set_xlabel("Pre-synaptic \n neuron index")
    ax.set_ylabel("Post-synaptic \n neuron index")
    tick_positions = [tick for tick in (0, 40, 80) if tick < N_total]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)

    #ax.set_title("Normalized weight matrix")

    return im  # Caller can attach a colorbar


def add_weight_matrix_colorbar(
    ax: plt.Axes,
    im,
    #label: str = "Normalized weight",
    size: str = "4%",
    pad: float = 0.05,
):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=pad)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(["-1", "0", "1"])
    #cbar.set_label(label)
    return cbar


def plot_weight_matrix(
    Wn: np.ndarray,
    cfg: Dict[str, Any],
 ) -> None:
    """
    Wrapper: eigener Plot für die Weight-Matrix.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    im = plot_weight_matrix_ax(ax, Wn, cfg)
    add_weight_matrix_colorbar(ax, im)
    plt.tight_layout()
    plt.show()



# ---------------------------------------------------------------------------
# K(t): mean weight change rate
# ---------------------------------------------------------------------------

def plot_K_ax(ax: plt.Axes,
              K: np.ndarray,
              bins: int = 50) -> None:
    """PDF der mittleren Gewichtsänderungsrate K(t)."""
    _plot_pdf_ax(
        ax,
        K,
        xlabel="Mean change rate of weights (Hz)",
        title="Distribution of mean weight change rate K(t)",
        bins=bins,
    )


def plot_K(K: np.ndarray, bins: int = 50) -> None:
    fig, ax = plt.subplots()
    plot_K_ax(ax, K, bins=bins)
    plt.tight_layout()
    plt.show()