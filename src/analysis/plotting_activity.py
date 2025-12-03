# src/analysis/plotting_activity.py

from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def _compute_grid_shape(n_units: int) -> Tuple[int, int]:
    """
    Finde eine (rows, cols)-Form, sodass rows * cols >= n_units
    und das Raster möglichst quadratisch ist.
    """
    if n_units <= 0:
        return 0, 0
    rows = int(np.floor(np.sqrt(n_units)))
    if rows == 0:
        rows = 1
    cols = int(np.ceil(n_units / rows))
    return rows, cols


def _binned_rates_from_indices(
    times_ms: np.ndarray,
    idx: np.ndarray,
    n_units: int,
    t_start_ms: float,
    t_stop_ms: float,
    bin_size_ms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Erzeuge Raten pro Neuron und Zeit-Bin.

    times_ms : Spikezeiten (ms)
    idx      : 0-basierte Neuronenindizes gleicher Länge wie times_ms
    n_units  : Anzahl Neuronen in dieser Population
    t_start_ms, t_stop_ms : Auswertefenster
    bin_size_ms : Binbreite

    Rückgabe:
    rates : (n_units, n_bins), in Hz
    t_centers : (n_bins,), mittlere Zeitpunkte
    """
    times = np.asarray(times_ms, float)
    idx = np.asarray(idx, int)

    mask = (times >= t_start_ms) & (times < t_stop_ms)
    times = times[mask]
    idx = idx[mask]

    if times.size == 0:
        # leere Population
        n_bins = int(np.floor((t_stop_ms - t_start_ms) / bin_size_ms))
        if n_bins <= 0:
            return np.zeros((n_units, 0)), np.array([], float)
        t_edges = t_start_ms + np.arange(n_bins + 1) * bin_size_ms
        t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
        return np.zeros((n_units, n_bins)), t_centers

    n_bins = int(np.floor((t_stop_ms - t_start_ms) / bin_size_ms))
    if n_bins <= 0:
        raise ValueError("Invalid time window or bin size")

    t_edges = t_start_ms + np.arange(n_bins + 1) * bin_size_ms
    t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])

    rates = np.zeros((n_units, n_bins), dtype=float)

    # Bin-Index pro Spike
    bin_idx = np.floor((times - t_start_ms) / bin_size_ms).astype(int)
    valid = (bin_idx >= 0) & (bin_idx < n_bins) & (idx >= 0) & (idx < n_units)
    bin_idx = bin_idx[valid]
    idx = idx[valid]

    # counts pro (unit, bin)
    np.add.at(rates, (idx, bin_idx), 1.0)

    # counts -> Hz
    T = bin_size_ms / 1000.0  # s
    rates /= T

    return rates, t_centers


def plot_population_activity_interactive(
    cfg: Dict[str, Any],
    data: Dict[str, Any],
    bin_size_ms: float = 50.0,
    t_start_ms: float | None = None,
    t_stop_ms: float | None = None,
) -> None:
    """
    Interaktive Visualisierung der Netzwerkaktivität (Firing Rates):

    - Unteres Panel: E-Population als Heatmap in (rows_E x cols_E)-Layout.
    - Oberes Panel: inhibitorische Neuronen (IA_1, IH, IA_2) als
      vertikal gestapelte Blöcke.
    - Ein Slider und ein Play/Pause-Button steuern den aktuellen Zeit-Bin.

    Annahmen:
      - E-GIDs: 1 .. N_E
      - Inhibitorische GIDs folgen direkt danach: IA_1, IH, IA_2
        also: [N_E+1 .. N_E+N_IA_1+N_IH+N_IA_2]
      - Spike-Daten: data["spikes_E"], data["spikes_IH"], data["spikes_IA"].
    """

    net_cfg = cfg["network"]
    N_E     = int(net_cfg["N_E"])
    N_IH    = int(net_cfg["N_IH"])
    N_IA_1  = int(net_cfg.get("N_IA_1", 0))
    N_IA_2  = int(net_cfg.get("N_IA_2", 0))
    N_IA    = N_IA_1 + N_IA_2
    N_I_total = N_IA_1 + N_IH + N_IA_2

    simtime_ms = float(cfg["experiment"]["simtime_ms"])
    if t_start_ms is None:
        t_start_ms = 0.0
    if t_stop_ms is None:
        t_stop_ms = simtime_ms

    # -----------------------
    # E-Spikes / Indizes
    # -----------------------
    spikes_E = data["spikes_E"]
    times_E = np.asarray(spikes_E["times"], float)
    senders_E = np.asarray(spikes_E["senders"], int)

    # E-GIDs 1..N_E → Indizes 0..N_E-1
    mask_E = (senders_E >= 1) & (senders_E <= N_E)
    times_E = times_E[mask_E]
    idx_E = senders_E[mask_E] - 1

    rates_E, t_bins = _binned_rates_from_indices(
        times_E,
        idx_E,
        n_units=N_E,
        t_start_ms=t_start_ms,
        t_stop_ms=t_stop_ms,
        bin_size_ms=bin_size_ms,
    )
    n_bins = rates_E.shape[1]
    if n_bins == 0:
        print("No time bins (check t_start/t_stop/bin_size).")
        return

    # -----------------------
    # Inhibitorische Spikes / Indizes
    # -----------------------
    have_I = ("spikes_IH" in data) and ("spikes_IA" in data) and (N_I_total > 0)
    if have_I:
        spikes_IH = data["spikes_IH"]
        spikes_IA = data["spikes_IA"]

        times_I = np.concatenate(
            [np.asarray(spikes_IH["times"], float),
             np.asarray(spikes_IA["times"], float)]
        )
        senders_I = np.concatenate(
            [np.asarray(spikes_IH["senders"], int),
             np.asarray(spikes_IA["senders"], int)]
        )

        # globale inhibitorische GIDs: N_E+1 .. N_E+N_I_total
        inh_first_gid = N_E + 1
        inh_last_gid  = N_E + N_I_total

        mask_I = (senders_I >= inh_first_gid) & (senders_I <= inh_last_gid)
        times_I = times_I[mask_I]
        # 0-basierte Indizes für I: 0..N_I_total-1 (IA_1, dann IH, dann IA_2)
        idx_I = senders_I[mask_I] - inh_first_gid

        rates_I_all, t_bins_I = _binned_rates_from_indices(
            times_I,
            idx_I,
            n_units=N_I_total,
            t_start_ms=t_start_ms,
            t_stop_ms=t_stop_ms,
            bin_size_ms=bin_size_ms,
        )

        # in Gruppen aufteilen: [IA_1 | IH | IA_2]
        rates_IA1 = rates_I_all[0:N_IA_1, :]
        rates_IH  = rates_I_all[N_IA_1:N_IA_1 + N_IH, :]
        rates_IA2 = rates_I_all[N_IA_1 + N_IH:N_I_total, :]
    else:
        rates_IA1 = np.zeros((0, n_bins))
        rates_IH  = np.zeros((0, n_bins))
        rates_IA2 = np.zeros((0, n_bins))

    # -----------------------
    # Frames bauen (E & I)
    # -----------------------

    # --- E: beliebiges Grid (z.B. 10x10 bei 100 Neuronen, 28x28 bei 784, ...) ---
    rows_E, cols_E = _compute_grid_shape(N_E)
    n_cells_E = rows_E * cols_E

    frames_E_flat = np.zeros((n_bins, n_cells_E), dtype=float)
    frames_E_flat[:, :N_E] = rates_E.T  # (n_bins, N_E)
    frames_E = frames_E_flat.reshape(n_bins, rows_E, cols_E)

    # --- Inhibitoren: IA_1, IH, IA_2 als vertikal gestapelte Blöcke ---
    def _build_block_frames(rates_block: np.ndarray, cols: int) -> tuple[np.ndarray, int]:
        """
        rates_block: (N_block, n_bins)
        cols: gemeinsame Spaltenzahl
        Rückgabe: (frames_block, rows_block)
        frames_block: (n_bins, rows_block, cols)
        """
        N_block = rates_block.shape[0]
        if N_block == 0:
            return np.zeros((n_bins, 0, cols), dtype=float), 0

        rows_block = int(np.ceil(N_block / cols))
        n_cells_block = rows_block * cols

        frames_flat = np.zeros((n_bins, n_cells_block), dtype=float)
        frames_flat[:, :N_block] = rates_block.T  # (n_bins, N_block)
        frames_block = frames_flat.reshape(n_bins, rows_block, cols)
        return frames_block, rows_block

    # Spaltenzahl für alle I-Blöcke gleich wählen (nach größtem Block)
    max_I_block = max(N_IA_1, N_IH, N_IA_2, 1)
    rows_Iref, cols_I = _compute_grid_shape(max_I_block)

    frames_IA1, rows_IA1 = _build_block_frames(rates_IA1, cols_I)
    frames_IH,  rows_IH  = _build_block_frames(rates_IH,  cols_I)
    frames_IA2, rows_IA2 = _build_block_frames(rates_IA2, cols_I)

    rows_I_total_img = rows_IA1 + rows_IH + rows_IA2

    if rows_I_total_img > 0:
        frames_I = np.zeros((n_bins, rows_I_total_img, cols_I), dtype=float)
        # IA_1 oben
        r0 = 0
        r1 = r0 + rows_IA1
        frames_I[:, r0:r1, :] = frames_IA1

        # IH darunter
        r0 = r1
        r1 = r0 + rows_IH
        frames_I[:, r0:r1, :] = frames_IH

        # IA_2 unten
        r0 = r1
        r1 = r0 + rows_IA2
        frames_I[:, r0:r1, :] = frames_IA2
    else:
        frames_I = None

    # gemeinsame Farbskala
    vmax = np.max(frames_E)
    if frames_I is not None:
        vmax = max(vmax, np.max(frames_I))
    if vmax == 0:
        vmax = 1.0

    # -----------------------
    # Matplotlib-Figur: 2 Panels + Slider + Button
    # -----------------------
    if frames_I is not None:
        fig, (ax_I, ax_E) = plt.subplots(
            2, 1,
            figsize=(6, 8),
            gridspec_kw={"height_ratios": [1, 2]},
        )
    else:
        fig, ax_E = plt.subplots(figsize=(6, 6))
        ax_I = None

    plt.subplots_adjust(left=0.1, bottom=0.2)

    # E-Panel
    im_E = ax_E.imshow(
        frames_E[0],
        origin="lower",
        vmin=0.0,
        vmax=vmax,
        cmap="viridis",
        interpolation="nearest",
    )
    ax_E.set_title(f"E activity, t ≈ {t_bins[0]:.1f} ms")
    ax_E.set_xlabel("E col")
    ax_E.set_ylabel("E row")

    # I-Panel (falls vorhanden)
    if ax_I is not None and frames_I is not None:
        im_I = ax_I.imshow(
            frames_I[0],
            origin="lower",
            vmin=0.0,
            vmax=vmax,
            cmap="viridis",
            interpolation="nearest",
        )
        ax_I.set_title("I activity (IA₁ / IH / IA₂)")
        ax_I.set_xlabel("I col")
        ax_I.set_ylabel("I row")
    else:
        im_I = None

    cbar = fig.colorbar(im_E, ax=[ax for ax in [ax_E, ax_I] if ax is not None])
    cbar.set_label("Firing rate (Hz)")

    # Slider
    ax_slider = plt.axes([0.1, 0.08, 0.65, 0.03])
    slider = Slider(
        ax=ax_slider,
        label="Time bin",
        valmin=0,
        valmax=n_bins - 1,
        valinit=0,
        valstep=1,
    )

    def update(_):
        idx = int(slider.val)
        im_E.set_data(frames_E[idx])
        ax_E.set_title(f"E activity, t ≈ {t_bins[idx]:.1f} ms")
        if im_I is not None and frames_I is not None:
            im_I.set_data(frames_I[idx])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Play/Pause
    ax_button = plt.axes([0.8, 0.02, 0.1, 0.05])
    button = Button(ax_button, "Play")
    state = {"playing": False}

    def toggle(event):
        state["playing"] = not state["playing"]
        button.label.set_text("Pause" if state["playing"] else "Play")

    button.on_clicked(toggle)

    # Timer
    timer = fig.canvas.new_timer(interval=100)  # 100 ms per Frame

    def on_timer():
        if not state["playing"]:
            return
        idx = int(slider.val)
        idx = (idx + 1) % n_bins
        slider.set_val(idx)  # triggert update()

    timer.add_callback(on_timer)
    timer.start()

    plt.show()