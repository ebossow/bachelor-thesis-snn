# src/setup/run_simulation.py

from typing import Dict, Any

import nest
import numpy as np
import time


class _PlasticityHandle:
    """Store original plasticity parameters to toggle learning on/off."""

    def __init__(self, connections: Any, parameter: str | None):
        self.connections = connections
        self.parameter = parameter
        self._enabled_values: np.ndarray | None = None
        self._disabled_values: np.ndarray | None = None
        self.valid = self._initialize()

    def _initialize(self) -> bool:
        if not self.parameter or self.connections is None or len(self.connections) == 0:
            return False
        try:
            values = np.array(self.connections.get(self.parameter), dtype=float)
        except Exception:
            return False
        self._enabled_values = values
        self._disabled_values = np.zeros_like(values)
        return True

    def apply(self, enabled: bool) -> None:
        if not self.valid or self._enabled_values is None or self._disabled_values is None:
            return
        values = self._enabled_values if enabled else self._disabled_values
        if len(values) == 0:
            return
        self.connections.set({self.parameter: values})


class _PlasticityController:
    """Coordinate multiple plasticity handles."""

    def __init__(self, handles: list[_PlasticityHandle]):
        self._handles = [h for h in handles if h and h.valid]
        self._state: bool | None = None

    def set_state(self, enabled: bool) -> None:
        if not self._handles:
            self._state = enabled
            return
        if self._state is enabled:
            return
        for handle in self._handles:
            handle.apply(enabled)
        self._state = enabled


def _build_phase_segments(total_ms: float, schedule: dict | None) -> list[dict[str, float | bool | str]]:
    """Create ordered segments describing when plasticity is enabled."""

    if not schedule:
        return [
            {
                "name": "full",
                "start_ms": 0.0,
                "end_ms": total_ms,
                "plasticity_enabled": True,
            }
        ]

    segments = schedule.get("segments") if isinstance(schedule, dict) else None
    cleaned: list[dict[str, float | bool | str]] = []

    if segments:
        for seg in segments:
            start = float(seg.get("start_ms", 0.0))
            end = float(seg.get("end_ms", start))
            cleaned.append(
                {
                    "name": str(seg.get("name", "segment")),
                    "start_ms": start,
                    "end_ms": end,
                    "plasticity_enabled": bool(seg.get("plasticity_enabled", True)),
                }
            )
    else:
        pre = max(0.0, float(schedule.get("pre_ms", 0.0)))
        post = max(0.0, float(schedule.get("post_ms", 0.0)))
        main = max(0.0, float(schedule.get("main_ms", total_ms - pre - post)))
        cursor = 0.0
        if pre > 0.0:
            cleaned.append(
                {
                    "name": "pre_frozen",
                    "start_ms": cursor,
                    "end_ms": cursor + pre,
                    "plasticity_enabled": False,
                }
            )
            cursor += pre
        cleaned.append(
            {
                "name": "main_plastic",
                "start_ms": cursor,
                "end_ms": cursor + main,
                "plasticity_enabled": True,
            }
        )
        cursor += main
        if post > 0.0:
            cleaned.append(
                {
                    "name": "post_frozen",
                    "start_ms": cursor,
                    "end_ms": cursor + post,
                    "plasticity_enabled": False,
                }
            )
            cursor += post

    # Ensure segments cover [0, total_ms] and are sorted.
    cleaned.sort(key=lambda seg: float(seg.get("start_ms", 0.0)))
    if not cleaned:
        cleaned.append(
            {
                "name": "full",
                "start_ms": 0.0,
                "end_ms": total_ms,
                "plasticity_enabled": True,
            }
        )

    last_end = cleaned[-1]["end_ms"]
    if abs(float(last_end) - float(total_ms)) > 1e-6:
        cleaned[-1]["end_ms"] = total_ms

    return cleaned


def _apply_weight_decay_clipped(conns,
                                decay_factor: float,
                                clip_min: float,
                                clip_max) -> None:
    """
    Subtract decay_factor from all weights and clip to [clip_min, clip_max].
    """
    weights = np.array(conns.get("weight"), dtype=float)
    #weights *= decay_factor
    weights -= decay_factor
    np.clip(weights, clip_min, clip_max, out=weights)
    conns.set({"weight": weights})


def run_simulation(
    simtime_ms: float,
    recording_devices: Dict[str, Any],
    populations: Dict[str, Any],
    synapse_cfg: Dict[str, Any],
    record_weight_trajectory: bool = False,
    snapshot_times_ms: list[float] | None = None,
    phase_schedule: dict | None = None,
) -> Dict[str, Any]:
    """
    Execute a NEST simulation with optional phase schedule and weight tracking.

    - Weight decay (if enabled) is applied after the configured chunk cadence
      while plasticity is active.
    - When record_weight_trajectory=True, weights are sampled at the requested
      snapshot times (plus the final time) for all phases.
    - phase_schedule allows temporarily freezing plasticity (learning rates).
    """
    sim_start = time.time()

    decay_cfg = synapse_cfg.get("weight_decay", {})
    decay_enabled = decay_cfg.get("enabled", False)

    # Populationen
    E  = populations["E"]
    IH = populations["IH"]
    IA = populations["IA"]

    # Synapsenmodell-Namen (müssen zu connect_synapses passen)
    model_E  = synapse_cfg["E_to_X"]["copy_model_name"]   # z.B. "stdp_ex_asym"
    model_IH = synapse_cfg["IH_to_X"]["copy_model_name"]  # z.B. "stdp_inh_sym"
    model_IA = synapse_cfg["IA_to_X"]["copy_model_name"]  # z.B. "stdp_inh_sym_anti"

    # ConnectionCollections pro Synapsetyp (brauchen wir für Decay und für Trajektorie)
    conn_E  = nest.GetConnections(source=E,  synapse_model=model_E)
    conn_IH = nest.GetConnections(source=IH, synapse_model=model_IH)
    conn_IA = nest.GetConnections(source=IA, synapse_model=model_IA)

    all_neurons = E + IH + IA
    conns = nest.GetConnections(all_neurons, all_neurons)
    w_now = np.array(conns.get("weight"), float)
    print(
        "In run_simulation:",
        "mean =", w_now.mean(),
        "std =", w_now.std(),
        "min =", w_now.min(),
        "max =", w_now.max(),
    )

    total_simtime_ms = float(simtime_ms)
    segments = _build_phase_segments(total_simtime_ms, phase_schedule)
    plasticity_controller = _PlasticityController(
        [
            _PlasticityHandle(conn_E, "lambda"),
            _PlasticityHandle(conn_IH, "eta"),
            _PlasticityHandle(conn_IA, "eta"),
        ]
    )

    # Gewichtstrajektorie (optional)
    weight_times: list[float] = []
    weight_snapshots: list[np.ndarray] = []
    weight_sources: np.ndarray | None = None
    weight_targets: np.ndarray | None = None

    def _snapshot_weights(current_time_ms: float) -> None:
        nonlocal weight_sources, weight_targets
        if not record_weight_trajectory:
            return
        wE  = np.array(conn_E.get("weight"),  dtype=float)
        wIH = np.array(conn_IH.get("weight"), dtype=float)
        wIA = np.array(conn_IA.get("weight"), dtype=float)
        if weight_sources is None or weight_targets is None:
            src_E = np.array(conn_E.get("source"), dtype=int)
            tgt_E = np.array(conn_E.get("target"), dtype=int)
            src_IH = np.array(conn_IH.get("source"), dtype=int)
            tgt_IH = np.array(conn_IH.get("target"), dtype=int)
            src_IA = np.array(conn_IA.get("source"), dtype=int)
            tgt_IA = np.array(conn_IA.get("target"), dtype=int)
            weight_sources = np.concatenate([src_E, src_IH, src_IA])
            weight_targets = np.concatenate([tgt_E, tgt_IH, tgt_IA])
        w_all = np.concatenate([wE, wIH, wIA])
        weight_times.append(current_time_ms)
        weight_snapshots.append(w_all)

    current_time = 0.0
    _snapshot_weights(current_time)  # initialer Snapshot bei t = 0

    total_duration_ms = float(segments[-1]["end_ms"]) if segments else total_simtime_ms
    eps = 1e-9

    effective_snapshot_times: list[float] = []
    if record_weight_trajectory:
        raw_times = snapshot_times_ms or []
        filtered = sorted({float(t) for t in raw_times if 0.0 < t <= total_duration_ms + eps})
        if not filtered or filtered[-1] < total_duration_ms - eps:
            filtered.append(total_duration_ms)
        effective_snapshot_times = filtered
    snapshot_idx = 0

    if decay_enabled:
        chunk_ms = float(decay_cfg.get("chunk_ms", total_duration_ms))
        if chunk_ms <= 0.0:
            raise ValueError("chunk_ms must be > 0 when decay is enabled")
        decay_every_n = max(1, int(decay_cfg.get("every_n_chunks", 1)))
        decay_summand = float(synapse_cfg.get("E_to_X", {}).get("decay_summand", decay_cfg.get("decay_summand", 0.0)))
        clip_max = np.array(conn_E.get("Wmax"), dtype=float) if len(conn_E) else np.array([])
    else:
        chunk_ms = 0.0
        decay_every_n = 0
        decay_summand = 0.0
        clip_max = np.array([])

    chunk_progress = 0.0
    chunk_counter = 0

    segment_idx = 0
    current_segment = segments[segment_idx]
    plasticity_enabled = bool(current_segment.get("plasticity_enabled", True))
    plasticity_controller.set_state(plasticity_enabled)
    segment_end = float(current_segment.get("end_ms", total_duration_ms))

    while current_time < total_duration_ms - eps:
        next_snapshot = (
            effective_snapshot_times[snapshot_idx]
            if snapshot_idx < len(effective_snapshot_times)
            else None
        )

        targets = [segment_end]
        if next_snapshot is not None:
            targets.append(next_snapshot)
        if decay_enabled and plasticity_enabled:
            remaining_chunk = chunk_ms - chunk_progress if chunk_progress > eps else chunk_ms
            targets.append(current_time + remaining_chunk)

        target_time = min(targets)
        dt = target_time - current_time
        if dt > eps:
            nest.Simulate(dt)
            if decay_enabled and plasticity_enabled:
                chunk_progress += dt
            current_time = target_time
        else:
            current_time = target_time

        # Apply weight decay whenever enough chunk time accumulated.
        if decay_enabled and plasticity_enabled and chunk_progress >= chunk_ms - eps:
            chunk_progress = 0.0
            chunk_counter += 1
            if chunk_counter % decay_every_n == 0 and len(conn_E):
                _apply_weight_decay_clipped(conn_E, decay_summand, 0.0, clip_max)

        if effective_snapshot_times and next_snapshot is not None and current_time >= next_snapshot - eps:
            _snapshot_weights(current_time)
            snapshot_idx += 1

        if current_time >= segment_end - eps:
            if segment_idx + 1 < len(segments):
                segment_idx += 1
                current_segment = segments[segment_idx]
                plasticity_enabled = bool(current_segment.get("plasticity_enabled", True))
                plasticity_controller.set_state(plasticity_enabled)
                chunk_progress = 0.0
                segment_end = float(current_segment.get("end_ms", total_duration_ms))
            else:
                segment_end = total_duration_ms

        if segment_idx >= len(segments) - 1 and current_time >= total_duration_ms - eps:
            break

    if record_weight_trajectory and snapshot_idx < len(effective_snapshot_times):
        while snapshot_idx < len(effective_snapshot_times):
            target = effective_snapshot_times[snapshot_idx]
            if target - current_time > eps:
                nest.Simulate(target - current_time)
                current_time = target
            _snapshot_weights(current_time)
            snapshot_idx += 1

    if segments:
        plasticity_controller.set_state(True)

    # Eventdaten einsammeln (wie bisher)
    data: Dict[str, Any] = {}

    for name, dev in recording_devices.items():
        events = nest.GetStatus(dev, "events")[0]
        data[name] = {
            "times": np.array(events["times"]),
            "senders": np.array(events["senders"]),
        }

    # Gewichtstrajektorie anhängen, falls aufgezeichnet
    if record_weight_trajectory and weight_snapshots:
        traj: Dict[str, Any] = {
            "times": np.array(weight_times, dtype=float),
            "weights": np.stack(weight_snapshots, axis=0).astype(float)
        }
        if weight_sources is not None and weight_targets is not None:
            traj["sources"] = weight_sources.astype(int)
            traj["targets"] = weight_targets.astype(int)
        data["weights_trajectory"] = traj

    sim_end = time.time()
    print(f"Simulation completed in {sim_end - sim_start:.2f} seconds.")

    return data