"""Helpers to derive stimulus-driven populations from recorded metadata."""

from __future__ import annotations

from typing import Any, Dict, List

MNIST_PATTERN_THRESHOLD_P_A = 1e-9


def extract_stimulus_populations(
    stim_metadata: Dict[str, Any] | None,
    max_groups: int = 2,
) -> List[Dict[str, Any]]:
    """Return up to ``max_groups`` populations that received patterned stimuli."""
    if not stim_metadata:
        return []

    dc_groups = _extract_dc_populations(stim_metadata.get("dc"))
    if dc_groups:
        return dc_groups[:max_groups]

    mnist_groups = _extract_mnist_populations(stim_metadata.get("mnist"))
    if mnist_groups:
        return mnist_groups[:max_groups]

    return []


def _extract_dc_populations(dc_meta: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(dc_meta, dict):
        return []
    groups: List[Dict[str, Any]] = []
    blocks = dc_meta.get("blocks", [])
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("population") != "E":
            continue
        neuron_ids = [int(nid) for nid in block.get("neuron_ids", [])]
        if not neuron_ids:
            continue
        pattern_idx = block.get("pattern_index")
        label = block.get("label") or (
            f"Pattern {pattern_idx}" if pattern_idx is not None else f"P{len(groups) + 1}"
        )
        groups.append(
            {
                "label": label,
                "display_label": f"P{len(groups) + 1}: {label}",
                "neuron_ids": sorted(neuron_ids),
                "pattern_index": pattern_idx,
                "source": "dc",
            }
        )
    groups.sort(key=lambda g: (g.get("pattern_index", -1), g["label"]))
    return groups


def _extract_mnist_populations(mnist_meta: Dict[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(mnist_meta, dict):
        return []
    e_meta = mnist_meta.get("E") or {}
    neuron_ids = [int(nid) for nid in e_meta.get("neuron_ids", [])]
    pattern_currents = e_meta.get("pattern_currents_pA") or []
    labels = mnist_meta.get("pattern_labels") or []

    if not neuron_ids or not pattern_currents:
        return []

    groups: List[Dict[str, Any]] = []
    for idx, currents in enumerate(pattern_currents):
        active_ids: List[int] = []
        for nid, curr in zip(neuron_ids, currents):
            try:
                curr_val = float(curr)
            except (TypeError, ValueError):
                continue
            if curr_val > MNIST_PATTERN_THRESHOLD_P_A:
                active_ids.append(int(nid))
        if not active_ids:
            continue
        label = labels[idx] if idx < len(labels) else idx
        groups.append(
            {
                "label": f"Pattern {label}",
                "display_label": f"P{len(groups) + 1}: Pattern {label}",
                "neuron_ids": sorted(active_ids),
                "pattern_index": idx,
                "source": "mnist",
            }
        )
    return groups
