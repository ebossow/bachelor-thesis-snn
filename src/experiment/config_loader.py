# src/experiment/config_loader.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_base_config(path: Path) -> Dict[str, Any]:
    """
    YAML-Config laden und als Dict zurückgeben.
    """
    with path.open("r") as f:
        cfg = yaml.safe_load(f)

    cfg = compute_derived_parameters(cfg)
    return cfg


def compute_derived_parameters(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dinge berechnen, die nicht direkt in YAML stehen sollen:
    - Rheobase aus neuron_model.params
    - absolute Excitability-Parameter (mean/upper/lower) aus Offsets
    """

    neuron_params = cfg["neuron_model"]["params"]
    g_L = neuron_params["g_L"]          # nS
    V_th = neuron_params["V_th"]        # mV
    E_L = neuron_params["E_L"]          # mV
    Delta_T = neuron_params["Delta_T"]  # mV

    # I_rheo in pA (nS * mV = pA)
    I_rheo = g_L * (V_th - E_L - Delta_T)

    cfg.setdefault("derived", {})
    cfg["derived"]["rheobase_pA"] = I_rheo

    # neuron_excitability aus Offsets ableiten
    ex_cfg = cfg.get("neuron_excitability", {})
    mean_offset = ex_cfg.get("mean_offset_pA", 0.0)
    std_pA = ex_cfg.get("std_pA", 0.0)
    upper_offset = ex_cfg.get("upper_offset_pA", 0.0)
    lower_offset = ex_cfg.get("lower_offset_pA", 0.0)

    ex_cfg["mean_pA"] = I_rheo + mean_offset
    ex_cfg["upper_bound_pA"] = I_rheo + upper_offset
    ex_cfg["lower_bound_pA"] = I_rheo + lower_offset
    ex_cfg["std_pA"] = std_pA

    cfg["neuron_excitability"] = ex_cfg

    return cfg