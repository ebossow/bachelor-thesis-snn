# src/setup/network.py

from typing import Dict, Any

import nest
import numpy as np

def _truncated_normal_upper(rng: np.random.Generator, mean: float, std: float, upper: float, lower: float, size: int):
    samples = rng.normal(loc=mean, scale=std, size=size)
    clipped_samples = np.minimum(samples, upper)
    clipped_samples = np.maximum(clipped_samples, lower)
    return clipped_samples

def build_populations(network_cfg: Dict[str, Any],
                      noise_cfg: Dict[str, Any],
                      neuron_model_cfg: Dict[str, Any],
                      excitability_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Neuronenpopulationen (E, IH, IA, ...) anlegen und I_e setzen.
    """

    N_E = network_cfg["N_E"]
    N_IH = network_cfg["N_IH"]
    N_IA = network_cfg["N_IA"]

    model_name = neuron_model_cfg["name"]
    neuron_params = neuron_model_cfg["params"]
    #print("Model Name: ", model_name, " Params: ", neuron_params)

    E = nest.Create(model_name, N_E, params=neuron_params)
    IH = nest.Create(model_name, N_IH, params=neuron_params)
    IA = nest.Create(model_name, N_IA, params=neuron_params)

    # neuron_excitability: I_e pro Neuron
    if excitability_cfg.get("enabled", False):
        rng = np.random.default_rng(
            excitability_cfg["rng_seed"] if excitability_cfg.get("use_seed", False) else None
        )

        mean = excitability_cfg["mean_pA"]
        std = excitability_cfg["std_pA"]

        I_e_values = _truncated_normal_upper(rng=rng, mean=mean, std=std, upper=excitability_cfg["upper_bound_pA"], lower=excitability_cfg["lower_bound_pA"], size=N_E + N_IH + N_IA)

        for gid, I_e in zip(E + IH + IA, I_e_values):
            nest.SetStatus(gid, {"I_e": I_e})

    # neuron_noise: Noise pro Neuron
    if noise_cfg.get("enabled", False):
        mean = noise_cfg["noise_mean"]
        std = noise_cfg["noise_std"]
        noise_w = noise_cfg["weight"]
        noise = nest.Create("noise_generator", params={
            "mean": mean,
            "std": std,
        })

        for gid in E + IH + IA:
            nest.Connect(noise, gid, syn_spec={"weight": noise_w})


    return {"E": E, "IH": IH, "IA": IA}


def connect_synapses(populations: Dict[str, Any],
                     synapse_cfg: Dict[str, Any]) -> None:
    """
    Projektionen gemäß synapse_cfg verbinden.
    synapse_cfg entspricht config['synapses'].
    """

    E = populations["E"]
    IH = populations["IH"]
    IA = populations["IA"]

    base_Wmax = synapse_cfg["base_Wmax"]
    base_LR = synapse_cfg["base_LR"]
    global_lr = synapse_cfg["global_lr"]

    # Beispiel: E_to_X (E -> E, IH, IA)
    e_to_x = synapse_cfg["E_to_X"]
    _connect_projection(
        src=E,
        targets=[E, IH, IA],
        cfg=e_to_x,
        base_Wmax=base_Wmax,
        base_LR=base_LR,
        global_lr=global_lr
    )

    ih_to_x = synapse_cfg["IH_to_X"]
    _connect_projection(
        src=IH,
        targets=[E, IH, IA],
        cfg=ih_to_x,
        base_Wmax=base_Wmax,
        base_LR=base_LR,
        global_lr=global_lr
    )

    ia_to_x = synapse_cfg["IA_to_X"]
    _connect_projection(
        src=IA,
        targets=[E, IH, IA],
        cfg=ia_to_x,
        base_Wmax=base_Wmax,
        base_LR=base_LR,
        global_lr=global_lr
    )


def _connect_projection(src, targets, cfg: Dict[str, Any], base_Wmax, base_LR, global_lr) -> None:
    """
    Hilfsfunktion: eine Projektion mit init-Gewichten aufbauen.
    """
    model = cfg["model"]
    copy_name = cfg["copy_model_name"]
    synapse_parameter = cfg["synapse_parameter"]
    synapse_parameter["Wmax"] = base_Wmax * cfg["Wmax_factor"]
    synapse_parameter["weight"] = synapse_parameter["Wmax"]
    if cfg["copy_model_name"] == "stdp_ex_asym":
        #synapse_parameter["lambda"] = base_LR * cfg["LR_factor"]
        synapse_parameter["lambda"] = 2.347 * global_lr # hardcoded mit max_change_rate_excitatory
    elif cfg["copy_model_name"] == "stdp_inh_sym_hebb":
        #synapse_parameter["eta"] = base_LR * cfg["LR_factor"]
        synapse_parameter["eta"] = -1 * (synapse_parameter["Wmax"] * global_lr * 3) # hardcoded mit max_change_rate_inhibitory * Wmax
    elif cfg["copy_model_name"] == "stdp_inh_sym_antihebb":
        synapse_parameter["eta"] = synapse_parameter["Wmax"] * global_lr * 3 # hardcoded mit max_change_rate_inhibitory * Wmax

    nest.CopyModel(model, copy_name, synapse_parameter)
    conn_all = {"rule": "all_to_all", "allow_autapses": False, "allow_multapses": True}
    syn_spec = {
        "synapse_model": copy_name,
        "weight": nest.random.normal(mean = 0.5, std = 0.2) * synapse_parameter["Wmax"],
        "delay": cfg["delay_ms"]
    }
    for target in targets:
        nest.Connect(src, target, conn_spec=conn_all, syn_spec=syn_spec)
    