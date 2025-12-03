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
    N_IA_1 = network_cfg["N_IA_1"]
    N_IH = network_cfg["N_IH"]
    N_IA_2 = network_cfg["N_IA_2"]
    N_IA = N_IA_1 + N_IA_2


    model_name = neuron_model_cfg["name"]
    neuron_params = neuron_model_cfg["params"]
    #print("Model Name: ", model_name, " Params: ", neuron_params)

    E = nest.Create(model_name, N_E, params=neuron_params)
    IA_1 = nest.Create(model_name, N_IA_1, params=neuron_params)
    IH = nest.Create(model_name, N_IH, params=neuron_params)
    IA_2 = nest.Create(model_name, N_IA_2, params=neuron_params)
    IA = IA_1 + IA_2

    #print(f"Created populations: E({E}), IH({IH}), IA({IA_1}+{IA_2})")
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
        noise_mean = noise_cfg["noise_mean"]
        noise_std = noise_cfg["noise_std"]
        noise_w = noise_cfg["weight"]
        noise = nest.Create("noise_generator", params={
            "mean": noise_mean,
            "std": noise_std,
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
    long_run = synapse_cfg.get("long_run", False)

    # Beispiel: E_to_X (E -> E, IH, IA)
    e_to_x = synapse_cfg["E_to_X"]
    _connect_projection(
        src=E,
        targets=[E, IH, IA],
        cfg=e_to_x,
        base_Wmax=base_Wmax,
        base_LR=base_LR,
        global_lr=global_lr,
        long_run=long_run
    )

    ih_to_x = synapse_cfg["IH_to_X"]
    _connect_projection(
        src=IH,
        targets=[E, IH, IA],
        cfg=ih_to_x,
        base_Wmax=base_Wmax,
        base_LR=base_LR,
        global_lr=global_lr,
        long_run=long_run
    )

    ia_to_x = synapse_cfg["IA_to_X"]
    _connect_projection(
        src=IA,
        targets=[E, IH, IA],
        cfg=ia_to_x,
        base_Wmax=base_Wmax,
        base_LR=base_LR,
        global_lr=global_lr,
        long_run=long_run
    )

    _init_projection_weights(e_to_x["copy_model_name"],  base_Wmax * e_to_x["Wmax_factor"])
    _init_projection_weights(ih_to_x["copy_model_name"], base_Wmax * ih_to_x["Wmax_factor"])
    _init_projection_weights(ia_to_x["copy_model_name"], base_Wmax * ia_to_x["Wmax_factor"])

    synapse_cfg["weight_decay"]["decay_summand"] = synapse_cfg["E_to_X"]["synapse_parameter"]["lambda"] * 0.03
    #print("Decay Summand set to: ", synapse_cfg["weight_decay"]["decay_summand"])

def _connect_projection(src, targets, cfg: Dict[str, Any], base_Wmax, base_LR, global_lr, long_run) -> None:
    """
    Hilfsfunktion: eine Projektion mit init-Gewichten aufbauen.
    """
    model = cfg["model"]
    copy_name = cfg["copy_model_name"]
    synapse_parameter = cfg["synapse_parameter"]
    synapse_parameter["Wmax"] = base_Wmax * cfg["Wmax_factor"]
    synapse_parameter["weight"] = synapse_parameter["Wmax"]

    manual_lambda = cfg.get("manual_lambda", None)
    manual_eta    = cfg.get("manual_eta", None)

    if copy_name == "stdp_ex_asym":
        if manual_lambda is not None:
            synapse_parameter["lambda"] = float(manual_lambda)
        else:
            # Standardformel aus dem Paper
            synapse_parameter["lambda"] = 2.347 * global_lr
        # Decay-Summand immer aus tatsächlicher lambda
        cfg["decay_summand"] = synapse_parameter["lambda"] * 0.03

    elif copy_name == "stdp_inh_sym_hebb":
        if manual_eta is not None:
            synapse_parameter["eta"] = float(manual_eta)
        else:
            synapse_parameter["eta"] = -1 * (synapse_parameter["Wmax"] * global_lr * 3)

    elif copy_name == "stdp_inh_sym_antihebb":
        if manual_eta is not None:
            synapse_parameter["eta"] = float(manual_eta)
        else:
            synapse_parameter["eta"] = synapse_parameter["Wmax"] * global_lr * 3

    if long_run:
        if cfg["copy_model_name"] == "stdp_ex_asym":
            synapse_parameter["Wmax"] = 5000
        else:
            synapse_parameter["Wmax"] = -5000

    nest.CopyModel(model, copy_name, synapse_parameter)
    conn_all = {"rule": "all_to_all", "allow_autapses": False, "allow_multapses": True}
    syn_spec = {
        "synapse_model": copy_name,
        #"weight": nest.random.normal(mean = 0, std = 0.2) * synapse_parameter["Wmax"],
        "delay": cfg["delay_ms"]
    }
    for target in targets:
        nest.Connect(src, target, conn_spec=conn_all, syn_spec=syn_spec)

    
def _init_projection_weights(copy_name: str, Wmax: float, std_rel: float = 0.2) -> None:
    conns = nest.GetConnections(synapse_model=copy_name)
    n = len(conns)
    if n == 0:
        return

    rng = np.random.default_rng()
    samples = rng.normal(loc=0.0, scale=std_rel, size=n) * Wmax
    #print("Wmax:", Wmax)

    if copy_name == "stdp_ex_asym":
        w_min, w_max = 0.0, float(Wmax)
    else:
        w_min, w_max = float(Wmax), 0.0

    weights = np.clip(samples, w_min, w_max)
    conns.set({"weight": weights})