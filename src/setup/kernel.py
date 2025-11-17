# src/setup/kernel.py

from typing import Dict, Any

import nest


def init_kernel(experiment_cfg: Dict[str, Any]) -> None:
    """
    NEST-Kernel initialisieren.
    Erwartet experiment_cfg wie aus config['experiment'].
    """
    simtime_ms = experiment_cfg["simtime_ms"]
    nest_cfg = experiment_cfg.get("nest", {})

    resolution = nest_cfg.get("resolution_ms", 0.1)
    threads = nest_cfg.get("threads", 1)
    seed = experiment_cfg.get("seed", None)

    nest.ResetKernel()
    nest.SetKernelStatus({
        "resolution": resolution,
        "local_num_threads": threads,
    })

    if seed is not None:
        # sehr einfache Seeding-Variante; kannst du später verfeinern
        nest.SetKernelStatus({
            "rng_seed": seed
        })