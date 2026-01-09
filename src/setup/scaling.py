"""Utilities for applying post-initialization scaling to synaptic weights."""

from __future__ import annotations

from typing import Any, Dict

import logging

import nest
import numpy as np

logger = logging.getLogger(__name__)


def _scale_block(source, target, factor: float) -> None:
    """Multiply all weights between ``source`` and ``target`` by ``factor``."""
    if factor == 1.0:
        return
    conns = nest.GetConnections(source=source, target=target)
    n_conn = len(conns)
    if n_conn == 0:
        return
    weights = np.asarray(conns.get("weight"), dtype=float)
    weights *= factor
    conns.set({"weight": weights})


def apply_post_init_scaling(populations: Dict[str, Any], scaling_cfg: Dict[str, Any] | None) -> None:
    """Apply alpha/beta scaling factors after network initialization."""
    if not scaling_cfg:
        return

    mode = scaling_cfg.get("apply_mode", "").lower()
    if mode not in ("post_init", "postinit", "after_init", ""):
        logger.debug("Scaling mode %s not handled", mode)
        return

    alpha = float(scaling_cfg.get("alpha", scaling_cfg.get("alpha_default", 1.0)))
    beta = float(scaling_cfg.get("beta", scaling_cfg.get("beta_default", 1.0)))

    pops_E = populations.get("E")
    pops_IH = populations.get("IH")
    pops_IA = populations.get("IA")

    if pops_E is None:
        return

    if alpha != 1.0:
        logger.info("Applying alpha scaling %.3f to E->E weights", alpha)
        _scale_block(pops_E, pops_E, alpha)

    if beta != 1.0 and pops_IH is not None:
        logger.info("Applying beta scaling %.3f to IH->E weights", beta)
        _scale_block(pops_IH, pops_E, beta)

    if beta != 1.0 and pops_IA is not None:
        logger.info("Applying beta scaling %.3f to IA->E weights", beta)
        _scale_block(pops_IA, pops_E, beta)
