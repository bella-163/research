from __future__ import annotations

from typing import Dict

import torch


def compute_gammas(
    method_cfg: dict,
    alignments: Dict[str, float],
    old_deltas: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    method = str(method_cfg.get("name", "adaptive")).lower()
    if method in {"none", "no_subtraction"}:
        return {k: 0.0 for k in old_deltas.keys()}
    if method in {"fixed", "fixed_subtraction"}:
        return {k: 1.0 for k in old_deltas.keys()}
    if method in {"adaptive", "gradient", "gradient_aware"}:
        rho = float(method_cfg.get("rho", 0.5))
        gmin = float(method_cfg.get("gamma_min", 0.5))
        gmax = float(method_cfg.get("gamma_max", 1.5))
        gammas = {}
        for k in old_deltas.keys():
            s = float(alignments.get(k, 0.0))
            gamma = 1.0 - rho * s
            gamma = max(gmin, min(gmax, gamma))
            gammas[k] = float(gamma)
        return gammas
    raise ValueError(f"Unknown method.name: {method}")
