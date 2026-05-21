from __future__ import annotations

from typing import Dict, List

import numpy as np


def average_accuracy(acc_matrix: np.ndarray, stage: int) -> float:
    vals = acc_matrix[stage, : stage + 1]
    vals = vals[~np.isnan(vals)]
    return float(np.mean(vals)) if len(vals) else float("nan")


def forgetting(acc_matrix: np.ndarray, stage: int) -> float:
    if stage <= 0:
        return 0.0
    fs = []
    for task_id in range(stage):
        best = np.nanmax(acc_matrix[: stage + 1, task_id])
        cur = acc_matrix[stage, task_id]
        if not np.isnan(best) and not np.isnan(cur):
            fs.append(best - cur)
    return float(np.mean(fs)) if fs else 0.0


def summarize_stage(acc_matrix: np.ndarray, stage: int, drift: float | None = None) -> Dict[str, float]:
    out = {
        "stage": int(stage),
        "average_accuracy": average_accuracy(acc_matrix, stage),
        "forgetting": forgetting(acc_matrix, stage),
    }
    if drift is not None:
        out["feature_drift"] = float(drift)
    return out
