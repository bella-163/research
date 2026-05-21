from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj: Dict[str, Any], path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def save_json(obj: Dict[str, Any], path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_value(raw: str) -> Any:
    low = raw.lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def set_by_dot(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = config
    parts = dotted_key.split(".")
    for p in parts[:-1]:
        if p not in cur or cur[p] is None:
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def apply_overrides(config: Dict[str, Any], overrides: Iterable[str] | None) -> Dict[str, Any]:
    cfg = deepcopy(config)
    if not overrides:
        return cfg
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item}")
        k, v = item.split("=", 1)
        set_by_dot(cfg, k, parse_value(v))
    return cfg


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
