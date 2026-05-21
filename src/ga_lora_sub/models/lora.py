from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A Linear layer with frozen base weight, optional subtraction delta, and trainable LoRA."""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank

        self.weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None

        self.lora_A = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, self.out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.register_buffer("sub_delta", torch.zeros_like(self.weight), persistent=True)
        self.reset_lora_parameters()

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def zero_lora_parameters(self) -> None:
        nn.init.zeros_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def get_delta_weight(self) -> torch.Tensor:
        return (self.lora_B.weight @ self.lora_A.weight) * self.scaling

    @torch.no_grad()
    def set_sub_delta(self, delta: torch.Tensor | None) -> None:
        if delta is None:
            self.sub_delta.zero_()
        else:
            self.sub_delta.copy_(delta.to(device=self.sub_delta.device, dtype=self.sub_delta.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight + self.sub_delta, self.bias)
        lora = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base + lora


def _get_parent_module(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def add_lora_to_model(
    model: nn.Module,
    target_modules: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float,
) -> List[str]:
    """Replace selected nn.Linear modules by LoRALinear.

    `target_modules` are substring patterns matched against full module names.
    Example for timm ViT: ["attn.qkv"] or ["attn.qkv", "attn.proj"].
    """
    patterns = list(target_modules)
    replaced: List[str] = []
    named = list(model.named_modules())
    for name, module in named:
        if not isinstance(module, nn.Linear):
            continue
        if not any(pat in name for pat in patterns):
            continue
        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(name)
    if not replaced:
        raise RuntimeError(f"No nn.Linear modules matched target_modules={patterns}")
    return replaced


def iter_lora_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def freeze_non_lora(model: nn.Module, train_head: bool = True) -> None:
    for name, p in model.named_parameters():
        p.requires_grad_(False)
    for _, module in iter_lora_modules(model):
        module.lora_A.weight.requires_grad_(True)
        module.lora_B.weight.requires_grad_(True)
    if train_head and hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad_(True)


def reset_lora_parameters(model: nn.Module) -> None:
    for _, module in iter_lora_modules(model):
        module.reset_lora_parameters()


def zero_lora_parameters(model: nn.Module) -> None:
    for _, module in iter_lora_modules(model):
        module.zero_lora_parameters()


@torch.no_grad()
def collect_lora_deltas(model: nn.Module, device: torch.device | str = "cpu") -> Dict[str, torch.Tensor]:
    return {name: module.get_delta_weight().detach().to(device) for name, module in iter_lora_modules(model)}


@torch.no_grad()
def empty_lora_deltas(model: nn.Module, device: torch.device | str = "cpu") -> Dict[str, torch.Tensor]:
    return {name: torch.zeros_like(module.weight, device=device) for name, module in iter_lora_modules(model)}


@torch.no_grad()
def add_delta_dict(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        if k in a and k in b:
            out[k] = a[k].detach().cpu() + b[k].detach().cpu()
        elif k in a:
            out[k] = a[k].detach().cpu().clone()
        else:
            out[k] = b[k].detach().cpu().clone()
    return out


@torch.no_grad()
def scale_delta_dict(a: Dict[str, torch.Tensor], scales: Dict[str, float] | float) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in a.items():
        s = scales if isinstance(scales, (float, int)) else scales.get(k, 1.0)
        out[k] = v.detach().cpu() * float(s)
    return out


@torch.no_grad()
def set_subtraction_deltas(model: nn.Module, deltas: Dict[str, torch.Tensor], sign: float = 1.0) -> None:
    for name, module in iter_lora_modules(model):
        delta = deltas.get(name)
        if delta is None:
            module.set_sub_delta(None)
        else:
            module.set_sub_delta(delta.to(module.weight.device) * float(sign))


@torch.no_grad()
def clear_subtraction_deltas(model: nn.Module) -> None:
    for _, module in iter_lora_modules(model):
        module.set_sub_delta(None)


def lora_state_dict_cpu(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    state = {}
    for name, module in iter_lora_modules(model):
        state[name] = {
            "A": module.lora_A.weight.detach().cpu().clone(),
            "B": module.lora_B.weight.detach().cpu().clone(),
            "sub_delta": module.sub_delta.detach().cpu().clone(),
        }
    return state
