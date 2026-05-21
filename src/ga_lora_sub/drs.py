from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm

from .models.lora import LoRALinear, iter_lora_modules, scale_delta_dict, set_subtraction_deltas


@dataclass
class DRSDiagnostics:
    ranks: Dict[str, int]
    counts: Dict[str, int]
    eigen_top: Dict[str, float]
    eigen_sum: Dict[str, float]


def _flatten_linear_input(x: torch.Tensor) -> torch.Tensor:
    """Return input as a 2D matrix [num_tokens_or_samples, in_features]."""
    if x.ndim < 2:
        raise ValueError(f"Expected linear input with ndim >= 2, got shape={tuple(x.shape)}")
    return x.detach().reshape(-1, x.shape[-1]).float()


def collect_input_covariances(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int = 8,
    max_rows_per_batch: int = 4096,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Collect uncentered input covariance X^T X for each LoRA-wrapped layer.

    DRS in LoRA-Subtraction is built from the input features of each linear
    layer under a subtracted model. We avoid storing all features by accumulating
    X^T X and the number of observed rows.
    """
    was_training = model.training
    model.eval()

    covs: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(module: LoRALinear, inputs, output):
            if not inputs:
                return
            x = _flatten_linear_input(inputs[0])
            if max_rows_per_batch > 0 and x.size(0) > max_rows_per_batch:
                # Deterministic enough for experiments; randomness is controlled by torch seed.
                idx = torch.randperm(x.size(0), device=x.device)[:max_rows_per_batch]
                x = x.index_select(0, idx)
            x_cpu = x.cpu()
            cov = x_cpu.t().matmul(x_cpu)
            if name not in covs:
                covs[name] = cov
                counts[name] = int(x_cpu.size(0))
            else:
                covs[name].add_(cov)
                counts[name] += int(x_cpu.size(0))
        return hook

    for name, module in iter_lora_modules(model):
        handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(loader, desc="collect DRS covariance", leave=False)):
            images = images.to(device, non_blocking=True)
            _ = model(images)
            if batch_idx + 1 >= max_batches:
                break

    for h in handles:
        h.remove()
    model.train(was_training)
    return covs, counts


def compute_projectors_from_covariances(
    covs: Dict[str, torch.Tensor],
    counts: Dict[str, int],
    rank: Optional[int] = None,
    energy: Optional[float] = 0.95,
    max_rank: Optional[int] = None,
    eps: float = 1e-8,
) -> Tuple[Dict[str, torch.Tensor], DRSDiagnostics]:
    """Compute per-layer DRS bases from uncentered covariance matrices.

    Each returned projector basis P has shape [in_features, k]. During training,
    LoRA A matrices are constrained by A <- A P P^T, so the low-rank update
    BA lies in the input subspace estimated by DRS.
    """
    projectors: Dict[str, torch.Tensor] = {}
    ranks: Dict[str, int] = {}
    eigen_top: Dict[str, float] = {}
    eigen_sum: Dict[str, float] = {}

    for name, cov in covs.items():
        n = max(int(counts.get(name, 0)), 1)
        c = (cov / float(n)).float()
        # Numerical symmetry helps eigendecomposition.
        c = 0.5 * (c + c.t())
        vals, vecs = torch.linalg.eigh(c)
        order = torch.argsort(vals, descending=True)
        vals = vals[order].clamp_min(0.0)
        vecs = vecs[:, order]

        dim = vals.numel()
        if rank is not None and int(rank) > 0:
            k = min(int(rank), dim)
        elif energy is not None and 0.0 < float(energy) < 1.0 and vals.sum().item() > eps:
            cum = torch.cumsum(vals, dim=0)
            k = int(torch.searchsorted(cum, float(energy) * vals.sum()).item()) + 1
            k = min(max(k, 1), dim)
        else:
            k = dim
        if max_rank is not None and int(max_rank) > 0:
            k = min(k, int(max_rank), dim)

        p = vecs[:, :k].contiguous()
        projectors[name] = p
        ranks[name] = int(k)
        eigen_top[name] = float(vals[0].item()) if dim > 0 else 0.0
        eigen_sum[name] = float(vals.sum().item())

    return projectors, DRSDiagnostics(ranks=ranks, counts=dict(counts), eigen_top=eigen_top, eigen_sum=eigen_sum)


def build_drs_projectors(
    model: nn.Module,
    loader,
    old_deltas: Dict[str, torch.Tensor],
    gammas: Dict[str, float],
    device: torch.device,
    max_batches: int = 8,
    max_rows_per_batch: int = 4096,
    rank: Optional[int] = None,
    energy: Optional[float] = 0.95,
    max_rank: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], DRSDiagnostics]:
    """Build DRS projectors using W0 - gamma * old_delta.

    This function temporarily changes sub_delta to the LoRA-subtracted model.
    The caller should set the desired training/evaluation deltas after return.
    """
    if not old_deltas or all(v.abs().sum().item() == 0 for v in old_deltas.values()):
        return {}, DRSDiagnostics(ranks={}, counts={}, eigen_top={}, eigen_sum={})

    subtraction = scale_delta_dict(old_deltas, gammas)
    set_subtraction_deltas(model, subtraction, sign=-1.0)
    covs, counts = collect_input_covariances(
        model,
        loader,
        device=device,
        max_batches=max_batches,
        max_rows_per_batch=max_rows_per_batch,
    )
    return compute_projectors_from_covariances(
        covs,
        counts,
        rank=rank,
        energy=energy,
        max_rank=max_rank,
    )


def _project_matrix_right(mat: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Project columns/input dimension of mat using P P^T.

    mat can be [..., in_features]. basis has shape [in_features, k].
    """
    p = basis.to(device=mat.device, dtype=mat.dtype)
    return mat.matmul(p).matmul(p.t())


@torch.no_grad()
def project_lora_a_weights(model: nn.Module, projectors: Dict[str, torch.Tensor]) -> None:
    """Constrain LoRA A matrices to the DRS input subspace after optimizer step."""
    if not projectors:
        return
    for name, module in iter_lora_modules(model):
        p = projectors.get(name)
        if p is None:
            continue
        module.lora_A.weight.copy_(_project_matrix_right(module.lora_A.weight, p))


def project_lora_a_gradients(model: nn.Module, projectors: Dict[str, torch.Tensor]) -> None:
    """Project LoRA A gradients to DRS before optimizer step."""
    if not projectors:
        return
    for name, module in iter_lora_modules(model):
        p = projectors.get(name)
        if p is None or module.lora_A.weight.grad is None:
            continue
        module.lora_A.weight.grad.copy_(_project_matrix_right(module.lora_A.weight.grad, p))


def drs_diagnostics_to_dict(diag: DRSDiagnostics) -> dict:
    return {
        "ranks": diag.ranks,
        "counts": diag.counts,
        "eigen_top": diag.eigen_top,
        "eigen_sum": diag.eigen_sum,
    }
