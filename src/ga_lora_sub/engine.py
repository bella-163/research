from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .models.lora import LoRALinear, iter_lora_modules
from .drs import project_lora_a_gradients, project_lora_a_weights


def _make_local_targets(targets: torch.Tensor, current_classes: Sequence[int]) -> torch.Tensor:
    mapping = {int(c): i for i, c in enumerate(current_classes)}
    local = torch.empty_like(targets)
    for cls, idx in mapping.items():
        local[targets == cls] = idx
    return local


def train_one_task(
    model: nn.Module,
    loader,
    current_classes: Sequence[int],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    amp: bool = False,
    drs_projectors: Optional[Dict[str, torch.Tensor]] = None,
    project_after_step: bool = True,
) -> Dict[str, float]:
    scaler = GradScaler(enabled=amp)
    model.train()
    last_loss = 0.0
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"train epoch {epoch + 1}/{epochs}", leave=False)
        running_loss = 0.0
        seen = 0
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            local_targets = _make_local_targets(targets, current_classes)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                logits = model(images)
                logits_cur = logits[:, list(current_classes)]
                loss = F.cross_entropy(logits_cur, local_targets)
            scaler.scale(loss).backward()
            if drs_projectors:
                # DRS constrains LoRA input-side updates. Projection is linear,
                # so it is safe even when AMP gradients are still scaled.
                project_lora_a_gradients(model, drs_projectors)
            scaler.step(optimizer)
            scaler.update()
            if drs_projectors and project_after_step:
                project_lora_a_weights(model, drs_projectors)

            bs = images.size(0)
            running_loss += float(loss.detach().cpu()) * bs
            seen += bs
            pbar.set_postfix(loss=running_loss / max(seen, 1))
        last_loss = running_loss / max(seen, 1)
    return {"loss": last_loss}


@torch.no_grad()
def extract_features(model: nn.Module, loader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    feats = []
    labels = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        z = model.forward_features(images)
        feats.append(z.detach().cpu())
        labels.append(targets.detach().cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


@torch.no_grad()
def update_prototypes(
    model: nn.Module,
    loader,
    device: torch.device,
    prototypes: Dict[int, torch.Tensor],
) -> Dict[int, torch.Tensor]:
    feats, labels = extract_features(model, loader, device)
    for c in labels.unique().tolist():
        mask = labels == int(c)
        prototypes[int(c)] = feats[mask].mean(dim=0)
    return prototypes


@torch.no_grad()
def evaluate_linear(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += int((pred == targets).sum().item())
        total += int(targets.numel())
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_ncm(model: nn.Module, loader, device: torch.device, prototypes: Dict[int, torch.Tensor]) -> float:
    model.eval()
    if not prototypes:
        return 0.0
    classes = sorted(prototypes.keys())
    proto = torch.stack([prototypes[c] for c in classes], dim=0).to(device)
    proto = F.normalize(proto, dim=1)
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        z = model.forward_features(images)
        z = F.normalize(z, dim=1)
        sim = z @ proto.t()
        pred_idx = sim.argmax(dim=1)
        pred = torch.tensor([classes[i] for i in pred_idx.tolist()], device=device, dtype=targets.dtype)
        correct += int((pred == targets).sum().item())
        total += int(targets.numel())
    return correct / max(total, 1)


def estimate_gradient_alignment(
    model: nn.Module,
    loader,
    current_classes: Sequence[int],
    old_deltas: Dict[str, torch.Tensor],
    device: torch.device,
    max_batches: int = 1,
) -> Dict[str, float]:
    """Estimate cos(-gradient, old_delta) for every LoRA-wrapped layer."""
    if not old_deltas:
        return {}
    was_training = model.training
    model.train()

    lora_modules: Dict[str, LoRALinear] = {name: m for name, m in iter_lora_modules(model)}
    for module in lora_modules.values():
        module.weight.requires_grad_(True)
        module.weight.grad = None

    model.zero_grad(set_to_none=True)
    batches = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        local_targets = _make_local_targets(targets, current_classes)
        logits = model(images)
        logits_cur = logits[:, list(current_classes)]
        loss = F.cross_entropy(logits_cur, local_targets) / max_batches
        loss.backward()
        batches += 1
        if batches >= max_batches:
            break

    alignments: Dict[str, float] = {}
    eps = 1e-12
    for name, module in lora_modules.items():
        old = old_deltas.get(name)
        grad = module.weight.grad
        if old is None or grad is None or old.abs().sum().item() == 0:
            alignments[name] = 0.0
            continue
        update_dir = -grad.detach().flatten().float().cpu()
        old_vec = old.detach().flatten().float().cpu()
        denom = update_dir.norm() * old_vec.norm() + eps
        alignments[name] = float(torch.dot(update_dir, old_vec) / denom)

    for module in lora_modules.values():
        module.weight.requires_grad_(False)
        module.weight.grad = None
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return alignments


@torch.no_grad()
def compute_feature_drift(
    before: Dict[int, torch.Tensor],
    after: Dict[int, torch.Tensor],
) -> float:
    common = sorted(set(before.keys()) & set(after.keys()))
    if not common:
        return 0.0
    vals = []
    for c in common:
        vals.append(torch.norm(before[c] - after[c], p=2).item())
    return float(sum(vals) / len(vals))


@torch.no_grad()
def class_centroids(model: nn.Module, loaders: List, device: torch.device) -> Dict[int, torch.Tensor]:
    out: Dict[int, List[torch.Tensor]] = {}
    for loader in loaders:
        feats, labels = extract_features(model, loader, device)
        for c in labels.unique().tolist():
            out.setdefault(int(c), []).append(feats[labels == int(c)])
    centroids = {}
    for c, chunks in out.items():
        all_feats = torch.cat(chunks, dim=0)
        centroids[c] = all_feats.mean(dim=0)
    return centroids
