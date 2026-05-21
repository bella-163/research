from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .data import build_pretrain_datasets, make_loader
from .models.timm_backbone import TimmClassifier
from .utils import apply_overrides, count_trainable_parameters, ensure_dir, load_yaml, save_yaml, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--set", nargs="*", default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        pred = model(images).argmax(dim=1)
        correct += int((pred == targets).sum())
        total += int(targets.numel())
    return correct / max(total, 1)


def main():
    args = parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.set)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    work_dir = ensure_dir(args.work_dir)
    save_yaml(cfg, work_dir / "config_resolved.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, test_set, class_order = build_pretrain_datasets(cfg, seed=seed)
    num_classes = len(class_order)
    batch_size = int(cfg["training"].get("batch_size", 64))
    num_workers = int(cfg["data"].get("num_workers", 4))
    train_loader = make_loader(train_set, batch_size, True, num_workers)
    test_loader = make_loader(test_set, batch_size, False, num_workers)

    model = TimmClassifier(
        model_name=cfg["model"]["name"],
        num_classes=num_classes,
        pretrained=bool(cfg["model"].get("pretrained", False)),
        allow_random_fallback=bool(cfg["model"].get("allow_random_fallback", True)),
    ).to(device)
    print(f"[pretrain] classes={num_classes}, trainable={count_trainable_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.05)),
    )
    epochs = int(cfg["training"].get("epochs", 50))
    amp = bool(cfg["training"].get("amp", False)) and torch.cuda.is_available()
    scaler = GradScaler(enabled=amp)
    records = []
    best_acc = -1.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0
        pbar = tqdm(train_loader, desc=f"pretrain epoch {epoch+1}/{epochs}")
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp):
                loss = F.cross_entropy(model(images), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = images.size(0)
            total_loss += float(loss.detach().cpu()) * bs
            total += bs
            pbar.set_postfix(loss=total_loss / max(total, 1))
        acc = evaluate(model, test_loader, device)
        rec = {"epoch": epoch + 1, "loss": total_loss / max(total, 1), "val_acc": acc}
        records.append(rec)
        pd.DataFrame(records).to_csv(work_dir / "pretrain_metrics.csv", index=False)
        print(f"[pretrain] epoch={epoch+1} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "config": cfg,
                    "class_order": class_order,
                    "backbone_state_dict": model.backbone.state_dict(),
                    "head_state_dict": model.head.state_dict(),
                    "val_acc": best_acc,
                },
                work_dir / "backbone.pt",
            )
    print(f"[pretrain done] best_acc={best_acc:.4f}, checkpoint={work_dir / 'backbone.pt'}")


if __name__ == "__main__":
    main()
