from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import build_cil_tasks, make_loader
from .engine import (
    class_centroids,
    compute_feature_drift,
    estimate_gradient_alignment,
    evaluate_linear,
    evaluate_ncm,
    train_one_task,
    update_prototypes,
)
from .methods import compute_gammas
from .metrics import summarize_stage
from .models.lora import (
    add_delta_dict,
    add_lora_to_model,
    clear_subtraction_deltas,
    collect_lora_deltas,
    empty_lora_deltas,
    freeze_non_lora,
    reset_lora_parameters,
    scale_delta_dict,
    set_subtraction_deltas,
)
from .models.timm_backbone import build_model
from .utils import apply_overrides, count_trainable_parameters, ensure_dir, load_yaml, save_json, save_yaml, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--set", nargs="*", default=None, help="Override config values, e.g. method.name=fixed training.epochs=1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.set)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    work_dir = ensure_dir(args.work_dir)
    save_yaml(cfg, work_dir / "config_resolved.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks, class_order, num_classes = build_cil_tasks(cfg, seed=seed)

    model = build_model(cfg, num_classes=num_classes)
    lora_cfg = cfg["model"]["lora"]
    replaced = add_lora_to_model(
        model,
        target_modules=lora_cfg.get("target_modules", ["attn.qkv"]),
        rank=int(lora_cfg.get("rank", 8)),
        alpha=float(lora_cfg.get("alpha", 16)),
        dropout=float(lora_cfg.get("dropout", 0.0)),
    )
    print(f"[lora] inserted into {len(replaced)} modules")
    for name in replaced[:10]:
        print(f"  - {name}")
    if len(replaced) > 10:
        print("  ...")

    model.to(device)
    freeze_non_lora(model, train_head=bool(cfg["training"].get("train_head", True)))
    print(f"[params] trainable parameters: {count_trainable_parameters(model):,}")

    batch_size = int(cfg["training"].get("batch_size", 64))
    num_workers = int(cfg["data"].get("num_workers", 4))
    epochs = int(cfg["training"].get("epochs", 10))
    amp = bool(cfg["training"].get("amp", False)) and torch.cuda.is_available()
    eval_classifier = str(cfg.get("evaluation", {}).get("classifier", "ncm")).lower()

    train_loaders = [make_loader(t.train_set, batch_size, True, num_workers) for t in tasks]
    eval_train_loaders = [make_loader(t.train_set, batch_size, False, num_workers) for t in tasks]
    test_loaders = [make_loader(t.test_set, batch_size, False, num_workers) for t in tasks]

    old_deltas: Dict[str, torch.Tensor] = empty_lora_deltas(model, device="cpu")
    prototypes: Dict[int, torch.Tensor] = {}
    acc_matrix = np.full((len(tasks), len(tasks)), np.nan, dtype=np.float32)
    records: List[dict] = []
    previous_centroids: Dict[int, torch.Tensor] = {}

    for task_idx, task in enumerate(tasks):
        print(f"\n[task {task_idx + 1}/{len(tasks)}] raw_classes={task.raw_classes}")
        reset_lora_parameters(model)
        clear_subtraction_deltas(model)

        # Estimate gamma with no subtraction first.
        alignments = estimate_gradient_alignment(
            model,
            train_loaders[task_idx],
            current_classes=task.global_classes,
            old_deltas=old_deltas,
            device=device,
            max_batches=int(cfg["method"].get("grad_batches", 1)),
        )
        gammas = compute_gammas(cfg["method"], alignments, old_deltas)
        save_json({"alignments": alignments, "gammas": gammas}, work_dir / f"gammas_task_{task_idx + 1}.json")

        # Train in drift-resistant space: W0 - gamma * old_delta + current LoRA.
        subtraction = scale_delta_dict(old_deltas, gammas)
        set_subtraction_deltas(model, subtraction, sign=-1.0)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(cfg["training"].get("lr", 5e-4)),
            weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
        )
        train_stats = train_one_task(
            model,
            train_loaders[task_idx],
            current_classes=task.global_classes,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            amp=amp,
        )

        # Merge current LoRA into cumulative delta bank, then evaluate with W0 + cumulative LoRA.
        current_delta = collect_lora_deltas(model, device="cpu")
        old_deltas = add_delta_dict(old_deltas, current_delta)
        reset_lora_parameters(model)
        set_subtraction_deltas(model, old_deltas, sign=1.0)

        # Update prototypes only for current task. Old prototypes are kept rehearsal-free.
        prototypes = update_prototypes(model, eval_train_loaders[task_idx], device, prototypes)

        # Feature drift is measured on old task test data for analysis only.
        current_centroids = class_centroids(model, test_loaders[: task_idx + 1], device)
        drift = compute_feature_drift(previous_centroids, current_centroids) if task_idx > 0 else 0.0
        previous_centroids = current_centroids

        for eval_task_idx in range(task_idx + 1):
            if eval_classifier == "linear":
                acc = evaluate_linear(model, test_loaders[eval_task_idx], device)
            elif eval_classifier == "ncm":
                acc = evaluate_ncm(model, test_loaders[eval_task_idx], device, prototypes)
            else:
                raise ValueError(f"Unknown evaluation.classifier: {eval_classifier}")
            acc_matrix[task_idx, eval_task_idx] = acc
            print(f"  eval task {eval_task_idx + 1}: acc={acc:.4f}")

        rec = summarize_stage(acc_matrix, task_idx, drift=drift)
        rec.update({
            "train_loss": train_stats["loss"],
            "method": cfg["method"]["name"],
            "num_seen_classes": (task_idx + 1) * int(cfg["data"]["classes_per_task"]),
        })
        records.append(rec)
        pd.DataFrame(records).to_csv(work_dir / "metrics.csv", index=False)
        np.save(work_dir / "accuracy_matrix.npy", acc_matrix)
        print(f"  average_accuracy={rec['average_accuracy']:.4f}, forgetting={rec['forgetting']:.4f}, drift={drift:.4f}")

        torch.save(
            {
                "config": cfg,
                "class_order": class_order,
                "task_idx": task_idx,
                "model_state_dict": model.state_dict(),
                "old_deltas": old_deltas,
                "prototypes": prototypes,
                "acc_matrix": acc_matrix,
            },
            work_dir / "checkpoint_last.pt",
        )

    print("\n[done]")
    print(pd.DataFrame(records).tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
