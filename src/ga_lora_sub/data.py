from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class TaskData:
    task_id: int
    raw_classes: List[int]
    global_classes: List[int]
    train_set: Dataset
    test_set: Dataset


class RemapClassSubset(Dataset):
    """Subset dataset by raw labels and remap them to global CIL label indices."""

    def __init__(
        self,
        base: Dataset,
        raw_classes: Sequence[int],
        raw_to_global: Dict[int, int],
        max_per_class: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.base = base
        self.raw_classes = list(raw_classes)
        self.raw_to_global = raw_to_global
        self.indices = self._select_indices(max_per_class=max_per_class, seed=seed)

    def _targets(self) -> List[int]:
        if hasattr(self.base, "targets"):
            return list(getattr(self.base, "targets"))
        if hasattr(self.base, "samples"):
            return [y for _, y in getattr(self.base, "samples")]
        raise AttributeError("Dataset must expose `targets` or `samples`.")

    def _select_indices(self, max_per_class: Optional[int], seed: int) -> List[int]:
        targets = self._targets()
        buckets: Dict[int, List[int]] = {c: [] for c in self.raw_classes}
        raw_set = set(self.raw_classes)
        for idx, y in enumerate(targets):
            if y in raw_set:
                buckets[y].append(idx)
        rng = random.Random(seed)
        selected: List[int] = []
        for c in self.raw_classes:
            inds = buckets[c]
            rng.shuffle(inds)
            if max_per_class is not None:
                inds = inds[: int(max_per_class)]
            selected.extend(inds)
        rng.shuffle(selected)
        return selected

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.base[self.indices[idx]]
        return x, self.raw_to_global[int(y)]


def build_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def _load_base_datasets(cfg: dict):
    name = cfg["dataset"].lower()
    image_size = int(cfg.get("image_size", 224))
    if name == "cifar100":
        root = cfg.get("root", "./data")
        download = bool(cfg.get("download", True))
        train = datasets.CIFAR100(root=root, train=True, download=download, transform=build_transforms(image_size, True))
        test = datasets.CIFAR100(root=root, train=False, download=download, transform=build_transforms(image_size, False))
        return train, test, 100
    if name == "cifar10":
        root = cfg.get("root", "./data")
        download = bool(cfg.get("download", True))
        train = datasets.CIFAR10(root=root, train=True, download=download, transform=build_transforms(image_size, True))
        test = datasets.CIFAR10(root=root, train=False, download=download, transform=build_transforms(image_size, False))
        return train, test, 10
    if name == "imagefolder":
        train_dir = cfg.get("train_dir")
        val_dir = cfg.get("val_dir")
        if train_dir is None or val_dir is None:
            raise ValueError("imagefolder dataset requires data.train_dir and data.val_dir")
        train = datasets.ImageFolder(train_dir, transform=build_transforms(image_size, True))
        test = datasets.ImageFolder(val_dir, transform=build_transforms(image_size, False))
        if train.class_to_idx != test.class_to_idx:
            raise ValueError("train_dir and val_dir must share the same class_to_idx mapping")
        return train, test, len(train.classes)
    raise ValueError(f"Unsupported dataset: {name}")


def resolve_class_order(cfg: dict, num_classes: int, seed: int) -> List[int]:
    mode = cfg.get("class_order", "sequential")
    if isinstance(mode, list):
        return [int(x) for x in mode]
    if mode == "sequential":
        return list(range(num_classes))
    if mode == "seeded":
        rng = np.random.default_rng(int(cfg.get("class_order_seed", seed)))
        order = np.arange(num_classes)
        rng.shuffle(order)
        return [int(x) for x in order]
    if mode == "range":
        start, end = cfg.get("class_range", [0, num_classes])
        return list(range(int(start), int(end)))
    raise ValueError(f"Unsupported class_order: {mode}")


def build_cil_tasks(cfg: dict, seed: int) -> Tuple[List[TaskData], List[int], int]:
    data_cfg = cfg["data"]
    train_base, test_base, n_raw_classes = _load_base_datasets(data_cfg)
    class_order = resolve_class_order(data_cfg, n_raw_classes, seed)
    num_tasks = int(data_cfg["num_tasks"])
    classes_per_task = int(data_cfg["classes_per_task"])
    total_needed = num_tasks * classes_per_task
    if total_needed > len(class_order):
        raise ValueError(f"Need {total_needed} classes but class_order only has {len(class_order)}")
    class_order = class_order[:total_needed]
    raw_to_global = {raw: i for i, raw in enumerate(class_order)}

    tasks: List[TaskData] = []
    for task_id in range(num_tasks):
        raw_classes = class_order[task_id * classes_per_task : (task_id + 1) * classes_per_task]
        global_classes = [raw_to_global[c] for c in raw_classes]
        train_set = RemapClassSubset(
            train_base,
            raw_classes,
            raw_to_global,
            max_per_class=data_cfg.get("max_train_per_class"),
            seed=seed + task_id,
        )
        test_set = RemapClassSubset(
            test_base,
            raw_classes,
            raw_to_global,
            max_per_class=data_cfg.get("max_test_per_class"),
            seed=seed + 1000 + task_id,
        )
        tasks.append(TaskData(task_id, raw_classes, global_classes, train_set, test_set))
    return tasks, class_order, len(class_order)


def build_pretrain_datasets(cfg: dict, seed: int):
    data_cfg = cfg["data"]
    train_base, test_base, n_raw_classes = _load_base_datasets(data_cfg)
    class_order = resolve_class_order(data_cfg, n_raw_classes, seed)
    raw_to_global = {raw: i for i, raw in enumerate(class_order)}
    train_set = RemapClassSubset(
        train_base,
        class_order,
        raw_to_global,
        max_per_class=data_cfg.get("max_train_per_class"),
        seed=seed,
    )
    test_set = RemapClassSubset(
        test_base,
        class_order,
        raw_to_global,
        max_per_class=data_cfg.get("max_test_per_class"),
        seed=seed + 1,
    )
    return train_set, test_set, class_order


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
