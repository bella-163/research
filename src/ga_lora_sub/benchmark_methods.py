from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ga_lora_sub.models.lora import add_lora_to_model, freeze_non_lora, iter_lora_modules
    from ga_lora_sub.models.timm_backbone import build_model
    from ga_lora_sub.utils import apply_overrides, load_yaml
else:
    from .models.lora import add_lora_to_model, freeze_non_lora, iter_lora_modules
    from .models.timm_backbone import build_model
    from .utils import apply_overrides, load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare method parameter size and inference latency.")
    parser.add_argument("--config", required=True, help="Base experiment config.")
    parser.add_argument(
        "--work-dirs",
        nargs="*",
        default=None,
        help="Run directories containing checkpoint_last.pt. If omitted, untrained models are benchmarked.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["none", "fixed", "adaptive"],
        help="Methods to benchmark when --work-dirs is omitted.",
    )
    parser.add_argument("--checkpoint-name", default="checkpoint_last.pt")
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--set", nargs="*", default=None, help="Config overrides, e.g. model.pretrained=false")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(name)


def count_named_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = 0
    sub_delta = 0
    for _, module in iter_lora_modules(model):
        lora += module.lora_A.weight.numel() + module.lora_B.weight.numel()
        sub_delta += module.sub_delta.numel()
    head = sum(p.numel() for p in getattr(model, "head", torch.nn.Module()).parameters())
    return {
        "total_params": total,
        "trainable_params": trainable,
        "lora_params": lora,
        "head_params": head,
        "sub_delta_buffer_elems": sub_delta,
    }


def parameter_bytes(model: torch.nn.Module, include_buffers: bool = False) -> int:
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    if not include_buffers:
        return params
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return params + buffers


def infer_method_from_work_dir(path: Path) -> str:
    name = path.name
    if "_" in name and name.startswith("s") and name.split("_", 1)[0][1:].isdigit():
        return name.split("_", 1)[1]
    return name


def build_lora_model(cfg: Dict[str, Any], num_classes: int, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg, num_classes=num_classes)
    lora_cfg = cfg["model"]["lora"]
    add_lora_to_model(
        model,
        target_modules=lora_cfg.get("target_modules", ["attn.qkv"]),
        rank=int(lora_cfg.get("rank", 8)),
        alpha=float(lora_cfg.get("alpha", 16)),
        dropout=float(lora_cfg.get("dropout", 0.0)),
    )
    freeze_non_lora(model, train_head=bool(cfg["training"].get("train_head", True)))
    model.to(device)
    model.eval()
    return model


def load_checkpoint_if_present(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> bool:
    if not checkpoint_path.exists():
        return False
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    return True


@torch.inference_mode()
def benchmark_latency(
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    batch_size: int,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)

    for _ in range(max(0, warmup)):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(max(1, iters)):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_ms = elapsed * 1000.0 / max(1, iters)
    return {
        "batch_latency_ms": latency_ms,
        "images_per_second": batch_size * 1000.0 / latency_ms,
    }


def make_specs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.work_dirs:
        specs = []
        for raw in args.work_dirs:
            work_dir = Path(raw)
            specs.append(
                {
                    "method": infer_method_from_work_dir(work_dir),
                    "work_dir": work_dir,
                    "checkpoint": work_dir / args.checkpoint_name,
                }
            )
        return specs
    return [{"method": method, "work_dir": None, "checkpoint": None} for method in args.methods]


def resolve_num_classes(cfg: Dict[str, Any], checkpoint: Path | None) -> int:
    if checkpoint is not None and checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "config" in ckpt:
            data_cfg = ckpt["config"].get("data", {})
            return int(data_cfg["num_tasks"]) * int(data_cfg["classes_per_task"])
    data_cfg = cfg["data"]
    return int(data_cfg["num_tasks"]) * int(data_cfg["classes_per_task"])


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_yaml(args.config), args.set)
    device = choose_device(args.device)
    image_size = int(cfg["data"].get("image_size", 224))

    rows = []
    for spec in make_specs(args):
        method = spec["method"]
        checkpoint = spec["checkpoint"]
        cfg_for_model = apply_overrides(cfg, [f"method.name={method}"])
        num_classes = resolve_num_classes(cfg_for_model, checkpoint)
        model = build_lora_model(cfg_for_model, num_classes=num_classes, device=device)
        loaded = load_checkpoint_if_present(model, checkpoint, device) if checkpoint is not None else False
        model.eval()

        counts = count_named_parameters(model)
        timings = benchmark_latency(
            model,
            device=device,
            image_size=image_size,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iters=args.iters,
        )
        checkpoint_mb = checkpoint.stat().st_size / (1024**2) if checkpoint is not None and checkpoint.exists() else 0.0
        rows.append(
            {
                "method": method,
                "checkpoint_loaded": loaded,
                "checkpoint_mb": checkpoint_mb,
                "device": str(device),
                "batch_size": args.batch_size,
                "image_size": image_size,
                "param_mb": parameter_bytes(model, include_buffers=False) / (1024**2),
                "param_plus_buffer_mb": parameter_bytes(model, include_buffers=True) / (1024**2),
                **counts,
                **timings,
            }
        )

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)


if __name__ == "__main__":
    main()
