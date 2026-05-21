from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import timm


class TimmClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, allow_random_fallback: bool = True):
        super().__init__()
        try:
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        except Exception as exc:
            if pretrained and allow_random_fallback:
                print(f"[warning] failed to load pretrained weights: {exc}")
                print("[warning] fallback to random initialization. For real experiments, provide a checkpoint or internet access.")
                self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
            else:
                raise
        self.num_features = int(getattr(self.backbone, "num_features"))
        self.head = nn.Linear(self.num_features, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x)
        return self.head(z)


def load_backbone_checkpoint(model: TimmClassifier, checkpoint_path: Optional[str]) -> None:
    if checkpoint_path in {None, "", "null"}:
        return
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "backbone_state_dict" in ckpt:
        state = ckpt["backbone_state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = {k.replace("backbone.", ""): v for k, v in ckpt["model_state_dict"].items() if k.startswith("backbone.")}
    else:
        state = ckpt
    missing, unexpected = model.backbone.load_state_dict(state, strict=False)
    print(f"[checkpoint] loaded backbone from {checkpoint_path}")
    if missing:
        print(f"[checkpoint] missing keys: {len(missing)}")
    if unexpected:
        print(f"[checkpoint] unexpected keys: {len(unexpected)}")


def build_model(cfg: dict, num_classes: int) -> TimmClassifier:
    model_cfg = cfg["model"]
    model = TimmClassifier(
        model_name=model_cfg["name"],
        num_classes=num_classes,
        pretrained=bool(model_cfg.get("pretrained", True)),
        allow_random_fallback=bool(model_cfg.get("allow_random_fallback", True)),
    )
    load_backbone_checkpoint(model, model_cfg.get("checkpoint"))
    return model
