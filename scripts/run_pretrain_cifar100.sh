#!/usr/bin/env bash
set -euo pipefail

python -m ga_lora_sub.pretrain \
  --config configs/pretrain/cifar100_base.yaml \
  --work-dir outputs/pretrain_cifar100_base
