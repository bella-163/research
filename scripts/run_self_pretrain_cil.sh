#!/usr/bin/env bash
set -euo pipefail

if [ ! -f outputs/pretrain_cifar100_base/backbone.pt ]; then
  echo "Pretrain checkpoint not found. Running pretraining first."
  bash scripts/run_pretrain_cifar100.sh
fi

for METHOD in none fixed adaptive; do
  python -m ga_lora_sub.train_cil \
    --config configs/full/cifar100_pretrain50_cil50.yaml \
    --work-dir outputs/full_cifar100_pretrain50_cil50/${METHOD} \
    --set method.name=${METHOD} model.checkpoint=outputs/pretrain_cifar100_base/backbone.pt
done
