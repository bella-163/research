#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/full/cifar100_10x10.yaml
OUT=outputs/full_cifar100

for METHOD in none fixed adaptive; do
  echo "========== full CIFAR-100 experiment: ${METHOD} =========="
  python -m ga_lora_sub.train_cil \
    --config ${CONFIG} \
    --work-dir ${OUT}/${METHOD} \
    --set method.name=${METHOD}
done

python - <<'PY'
from pathlib import Path
import pandas as pd
rows=[]
for p in Path('outputs/full_cifar100').glob('*/metrics.csv'):
    df=pd.read_csv(p)
    last=df.tail(1).copy()
    last.insert(0,'run',p.parent.name)
    rows.append(last)
if rows:
    out=pd.concat(rows,ignore_index=True)
    print(out.to_string(index=False))
    out.to_csv('outputs/full_cifar100/summary.csv',index=False)
PY
