#!/usr/bin/env bash
set -euo pipefail

CONFIG=configs/quick/cifar100_debug.yaml
OUT=outputs/quick

for METHOD in none fixed adaptive; do
  echo "========== quick experiment: ${METHOD} =========="
  python -m ga_lora_sub.train_cil \
    --config ${CONFIG} \
    --work-dir ${OUT}/${METHOD} \
    --set method.name=${METHOD}
done

python - <<'PY'
from pathlib import Path
import pandas as pd
rows=[]
for p in Path('outputs/quick').glob('*/metrics.csv'):
    df=pd.read_csv(p)
    last=df.tail(1).copy()
    last.insert(0,'run',p.parent.name)
    rows.append(last)
if rows:
    out=pd.concat(rows,ignore_index=True)
    print(out.to_string(index=False))
    out.to_csv('outputs/quick/summary.csv',index=False)
PY
