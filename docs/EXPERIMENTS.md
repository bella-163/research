# 實驗規劃

## Baselines

| 名稱 | 說明 |
|---|---|
| none | 不做 DRS projection，直接在累積模型 `W0 + old_delta` 上訓練目前 LoRA |
| fixed | 使用 `W0 - old_delta` 建立 DRS，並在訓練時投影 LoRA A 的 gradient / weight |
| adaptive | 使用 gradient alignment 估計每層 `gamma`，再用 `W0 - gamma * old_delta` 建立 DRS |

## Metrics

| 指標 | 說明 |
|---|---|
| Average Accuracy | 每個 stage 對所有已見 task 的平均準確率 |
| Final Accuracy | 最後 stage 的平均準確率 |
| Forgetting | 舊 task 歷史最佳準確率與目前準確率的差距 |
| Feature Drift | 舊 task feature centroid 在模型更新前後的距離 |
| Trainable Params | LoRA 與 classifier 的可訓練參數量 |
| DRS Rank | 每層 DRS basis 的維度，存於 `gammas_task_*.json` |
| Gamma Distribution | adaptive subtraction strength 的分布，存於 `gammas_task_*.json` |

## 建議表格

1. CIFAR-100 10 tasks x 10 classes：主表。
2. CIFAR-100 20 tasks x 5 classes：長序列穩定性。
3. ImageNet-R 10 tasks x 20 classes：PTM-based CIL 常見設定。
4. self-pretrain 50 classes -> CIL 50 classes：沒有外部 PTM 時的完整 pipeline。

## 建議 ablation

- DRS：none vs fixed vs adaptive
- adaptive 的 rho：0.25, 0.5, 1.0
- DRS energy：0.90, 0.95, 0.99
- DRS max_rank：16, 32, 64, 128
- LoRA rank：2, 4, 8
- LoRA target module：qkv, qkv+proj, qkv+proj+mlp

## 檢查 DRS 是否有正常運作

```bash
python - <<'PY'
import json
from pathlib import Path

for f in sorted(Path('outputs/quick/adaptive').glob('gammas_task_*.json')):
    obj = json.load(open(f))
    ranks = obj.get('drs', {}).get('ranks', {})
    gammas = obj.get('gammas', {})
    if ranks:
        print(f.name, 'avg_rank=', sum(ranks.values())/len(ranks), 'gamma_range=', (min(gammas.values()), max(gammas.values())))
    else:
        print(f.name, 'no DRS, usually task 1 or method none')
PY
```
