# 實驗規劃

## Baselines

| 名稱 | 說明 |
|---|---|
| none | 不做 LoRA Subtraction，直接在新 task 上訓練 LoRA |
| fixed | 固定 gamma=1，對舊 LoRA 做 subtraction |
| adaptive | 使用 gradient alignment 動態估計每層 gamma |

## Metrics

| 指標 | 說明 |
|---|---|
| Average Accuracy | 每個 stage 對所有已見 task 的平均準確率 |
| Final Accuracy | 最後 stage 的平均準確率 |
| Forgetting | 舊 task 歷史最佳準確率與目前準確率的差距 |
| Feature Drift | 舊 task feature centroid 在模型更新前後的距離 |
| Trainable Params | LoRA 與 classifier 的可訓練參數量 |

## 建議表格

1. CIFAR-100 10 tasks x 10 classes：主表。
2. CIFAR-100 20 tasks x 5 classes：長序列穩定性。
3. self-pretrain 50 classes -> CIL 50 classes：沒有外部 PTM 時的完整 pipeline。

## 建議 ablation

- gamma 固定值：0, 0.5, 1.0, 1.5
- adaptive 的 rho：0.25, 0.5, 1.0
- LoRA rank：2, 4, 8
- LoRA target module：qkv, qkv+proj, qkv+proj+mlp
