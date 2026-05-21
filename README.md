# Adaptive LoRA Subtraction for Class-Incremental Learning

這個 repo 是用來跑 **Gradient-Aware Adaptive LoRA Subtraction** 的快速實驗與完整實驗。
研究目標是：在 rehearsal-free Class-Incremental Learning 中，透過 LoRA Subtraction 建立較穩定的 Drift-Resistant Space，並進一步用 gradient alignment 自動調整每一層的 subtraction strength，降低舊類別 feature drift。

## 方法摘要

對第 `t` 個 task，第 `l` 層舊任務累積 LoRA 更新為：

```text
V_old^l = sum_{j=1}^{t-1} ΔW_j^l
```

固定版 LoRA Subtraction：

```text
W_tilde^l = W_0^l - V_old^l
```

本專案實作的 Adaptive LoRA Subtraction：

```text
s_t^l = cos(-grad_t^l, V_old^l)
gamma_t^l = clip(1 - rho * s_t^l, gamma_min, gamma_max)
W_tilde^l = W_0^l - gamma_t^l * V_old^l
```

直覺：

- 新任務更新方向與舊 LoRA 相似：少扣一點，保留可轉移知識。
- 新任務更新方向與舊 LoRA 衝突：多扣一點，減少干擾。
- 不確定：接近固定 subtraction。

## 安裝

```bash
git clone https://github.com/bella-163/research.git
cd research
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## 快速實驗

快速實驗使用 CIFAR-100 的少量類別與少量樣本，適合檢查程式能不能跑通。

```bash
bash scripts/run_quick.sh
```

預設會跑三個方法：

```text
none      : 不做 LoRA Subtraction
fixed     : 固定 gamma = 1 的 LoRA Subtraction
adaptive  : gradient-aware adaptive LoRA Subtraction
```

結果會輸出到：

```text
outputs/quick/
```

## 完整實驗：CIFAR-100 10 tasks x 10 classes

```bash
bash scripts/run_full_cifar100.sh
```

結果會輸出到：

```text
outputs/full_cifar100/
```

## Pre-trained model 還沒訓練怎麼辦？

有兩種路線：

### 路線 A：使用 timm 已提供的 ImageNet pre-trained ViT

這是最快、最接近 PTM-based CIL 論文設定的方式。設定檔中：

```yaml
model:
  pretrained: true
```

第一次跑時會自動下載 timm 權重。若環境沒有網路，可以先在有網路的環境下載或改用路線 B。

### 路線 B：自己先做 supervised pretraining

下面指令會用 CIFAR-100 的前 50 類訓練一個 backbone checkpoint，再用剩下類別做 CIL。這不是大型 PTM，但適合在沒有現成 pre-trained weights 時做完整 pipeline 驗證。

```bash
bash scripts/run_pretrain_cifar100.sh
```

pretrain checkpoint 預設輸出：

```text
outputs/pretrain_cifar100_base/backbone.pt
```

接著跑 self-pretrain 後的 CIL：

```bash
python -m ga_lora_sub.train_cil \
  --config configs/full/cifar100_pretrain50_cil50.yaml \
  --work-dir outputs/full_cifar100_pretrain50_cil50 \
  --set model.checkpoint=outputs/pretrain_cifar100_base/backbone.pt
```

## 重要輸出

每個實驗資料夾會包含：

```text
config_resolved.yaml          # 實際使用設定
metrics.csv                   # 每個 stage 的 accuracy / forgetting / drift
accuracy_matrix.npy           # acc_matrix[train_stage, eval_task]
gammas_task_*.json            # adaptive subtraction 的每層 gamma
checkpoint_last.pt            # 最後模型與 LoRA 狀態
```

## 建議的實驗順序

1. 先跑 `bash scripts/run_quick.sh` 確認環境與程式正常。
2. 再跑 `bash scripts/run_full_cifar100.sh` 建立主要表格。
3. 若沒有 ImageNet pre-trained 權重，先跑 `bash scripts/run_pretrain_cifar100.sh`。
4. 最後比較 `none / fixed / adaptive` 的：
   - Average Accuracy
   - Final Accuracy
   - Forgetting
   - Feature Drift Distance

## 注意事項

- 本專案預設是 rehearsal-free：訓練每個 task 時不使用舊 task 的訓練資料。
- Feature drift metric 只用於分析與報告，不會回傳到訓練 loss。
- 若要嚴格遵守 benchmark，pretraining dataset 應避免與 downstream CIL 類別重疊。
