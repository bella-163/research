# Adaptive LoRA-DRS for Class-Incremental Learning

這個 repo 用來跑 **Adaptive LoRA Subtraction with Drift-Resistant Space (DRS)** 的快速實驗與完整實驗。
研究目標是：在 rehearsal-free Class-Incremental Learning 中，使用 frozen pre-trained ViT 作為 backbone，只訓練 LoRA 與 classifier，並透過 DRS gradient projection 降低舊類別 feature drift。

## 方法摘要

對第 `t` 個 task，第 `l` 層舊任務累積 LoRA 更新為：

```text
V_old^l = sum_{j=1}^{t-1} ΔW_j^l
```

固定版 LoRA Subtraction 用：

```text
W_tilde^l = W_0^l - V_old^l
```

本專案的 Adaptive 版本用：

```text
s_t^l = cos(-grad_t^l, V_old^l)
gamma_t^l = clip(1 - rho * s_t^l, gamma_min, gamma_max)
W_tilde^l = W_0^l - gamma_t^l * V_old^l
```

但請注意，這版**不再直接用 `W_tilde` 當訓練模型**。正確流程是：

```text
1. 用 W_0 - gamma * V_old 跑目前 task data
2. 收集每個 LoRA-wrapped linear layer 的 input covariance
3. 由 covariance 取 DRS basis P_t^l
4. 實際訓練時使用 W_0 + V_old + ΔW_t
5. 每一步把 LoRA A 的 gradient / weight 投影到 P_t^l(P_t^l)^T
6. 訓練完後把目前 LoRA merge 到 cumulative old_delta
```

也就是說，subtraction 只負責**建立更新子空間**，真正 forward / evaluation 還是使用累積模型 `W_0 + old_delta`。這避免了 naive forward-subtraction 造成的 train/eval mismatch。

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

## 快速實驗：使用現成 pretrained model

快速實驗使用 CIFAR-100 的少量類別與少量樣本，適合檢查程式能不能跑通。

```bash
bash scripts/run_quick.sh
```

若你要嚴格避免 fallback 到 random backbone，可以改用：

```bash
for METHOD in none fixed adaptive; do
  python -m ga_lora_sub.train_cil \
    --config configs/quick/cifar100_debug.yaml \
    --work-dir outputs/quick_ptm/${METHOD} \
    --set method.name=${METHOD} \
          model.pretrained=true \
          model.allow_random_fallback=false \
          model.checkpoint=null
done
```

預設會跑三個方法：

```text
none      : 不做 DRS projection
fixed     : gamma = 1，使用固定 LoRA Subtraction 建立 DRS
adaptive  : 使用 gradient alignment 動態估計每層 gamma，再建立 DRS
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

第一次跑時會自動下載 timm 權重。若環境沒有網路，可以先在有網路的環境下載，或改用路線 B。

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
gammas_task_*.json            # 每層 gamma 與 DRS rank / covariance diagnostics
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
   - DRS rank / gamma 分布

## 注意事項

- 本專案預設是 rehearsal-free：訓練每個 task 時不使用舊 task 的訓練資料。
- Feature drift metric 只用於分析與報告，不會回傳到訓練 loss。
- DRS 是由目前 task data 在 subtracted model 下的 layer input covariance 建立，沒有存舊 task samples。
- 若要嚴格遵守 benchmark，pretraining dataset 應避免與 downstream CIL 類別重疊。
