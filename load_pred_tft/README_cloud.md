# load_pred_tft — TFT 路线A Phase 0（云端运行说明）

Temporal Fusion Transformer 移植版。与 `load_pred`（LightGBM v6）/ `load_pred_tcn`（TCN）同构：
**仅模型方法换为 TFT 序列建模**，数据/特征/集成结构/OOF 校正/6 条泄露不变量全部逐行一致。

---

## 1. 这是什么

路线A Phase 0：验证 TFT 能否突破 v6 的 **raw val 天花板 1515 MW**（fit_diag）。
- v6 LightGBM 校正后 val = **1445.62 MW**（生产基线）
- v6 raw val（无 OOF 校正）≈ 1512 MW
- TCN raw ceiling ≈ 1640（已 NO-GO）

TFT 的增量来源（非"替代手工 lag"，lag 仍受不变量约束）：
- **attention 自动学 pred_load×NWP×calendar 交互**（v6 是手工交互特征）
- **multi-horizon**：一次前向输出 D+1 全天 96 点（信息共享）
- **Variable Selection Network** 自动学变量权重
- **residual target = actual − MOS_anchor**（保留 MOS 收益）

**关键约束（不变量#2）**：encoder 历史负荷只能用 `pred_load`（外部预测），**不能用 actual**。
所以 TFT 学的是 pred_load 序列依赖 + NWP 交互，与 v6 lag 同源但 attention 自动组合。

---

## 2. 云端环境要求

| 项 | 要求 |
|---|---|
| Python | 3.14（与本地一致；3.11+ 即可） |
| PyTorch | **GPU 版**（CUDA 11.8/12.1/12.4 任一，匹配云端 GPU 驱动） |
| GPU | 显存 ≥ 4GB（TFT hidden=64, batch=16，占用适中；4090/3060 均可） |
| 依赖 | `pip install torch pandas numpy scikit-learn` （lightgbm 不需要） |

**本地 torch 是 CPU 版（2.12.0+cpu），不可本地训练 TFT**——必须上云端 GPU。
Phase 0（2 成员 × 30 epochs）云端 GPU 约 10-20 min；生产 40 成员约 3-6 h。

### 安装 GPU 版 PyTorch（云端）
```bash
# CUDA 12.1 示例（按云端 GPU 驱动选 cu118/cu121/cu124）
pip install torch --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 应输出 True
```

---

## 3. 上传文件

上传整个项目即可，TFT 包只读共享 `data/`，模型/输出写在本包内（不覆盖 `load_pred` 的）：

```
load_prediction/
├── data/                          # 共享输入（只读）
│   ├── direct_load_latest.csv     # 负荷（UTF-8-BOM）
│   └── shandong_weather_15min.csv # 气象
├── load_pred_tft/                 # ← TFT 包（本次新增）
│   ├── __init__.py
│   ├── config.py                  # TFT 超参 + 路径（包内 models/output）
│   ├── data_loader.py             # 复用（共享 data/）
│   ├── features.py                # 复用（build_features，不变量#5）
│   ├── tft.py                     # TFT 模型 + 序列数据构建 + 训练 + 推理
│   ├── model.py                   # EnsembleModel（TFT 成员 + OOF 后处理逐时刻复用）
│   ├── train.py                   # 训练入口（40 成员生产版）
│   ├── predict.py                 # D+1 预测入口
│   ├── exp_tft_phase0.py          # ← Phase 0 raw gate（先跑这个）
│   └── README_cloud.md            # 本文件
├── load_pred/  load_pred_tcn/     # 其他包（可不传，TFT 不依赖）
└── ...
```

**最小上传**：`data/` + `load_pred_tft/` 即可运行（TFT 不依赖其他包）。

---

## 4. 运行命令（在项目根目录 `load_prediction/` 下执行）

### 步骤 1：Phase 0 raw gate（先跑，决定生死）
```bash
python -m load_pred_tft.exp_tft_phase0
```
输出关键行：
```
TFT raw val        = XXXX.XX  (v6 raw≈1512, gate<1515)
TFT OOF校正后 val  = XXXX.XX  (v6=1445.62, Δv6 ±XX.XX)
OOF/val 一致率     = XX.X%
-> RAW GATE 通过/未通过 ...
```

**判定**：
- `raw val < 1515` → **RAW GATE 通过**，TFT 有潜力突破 raw 地板 → 进步骤 2（Phase 1 调优）
- `raw val >= 1515` → **RAW GATE 未通过**，TFT 同 TCN 无法突破 raw 地板 → 路线A NO-GO，回 v6

### 步骤 2：生产训练（仅 Phase 0 gate 通过才跑，40 成员，约 3-6 h）
```bash
python -m load_pred_tft.train
```
产出（写在本包内）：
- `load_pred_tft/models/model_bundle.pkl` + `load_pred_tft/models/boosters/member_NNN.pt`（40 个）
- `load_pred_tft/output/evaluation_metrics.txt`（val MAE，<1500 PASS）
- `load_pred_tft/output/full_predictions.csv` / `full_mae.csv`
- `train.main()` val MAE ≥ 1500 退出非零（pass/fail gate，与 v6 一致）

### 步骤 3：D+1 预测（加载已训练模型，不重训）
```bash
python -m load_pred_tft.predict --run-date 2026-05-18   # 指定运行日 D
python -m load_pred_tft.predict                         # 默认 D=今天
python -m load_pred_tft.predict --run-hour 21           # 历史回测用 21 时（复现 val 的 20:00-D-issue 条件）
```
产出：
- `load_pred_tft/output/prediction_YYYYMMDD.csv`（YYYYMMDD=运行日 D）
- `load_pred_tft/output/latest_prediction.csv`（固定名，下游读取）
- 格式：`时间, 预测负荷` 96 行（D+1 全天），UTF-8-BOM，2 位小数（与 v6 一致）

---

## 5. 产出物格式（与 v6 完全一致，不破坏工程约定）

| 文件 | 格式 | 说明 |
|---|---|---|
| `evaluation_metrics.txt` | UTF-8 文本 | MAE/RMSE/R2/MAPE/Bias/分位 + PASS/FAIL(<1500) |
| `full_predictions.csv` | UTF-8-BOM, `时间,预测负荷,实际负荷` | full 范围每时刻 |
| `full_mae.csv` | UTF-8-BOM, `时间,预测负荷,实际负荷,MAE` | 逐时刻绝对误差 |
| `prediction_YYYYMMDD.csv` | UTF-8-BOM, `时间,预测负荷` | D+1 96 点 |
| `latest_prediction.csv` | 同上 | 固定名 |
| `model_bundle.pkl` + `boosters/*.pt` | pickle + state_dict | 集成 + 40 成员 |

---

## 6. 不变量保持（6 条泄露不变量全保持）

| # | 不变量 | TFT 如何保持 |
|---|---|---|
| 1 | actual eval-only | actual 仅作 target / MOS 目标 / eval；features.py 不读 actual；TFT encoder 不用 actual |
| 2 | pred_load lags，lag_192 必含 | encoder_len=288(3天) > lag_192(2天)；lag_672 仍作逐时刻特征；encoder 历史用 pred_load 非 actual |
| 3 | weather dedup 三阶段一致 | 复用 data_loader.load_weather_dedup（train run_time=None / predict run_time=09:00） |
| 4 | 时间边界 TRAIN_END=2026-02-28 | usable_mask 限制训练；val eval-only；OOF 3 折全在训练期 |
| 5 | 共享 build_features | 复用 load_pred/features.py（train/serve 一致，predict 14 天回看窗口防 skew） |
| 6 | 模块结构 | `python -m load_pred_tft.xxx`；模型/输出写本包内；predict 只 load 不重训 |

---

## 7. Phase 1 调优建议（仅 Phase 0 gate 通过后）

修改 `load_pred_tft/config.py` 的 `TRAIN_CONFIG`：

| 参数 | 默认 | 调优方向 |
|---|---|---|
| `encoder_len` | 288 | 试 96/192/384（288 在 v6 已用，但 TFT attention 可能不同） |
| `hidden_size` | 64 | 64/128（云端 GPU 显存够可上调） |
| `best_it_fixed` | 30 | 30/50/80（防过拟合，与 v6 best_it=80 同哲学） |
| `lr` | 1e-3 | 1e-3/5e-4 |
| `dropout` | 0.1 | 0.1/0.2（过拟合时上调） |
| `batch_size` | 16 | 16/32 |

**验证纪律（P2-8 教训，强制）**：所有调优须 OOF 3 折选超参 + val 验证迁移，OOF/val 一致率 ≥50% 才采纳。
`exp_tft_phase0.py` 已内置 OOF/val 一致性诊断，可复制为 `exp_tft_phase1_xxx.py` 做调优实验（throwaway，纯 stdout）。

**数值稳定性**：特征尺度跨度大（feat_std 0.0017~1.7e7，标准化已处理）。若 Phase 0 loss 不稳或 val 异常，
可在 `tft.py _standardize_fit` 改用 robust scaling（中位数/MAD）或对 `pred_load` 列先 log1p。

---

## 8. 决策点

```
Phase 0 (exp_tft_phase0.py)
  ├── raw val < 1515? 
  │     是 → Phase 1 调优 → OOF 校正后 val < v6 1445.62 且 OOF/val 一致?
  │     │       是 → TFT 路线成功，并入生产候选（生产 train.py 40 成员）
  │     │       否 → Phase 1 调优未破 v6，NO-GO 回 v6
  │     否 → 路线A NO-GO（同 TCN），回 v6 / 转路线B
```

无论结果如何，与 v6 体系不冲突（本包独立，不覆盖 load_pred 的 models/output）。
