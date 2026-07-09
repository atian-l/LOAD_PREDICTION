# load_pred_v2 — 山东省日前(D+1)直调负荷预测（归档版）

本目录是 **load_pred_v2** 归档：一个可独立部署的预测项目，锁定 2026-07-07 训练得到的
生产模型（验证集 MAE = **1512.63 MW**，R² = 0.923）。归档时未修改任何原有脚本/代码，
`load_pred/` 为字节级副本，模型/数据为原样拷贝。

## 目录结构

```
load_pred_v2/
├── load_pred/                 # 生产代码包（与原项目字节一致）
│   ├── __init__.py
│   ├── config.py              # 配置（路径以 __file__ 锚定到本归档根，自包含）
│   ├── data_loader.py         # 负荷/气象读取（含气象去重）
│   ├── features.py            # 特征工程（train/predict 共享，含 MismatchModel）
│   ├── model.py               # EnsembleModel（40 LightGBM 成员 + per-hour + drift_corr）
│   ├── train.py               # 训练入口
│   ├── predict.py             # 预测入口（加载已存模型，不重训练）
│   └── README.md              # 包内说明
├── data/                      # 输入数据（UTF-8-BOM）
│   ├── direct_load_latest.csv     # 负荷：时间,预测负荷,实际负荷
│   └── shandong_weather_15min.csv # 气象预报（含起报/预测时间 + 集成分位）
├── models/                    # 已训练模型（1512.63 MW）
│   ├── model_bundle.pkl           # EnsembleModel + MismatchModel + hour_bias + drift_corr
│   └── boosters/member_000..039.txt  # 40 个 LightGBM booster
├── output/                    # 参考输出（归档时生成）
│   ├── evaluation_metrics.txt      # 验证集指标（MAE=1512.6251, R²=0.923）
│   ├── full_predictions.csv        # 2023/02/01 ~ 2026/06/15 全量预测
│   ├── full_mae.csv                # 逐点 MAE
│   ├── latest_prediction.csv       # 最近一次 D+1 预测
│   └── prediction_20260518.csv     # run-date=2026-05-18 的 D+1 预测样本
└── README.md                  # 本文件
```

## 部署与运行

环境：Python 3.14、lightgbm 4.6、pandas 3.0、numpy 2.4、scikit-learn 1.9。
**必须以模块方式运行，且工作目录为本归档根**（`load_pred_v2/`）：

```bash
cd load_pred_v2

# 预测 D+1（96 点，不重训练，仅用运行时可获数据）
python -m load_pred.predict --run-date 2026-05-18      # 预测 2026-05-19 全天
python -m load_pred.predict                            # 默认 run-date = 今天

# 重新训练（会覆盖 models/ 与 output/；固定种子下结果可复现）
python -m load_pred.train
```

预测产物：
- `output/prediction_YYYYMMDD.csv` —— 对应 run-date 的 D+1 96 点预测（`时间,预测负荷`）
- `output/latest_prediction.csv` —— 同上，便于下游取用

`predict.py` 仅调用 `EnsembleModel.load()`，**不触碰实际负荷**，不重训练。

## 验证集表现（官方窗口 2026/03/01 ~ 2026/06/15）

| 指标 | 值 |
|---|---|
| MAE | 1512.6251 MW |
| R² | 0.923005 |
| RMSE | 2372.4151 MW |
| MAPE | 2.5297 % |
| MAE q50 / q90 / q95 | 853.0 / 3799.2 / 5273.7 MW |

**关于 <1500 MW 目标：** 本模型 MAE=1512.63，距 1500 目标差 ~12.6 MW。经 exp52–55 系统诊断
证明：在“无数据泄露”约束下（实际负荷仅作评估基准），该差距为不可约的午间光伏/需求预测噪声；
即便放宽约束#1允许使用“过去实际负荷”做 MOS 偏置校正，因预测偏置逐日漂移亦无法突破（详见
原项目 `exp52.log`–`exp55.log` 与记忆 `drift-blocker-1500-infeasible`）。1512.63 为当前
特征/模型族的无泄露天花板。

## 数据泄露不变量（部署须遵守）

1. **实际负荷仅作评估** —— 绝不作为输入/特征/滞后/滚动/统计/归一化/编码/中间量进入任何训练或预测过程。
2. **预测负荷**可用作输入与滞后特征；运行日 D 最远仅获 D+1。
3. **滞后仅基于预测负荷**，最短含 `lag_192`（2 天）。
4. **气象去重**：同一预测时间仅保留最晚起报版本；预测模式额外过滤 `起报时间 ≤ 运行时刻`。
5. **时间边界**：训练 < 2026-03-01；预测不得使用未来实际负荷/气象/D+2 及以后预报。
6. 代码在 `load_pred/`，输出在 `output/`，模型在 `models/`，数据在 `data/`；train/predict 解耦。

## 可复现性

固定随机种子（`seeds=[42,7,123,2024,99]`）、`best_it_fixed=80`。从本归档运行
`python -m load_pred.predict --run-date 2026-05-18` 所得预测与 `output/prediction_20260518.csv`
**逐字节一致**（已验证）。
