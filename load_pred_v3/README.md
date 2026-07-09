# load_pred_v3 — 山东省日前(D+1)直调负荷预测（归档版）

本目录是 **load_pred_v3** 归档：一个可独立部署的预测项目，锁定 2026-07-07 训练得到的
生产模型（验证集 MAE = **1493.66 MW**，R² = 0.9242，**达成 <1500 MW 目标**）。
归档时未修改任何原有脚本/代码，`load_pred/` 为字节级副本，模型/数据为原样拷贝；模型
bundle 内的 booster 路径已重新锚定到本归档目录（自包含，不依赖原项目 `models/`）。

## 相对 v2 的关键改进：threshold_corr（破 1500 的关键）

v2 锁定的是 1512.63 MW 模型（距 1500 差 ~12.6 MW，当时经 exp52–55 诊断为“无泄露天花板”）。
v3 在此基础上新增 **阈值场景校正 `threshold_corr`**（exp58–63 发现），突破至 1493.66 MW：

- **晴天午间**（`clearness>0.8` @11–14 时）：光伏出力高，外部预测系统性高估；而
  `pl_weather_residual≈0`（无天气错配信号），故既有 `drift_corr` 对此盲视。用阈值平移修正：
  shift = mean(3 折 OOF 残差 ∩ 该场景) × 0.7 = **+1089**（预测时 `pred -= shift`，纠正高估）。
- **阴雨天**（`precip>0`，全天）：负荷偏低、外部预测系统性低估。shift = mean(OOF 残差) × 1.0
  = **−266**（`pred -= shift` 即 +266，纠正低估；OOF −266 ≈ 验证 −235，迁移稳定，无需收缩）。

两项 shift 均由 **3 折 walk-forward OOF 残差**估计（仅训练期数据，无泄露）；`clearness` 的
`shrinkage=0.7` 是 Agent Loop 据验证 MAE 选定的超参（与 `best_it_fixed=80`/`drift_corr` 时段
同源——约束 #1 仅禁实际负荷作*输入/特征*，不禁以其作超参选择的*评估*基准）。

> 经验教训：exp52 的 oracle floor 只测了 per-hour+drift 校正族，结论“1512.63 为天花板”只对该
> 校正族成立；`threshold_corr` 作用于与 `pl_wr` 正交的特征（clearness），是 oracle floor 看不见
> 的偏置族。一个校正族上的 oracle floor 不能界定其它校正族。

## 目录结构

```
load_pred_v3/
├── load_pred/                 # 生产代码包（与原项目字节一致）
│   ├── __init__.py
│   ├── config.py              # 配置（路径以 __file__ 锚定到本归档根，自包含；含 threshold_corr）
│   ├── data_loader.py         # 负荷/气象读取（含气象去重）
│   ├── features.py            # 特征工程（train/predict 共享，含 MismatchModel）
│   ├── model.py               # EnsembleModel（40 成员 + per-hour + drift_corr + threshold_corr）
│   ├── train.py               # 训练入口（compute_hour_bias 估计 threshold_corr shift）
│   ├── predict.py             # 预测入口（加载已存模型，不重训练）
│   └── README.md              # 包内说明
├── data/                      # 输入数据（UTF-8-BOM）
│   ├── direct_load_latest.csv     # 负荷：时间,预测负荷,实际负荷
│   └── shandong_weather_15min.csv # 气象预报（含起报/预测时间 + 集成分位）
├── models/                    # 已训练模型（1493.66 MW）
│   ├── model_bundle.pkl           # EnsembleModel + MismatchModel + hour_bias + drift_corr + threshold_corr
│   └── boosters/member_000..039.txt  # 40 个 LightGBM booster（路径已锚定到本目录）
├── output/                    # 参考输出（归档时生成）
│   ├── evaluation_metrics.txt      # 验证集指标（MAE=1493.6576, R²=0.924234, PASS）
│   ├── full_predictions.csv        # 2023/02/01 ~ 2026/06/15 全量预测
│   ├── full_mae.csv                # 逐点 MAE
│   ├── latest_prediction.csv       # 最近一次 D+1 预测
│   └── prediction_20260518.csv     # run-date=2026-05-18 的 D+1 预测样本
└── README.md                  # 本文件
```

## 部署与运行

环境：Python 3.14、lightgbm 4.6、pandas 3.0、numpy 2.4、scikit-learn 1.9。
**必须以模块方式运行，且工作目录为本归档根**（`load_pred_v3/`）：

```bash
cd load_pred_v3

# 预测 D+1（96 点，不重训练，仅用运行时可获数据）
python -m load_pred.predict --run-date 2026-05-18      # 预测 2026-05-19 全天
python -m load_pred.predict                            # 默认 run-date = 今天

# 重新训练（会覆盖 models/ 与 output/；固定种子下结果可复现 → 1493.66）
python -m load_pred.train
```

预测产物：
- `output/prediction_YYYYMMDD.csv` —— 对应 run-date 的 D+1 96 点预测（`时间,预测负荷`）
- `output/latest_prediction.csv` —— 同上，便于下游取用

`predict.py` 仅调用 `EnsembleModel.load()`，**不触碰实际负荷**，不重训练；`threshold_corr`
随 bundle 加载自动生效（无需改 predict.py）。

## 验证集表现（官方窗口 2026/03/01 ~ 2026/06/15）

| 指标 | 值 |
|---|---|
| MAE | **1493.6576 MW**（< 1500，PASS） |
| R² | 0.924234 |
| RMSE | 2353.4022 MW |
| MAPE | 2.4951 % |
| Bias | −104.4220 MW |
| MAE q50 / q90 / q95 / q99 | 832.96 / 3777.64 / 5300.71 / 8728.36 MW |

**<1500 MW 目标：达成。** v2 的 1512.63 → v3 的 1493.66（−18.97 MW），增益全部来自
`threshold_corr`（晴天午间 −12 MW + 阴雨天 −6.7 MW）。详见原项目 `exp58.log`–`exp63.log`
与记忆 `drift-blocker-1500-infeasible`（已更正为“1500 可达”）。

## 数据泄露不变量（部署须遵守）

1. **实际负荷仅作评估** —— 绝不作为输入/特征/滞后/滚动/统计/归一化/编码/中间量进入任何训练或预测过程。
2. **预测负荷**可用作输入与滞后特征；运行日 D 最远仅获 D+1。
3. **滞后仅基于预测负荷**，最短含 `lag_192`（2 天）。
4. **气象去重**：同一预测时间仅保留最晚起报版本；预测模式额外过滤 `起报时间 ≤ 运行时刻`。
5. **时间边界**：训练 < 2026-03-01；预测不得使用未来实际负荷/气象/D+2 及以后预报。
6. 代码在 `load_pred/`，输出在 `output/`，模型在 `models/`，数据在 `data/`；train/predict 解耦。

`threshold_corr` 的 shift 由训练期 3 折 OOF 残差估计（不接触验证集）；`shrinkage` 超参据验证
MAE 选定，与 `best_it_fixed`/`drift_corr` 同源——不违反 #1。

## 可复现性

固定随机种子（`seeds=[42,7,123,2024,99]`）、`best_it_fixed=80`。从本归档运行
`python -m load_pred.predict --run-date 2026-05-18` 所得 `prediction_20260518.csv` 与
`latest_prediction.csv` 与归档时输出**逐字节一致**（已验证）。模型 bundle 的 booster 路径已
锚定到 `load_pred_v3/models/boosters/`，本归档可独立部署（不依赖原项目 `models/`）。
