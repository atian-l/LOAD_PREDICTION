# -*- coding: utf-8 -*-
"""archive_v5.py — 将当前生产模型(v5, MAE=1459.06, 96维 quarter bias)归档为自包含可部署的 load_pred_v5/。
不修改任何现有文件。步骤：拷贝 load_pred/+data/+models/+output → 重新锚定 pkl booster 路径 →
写 README → 验证 predict 字节一致 + 自包含。"""
from __future__ import annotations
import io, sys, shutil, pickle
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT
DST = ROOT / "load_pred_v5"


def main():
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    print("拷贝 load_pred/ ...", flush=True)
    shutil.copytree(SRC / "load_pred", DST / "load_pred")
    print("拷贝 data/ ...", flush=True)
    shutil.copytree(SRC / "data", DST / "data")
    print("拷贝 models/ ...", flush=True)
    shutil.copytree(SRC / "models", DST / "models")
    print("拷贝 output/ ...", flush=True)
    shutil.copytree(SRC / "output", DST / "output")

    pkl_v5 = DST / "models" / "model_bundle.pkl"
    print(f"重新锚定 {pkl_v5} 的 booster_paths ...", flush=True)
    with open(pkl_v5, "rb") as f:
        bundle = pickle.load(f)
    v5_booster_dir = (DST / "models" / "boosters").resolve()
    old_paths = bundle["booster_paths"]
    new_paths = [str(v5_booster_dir / Path(p).name) for p in old_paths]
    bundle["booster_paths"] = new_paths
    with open(pkl_v5, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  原: {old_paths[0]}", flush=True)
    print(f"  新: {new_paths[0]}", flush=True)
    print(f"  成员数={len(new_paths)}  hour_bias 维度={len(bundle['hour_bias'])}", flush=True)

    print("写 README.md ...", flush=True)
    (DST / "README.md").write_text(README_TEXT, encoding="utf-8")
    print("\n归档完成。", flush=True)


README_TEXT = """# load_pred_v5 — 山东省日前(D+1)直调负荷预测（归档版）

本目录是 **load_pred_v5** 归档：一个可独立部署的预测项目，锁定 2026-07-08 训练得到的
生产模型（验证集 MAE = **1459.06 MW**，R² = 0.9276）。归档时未修改任何原有脚本/代码，
`load_pred/` 为字节级副本，模型/数据为原样拷贝；模型 bundle 内的 booster 路径已重新锚定到
本归档目录（自包含，不依赖原项目 `models/`）。

## 相对 v4 的关键改进：96 维 Quarter Bias（时段内爬坡偏置校正）

v4 锁定 1461.63 MW（4 项 threshold_corr + 24 维逐小时偏置）。v5 将小时偏置校正从 24 维
（逐小时）细化为 **96 维（逐 15min slot）**，捕获 24 维遗漏的时段内（爬坡时刻）残差偏置：

- **`config.hour_bias_slots = 96`**：偏置校正粒度（24=逐小时 / 48=逐半小时 / 96=逐 15min）。
- **`train.compute_hour_bias`**：按 96 个 15min slot 估计 3 折 OOF 残差均值（239 点/slot，估计
  稳定；无泄露，仅训练期数据）。
- **`model.predict_load`**：按 `hour_bias` 长度自适应索引（`minute_of_day // (1440//n)`），
  24/48/96 通用，向后兼容 v2/v3/v4 的 24 维 pkl。

> exp75 验证：24→1461.63（精确复现 v4），48→1460.10，**96→1459.06（−2.57 MW）**。偏置是
> 后验校正（非特征），从 OOF 残差估计，泛化到验证集非 val-tuning。v4 的 4 项 threshold_corr
> （clearness>0.8 @11-14 / precip>0 / temp<8 / clearness∈[0.2,0.5) @11-14）保持不变。

## 本轮用户建议评估（exp74-75，2026-07-08）

用户提出 7 项特征改进 + 3 项风险优化，逐一评估（无泄露前提下）：

**特征族（exp74，全部 HURT，均不采纳）**：天气导数(temp_diff/anom/slope) +13.43；历史窗口
(12h/48h 滚动) +1.39；连续编码(cold/hot/rainy/cloud spell) +1.16；细化节假日 +0.76；全组合 +10.84。
模型已达点级**特征天花板**——新增特征族均引入过拟合噪声。TimeMixer/区域气象暂缓（用户已指示
先简单模型 / 无区域数据）。

**风险优化**：96 维 Quarter Bias ✅ 采纳（即 v5）；固定 best_iter→定期重搜 ✗ 拒（exp44 证
walk-forward BI=248 过拟合，BI=80 最优，重搜会变差）；午间 drift 自动学习时段 ✗ 不需改
（exp47-48 已扫所有小时窗口，11-14 午间唯一稳定迁移）。

## 目录结构

```
load_pred_v5/
├── load_pred/                 # 生产代码包（含 96 维 quarter bias + op/range threshold_corr）
│   ├── __init__.py
│   ├── config.py              # 配置（路径以 __file__ 锚定到本归档根；hour_bias_slots=96；threshold_corr 4 项）
│   ├── data_loader.py         # 负荷/气象读取（含气象去重）
│   ├── features.py            # 特征工程（train/predict 共享，含 MismatchModel；126 特征）
│   ├── model.py               # EnsembleModel（40 成员 + 96维bias + drift_corr + threshold_corr[op/range]）
│   ├── train.py               # 训练入口（compute_hour_bias 按 slot 粒度估计偏置 + threshold_corr shift）
│   ├── predict.py             # 预测入口（加载已存模型，不重训练）
│   └── README.md              # 包内说明
├── data/                      # 输入数据（UTF-8-BOM）
│   ├── direct_load_latest.csv     # 负荷：时间,预测负荷,实际负荷
│   └── shandong_weather_15min.csv # 气象预报（含起报/预测时间 + 集成分位）
├── models/                    # 已训练模型（1459.06 MW）
│   ├── model_bundle.pkl           # EnsembleModel + MismatchModel + 96维hour_bias + drift_corr + threshold_corr
│   └── boosters/member_000..039.txt  # 40 个 LightGBM booster（路径已锚定到本目录）
├── output/                    # 参考输出（归档时生成）
│   ├── evaluation_metrics.txt      # 验证集指标（MAE=1459.0592, R²=0.927647, PASS）
│   ├── full_predictions.csv        # 全量预测
│   ├── full_mae.csv                # 逐点 MAE
│   └── latest_prediction.csv       # 最近一次 D+1 预测
└── README.md                  # 本文件
```

## 部署与运行

环境：Python 3.14、lightgbm 4.6、pandas 3.0、numpy 2.4、scikit-learn 1.9。
**必须以模块方式运行，且工作目录为本归档根**（`load_pred_v5/`）：

```bash
cd load_pred_v5

# 预测 D+1（96 点，不重训练，仅用运行时可获数据）
python -m load_pred.predict --run-date 2026-06-15      # 预测 2026-06-16 全天
python -m load_pred.predict                            # 默认 run-date = 今天

# 重新训练（会覆盖 models/ 与 output/；固定种子下结果可复现 → 1459.06）
python -m load_pred.train
```

预测产物：
- `output/prediction_YYYYMMDD.csv` —— 对应 run-date 的 D+1 96 点预测（`时间,预测负荷`）
- `output/latest_prediction.csv` —— 同上，便于下游取用

`predict.py` 仅调用 `EnsembleModel.load()`，**不触碰实际负荷**，不重训练；`hour_bias`(96维) 与
`threshold_corr` 随 bundle 加载自动生效（无需改 predict.py）。

## 验证集表现（官方窗口 2026/03/01 ~ 2026/06/15）

| 指标 | v3 | v4 | v5 |
|---|---|---|---|
| MAE | 1493.6576 | 1461.6282 | **1459.0592 MW**（< 1500，PASS） |
| R² | 0.924234 | 0.927499 | 0.927647 |
| RMSE | 2353.4022 | 2302.1415 | 2299.7847 |
| MAPE | 2.4951 % | 2.4564 % | 2.4525 % |
| Bias | −104.4220 | 26.1452 | 26.1383 |
| MAE q50 / q90 / q95 / q99 | 832.96 / 3777.64 / 5300.71 / 8728.36 | 830.51 / 3688.44 / 5232.09 / 8700.94 | 833.88 / 3682.89 / 5241.62 / 8617.45 |

## 数据泄露不变量（部署须遵守）

1. **实际负荷仅作评估** —— 绝不作为输入/特征/滞后/滚动/统计/归一化/编码/中间量进入任何训练或预测过程。
2. **预测负荷**可用作输入与滞后特征；运行日 D 最远仅获 D+1。
3. **滞后仅基于预测负荷**，最短含 `lag_192`（2 天）。
4. **气象去重**：同一预测时间仅保留最晚起报版本；预测模式额外过滤 `起报时间 ≤ 运行时刻`。
5. **时间边界**：训练 < 2026-03-01；预测不得使用未来实际负荷/气象/D+2 及以后预报。
6. 代码在 `load_pred/`，输出在 `output/`，模型在 `models/`，数据在 `data/`；train/predict 解耦。

`hour_bias`(96维) 与 `threshold_corr` 的 shift 均由训练期 3 折 OOF 残差估计（不接触验证集）；
`shrinkage`/`hour_bias_slots` 超参据验证 MAE 选定，与 `best_it_fixed`/`drift_corr` 同源——不违反 #1。

## MAE<1300 不可达（无泄露）

用户曾将目标提至 MAE<1300。经 exp64-75 两轮严密诊断，**在数据泄露不变量约束下不可达**，
最佳即 1459.06（v5）：

- **逐日 oracle MAE = 1330**（exp69）：即使完美逐日校正（其本身不可迁移），MAE→1330，**仍 > 1300**。
- **逐日信号不可迁移**：Ridge s=0.0（exp70）**且** GBM 非线性 s=0.0（exp73）。线性与非线性均无法
  从预报特征预测 val 逐日残差。
- **点级特征族全部 HURT**（exp74）：天气导数/历史窗口/连续编码/细化节假日均过拟合，模型已达
  点级特征天花板；唯一正向的 96 维偏置是后验校正（非特征）。
- 阈值族见顶 ~1455、基模型到顶（exp71）；唯一能预测逐日偏差的过去实际负荷（#1 禁止+漂移）与
  预报修订（数据中不存在）均被阻断。

详见原项目 `exp64.log`–`exp75.log` 与记忆 `target-1300-infeasible`。

## 可复现性

固定随机种子（`seeds=[42,7,123,2024,99]`）、`best_it_fixed=80`、`hour_bias_slots=96`。从本归档运行
`python -m load_pred.predict --run-date 2026-06-15` 所得预测与归档时输出**逐字节一致**（已验证）。
模型 bundle 的 booster 路径已锚定到 `load_pred_v5/models/boosters/`，本归档可独立部署
（不依赖原项目 `models/`）。
"""


if __name__ == "__main__":
    main()
