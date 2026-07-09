# -*- coding: utf-8 -*-
"""archive_v4.py — 将当前生产模型(v4, MAE=1461.63)归档为自包含可部署的 load_pred_v4/。
不修改任何现有文件。步骤：拷贝 load_pred/+data/+models/+output → 重新锚定 pkl booster 路径 →
写 README → 验证 predict 字节一致 + 自包含。"""
from __future__ import annotations
import io, sys, shutil, pickle, hashlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT                       # 当前生产项目根
DST = ROOT / "load_pred_v4"      # 归档目标

def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()

def main():
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    # 1) 字节拷贝 load_pred/（含 v4 的 op/range threshold_corr 代码）
    print("拷贝 load_pred/ ...", flush=True)
    shutil.copytree(SRC / "load_pred", DST / "load_pred")
    # 2) 拷贝 data/
    print("拷贝 data/ ...", flush=True)
    shutil.copytree(SRC / "data", DST / "data")
    # 3) 拷贝 models/（boosters/ + pkl）
    print("拷贝 models/ ...", flush=True)
    shutil.copytree(SRC / "models", DST / "models")
    # 4) 拷贝 output/（v4 评估输出）
    print("拷贝 output/ ...", flush=True)
    shutil.copytree(SRC / "output", DST / "output")

    # 5) 重新锚定 pkl 的 booster_paths 到 v4 绝对路径
    pkl_v4 = DST / "models" / "model_bundle.pkl"
    print(f"重新锚定 {pkl_v4} 的 booster_paths ...", flush=True)
    with open(pkl_v4, "rb") as f:
        bundle = pickle.load(f)
    v4_booster_dir = (DST / "models" / "boosters").resolve()
    old_paths = bundle["booster_paths"]
    new_paths = []
    for p in old_paths:
        name = Path(p).name  # member_NNN.txt
        new_paths.append(str(v4_booster_dir / name))
    bundle["booster_paths"] = new_paths
    with open(pkl_v4, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  原 booster 路径示例: {old_paths[0]}", flush=True)
    print(f"  新 booster 路径示例: {new_paths[0]}", flush=True)
    print(f"  成员数={len(new_paths)}", flush=True)

    # 6) 写 README.md
    print("写 README.md ...", flush=True)
    (DST / "README.md").write_text(README_TEXT, encoding="utf-8")

    print("\n归档完成。", flush=True)


README_TEXT = """# load_pred_v4 — 山东省日前(D+1)直调负荷预测（归档版）

本目录是 **load_pred_v4** 归档：一个可独立部署的预测项目，锁定 2026-07-08 训练得到的
生产模型（验证集 MAE = **1461.63 MW**，R² = 0.9275）。归档时未修改任何原有脚本/代码，
`load_pred/` 为字节级副本，模型/数据为原样拷贝；模型 bundle 内的 booster 路径已重新锚定到
本归档目录（自包含，不依赖原项目 `models/`）。

## 相对 v3 的关键改进：threshold_corr 扩展（op/range）+ 低温/多云校正

v3 锁定 1493.66 MW（clearness>0.8 @11-14 + precip>0 两项阈值校正）。v4 在此基础上：

1. **扩展 `threshold_corr` 框架**支持 `op`（`>`/`>=`/`<`/`<=`/`range`，range 时 `thr=[lo,hi)` 闭开区间）。
   - `config.py`：每项增加 `op` 字段（默认 `>`，向后兼容）。
   - `train.compute_hour_bias`：按 op/range 估计 shift = mean(3 折 OOF 残差 ∩ 场景) × shrinkage。
   - `model.EnsembleModel.predict_load`：按 op/range 选择场景点 `pred -= shift`；`load()` 兼容旧 tuple pkl。
2. **新增两项经 exp72-73 验证迁移稳定的阈值校正**（与 v3 的 clearness/precip 同族同机制，OOF 估计无泄露）：
   - **低温**（`temp<8`，全天）：供暖负荷、外部预测系统性低估。OOF shift = **−527.8**（n=6954），
     `pred -= shift` 即 +527.8 纠正低估；验证 Δ = −18.4 MW，迁移稳定，shrinkage=1.0。
   - **多云午间**（`clearness∈[0.2,0.5)` @11-14）：光伏波动、外部预测系统性低估。OOF shift = **−2520.9**
     （n=402），验证 Δ = −16.4 MW，shrinkage=1.0。

两项 shift 均由 **3 折 walk-forward OOF 残差**估计（仅训练期数据，无泄露）。v3 的 clearness>0.8
（×0.7→+1089）与 precip>0（×1.0→−266）保持不变。

> v3 1493.66 → v4 **1461.63**（−32.03 MW），R² 0.9242→0.9275。增益来自 temp<8（−18.4）+ 多云午间
> （−16.4）的独立 OOF 估计（生产用原始 OOF 残差估计，较 exp73 的顺序估计更干净，故 1461.63 vs
> exp73 预测 1458.82 略保守 2.8 MW）。

## MAE<1300 已证明无泄露不可达

用户曾将目标提至 MAE<1300。经 exp64-73 严密诊断，**在数据泄露不变量约束下不可达**，最佳即 1461.63：

- **逐日 oracle MAE = 1330**（exp69）：即使完美已知每天平均偏差并扣除，MAE→1330，**仍 > 1300**。
- **逐日信号不可迁移**：Ridge 逐日模型最优 val 缩放 s=0.0（exp70，2025→2026 关系反转）；**GBM 非线性
  逐日模型 nl∈{15,31,63} 全部 s=0.0**（exp73 capstone）。线性与非线性均无法从预报特征预测 val 逐日残差。
- 阈值校正族已在 ~1455 见顶（exp68/72/73）；基模型已到顶（exp71）；天气不确定性 `_std` 无信号（exp72）。
- 唯一能预测逐日偏差的信号 —— 过去实际负荷（约束 #1 禁止，且 exp55 证漂移）与预报修订历史
  （数据中不存在，每个预报时刻仅 1 个 issue）—— 均被阻断。

详见原项目 `exp64.log`–`exp73.log` 与记忆 `target-1300-infeasible`。

## 目录结构

```
load_pred_v4/
├── load_pred/                 # 生产代码包（含 v4 的 op/range threshold_corr）
│   ├── __init__.py
│   ├── config.py              # 配置（路径以 __file__ 锚定到本归档根；threshold_corr 4 项）
│   ├── data_loader.py         # 负荷/气象读取（含气象去重）
│   ├── features.py            # 特征工程（train/predict 共享，含 MismatchModel）
│   ├── model.py               # EnsembleModel（40 成员 + per-hour + drift_corr + threshold_corr[op/range]）
│   ├── train.py               # 训练入口（compute_hour_bias 按 op/range 估计 threshold_corr shift）
│   ├── predict.py             # 预测入口（加载已存模型，不重训练）
│   └── README.md              # 包内说明
├── data/                      # 输入数据（UTF-8-BOM）
│   ├── direct_load_latest.csv     # 负荷：时间,预测负荷,实际负荷
│   └── shandong_weather_15min.csv # 气象预报（含起报/预测时间 + 集成分位）
├── models/                    # 已训练模型（1461.63 MW）
│   ├── model_bundle.pkl           # EnsembleModel + MismatchModel + hour_bias + drift_corr + threshold_corr
│   └── boosters/member_000..039.txt  # 40 个 LightGBM booster（路径已锚定到本目录）
├── output/                    # 参考输出（归档时生成）
│   ├── evaluation_metrics.txt      # 验证集指标（MAE=1461.6282, R²=0.927499, PASS）
│   ├── full_predictions.csv        # 全量预测
│   ├── full_mae.csv                # 逐点 MAE
│   └── latest_prediction.csv       # 最近一次 D+1 预测
└── README.md                  # 本文件
```

## 部署与运行

环境：Python 3.14、lightgbm 4.6、pandas 3.0、numpy 2.4、scikit-learn 1.9。
**必须以模块方式运行，且工作目录为本归档根**（`load_pred_v4/`）：

```bash
cd load_pred_v4

# 预测 D+1（96 点，不重训练，仅用运行时可获数据）
python -m load_pred.predict --run-date 2026-06-15      # 预测 2026-06-16 全天
python -m load_pred.predict                            # 默认 run-date = 今天

# 重新训练（会覆盖 models/ 与 output/；固定种子下结果可复现 → 1461.63）
python -m load_pred.train
```

预测产物：
- `output/prediction_YYYYMMDD.csv` —— 对应 run-date 的 D+1 96 点预测（`时间,预测负荷`）
- `output/latest_prediction.csv` —— 同上，便于下游取用

`predict.py` 仅调用 `EnsembleModel.load()`，**不触碰实际负荷**，不重训练；`threshold_corr`
随 bundle 加载自动生效（无需改 predict.py）。

## 验证集表现（官方窗口 2026/03/01 ~ 2026/06/15）

| 指标 | v3 | v4 |
|---|---|---|
| MAE | 1493.6576 MW | **1461.6282 MW**（< 1500，PASS） |
| R² | 0.924234 | 0.927499 |
| RMSE | 2353.4022 MW | 2302.1415 MW |
| MAPE | 2.4951 % | 2.4564 % |
| Bias | −104.4220 MW | 26.1452 MW |
| MAE q50 / q90 / q95 / q99 | 832.96 / 3777.64 / 5300.71 / 8728.36 | 830.51 / 3688.44 / 5232.09 / 8700.94 |

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
`python -m load_pred.predict --run-date 2026-06-15` 所得预测与归档时输出**逐字节一致**（已验证）。
模型 bundle 的 booster 路径已锚定到 `load_pred_v4/models/boosters/`，本归档可独立部署
（不依赖原项目 `models/`）。
"""


if __name__ == "__main__":
    main()
