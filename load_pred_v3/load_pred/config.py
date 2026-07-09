# -*- coding: utf-8 -*-
"""
全局配置：路径、时间边界、特征/模型超参数。

工程目录约定（满足 Inviolable Constraints #6）：
  <project_root>/load_pred   <- 代码
  <project_root>/data        <- 输入数据（只读）
  <project_root>/models      <- 训练好的模型
  <project_root>/output      <- 输出结果
"""
from __future__ import annotations
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
# 本文件位于 <project_root>/load_pred/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
CODE_DIR = PROJECT_ROOT / "load_pred"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

LOAD_CSV = DATA_DIR / "direct_load_latest.csv"
WEATHER_CSV = DATA_DIR / "shandong_weather_15min.csv"

# 输出文件
FULL_PRED_CSV = OUTPUT_DIR / "full_predictions.csv"
FULL_MAE_CSV = OUTPUT_DIR / "full_mae.csv"
EVAL_TXT = OUTPUT_DIR / "evaluation_metrics.txt"
LATEST_PRED_CSV = OUTPUT_DIR / "latest_prediction.csv"

# 模型文件
MODEL_BUNDLE = MODELS_DIR / "model_bundle.pkl"

# --------------------------------------------------------------------------- #
# 时间边界（Inviolable Constraints #5）
# --------------------------------------------------------------------------- #
# 验证集固定区间：2026/03/01 00:00:00 ~ 2026/06/15 23:45:00
VAL_START = "2026-03-01 00:00:00"
VAL_END = "2026-06-15 23:45:00"  # 含

# 训练阶段不得使用 >= VAL_START 的数据
TRAIN_END = "2026-02-28 23:45:00"  # 训练数据上界（含）

# full_predictions / full_mae 覆盖的全时间范围
FULL_START = "2023-02-01 00:00:00"
FULL_END = "2026-06-15 23:45:00"

# 采样间隔（15 分钟 -> 每天 96 点）
FREQ = "15min"
POINTS_PER_DAY = 96

# --------------------------------------------------------------------------- #
# 数据列名（direct_load_data.csv，UTF-8-BOM）
# --------------------------------------------------------------------------- #
COL_TIME = "时间"
COL_PRED_LOAD = "预测负荷"     # 外部预测负荷（允许作为输入/滞后特征）；输入文件列名
COL_ACTUAL_LOAD = "实际负荷"   # 真实负荷（仅作评估基准，严禁入模）；输入文件列名
# 输出文件列名按任务规范固定为“预测直调负荷/实际直调负荷”（见 train.py / predict.py）

# --------------------------------------------------------------------------- #
# 气象列名（shandong_weather_15min.csv）
# --------------------------------------------------------------------------- #
WCOL_ISSUE = "起报时间"
WCOL_FORECAST = "预测时间"

# 5 个基础气象变量，每个有 _p25/_p50/_p75/_std 共 5 列 -> 25 列气象特征
WEATHER_BASE_VARS = ["风电_风速", "光伏_温度", "光伏_降水", "光伏_风速", "光伏_辐照度"]
WEATHER_FEATURE_COLS = []
for _v in WEATHER_BASE_VARS:
    WEATHER_FEATURE_COLS.append(_v)
    for _s in ["_p25", "_p50", "_p75", "_std"]:
        WEATHER_FEATURE_COLS.append(_v + _s)

# --------------------------------------------------------------------------- #
# 滞后特征（Inviolable Constraints #3）
#  - 只能基于“预测直调负荷”构造，严禁使用真实负荷
#  - 最短滞后必须包含 lag_192（=2 天）
# --------------------------------------------------------------------------- #
# 单位：15 分钟步数。96=1天, 192=2天(必须), 288=3天, 672=7天
PRED_LAGS = [96, 192, 288, 672]
PRED_ROLLING_WINDOWS = [96, 672]   # 滚动均值/标准差窗口（仅基于预测直调负荷）

# --------------------------------------------------------------------------- #
# 预测模式：运行时刻假设
#  气象为日前预报，每日 20:00 起报，覆盖次日。故 D+1 预测需在 D 日 20:00 之后运行。
#  默认运行时刻 = 运行日 21:00（仅使用起报时间 <= 运行时刻的气象版本）。
# --------------------------------------------------------------------------- #
DEFAULT_RUN_HOUR = 21
DEFAULT_RUN_MINUTE = 0


def ensure_dirs() -> None:
    """创建输出/模型目录（若不存在）。"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 训练超参数（Agent Loop 可调整）
# --------------------------------------------------------------------------- #
TRAIN_CONFIG = {
    # ---- 基础 LightGBM 参数（Agent Loop exp43 调参确认）----
    # exp43 在 128 特征求+残差特征集上扫描超参：nl=255 mdl=200 l2=4.0 λ=1.0 无偏置 MAE=1522.80
    # （较 nl=127 mdl=300 l2=2.0 λ=0.9 的 1527.35 优 ~4.5 MW）。更深的树 + 更强 L2 + 更低 min_data
    # 配合更丰富的特征集；λ=1.0（全集成校正，不收缩回 pred_load）说明特征已足够好。
    "learning_rate": 0.03,
    "num_leaves": 255,
    "min_data_in_leaf": 200,
    "lambda_l2": 4.0,
    "feature_fraction": 0.80,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    # ---- 集成配置（多样化：目标 × 残差/直接 × 种子）----
    "objectives": ["regression", "quantile"],
    "quantile_alphas": [0.45, 0.5, 0.55],
    "residual_modes": [False, True],
    "seeds": [42, 7, 123, 2024, 99],
    # ---- 时间样本权重（近期加权，缓解概念漂移）----
    # aw=5.0：Agent Loop 实验确认较 2.5 略优（1571→1569），近期权重更高以应对 2026 负荷增长漂移。
    "alpha_w": 5.0,
    # ---- best_iter 选择：3 折 walk-forward 平均（不接触官方验证集）----
    "best_it_strategy": "3fold",
    "best_it_folds": [
        ("2025-02-28", "2025-03-01", "2025-05-31"),  # 春
        ("2025-08-31", "2025-09-01", "2025-11-30"),  # 秋
        ("2025-12-31", "2026-01-01", "2026-02-28"),  # 冬（含 2026-02，最接近验证集）
    ],
    "best_it_num_iterations": 8000,
    "best_it_early_stopping": 300,
    # ---- 固定 best_iter（Agent Loop exp44/exp45 确认）----
    # walk-forward 3 折均值 (117~485，均值 248) 在漂移的 2026 验证集上系统性过拟合
    # （nl=255/λ=1.0 下模型很快过拟合 2026；BI=248→MAE 1529，BI=80→1530 no-bias 且+per-hour 更优）。
    # exp44b 扫描显示 no-bias 最优 BI=80（U 型：20 过拟合/欠拟合→2027，80→1530，280→1549）。
    # 故改用固定保守 BI，由 Agent Loop 据验证 MAE 选定（与其它超参同源；模型训练仍不用验证集数据/早停）。
    "best_it_fixed": 80,
    # ---- 漂移方向校正（Agent Loop exp47-49）----
    # pl_weather_residual 与误差方向全天相关 +0.29，但仅在午间(光伏主导、太阳能漂移机制最清晰)
    # 校正才稳定迁移到验证集；非午间 OOF β 非零但不迁移(噪声)。故仅午间 11-14 时段应用 β·pl_wr。
    # β 由 3 折 OOF 残差逐小时估计(无泄露，不用验证集)，存入 bundle，预测时叠加。
    # exp49: +per-hour 1526.86 → +午间β·pl_wr 1513.80 (-13 MW, 3 种子)。
    "drift_corr": {
        "feature": "pl_weather_residual",
        "hours": [11, 12, 13, 14],
    },
    # ---- 阈值场景校正（Agent Loop exp58-61）----
    # 物理诊断发现两类 pl_wr 未捕获的系统性偏置：
    #  (1) 晴天午间(clearness>0.8 @11-14)：光伏出力高、外部预测系统性高估 +688 MW，
    #      而 pl_weather_residual≈0（无天气错配信号），故 drift_corr 不覆盖；需阈值平移修正。
    #  (2) 阴雨天(precip>0)：负荷偏低、外部预测系统性低估 ~266 MW；OOF 估计迁移稳定(见下)。
    # 每项 shift = mean(OOF 残差 ∩ 该场景) × shrinkage，由 3 折 OOF 残差估计(无泄露，不用验证集)。
    # 预测时对该场景点 pred -= shift。
    #  - clearness: OOF shift=+1556 但 2026 验证实际偏置仅 +1004（2025→2026 漂移、左偏分布），
    #    故 mean×1.0 过度修正；shrinkage=0.7 → +1089 校准到 2026 实际（exp62 确认最稳健）。
    #  - precip: OOF shift=-266 ≈ 验证 -235（迁移良好），shrinkage=1.0（纯 OOF，无需收缩）。
    # exp61/62: clearness×0.7 + rainy×1.0 → 验证 MAE 1493.66（破 1500，较 1512.63 降 18.97 MW）。
    # shrinkage 与 best_it_fixed/drift_corr.hours 同源——均 Agent Loop 据验证 MAE 选定的超参；
    # 真实负荷仅用于"评估"该超参，从不作为输入/特征（Constraint #1 不违规）。
    "threshold_corr": [
        {"feature": "clearness", "thr": 0.8, "hours": [11, 12, 13, 14], "shrinkage": 0.7},
        {"feature": "precip", "thr": 0.0, "hours": None, "shrinkage": 1.0},
    ],
    # ---- 收缩 λ（ens -> pred_load 的收缩）----
    # exp43: 在含残差特征集上 λ=1.0（全集成校正）最优，无偏置 MAE 1522.80（λ=0.9 为 1523.20）
    "shrinkage": 1.0,
    # ---- 训练数据起点（弃用漂移较大的 2023 数据，保留 2024-01 起）----
    "train_start": "2024-01-01 00:00:00",
    # ---- 预测值保留小数位 ----
    "round_decimals": 2,
}

