# -*- coding: utf-8 -*-
"""
全局配置：路径、时间边界、特征/模型超参数（TCN 变体）。

本包是 load_pred（LightGBM 集成）的 TCN 移植版：除"模型方法 LightGBM -> TCN"外，
数据/特征/集成结构/OOF 校正/泄露不变量全部与 load_pred 逐行一致。

工程目录约定（满足 Inviolable Constraints #6；为不修改 load_pred_tcn 以外的文件）：
  <project_root>/load_pred_tcn   <- 代码 + 本包私有 models/ output/
  <project_root>/data            <- 输入数据（共享，只读；不写入）
  <project_root>/load_pred_tcn/models   <- 本包训练好的模型（写在本包内）
  <project_root>/load_pred_tcn/output   <- 本包输出结果（写在本包内）
注：模型/输出写在本包内，避免覆盖 load_pred 的 models/output；数据仍读共享 data/（只读）。
"""
from __future__ import annotations
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
# 本文件位于 <project_root>/load_pred_tcn/config.py
PKG_DIR = Path(__file__).resolve().parent          # load_pred_tcn/
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # <project_root>/（用于读共享数据）

DATA_DIR = PROJECT_ROOT / "data"          # 共享输入数据（只读）
CODE_DIR = PKG_DIR                         # 本包代码
MODELS_DIR = PKG_DIR / "models"            # 本包私有模型目录（写在包内）
OUTPUT_DIR = PKG_DIR / "output"            # 本包私有输出目录（写在包内）

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
COL_ACTUAL_LOAD = "实际负荷"   # 真实负荷（仅作评估基准，严禁入模）；输入文件名
# 输出文件列名固定为“预测负荷/实际负荷”，与输入列名一致（见 train.py / predict.py）

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
# 预测模式：运行时刻假设（部署 = 每日 09:00 前后运行）
#  气象为日前预报，历史数据中每日 20:00 起报、覆盖次日（20:00 D issue 覆盖 D+1）。
#  部署在 D 日 09:00 运行预测 D+1：此时 20:00 D issue 尚未起报（11h 后），覆盖 D+1 的
#  气象需依赖"晨间起报"（如 08:00 D issue，覆盖 D+1），由部署端气象管线提供（用户确认可得）。
#  - val 评估(train.py) 用 run_time=None：取每个预测时刻"最晚起报版本"=20:00 D issue 覆盖 D+1，
#    作为晨间 issue 的代理（同为"运行时可得的最佳 D+1 预报"），val MAE 代表部署条件。
#  - predict.py 用 run_time=09:00 过滤 起报时间<=09:00（Constraint #4/#5）：部署端管线含晨间
#    issue 时 D+1 气象可得->≈val；若晨间 issue 不可得则 D+1 气象 NaN->模型退化为 pred_load+
#    calendar 基线（TCN 用列均值填 NaN，不崩溃）。历史 CSV 仅含 20:00 issue，故历史
#    predict 回测需用 --run-hour 21 复现 val 的 20:00-D-issue 条件。
# --------------------------------------------------------------------------- #
DEFAULT_RUN_HOUR = 9
DEFAULT_RUN_MINUTE = 0


def ensure_dirs() -> None:
    """创建输出/模型目录（若不存在）。"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 训练超参数（TCN 变体；Agent Loop 可调整）
# --------------------------------------------------------------------------- #
TRAIN_CONFIG = {
    # ---- TCN 基础参数（替代 LightGBM 的 num_leaves/lambda_l2/feature_fraction/bagging）----
    # 因果膨胀卷积：output[t] 仅依赖 input[<=t]（满足 Inviolable Constraints #5，无未来信息）。
    # 4 个残差块 dilations=[1,2,4,8]，kernel=7 -> 感受野 RF=1+(7-1)·(1+2+4+8)=91 步(≈23h)。
    # 长程信息已由 PRED_LAGS=[96,192,288,672] 等滞后特征编码，TCN 在此之上做局部时序平滑。
    "learning_rate": 1e-3,               # Adam 学习率（≈ LightGBM learning_rate 的对偶）
    "weight_decay": 1e-5,                # L2 正则（≈ LightGBM lambda_l2）
    "num_channels": [64, 64, 64, 64],    # 各残差块通道数（=块数决定深度与感受野）
    "kernel_size": 7,                    # 因果卷积核长
    "dropout": 0.1,                      # 残差块内 Dropout（正则）
    "seq_len": 480,                      # 训练滑窗长度（15min 步，=5 天；> RF=91）
    "stride": 96,                        # 滑窗步长（=1 天，重叠采样增梯度步；勿过小致 OOM）
    "batch_size": 64,                    # 每批窗口数
    "grad_clip": 5.0,                    # 梯度裁剪
    "device": "auto",                    # "auto"=cuda if available else cpu；训练建议 GPU
    # ---- 集成配置（与 v6 LightGBM 完全一致：目标 × 残差/直接 × 种子 = 40 成员）----
    # 多样化来源不变：{regression, quantile(0.45/0.5/0.55)} × {direct, residual} × 5 seeds。
    # quantile 成员用 pinball 损失（同 LightGBM quantile 目标），regression 用 MSE。
    "objectives": ["regression", "quantile"],
    "quantile_alphas": [0.45, 0.5, 0.55],
    "residual_modes": [False, True],
    "seeds": [42, 7, 123, 2024, 99],
    # ---- 时间样本权重（近期加权，缓解概念漂移；与 v6 一致）----
    "alpha_w": 5.0,
    # ---- 联合样本权重：负荷加权（v6 exp82；用户进阶建议⑤；输入仅 pred_load，合规#2）----
    "weight_load_gamma": 1.0,
    # ---- best_iter 选择：3 折 walk-forward（不接触官方验证集）----
    # 与 v6 同哲学：walk-forward 在漂移 val 上系统性过拟合（exp44），故用固定 best_it_fixed。
    # TCN 下 best_it_fixed 即"固定训练 epochs"（替代 LightGBM 的固定 num_boost_round=80）。
    "best_it_strategy": "3fold",
    "best_it_folds": [
        ("2025-02-28", "2025-03-01", "2025-05-31"),  # 春
        ("2025-08-31", "2025-09-01", "2025-11-30"),  # 秋
        ("2025-12-31", "2026-01-01", "2026-02-28"),  # 冬（含 2026-02，最接近验证集）
    ],
    "best_it_num_iterations": 200,   # TCN walk-forward 上限 epochs（best_it_fixed 启用时未用）
    "best_it_early_stopping": 20,   # walk-forward 早停耐心（best_it_fixed 启用时未用）
    # ---- 固定 epochs（与 v6 best_it_fixed=80 同源哲学：固定保守迭代，不在 val 早停）----
    "best_it_fixed": 60,            # TCN 固定训练 epochs
    # ---- 小时偏置校正粒度（v6 exp75；模型无关）----
    # hour_bias 由 3 折 OOF 残差逐 slot 估计（无泄露）。96=逐 15min slot。
    # exp75: 24->1461.63, 96->1459.06（-2.57 MW）。模型按 len 自适应索引。
    "hour_bias_slots": 96,
    # ---- 漂移方向校正（v6 exp47-49；模型无关）----
    # pl_weather_residual 与误差方向全天相关 +0.29，但仅在午间(11-14)校正才稳定迁移到验证集。
    # β 由 3 折 OOF 残差逐小时估计(无泄露)，存入 bundle，预测时叠加（符号 +=，勿改，见 model.py）。
    "drift_corr": {
        "feature": "pl_weather_residual",
        "hours": [11, 12, 13, 14],
    },
    # ---- 阈值场景校正（v6 exp58-61, exp72-73；模型无关）----
    # 物理诊断发现 pl_wr 未捕获的系统性偏置，逐场景 OOF 估计 shift（3 折 OOF 残差，无泄露）。
    # 预测时对该场景点 pred -= shift。每项 op: ">"(默认)/"<"/"<="/">=" 或 "range"(thr=[lo,hi))。
    "threshold_corr": [
        {"feature": "clearness", "op": ">", "thr": 0.8, "hours": [11, 12, 13, 14], "shrinkage": 0.7},
        {"feature": "precip", "op": ">", "thr": 0.0, "hours": None, "shrinkage": 1.0},
        {"feature": "temp", "op": "<", "thr": 8.0, "hours": None, "shrinkage": 1.0},
        {"feature": "clearness", "op": "range", "thr": [0.2, 0.5], "hours": [11, 12, 13, 14], "shrinkage": 1.0},
    ],
    # ---- 收缩 λ（ens -> anchor 的收缩；v6 exp43 λ=1.0 最优）----
    "shrinkage": 1.0,
    # ---- 集成聚合方式（v6 exp78/exp79；median 最优）----
    "aggregation": "median",
    "trim_frac": 0.2,
    # ---- 两级系统 Stage1 MOS（v6 exp80；用户进阶建议⑥；模型无关）----
    # Ridge(actual ~ pred_load + 气象 + 日历 + pl_weather_residual)：target=actual（仅作目标，合规#1）。
    # corrected_pred 作为残差成员的"锚"。cols=None 用 MosModel.DEFAULT_COLS(15 列)。
    "mos": {"cols": None, "alpha": 1.0},
    # ---- 训练数据起点（弃用漂移较大的 2023 数据，保留 2024-01 起）----
    "train_start": "2024-01-01 00:00:00",
    # ---- 预测值保留小数位 ----
    "round_decimals": 2,
}
