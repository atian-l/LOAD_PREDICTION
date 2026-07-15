# -*- coding: utf-8 -*-
"""
全局配置：路径、时间边界、特征/模型超参数（TFT 变体）。

本包是 load_pred（LightGBM 集成）的 TFT 移植版：除"模型方法 LightGBM -> TFT 序列建模"外，
数据/特征/集成结构/OOF 校正/泄露不变量全部与 load_pred 逐行一致。

工程目录约定（满足 Inviolable Constraints #6；为不修改 load_pred_tft 以外的文件）：
  <project_root>/load_pred_tft   <- 代码 + 本包私有 models/ output/
  <project_root>/data            <- 输入数据（共享，只读；不写入）
  <project_root>/load_pred_tft/models   <- 本包训练好的模型（写在本包内）
  <project_root>/load_pred_tft/output   <- 本包输出结果（写在本包内）
注：模型/输出写在本包内，避免覆盖 load_pred 的 models/output；数据仍读共享 data/（只读）。
"""
from __future__ import annotations
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
PKG_DIR = Path(__file__).resolve().parent              # load_pred_tft/
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # <project_root>/（读共享数据）

DATA_DIR = PROJECT_ROOT / "data"          # 共享输入数据（只读）
CODE_DIR = PKG_DIR                         # 本包代码
MODELS_DIR = PKG_DIR / "models"            # 本包私有模型目录
OUTPUT_DIR = PKG_DIR / "output"            # 本包私有输出目录

LOAD_CSV = DATA_DIR / "direct_load_latest.csv"
WEATHER_CSV = DATA_DIR / "shandong_weather_15min.csv"

# 输出文件（与 load_pred 同名同格式，写在本包 output/ 内）
FULL_PRED_CSV = OUTPUT_DIR / "full_predictions.csv"
FULL_MAE_CSV = OUTPUT_DIR / "full_mae.csv"
EVAL_TXT = OUTPUT_DIR / "evaluation_metrics.txt"
LATEST_PRED_CSV = OUTPUT_DIR / "latest_prediction.csv"

# 模型文件
MODEL_BUNDLE = MODELS_DIR / "model_bundle.pkl"

# --------------------------------------------------------------------------- #
# 时间边界（Inviolable Constraints #5）
# --------------------------------------------------------------------------- #
VAL_START = "2026-03-01 00:00:00"
VAL_END = "2026-06-15 23:45:00"  # 含
TRAIN_END = "2026-02-28 23:45:00"  # 训练数据上界（含）

FULL_START = "2023-02-01 00:00:00"
FULL_END = "2026-06-15 23:45:00"

FREQ = "15min"
POINTS_PER_DAY = 96

# --------------------------------------------------------------------------- #
# 数据列名（direct_load_latest.csv，UTF-8-BOM）
# --------------------------------------------------------------------------- #
COL_TIME = "时间"
COL_PRED_LOAD = "预测负荷"     # 外部预测负荷（允许作为输入/滞后特征）
COL_ACTUAL_LOAD = "实际负荷"   # 真实负荷（仅作评估基准，严禁入模）

# --------------------------------------------------------------------------- #
# 气象列名（shandong_weather_15min.csv）
# --------------------------------------------------------------------------- #
WCOL_ISSUE = "起报时间"
WCOL_FORECAST = "预测时间"

WEATHER_BASE_VARS = ["风电_风速", "光伏_温度", "光伏_降水", "光伏_风速", "光伏_辐照度"]
WEATHER_FEATURE_COLS = []
for _v in WEATHER_BASE_VARS:
    WEATHER_FEATURE_COLS.append(_v)
    for _s in ["_p25", "_p50", "_p75", "_std"]:
        WEATHER_FEATURE_COLS.append(_v + _s)

# --------------------------------------------------------------------------- #
# 滞后特征（Inviolable Constraints #2/#3）
#  - 只能基于"预测直调负荷"构造，严禁使用真实负荷
#  - 最短滞后必须包含 lag_192（=2 天）
#  - TFT encoder_len=288(3天) 已覆盖 lag_192/288；lag_672(7天) 仍作为逐时刻特征保留
# --------------------------------------------------------------------------- #
PRED_LAGS = [96, 192, 288, 672]
PRED_ROLLING_WINDOWS = [96, 672]

# --------------------------------------------------------------------------- #
# 预测模式：运行时刻假设（部署 = 每日 09:00 前后运行）-- 与 load_pred 一致
# --------------------------------------------------------------------------- #
DEFAULT_RUN_HOUR = 9
DEFAULT_RUN_MINUTE = 0


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# TFT 静态协变量（对整个预测日 D 恒定的日历特征；TFT static metadata）
#  - 取预测日 D 内任意时刻的值（日级恒定），用于 static covariate encoder
#  - hour/min 是时变（随 decoder 96 步变化），不在此列，作为时变输入
# --------------------------------------------------------------------------- #
STATIC_COLS = ["is_holiday", "is_weekend",
               "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos"]


# --------------------------------------------------------------------------- #
# 训练超参数（TFT 变体）
# --------------------------------------------------------------------------- #
TRAIN_CONFIG = {
    # ---- TFT 架构参数（替代 LightGBM/TCN 的树/卷积参数）----
    # 序列建模：encoder 吃过去 H_enc 步 pred_load+weather+calendar（observed past，不含 actual，
    #   合规#1/#2），decoder 吃未来 96 步 weather+calendar+pred_load（known future，日前可得，合规#4）。
    #   static 用日级日历（STATIC_COLS）。multi-horizon 一次输出 96 点（用户建议 Method A 思想的序列版）。
    "encoder_len": 288,                 # encoder 历史长度（15min 步，=3 天；> lag_192=2 天，满足#2/#3）
    "decoder_len": 96,                  # decoder/输出长度（=D+1 全天 96 点，multi-horizon）
    "hidden_size": 64,                  # TFT 隐维（控制显存；云端 GPU 可上调）
    "num_heads": 2,                     # interpretable multi-head attention 头数
    "num_lstm_layers": 1,               # encoder-decoder LSTM 层数
    "dropout": 0.1,                     # VSN/GRN/attention dropout
    "lr": 1e-3,                         # Adam 学习率
    "weight_decay": 1e-4,               # L2 正则
    "lr_schedule": "cosine",            # LR 调度："cosine" 退火 / "none" 恒定
    "lr_eta_min": 1e-5,                 # cosine 退火终点
    "batch_size": 16,                   # 每批预测日样本数（序列模型显存大，小 batch）
    "grad_clip": 5.0,                   # 梯度裁剪
    "device": "auto",                   # "auto"=cuda if available else cpu；Phase0 建议 GPU
    "num_workers": 0,                   # DataLoader 工作进程（云端可上调）

    # ---- 集成配置（与 v6 LightGBM 完全一致：目标 × 残差/直接 × 种子 = 40 成员）----
    # 多样化来源不变：{regression, quantile(0.45/0.5/0.55)} × {direct, residual} × 5 seeds。
    # quantile 成员用 pinball 损失，regression 用 MSE。残差目标 = actual - MOS_anchor（保留 MOS 收益）。
    "objectives": ["regression", "quantile"],
    "quantile_alphas": [0.45, 0.5, 0.55],
    "residual_modes": [False, True],
    "seeds": [42, 7, 123, 2024, 99],
    # ---- 时间样本权重（近期加权，缓解概念漂移；与 v6 一致）----
    "alpha_w": 5.0,
    # ---- 联合样本权重：负荷加权（v6 exp82；输入仅 pred_load，合规#2）----
    "weight_load_gamma": 1.0,
    # ---- best_iter 选择：3 折 walk-forward（不接触官方验证集）----
    # 与 v6 同哲学：walk-forward 在漂移 val 上系统性过拟合（exp44），故用固定 best_it_fixed。
    # TFT 下 best_it_fixed 即"固定训练 epochs"。
    "best_it_strategy": "3fold",
    "best_it_folds": [
        ("2025-02-28", "2025-03-01", "2025-05-31"),  # 春
        ("2025-08-31", "2025-09-01", "2025-11-30"),  # 秋
        ("2025-12-31", "2026-01-01", "2026-02-28"),  # 冬（含 2026-02，最接近验证集）
    ],
    "best_it_num_iterations": 200,
    "best_it_early_stopping": 20,
    # ---- 固定 epochs（Phase 0 保守值；与 v6 best_it_fixed=80 同源哲学：固定迭代，不在 val 早停）----
    "best_it_fixed": 30,
    # ---- 小时偏置校正粒度（v6 exp75；模型无关，TFT 预测 96 点后逐时刻复用）----
    "hour_bias_slots": 96,
    # ---- 漂移方向校正（v6 exp47-49；模型无关）----
    "drift_corr": {
        "feature": "pl_weather_residual",
        "hours": [11, 12, 13, 14],
    },
    # ---- 阈值场景校正（v6 exp58-61, exp72-73；模型无关）----
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
    # ---- 两级系统 Stage1 MOS（v6 exp80；模型无关）----
    "mos": {"cols": None, "alpha": 1.0},
    # ---- 训练数据起点（弃用漂移较大的 2023 数据，保留 2024-01 起）----
    "train_start": "2024-01-01 00:00:00",
    # ---- 预测值保留小数位 ----
    "round_decimals": 2,
}
