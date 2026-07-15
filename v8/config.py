# -*- coding: utf-8 -*-
"""v8 配置：段定义、天气相似度、trigger/α/w 网格、base B 配置、路径。

复用 load_pred.config（as LC）的时间边界/路径/特征列定义，v8 仅追加五层架构参数。
六条 leakage 不变量全部继承自 load_pred（v8 不改 features/data_loader/时间边界）。
"""
from __future__ import annotations
from pathlib import Path

from load_pred import config as LC

# --------------------------------------------------------------------------- #
# v8 路径（v8 自有模型/输出，不污染根 models/）
# --------------------------------------------------------------------------- #
V8_DIR = Path(__file__).resolve().parent
V8_MODELS_DIR = V8_DIR / "models"
V8_OUTPUT_DIR = V8_DIR / "output"
V8_BUNDLE = V8_MODELS_DIR / "v8_bundle.pkl"
V8_CORR_DIR = V8_MODELS_DIR / "correction_boosters"

# base A = v6，直接加载根 models/model_bundle.pkl（= 目前最优版本，不重训）
BASE_A_BUNDLE = LC.MODEL_BUNDLE


def ensure_dirs() -> None:
    V8_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    V8_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    V8_CORR_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 第一层：分段建模（按 hour of day）
# --------------------------------------------------------------------------- #
# night: 00-08, day: 08-18（含午间 11-14 高误差区）, evening: 18-24
SEGMENTS = ["night", "day", "evening"]
SEGMENT_HOURS = {
    "night": (0, 8),     # [0, 8)
    "day": (8, 18),      # [8, 18)
    "evening": (18, 24), # [18, 24)
}


def segment_of_hour(hour: int) -> str:
    for name, (lo, hi) in SEGMENT_HOURS.items():
        if lo <= hour < hi:
            return name
    return "evening"


# --------------------------------------------------------------------------- #
# 统一天气相似度（服务于 adaptive/trigger/α/w 四层）
# --------------------------------------------------------------------------- #
# 日级天气向量（跨年稳定的物理量）。temp_day_mean/irrad_day_sum/temp_day_range 来自
# features.day_level_features（日级列，每日常数）；clearness/precip 为时点列，按日聚合。
WEATHER_SIM_COLS = ["temp_day_mean", "irrad_day_sum", "clearness_day_mean",
                    "precip_day_sum", "temp_day_range"]
WEATHER_SIM_K = 40            # KNN 邻居数
WEATHER_SIM_TAU_GRID = [0.5, 1.0, 2.0, 4.0]  # softmax 温度，OOF 选（非 val）


# --------------------------------------------------------------------------- #
# 第三层：Correction Trigger
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 第三层：Correction Trigger
# --------------------------------------------------------------------------- #
# 跨年可信度门槛（固定，非 val 调）：要求 ≥80% 天气相似历史日从修正中获益才触发
# （overwhelming consensus）。诊断 diag_trigger 证实：残差目标 = 外部预测误差，
# 跨年 R² 为负（memory: info_source_assessment，2025春 +1971 -> 2026春 -572 符号翻转），
# 任何触发比例都恶化 val（tf=0.7 仅 3.9% 触发仍 +3.28MW）。故用 80% 共识门槛保守不激活：
# OOF 显示无 (date,seg) 达标 -> 修正层当前不触发；待新能源出力等跨年可学信号引入后自动启用。
# 此为基于残差跨年不可学这一既存性质的工程设计决策，非验证集调参。
TRIG_MIN_FRAC = 0.80
MIN_GAIN_GRID = [0.0, 30.0, 60.0, 120.0, 240.0]   # 加权平均改善阈值（MW），minimax 选

# --------------------------------------------------------------------------- #
# 第四层：Shrink α（动态，非固定）
# --------------------------------------------------------------------------- #
ALPHA_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]  # 局部 KNN grid search

# --------------------------------------------------------------------------- #
# 第二层：Adaptive Model Selection（base A vs base B）
# --------------------------------------------------------------------------- #
# 天气型分桶：clearness_day_mean × temp_day_mean 9 宫格 + precip>0 雨型
CLEARNESS_BINS = [0.3, 0.7]      # <0.3 / [0.3,0.7) / >=0.7
TEMP_BINS = [8.0, 22.0]          # <8 / [8,22) / >=22
ADAPTIVE_MIN_MARGIN = 0.01       # B 优 A 需超 1% 才切换
ADAPTIVE_MIN_N = 200             # 桶最小样本数


# --------------------------------------------------------------------------- #
# Base B 配置（reg_only LightGBM，diversity 备选）
# --------------------------------------------------------------------------- #
# 去 quantile 成员，仅 regression × {direct,residual} × 5 seeds = 10 成员；其余同 v6。
BASE_B_CFG_OVERRIDE = {
    "objectives": ["regression"],
    "quantile_alphas": [],
}


# --------------------------------------------------------------------------- #
# Correction model（残差 LightGBM，3 段各一）
# --------------------------------------------------------------------------- #
CORR_CFG = {
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_data_in_leaf": 300,   # 残差量级小，加强正则防过拟合
    "lambda_l2": 8.0,
    "feature_fraction": 0.80,
    "bagging_fraction": 0.80,
    "bagging_freq": 1,
    "best_it_fixed": 80,
    "seeds": [42, 7, 123],     # 残差模型用 3 种子中位（稳）
}
