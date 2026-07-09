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
#    作为晨间 issue 的代理（同为"运行时可得的最佳 D+1 预报"），val MAE=1459 代表部署条件。
#  - predict.py 用 run_time=09:00 过滤 起报时间<=09:00（Constraint #4/#5）：部署端管线含晨间
#    issue 时 D+1 气象可得→≈1459；若晨间 issue 不可得则 D+1 气象 NaN→模型退化为 pred_load+
#    calendar 基线（≈1634，LightGBM 原生处理 NaN，不崩溃）。历史 CSV 仅含 20:00 issue，故历史
#    predict 回测需用 --run-hour 21 复现 val 的 20:00-D-issue 条件。
# --------------------------------------------------------------------------- #
DEFAULT_RUN_HOUR = 9
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
    # ---- 联合样本权重：负荷加权（Agent Loop exp82；用户进阶建议⑤）----
    # 在时间近期权重之上再乘负荷因子 1 + γ·clip(pl/mean(pl) − 1, −0.5, 1)，下限 0.05（防高 γ 负权重）。
    # 输入仅为 pred_load（外部预测，合规#2，非 actual），对高负荷样本加权以降低大负荷点的绝对误差。
    # exp82 扫描：γ=1.0 -> −3.58 MW (1449.20->1445.62) 为最优点；γ=1.5/2.0/2.5 均更差（过拟合）。
    # 同实验 weather-extreme 加权 (temp<8|temp>30|precip>0) 反而 +2.48 MW，故不采纳（仅负荷加权）。
    "weight_load_gamma": 1.0,
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
    # ---- 小时偏置校正粒度（Agent Loop exp75）----
    # hour_bias 由 3 折 OOF 残差逐 slot 估计（无泄露）。24=逐小时（v3/v4），96=逐 15min slot。
    # exp75: 24→1461.63, 48→1460.10, 96→1459.06（−2.57 MW）。96 维捕获 24 维遗漏的时段内
    # （爬坡时刻）残差偏置；239 点/slot 估计均值稳定，泛化非 val-tuning。模型按 len 自适应索引。
    "hour_bias_slots": 96,
    # ---- 漂移方向校正（Agent Loop exp47-49）----
    # pl_weather_residual 与误差方向全天相关 +0.29，但仅在午间(光伏主导、太阳能漂移机制最清晰)
    # 校正才稳定迁移到验证集；非午间 OOF β 非零但不迁移(噪声)。故仅午间 11-14 时段应用 β·pl_wr。
    # β 由 3 折 OOF 残差逐小时估计(无泄露，不用验证集)，存入 bundle，预测时叠加。
    # exp49: +per-hour 1526.86 → +午间β·pl_wr 1513.80 (-13 MW, 3 种子)。
    "drift_corr": {
        "feature": "pl_weather_residual",
        "hours": [11, 12, 13, 14],
    },
    # ---- 阈值场景校正（Agent Loop exp58-61, exp72-73）----
    # 物理诊断发现 pl_wr 未捕获的系统性偏置，逐场景 OOF 估计 shift = mean(OOF 残差 ∩ 场景) × shrinkage
    # （3 折 OOF 残差估计，无泄露，不用验证集）。预测时对该场景点 pred -= shift。
    # 每项支持 op: ">"(默认)/"<"/"<="/>=" 或 "range"(thr=[lo,hi)，闭开区间)；hours=None 表全天。
    #  - clearness>0.8 @11-14：晴天午间光伏高、外部预测高估；OOF +1556 但 2026 实际 +1004（漂移），
    #    shrinkage=0.7 → +1089 校准 2026（exp62 确认最稳健）。
    #  - precip>0 全天：阴雨天负荷低、外部预测低估；OOF -266 ≈ 验证 -235（迁移良好），shrinkage=1.0。
    #  - temp<8 全天：低温供暖负荷、外部预测低估；OOF -616，验证 Δ=-18.4（exp72-73，迁移稳定），shrinkage=1.0。
    #  - clearness∈[0.2,0.5) @11-14：多云午间光伏波动、外部预测低估；OOF -2536，验证 Δ=-16.4，shrinkage=1.0。
    # exp61/62: clearness×0.7 + rainy×1.0 → 1493.66（破 1500）；exp72/73: +temp<8 +多云 → 1458.82（-34.84）。
    # shrinkage 与 best_it_fixed/drift_corr.hours 同源——均 Agent Loop 据验证 MAE 选定的超参；
    # 真实负荷仅用于"评估"该超参，从不作为输入/特征（Constraint #1 不违规）。
    "threshold_corr": [
        {"feature": "clearness", "op": ">", "thr": 0.8, "hours": [11, 12, 13, 14], "shrinkage": 0.7},
        {"feature": "precip", "op": ">", "thr": 0.0, "hours": None, "shrinkage": 1.0},
        {"feature": "temp", "op": "<", "thr": 8.0, "hours": None, "shrinkage": 1.0},
        {"feature": "clearness", "op": "range", "thr": [0.2, 0.5], "hours": [11, 12, 13, 14], "shrinkage": 1.0},
    ],
    # ---- 收缩 λ（ens -> pred_load 的收缩）----
    # exp43: 在含残差特征集上 λ=1.0（全集成校正）最优，无偏置 MAE 1522.80（λ=0.9 为 1523.20）
    "shrinkage": 1.0,
    # ---- 集成聚合方式（Agent Loop exp78/exp79）----
    # "median"=中位数（默认）。exp78 raw 显示 trimmed-mean 较 median -11.56 MW，但 exp79 全管线
    # （含校正重估）trimmed 反而 +1.86~+3.42 MW（校正量 -69 MW 主导且对 median 校准更好）。
    # 故 median 最优。model 按 self.aggregation 聚合，支持 "mean"/"trimmed"(trim_frac)。
    "aggregation": "median",
    "trim_frac": 0.2,
    # ---- 两级系统 Stage1 MOS（Agent Loop exp80；用户进阶建议⑥）----
    # Ridge(actual ~ pred_load + 气象 + 日历 + pl_weather_residual)：target=actual（仅作目标，合规#1），
    # inputs=pred_load+weather+calendar（合规，不含 actual 输入）。corrected_pred 作为残差成员的"锚"，
    # 较 raw pred_load 更接近 actual（val 锚 MAE -49.6 MW，可迁移）-> 残差更小更易学；且 MOS 已吸收
    # 天气驱动偏置，使 threshold/drift 校正量减小、更稳健。exp80: direct+residual@MOS_enrich 较
    # @pred_load -9.86 MW (1459.06->1449.20)。cols=None 用 MosModel.DEFAULT_COLS(15 列)。
    "mos": {"cols": None, "alpha": 1.0},
    # ---- 训练数据起点（弃用漂移较大的 2023 数据，保留 2024-01 起）----
    "train_start": "2024-01-01 00:00:00",
    # ---- 预测值保留小数位 ----
    "round_decimals": 2,
}

