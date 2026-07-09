# -*- coding: utf-8 -*-
"""
数据加载与清洗。

核心合规点（Inviolable Constraints）：
  #1 数据隔离：真实负荷（实际直调负荷）只作为评估基准，本模块仅把它原样读出，
     供 train.py 拼接评估用；features.py 绝不读取该列。
  #4 气象去重：对相同“预测时间”仅保留“起报时间”最晚的一条记录。
  #2/#3 外部预测（预测直调负荷）可作为输入/滞后特征。
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from . import config as C


# --------------------------------------------------------------------------- #
# 负荷数据
# --------------------------------------------------------------------------- #
def load_load_data() -> pd.DataFrame:
    """
    读取 direct_load_latest.csv。

    返回 DataFrame，列：时间(datetime), 预测负荷(float64), 实际负荷(float64)
    时间为 15 分钟等间隔（2023-01-01 ~ 2026-07-07）。

    注意：实际负荷仅用于评估，不参与任何特征工程。
    """
    df = pd.read_csv(C.LOAD_CSV, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df[C.COL_TIME] = pd.to_datetime(df[C.COL_TIME])
    df[C.COL_PRED_LOAD] = pd.to_numeric(df[C.COL_PRED_LOAD], errors="coerce")
    df[C.COL_ACTUAL_LOAD] = pd.to_numeric(df[C.COL_ACTUAL_LOAD], errors="coerce")
    df = df.sort_values(C.COL_TIME).reset_index(drop=True)
    return df


def load_actual_load_strings() -> pd.DataFrame:
    """
    以原始字符串读取真实负荷（用于输出文件“完全一致”比对）。

    返回 DataFrame：时间(datetime), actual_str(原始字符串或 NA)
    """
    df = pd.read_csv(
        C.LOAD_CSV,
        encoding="utf-8-sig",
        dtype={C.COL_ACTUAL_LOAD: str, C.COL_PRED_LOAD: str},
    )
    df.columns = [c.strip() for c in df.columns]
    df[C.COL_TIME] = pd.to_datetime(df[C.COL_TIME])
    s = df[C.COL_ACTUAL_LOAD].astype("string").str.strip()
    # 空字符串/纯空白 -> NA
    s = s.where(s.str.len() > 0, pd.NA)
    out = pd.DataFrame({C.COL_TIME: df[C.COL_TIME], "actual_str": s})
    return out


# --------------------------------------------------------------------------- #
# 气象数据
# --------------------------------------------------------------------------- #
def load_weather_dedup(run_time: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    读取气象数据并执行“相同预测时间保留最晚起报时间”的去重（Constraint #4）。

    参数
    ----
    run_time : 若给定（预测模式），仅保留 起报时间 <= run_time 的记录后再去重，
               模拟运行时仅能获得已起报版本（Constraint #5）。
               若为 None（训练模式），使用全部历史起报版本。

    返回
    ----
    DataFrame，索引=预测时间(datetime)，列=25 个气象特征。
    同一预测时间仅保留最晚起报版本的一行。
    """
    w = pd.read_csv(C.WEATHER_CSV, encoding="utf-8-sig")
    w.columns = [c.strip() for c in w.columns]
    w[C.WCOL_ISSUE] = pd.to_datetime(w[C.WCOL_ISSUE])
    w[C.WCOL_FORECAST] = pd.to_datetime(w[C.WCOL_FORECAST])

    # 运行时可获得性：仅保留起报时间 <= run_time
    if run_time is not None:
        w = w[w[C.WCOL_ISSUE] <= run_time].copy()

    # 关键去重：同一预测时间，保留起报时间最晚的一条
    w = w.sort_values(C.WCOL_ISSUE)
    w = w.drop_duplicates(subset=[C.WCOL_FORECAST], keep="last")

    w = w.set_index(C.WCOL_FORECAST)
    # 仅保留 25 个气象特征列
    w = w[C.WEATHER_FEATURE_COLS].sort_index()
    return w


# --------------------------------------------------------------------------- #
# 时间轴
# --------------------------------------------------------------------------- #
def full_time_index() -> pd.DatetimeIndex:
    """full_predictions / full_mae 覆盖的完整 15 分钟时间轴。"""
    return pd.date_range(C.FULL_START, C.FULL_END, freq=C.FREQ)


def pred_load_series() -> pd.Series:
    """
    返回以时间为索引的“预测直调负荷”序列（外部预测，全历史）。

    用于构造滞后/滚动特征；该序列不含任何真实负荷信息。
    """
    df = load_load_data()
    s = df.set_index(C.COL_TIME)[C.COL_PRED_LOAD]
    # 重采样到标准 15 分钟网格（数据本身即 15 分钟），保留原值
    s = s.reindex(full_time_index().union(s.index)).sort_index()
    return s
