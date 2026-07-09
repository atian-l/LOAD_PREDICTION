# -*- coding: utf-8 -*-
"""exp74 — 评估用户建议的点级天气/日历特征族（无泄露、仅过去）。

测试 5 族特征（在 v4=1461.63 基线上增量）：
  A 天气导数: temp_diff_2h/6h, temp_anom_96/672, temp_slope_6h
  B 历史窗口: temp_mean/min/max/std_12h, temp_mean_48h, irrad_mean_12h
  C 连续编码: cold_spell( temp<8)/hot_spell(>30)/rainy_spell(precip>0)/cloud_spell(clearness<0.5)
  D 细化节假日: post_holiday_first_workday, days_into_holiday, holiday_length
  E 96维 Quarter Bias（模型改动，单独测）
全部仅用 T 及之前的预报天气/日历，无实际负荷、无未来信息。仅诊断，不写产物。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from load_pred import config as C, features as F, train as T


def _run_length(cond_series: pd.Series) -> pd.Series:
    """连续 True 的长度（向后看，仅含 T 及之前；无泄露）。False 处为 0。"""
    cond = cond_series.astype(int)
    g = (cond.diff() != 0).cumsum()
    return cond.groupby(g).cumsum()


def add_weather_derivatives(X: pd.DataFrame) -> pd.DataFrame:
    temp = X["temp"].astype(float)
    X = X.copy()
    X["temp_diff_2h"] = (temp - temp.shift(8)).values       # 2h = 8 步
    X["temp_diff_6h"] = (temp - temp.shift(24)).values
    X["temp_anom_96"] = (temp - temp.rolling(96, min_periods=24).mean()).values
    X["temp_anom_672"] = (temp - temp.rolling(672, min_periods=96).mean()).values
    # 过去 6h(24步) 线性斜率（最小二乘，仅用过去窗口）
    y = temp.rolling(24, min_periods=12)
    # 用简单端点差分近似斜率（避免逐窗 lstsq 开销）：(mean_recent - mean_old)
    recent = temp.rolling(12, min_periods=6).mean()
    older = temp.shift(12).rolling(12, min_periods=6).mean()
    X["temp_slope_6h"] = (recent - older).values
    return X


def add_history_window(X: pd.DataFrame) -> pd.DataFrame:
    temp = X["temp"].astype(float)
    irrad = X["irrad"].astype(float)
    X = X.copy()
    r12 = temp.rolling(48, min_periods=16)   # 12h
    X["temp_mean_12h"] = r12.mean().values
    X["temp_min_12h"] = r12.min().values
    X["temp_max_12h"] = r12.max().values
    X["temp_std_12h"] = r12.std().values
    X["temp_mean_48h"] = temp.rolling(192, min_periods=48).mean().values  # 48h
    X["irrad_mean_12h"] = irrad.rolling(48, min_periods=16).mean().values
    X["irrad_sum_12h"] = irrad.rolling(48, min_periods=16).sum().values
    return X


def add_consecutive_spell(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["cold_spell"] = _run_length(X["temp"].astype(float) < 8.0).values
    X["hot_spell"] = _run_length(X["temp"].astype(float) > 30.0).values
    X["rainy_spell"] = _run_length(X["precip"].astype(float) > 0.0).values
    X["cloud_spell"] = _run_length(X["clearness"].astype(float) < 0.5).values
    # 截断长尾（>48h 的极端持续用 48 封顶，防离群）
    for c in ["cold_spell", "hot_spell", "rainy_spell", "cloud_spell"]:
        X[c] = X[c].clip(upper=48.0)
    return X


def add_refined_holiday(X: pd.DataFrame) -> pd.DataFrame:
    """节后第一工作日 / 当日假期序号 / 当次假期总长。确定性日历，无泄露。"""
    t = pd.DatetimeIndex(X.index)
    dates = t.normalize()
    is_hol = pd.Series(F._holiday_flag(t), index=t)
    # 假期段：连续 is_hol==1 的日期归为一段
    g = (is_hol.values != np.roll(is_hol.values, 1)).cumsum()
    df = pd.DataFrame({"hol": is_hol.values, "g": g}, index=pd.DatetimeIndex(dates))
    # 每段总长
    seg_len = df.groupby("g")["hol"].transform("size")
    seg_len = seg_len.where(df["hol"] == 1, 0)
    # 段内序号（第几天，0 基；非假期为 0）
    seg_idx = df.groupby("g").cumcount()
    days_into = seg_idx.where(df["hol"] == 1, 0)
    # 节后第一工作日：今日非假期 且 昨日为假期
    prev_hol = is_hol.shift(1).fillna(0).astype(int)
    post_holiday = ((is_hol.values == 0) & (prev_hol.values == 1)).astype(int)
    X = X.copy()
    X["days_into_holiday"] = days_into.values
    X["holiday_length"] = seg_len.values
    X["post_holiday"] = post_holiday
    return X


def evaluate(tag, augment_fn, times, X_base, pred_load, actual, usable, cfg, base_mae):
    X = augment_fn(X_base.copy())
    model = T.train_ensemble(times, X, pred_load, actual, usable, cfg, 80)
    model.hour_bias, model.drift_corr, model.threshold_corr = T.compute_hour_bias(
        times, X, pred_load, actual, usable, cfg, 80)
    val = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END)) & actual.notna()).values
    pred_v = model.predict_load(X[val], pred_load[val])
    mae = np.abs(pred_v - actual[val].values).mean()
    print(f"  [{tag}] val MAE={mae:.2f}  Δ={mae-base_mae:+.2f}  (新增列 {X.shape[1]-X_base.shape[1]})", flush=True)
    return mae, X


def main():
    times, X0, pred_load, actual = T.build_dataset()
    usable = T.usable_mask(times, pred_load, actual)
    mm = F.MismatchModel().fit(X0, usable); X_base = mm.transform(X0)
    cfg = dict(C.TRAIN_CONFIG); cfg["best_it_fixed"] = 80
    # 基线（应≈1461.63）
    val = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END)) & actual.notna()).values
    model = T.train_ensemble(times, X_base, pred_load, actual, usable, cfg, 80)
    model.hour_bias, model.drift_corr, model.threshold_corr = T.compute_hour_bias(
        times, X_base, pred_load, actual, usable, cfg, 80)
    base_mae = np.abs(model.predict_load(X_base[val], pred_load[val]) - actual[val].values).mean()
    print(f"基线(v4)={base_mae:.2f}  特征数={X_base.shape[1]}\n", flush=True)

    print("=== A 天气导数 ===", flush=True)
    evaluate("A 天气导数", add_weather_derivatives, times, X_base, pred_load, actual, usable, cfg, base_mae)
    print("=== B 历史窗口 ===", flush=True)
    evaluate("B 历史窗口", add_history_window, times, X_base, pred_load, actual, usable, cfg, base_mae)
    print("=== C 连续编码 ===", flush=True)
    evaluate("C 连续编码", add_consecutive_spell, times, X_base, pred_load, actual, usable, cfg, base_mae)
    print("=== D 细化节假日 ===", flush=True)
    evaluate("D 细化节假日", add_refined_holiday, times, X_base, pred_load, actual, usable, cfg, base_mae)

    # 组合：逐族叠加优胜者
    print("\n=== E 组合(全加) ===", flush=True)
    def augment_all(X):
        X = add_weather_derivatives(X)
        X = add_history_window(X)
        X = add_consecutive_spell(X)
        X = add_refined_holiday(X)
        return X
    evaluate("E 全组合", augment_all, times, X_base, pred_load, actual, usable, cfg, base_mae)


if __name__ == "__main__":
    main()
