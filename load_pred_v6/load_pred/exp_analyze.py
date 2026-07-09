# -*- coding: utf-8 -*-
"""误差归因分析（无训练）：按小时/午间/阴雨/多云分解外部预测(pred_load)误差。
用于指导 CatBoost 建模与午间/阴雨天场景特征设计。"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C
from . import data_loader as dl
from . import features as F


def main():
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)
    X = F.build_features(times, pred_load, weather)

    vm = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END)) & actual.notna()).values
    t = times[vm]
    pl = pred_load.values[vm]
    ac = actual.values[vm]
    err = pl - ac  # 正=高估

    dt = pd.DatetimeIndex(t)
    hour = dt.hour.values
    precip = X["precip"].values[vm]
    irrad = X["irrad"].values[vm]

    def blk(m, name):
        e = err[m]; n = m.sum()
        if n == 0:
            print(f"  {name:28s}: n=0"); return
        print(f"  {name:28s}: n={n:5d}  MAE={np.mean(np.abs(e)):.1f}  bias={np.mean(e):+.1f}  rmse={np.sqrt(np.mean(e**2)):.1f}")

    print("=== 验证集整体（外部预测 pred_load）===")
    blk(np.ones(len(err), bool), "all")

    print("\n=== 按小时（MAE 最高的时段=难点）===")
    rows = []
    for h in range(24):
        m = hour == h
        if m.sum() == 0: continue
        rows.append((h, m.sum(), np.mean(np.abs(err[m])), np.mean(err[m])))
    dfh = pd.DataFrame(rows, columns=["h", "n", "mae", "bias"]).sort_values("mae", ascending=False)
    print(dfh.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    print("\n=== 午间 vs 非午间（11-13 时）===")
    midday = (hour >= 11) & (hour <= 13)
    blk(midday, "午间 11-13")
    blk(~midday, "非午间")

    print("\n=== 阴雨（precip_p50 > 0.1）===")
    rain = precip > 0.1
    blk(rain, "阴雨")
    blk(~rain, "非阴雨")

    print("\n=== 白天多云（hour 8-16 且 irrad 低于该小时中位数）===")
    day = (hour >= 8) & (hour <= 16)
    cloudy = np.zeros(len(err), bool)
    for h in range(8, 17):
        mh = (hour == h) & day
        if mh.sum() == 0: continue
        med = np.median(irrad[mh])
        cloudy = cloudy | (mh & (irrad < med * 0.5))
    blk(cloudy, "白天多云")
    blk(day & ~cloudy, "白天晴")
    blk(~day, "夜间")

    print("\n=== 午间 × 阴雨 交叉 ===")
    blk(midday & rain, "午间&阴雨")
    blk(midday & ~rain, "午间&非雨")
    blk(~midday & rain, "非午间&阴雨")

    print("\n=== 午间 × 多云 交叉 ===")
    blk(midday & cloudy, "午间&多云")
    blk(midday & ~cloudy, "午间&晴")

    print("\n=== 按月 ===")
    mo = dt.month.values
    for mm in sorted(set(mo)):
        m = mo == mm
        blk(m, f"月 {mm}")


if __name__ == "__main__":
    main()
