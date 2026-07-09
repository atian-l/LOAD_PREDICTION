# -*- coding: utf-8 -*-
"""exp66: 日级偏置结构分析 + 温度响应。

exp64 顶偏置日（2000-5500 MW 日偏置）是重尾主因。分析：
  1. 正确扫描 temp（X['temp']，非 光伏_温度）的偏置结构。
  2. 日级残差 vs 日级天气（temp/irrad/clearness/precip/wind/pred_load）相关性。
  3. 顶偏置日的天气特征 —— 是否可由预报天气识别。
若日级偏置与预报温度强相关（如冷天欠预测），可加日级温度响应校正。仅诊断。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from load_pred import config as C, features as F, train as T


def main():
    times, X, pred_load, actual = T.build_dataset()
    usable = T.usable_mask(times, pred_load, actual)
    mm = F.MismatchModel().fit(X, usable); X = mm.transform(X)
    val = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END)) & actual.notna()).values
    cfg = dict(C.TRAIN_CONFIG); cfg["best_it_fixed"] = 80
    model = T.train_ensemble(times, X, pred_load, actual, usable, cfg, 80)
    model.hour_bias, model.drift_corr, model.threshold_corr = T.compute_hour_bias(
        times, X, pred_load, actual, usable, cfg, 80)
    pred_v = model.predict_load(X[val], pred_load[val])
    act_v = actual[val].values
    resid_v = pred_v - act_v
    print(f"基线 MAE={np.abs(resid_v).mean():.2f}", flush=True)

    t_idx = pd.DatetimeIndex(times[val])
    temp_v = X["temp"].values[val]
    irrad_v = np.nan_to_num((X["irrad"].values if "irrad" in X.columns else X["光伏_辐照度"].values).astype(float)[val], nan=0.0)
    clear_v = np.nan_to_num(X["clearness"].values.astype(float)[val], nan=0.0)
    precip_v = np.nan_to_num(X["precip"].values.astype(float)[val], nan=0.0)
    pl_v = pred_load.values[val]

    # 1) temp 扫描（正确列）
    print("\n=== 按 temp 分桶 ===", flush=True)
    bins = [-99, 0, 5, 10, 15, 20, 25, 30, 99]
    bk = np.digitize(temp_v, bins)
    for b in range(1, len(bins)):
        m = bk == b
        if m.sum() < 30: continue
        print(f"  {bins[b-1]:>4}~{bins[b]:>4}C  n={m.sum():>5}  bias={resid_v[m].mean():>8.1f}  MAE={np.abs(resid_v[m]).mean():.1f}", flush=True)

    # 2) 日级聚合
    df = pd.DataFrame({
        "d": t_idx.date, "r": resid_v, "temp": temp_v, "irrad": irrad_v,
        "clear": clear_v, "precip": precip_v, "pl": pl_v, "hour": t_idx.hour,
    })
    day = df.groupby("d").agg(day_r=("r","mean"), day_abs=("r", lambda x: np.abs(x).mean()),
                              temp=("temp","mean"), temp_min=("temp","min"),
                              irrad=("irrad","mean"), clear=("clear","mean"),
                              precip=("precip","mean"), pl=("pl","mean"))
    print("\n=== 日级残差 vs 日级天气 相关 ===", flush=True)
    for col in ["temp","temp_min","irrad","clear","precip","pl"]:
        c = day["day_r"].corr(day[col])
        print(f"  corr(day_bias, {col:9s}) = {c:+.3f}", flush=True)

    # 3) 顶偏置日
    print("\n=== 顶偏置日（|day_r| 最大 15）===", flush=True)
    day["abs"] = day["day_r"].abs()
    top = day.sort_values("abs", ascending=False).head(15)
    print(top[["day_r","temp","temp_min","irrad","clear","precip","pl"]].round(1).to_string(), flush=True)

    # 4) 冷天/热天 日级偏置
    print("\n=== 日级 temp 分桶 ===", flush=True)
    for lo, hi in [(-99,5),(5,10),(10,15),(15,20),(20,25),(25,99)]:
        m = (day["temp"]>=lo) & (day["temp"]<hi)
        if m.sum()<3: continue
        print(f"  daytemp {lo:>3}~{hi:>3}C  n_days={m.sum():>3}  day_bias={day.loc[m,'day_r'].mean():>8.1f}  day_MAE={day.loc[m,'day_abs'].mean():.1f}", flush=True)

    # 5) 测试：日级 temp 响应校正（用 OOF 日级残差对 temp 回归）
    # 先拿 OOF 已校正残差
    oof = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0: continue
        fm = T.train_ensemble(times, X, pred_load, actual, ftr, cfg, 80)
        oof[fva] = fm.predict_load(X[fva], pred_load[fva])
    oof_mask = usable & oof.notna().values
    h_all = pd.DatetimeIndex(times).hour.values.astype(int)
    oof_cor = oof.values - model.hour_bias[h_all]
    for fn, beta in model.drift_corr:
        oof_cor = oof_cor + np.asarray(beta, dtype=float)[h_all] * X[fn].values.astype(float)
    for fn, thr, hrs, sh in model.threshold_corr:
        fv = X[fn].values.astype(float); sel = fv > thr
        if hrs is not None: sel = sel & np.isin(h_all, hrs)
        oof_cor[sel] = oof_cor[sel] - sh
    oof_resid = oof_cor - actual.values
    # OOF 日级
    od = pd.DataFrame({"d": pd.DatetimeIndex(times).date, "r": oof_resid, "temp": X["temp"].values}, index=times)
    od = od[oof_mask]
    oday = od.groupby("d").agg(r=("r","mean"), temp=("temp","mean"))
    # 日级 temp 回归（OOF）
    x = oday["temp"].values; y = oday["r"].values
    A = np.vstack([np.ones_like(x), x, x**2]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    print(f"\n=== OOF 日级残差 ~ temp + temp² 回归 ===", flush=True)
    print(f"  intercept={coef[0]:.1f}  temp={coef[1]:.2f}  temp²={coef[2]:.4f}", flush=True)
    # 应用：日级 temp 校正到 val（用预报日级 temp）
    vday = pd.DataFrame({"d": t_idx.date, "temp": temp_v})
    vday_mean = vday.groupby("d")["temp"].mean()
    day_temp = vday_mean.reindex(t_idx.date).values
    corr = coef[0] + coef[1]*day_temp + coef[2]*day_temp**2
    # 减去均值（避免整体偏移被 hour_bias 重复）
    corr = corr - corr.mean()
    p = pred_v - corr
    print(f"  应用日级temp校正(去均值) → MAE={np.abs(p-act_v).mean():.2f}  Δ={np.abs(p-act_v).mean()-np.abs(resid_v).mean():+.2f}", flush=True)
    # 不去均值
    p2 = pred_v - (coef[0] + coef[1]*day_temp + coef[2]*day_temp**2)
    print(f"  应用日级temp校正(不去均值) → MAE={np.abs(p2-act_v).mean():.2f}  Δ={np.abs(p2-act_v).mean()-np.abs(resid_v).mean():+.2f}", flush=True)


if __name__ == "__main__":
    main()
