# -*- coding: utf-8 -*-
"""exp70: 预报特征日级偏差模型（无泄露）。

exp69: oracle per-day MAE=1330（163 MW 日级信号）。past-actuals 持续性被 #1 禁止且 drift。
但 D 21:00 预测 D+1 时，D+1 全天预报天气/预测负荷可用 → 可构造日级预报特征。
用训练期 OOF 日级残差 ~ 日级预报特征 训练日级模型，val 应用。无泄露（仅预报特征）。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
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
    base = np.abs(pred_v - act_v).mean()
    print(f"基线(生产) = {base:.2f}", flush=True)

    # OOF 已校正残差
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

    # 日级特征（全天均值/最值）
    dt = pd.DatetimeIndex(times)
    df_all = pd.DataFrame({
        "d": dt.date, "h": dt.hour, "oofmask": oof_mask,
        "oof_r": oof_resid,
        "temp": np.nan_to_num(X["temp"].values.astype(float), nan=0.0),
        "irrad": np.nan_to_num((X["irrad"].values if "irrad" in X.columns else X["光伏_辐照度"].values).astype(float), nan=0.0),
        "clear": np.nan_to_num(X["clearness"].values.astype(float), nan=0.0),
        "precip": np.nan_to_num(X["precip"].values.astype(float), nan=0.0),
        "pl": np.nan_to_num(pred_load.values.astype(float), nan=0.0),
        "plwr": np.nan_to_num(X["pl_weather_residual"].values.astype(float), nan=0.0),
    })
    wind_col = "风电_风速" if "风电_风速" in X.columns else ("w_风电_风速" if "w_风电_风速" in X.columns else None)
    if wind_col: df_all["wind"] = np.nan_to_num(X[wind_col].values.astype(float), nan=0.0)

    # 训练日级（OOF 期内）
    oof_day = df_all[df_all["oofmask"]].groupby("d").agg(
        day_r=("oof_r","mean"), temp=("temp","mean"),
        irrad=("irrad","mean"), clear=("clear","mean"), precip=("precip","mean"),
        pl=("pl","mean"), plwr=("plwr","mean"))
    if "wind" in df_all: oof_day["wind"] = df_all[df_all["oofmask"]].groupby("d")["wind"].mean()
    oof_day["month"] = pd.to_datetime(oof_day.index).month
    print(f"训练日数: {len(oof_day)}", flush=True)
    print(f"OOF 日级残差 vs 日级特征相关:", flush=True)
    for c in ["temp","irrad","clear","precip","pl","plwr"]+(["wind"] if "wind" in oof_day else []):
        print(f"  corr(day_r, {c}) = {oof_day['day_r'].corr(oof_day[c]):+.3f}", flush=True)

    feat_cols = ["temp","irrad","clear","precip","pl","plwr"]+(["wind"] if "wind" in oof_day else [])
    # val 日级特征
    df_val = pd.DataFrame({"d": pd.DatetimeIndex(times[val]).date,
                           "temp": np.nan_to_num(X["temp"].values[val], nan=0.0),
                           "irrad": np.nan_to_num((X["irrad"].values if "irrad" in X.columns else X["光伏_辐照度"].values).astype(float)[val], nan=0.0),
                           "clear": np.nan_to_num(X["clearness"].values[val], nan=0.0),
                           "precip": np.nan_to_num(X["precip"].values[val], nan=0.0),
                           "pl": pred_load.values[val],
                           "plwr": np.nan_to_num(X["pl_weather_residual"].values[val], nan=0.0)})
    if "wind" in df_all: df_val["wind"] = np.nan_to_num(X[wind_col].values[val], nan=0.0)
    val_day = df_val.groupby("d")[feat_cols].mean()

    # Ridge 日级模型
    Xtr = oof_day[feat_cols].values; ytr = oof_day["day_r"].values
    Xva = val_day[feat_cols].reindex(pd.DatetimeIndex(times[val]).date).values
    print(f"\n=== Ridge 日级偏差模型 ===", flush=True)
    for alpha in [0.1, 1.0, 10.0, 50.0]:
        rg = Ridge(alpha=alpha).fit(Xtr, ytr)
        day_pred = rg.predict(Xva)
        # 去整体均值（避免与 hour_bias 重复整体偏移）
        day_pred_dm = day_pred - day_pred.mean()
        p = pred_v - day_pred_dm
        # 也试不去均值
        p2 = pred_v - day_pred
        # 按 OOF 估缩放
        print(f"  alpha={alpha}: 去均值 MAE={np.abs(p-act_v).mean():.2f} Δ={np.abs(p-act_v).mean()-base:+.2f} | 不去均值 MAE={np.abs(p2-act_v).mean():.2f} Δ={np.abs(p2-act_v).mean()-base:+.2f}", flush=True)

    # 缩放搜索（OOF shift 可能过估/低估，找最优缩放）
    print(f"\n=== 最优缩放搜索（val 调参，界定上界）===", flush=True)
    rg = Ridge(alpha=1.0).fit(Xtr, ytr)
    day_pred = rg.predict(Xva)
    best_s, best_m = 0, base
    for s in np.arange(0, 1.51, 0.1):
        p = pred_v - s*day_pred
        m = np.abs(p-act_v).mean()
        if m < best_m: best_m, best_s = m, s
    print(f"  最优缩放 s={best_s:.1f} → MAE={best_m:.2f} Δ={best_m-base:+.2f} (val调参上界)", flush=True)

    # 日级 + 贪心最佳阈值校正组合
    print(f"\n=== 日级模型 + temp/irrad 阈值校正组合 ===", flush=True)
    rg = Ridge(alpha=1.0).fit(Xtr, ytr)
    day_pred = rg.predict(Xva); day_pred_dm = day_pred - day_pred.mean()
    p = pred_v - day_pred_dm
    temp_all = X["temp"].values.astype(float); temp_v = temp_all[val]
    irrad_all = (X["irrad"].values if "irrad" in X.columns else X["光伏_辐照度"].values).astype(float); irrad_v = irrad_all[val]
    hours_v = pd.DatetimeIndex(times[val]).hour.values.astype(int)
    # temp<8 shift
    sh_cold = float(np.nanmean(oof_resid[oof_mask & (temp_all<8)]))
    p2 = p.copy(); m_v = temp_v<8; p2[m_v] = p2[m_v] - sh_cold
    print(f"  日级 + temp<8(shift={sh_cold:.0f}) → MAE={np.abs(p2-act_v).mean():.2f} Δ={np.abs(p2-act_v).mean()-base:+.2f}", flush=True)


if __name__ == "__main__":
    main()
