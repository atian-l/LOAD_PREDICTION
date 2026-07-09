# -*- coding: utf-8 -*-
"""exp73: 最终判定 —— GBM 日级模型(非线性)是否迁移 + 实现就绪清洁配置。

Ridge(线性)日级模型 s=0.0 不迁移。测 GBM(非线性)日级模型是否亦失败。
若 GBM 也 s=0.0，则日级信号确证不可由预报特征迁移 → 1300 无泄露不可达。
另确认实现就绪的清洁 threshold 配置(含 <= 与 band)的精确 MAE。仅诊断。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import lightgbm as lgb
from load_pred import config as C, features as F, train as T


def main():
    times, X, pred_load, actual = T.build_dataset()
    usable = T.usable_mask(times, pred_load, actual)
    mm = F.MismatchModel().fit(X, usable); X = mm.transform(X)
    val = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END)) & actual.notna()).values
    cfg = dict(C.TRAIN_CONFIG); cfg["best_it_fixed"]=80
    model = T.train_ensemble(times, X, pred_load, actual, usable, cfg, 80)
    model.hour_bias, model.drift_corr, model.threshold_corr = T.compute_hour_bias(times, X, pred_load, actual, usable, cfg, 80)
    pred_v = model.predict_load(X[val], pred_load[val]); a = actual[val].values
    base = np.abs(pred_v-a).mean()
    print(f"基线={base:.2f}", flush=True)

    oof = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te); fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum()==0: continue
        fm = T.train_ensemble(times, X, pred_load, actual, ftr, cfg, 80)
        oof[fva] = fm.predict_load(X[fva], pred_load[fva])
    oof_mask = usable & oof.notna().values
    h_all = pd.DatetimeIndex(times).hour.values.astype(int)
    oof_cor = oof.values - model.hour_bias[h_all]
    for fn, beta in model.drift_corr:
        oof_cor = oof_cor + np.asarray(beta,float)[h_all]*X[fn].values.astype(float)
    for fn, thr, hrs, sh in model.threshold_corr:
        fv = X[fn].values.astype(float); sel = fv > thr
        if hrs is not None: sel = sel & np.isin(h_all, hrs)
        oof_cor[sel] = oof_cor[sel] - sh
    oof_resid = oof_cor - actual.values

    # ---- GBM 日级模型 ----
    dt = pd.DatetimeIndex(times)
    df_all = pd.DataFrame({"d": dt.date, "oofmask": oof_mask, "oof_r": oof_resid,
        "temp": np.nan_to_num(X["temp"].values.astype(float), nan=0.0),
        "irrad": np.nan_to_num((X["irrad"].values if "irrad" in X.columns else X["光伏_辐照度"].values).astype(float), nan=0.0),
        "clearness": np.nan_to_num(X["clearness"].values.astype(float), nan=0.0),
        "precip": np.nan_to_num(X["precip"].values.astype(float), nan=0.0),
        "pl": np.nan_to_num(pred_load.values.astype(float), nan=0.0),
        "plwr": np.nan_to_num(X["pl_weather_residual"].values.astype(float), nan=0.0)})
    feat = ["temp","irrad","clearness","precip","pl","plwr"]
    oday = df_all[df_all["oofmask"]].groupby("d").agg(day_r=("oof_r","mean"), **{c:(c,"mean") for c in feat})
    oday["month"] = pd.to_datetime(oday.index).month
    feat2 = feat+["month"]
    COLMAP = {"temp":"temp","irrad":("irrad" if "irrad" in X.columns else "光伏_辐照度"),
              "clearness":"clearness","precip":"precip","plwr":"pl_weather_residual"}
    # val 日级特征(直接重新取)
    vday = pd.DataFrame({"d": pd.DatetimeIndex(times[val]).date})
    for c in feat:
        if c=="pl": arr = pred_load.values.astype(float)
        else: arr = X[COLMAP[c]].values.astype(float)
        vday[c] = np.nan_to_num(arr[val], nan=0.0)
    val_day = vday.groupby("d").mean()
    val_day["month"] = pd.to_datetime(val_day.index).month

    print("\n=== GBM 日级模型 ===", flush=True)
    Xtr = oday[feat2].values; ytr = oday["day_r"].values
    Xva_full = val_day[feat2].reindex(pd.DatetimeIndex(times[val]).date).values
    for nl, bi in [(31, 50), (15, 30), (63, 100)]:
        gb = lgb.train({"objective":"regression","metric":"mae","num_leaves":nl,"learning_rate":0.05,
                        "min_data_in_leaf":5,"lambda_l2":5.0,"verbose":-1,"force_col_wise":True},
                       lgb.Dataset(Xtr, label=ytr), num_boost_round=bi)
        dp = gb.predict(Xva_full)
        # 缩放搜索
        best_s, best_m = 0, base
        for s in np.arange(0, 1.51, 0.1):
            m = np.abs(pred_v - s*dp - a).mean()
            if m < best_m: best_m, best_s = m, s
        print(f"  nl={nl} bi={bi}: 最优缩放 s={best_s:.1f} → MAE={best_m:.2f} Δ={best_m-base:+.2f}", flush=True)

    # ---- 实现就绪清洁配置 ----
    print("\n=== 实现就绪清洁配置 ===", flush=True)
    hours_v = pd.DatetimeIndex(times[val]).hour.values.astype(int)
    temp_all = np.nan_to_num(X["temp"].values.astype(float), nan=0.0); temp_v = temp_all[val]
    clear = np.nan_to_num(X["clearness"].values.astype(float), nan=0.0); clear_v = clear[val]
    irrad_all = np.nan_to_num((X["irrad"].values if "irrad" in X.columns else X["光伏_辐照度"].values).astype(float), nan=0.0)
    def shift_of(m): r=oof_resid[m]; return float(np.nanmean(r)) if np.isfinite(r).sum()>20 else 0.0
    def mae(p): return np.abs(p-a).mean()
    # temp<8 (op <), clr0.2-0.5 band, irrad<400 (op <), temp>30 (op >)
    p = pred_v.copy()
    # temp<8 全天
    sh = shift_of(oof_mask & (temp_all<8)); m=temp_v<8; p[m]-=sh
    print(f"  +temp<8(shift={sh:+.0f},n={m.sum()}) → {mae(p):.2f}", flush=True)
    # clr 0.2-0.5 @11-14 (band: >0.2 & <0.5)
    sh = shift_of(oof_mask & np.isin(h_all,[11,12,13,14]) & (clear>0.2)&(clear<0.5))
    m = np.isin(hours_v,[11,12,13,14])&(clear_v>0.2)&(clear_v<0.5); p[m]-=sh
    print(f"  +clr0.2-0.5@11-14(shift={sh:+.0f},n={m.sum()}) → {mae(p):.2f}", flush=True)
    # irrad<400 @9-15
    sh = shift_of(oof_mask & np.isin(h_all,list(range(9,16))) & (irrad_all<400))
    m = np.isin(hours_v,list(range(9,16)))&(irrad_all[val]<400); p[m]-=sh
    print(f"  +irrad<400@9-15(shift={sh:+.0f},n={m.sum()}) → {mae(p):.2f}", flush=True)
    # temp>30 全天
    sh = shift_of(oof_mask & (temp_all>30)); m=temp_v>30; p[m]-=sh
    print(f"  +temp>30(shift={sh:+.0f},n={m.sum()}) → {mae(p):.2f}  (Δ基线 {mae(p)-base:+.2f})", flush=True)
    # 仅 robust 子集(temp<8 + clr0.2-0.5)
    p2 = pred_v.copy()
    sh=shift_of(oof_mask&(temp_all<8)); m=temp_v<8; p2[m]-=sh
    sh=shift_of(oof_mask&np.isin(h_all,[11,12,13,14])&(clear>0.2)&(clear<0.5)); m=np.isin(hours_v,[11,12,13,14])&(clear_v>0.2)&(clear_v<0.5); p2[m]-=sh
    print(f"  robust子集(temp<8+clr0.2-0.5) → {mae(p2):.2f} (Δ {mae(p2)-base:+.2f})", flush=True)


if __name__=="__main__":
    main()
