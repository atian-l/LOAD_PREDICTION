# -*- coding: utf-8 -*-
"""实验30：丰富交互特征。exp29 发现 pl×calendar(-6) + pl×weather(-5) 抗漂移。
测更多：pl×cyclic, pl×weather_quantiles精选, 三阶交互, pl比值。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from . import data_loader as dl
from . import features as F


def build(mode="baseline"):
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)
    X = F.build_features(times, pred_load, weather)
    pl = X["pred_load"]
    pl_norm = pl / (pl.rolling(672, min_periods=96).mean())

    base_interactions = {
        # pl × weather
        "pl_x_temp": pl * X["temp"], "pl_x_hdd": pl * X["hdd"], "pl_x_cdd": pl * X["cdd"],
        "pl_x_irrad": pl * X["irrad"], "pl_x_wind": pl * X["wind"],
        "plnorm_x_temp": pl_norm * X["temp"], "plnorm_x_hdd": pl_norm * X["hdd"], "plnorm_x_cdd": pl_norm * X["cdd"],
        # pl × calendar
        "pl_x_hour": pl * X["hour"], "pl_x_dow": pl * X["dayofweek"], "pl_x_month": pl * X["month"],
        "plnorm_x_hour": pl_norm * X["hour"],
        # anomaly
        "plvsmean_x_temp": X.get("pred_load_vs_mean_96", pl - pl.rolling(96, min_periods=24).mean()) * X["temp"],
        "plvsmean_x_hdd": X.get("pred_load_vs_mean_96", pl - pl.rolling(96, min_periods=24).mean()) * X["hdd"],
    }
    if mode == "all":
        for k, v in base_interactions.items():
            X[k] = v
    elif mode == "cyclic":
        for k, v in base_interactions.items():
            X[k] = v
        X["pl_x_hour_sin"] = pl * X["hour_sin"]
        X["pl_x_hour_cos"] = pl * X["hour_cos"]
        X["pl_x_doy_sin"] = pl * X["doy_sin"]
        X["pl_x_doy_cos"] = pl * X["doy_cos"]
        X["pl_x_month_sin"] = pl * X["month_sin"]
        X["pl_x_month_cos"] = pl * X["month_cos"]
        X["plnorm_x_doy_sin"] = pl_norm * X["doy_sin"]
        X["plnorm_x_doy_cos"] = pl_norm * X["doy_cos"]
    elif mode == "triple":
        for k, v in base_interactions.items():
            X[k] = v
        # 三阶：pl × weather × hour
        X["pl_x_temp_x_hour"] = pl * X["temp"] * X["hour"]
        X["pl_x_cdd_x_hour"] = pl * X["cdd"] * X["hour"]
        X["pl_x_hdd_x_hour"] = pl * X["hdd"] * X["hour"]
        X["pl_x_temp_x_irrad"] = pl * X["temp"] * X["irrad"]
        X["plnorm_x_temp_x_hour"] = pl_norm * X["temp"] * X["hour"]
    elif mode == "ratios":
        for k, v in base_interactions.items():
            X[k] = v
        for lag in [96, 192, 672]:
            if f"pred_load_lag_{lag}" in X.columns:
                X[f"pl_ratio_{lag}"] = pl / (X[f"pred_load_lag_{lag}"] + 1e-6)
                X[f"pl_ratio_{lag}_x_temp"] = X[f"pl_ratio_{lag}"] * X["temp"]
        # 滚动均值比值
        if "pred_load_roll_mean_96" in X.columns:
            X["pl_over_roll96"] = pl / (X["pred_load_roll_mean_96"] + 1e-6)
            X["pl_over_roll96_x_temp"] = X["pl_over_roll96"] * X["temp"]
    elif mode == "rich":
        # 全部交互
        for k, v in base_interactions.items():
            X[k] = v
        X["pl_x_hour_sin"] = pl * X["hour_sin"]; X["pl_x_hour_cos"] = pl * X["hour_cos"]
        X["pl_x_doy_sin"] = pl * X["doy_sin"]; X["pl_x_doy_cos"] = pl * X["doy_cos"]
        X["pl_x_month_sin"] = pl * X["month_sin"]; X["pl_x_month_cos"] = pl * X["month_cos"]
        X["pl_x_temp_x_hour"] = pl * X["temp"] * X["hour"]
        X["pl_x_cdd_x_hour"] = pl * X["cdd"] * X["hour"]
        X["pl_x_hdd_x_hour"] = pl * X["hdd"] * X["hour"]
        X["pl_x_temp_x_irrad"] = pl * X["temp"] * X["irrad"]
        for lag in [96, 192, 672]:
            if f"pred_load_lag_{lag}" in X.columns:
                X[f"pl_ratio_{lag}"] = pl / (X[f"pred_load_lag_{lag}"] + 1e-6)
        if "pred_load_roll_mean_96" in X.columns:
            X["pl_over_roll96"] = pl / (X["pred_load_roll_mean_96"] + 1e-6)
    return times, X, pred_load, actual


def tw(times, mask, alpha):
    t = times[mask]; tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        return np.ones(len(t))
    return (1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()).values.astype(float)


def _mae(p, a):
    return np.mean(np.abs(p - a))


PP = dict(learning_rate=0.02, num_leaves=127, min_data_in_leaf=300, lambda_l2=1.0,
          feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1, verbose=-1, force_col_wise=True)
QA = [0.45, 0.5, 0.55]; SEEDS = [42, 7, 123]; AW = 5.0; BI = 221; LAM = 0.8


def train_members(times, X, pred_load, actual, feat_cols, train_mask):
    y_dir = actual; y_res = actual - pred_load
    Xtr = X[train_mask][feat_cols]; wtr = tw(times, train_mask, AW)
    member_preds = []
    objs = [("regression", None)] + [("quantile", q) for q in QA]
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[train_mask]
        d = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj, qa_ in objs:
            for s in SEEDS:
                p = dict(PP, objective=obj, seed=s)
                if obj == "quantile":
                    p["alpha"] = qa_
                bst = lgb.train(p, d, num_boost_round=int(BI))
                raw = bst.predict(X[feat_cols])
                member_preds.append((pred_load.values + raw) if residual else raw)
    return np.array(member_preds)


def main():
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    vm = None; av = None; pv_full = None
    for mode in ["all", "cyclic", "triple", "ratios", "rich"]:
        times, X, pred_load, actual = build(mode=mode)
        if vm is None:
            vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
            av = actual.values[vm]; pv_full = pred_load.values
        full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
        feat_cols = list(X.columns)
        M = train_members(times, X, pred_load, actual, feat_cols, full_mask)
        base = np.clip(pv_full + LAM*(np.median(M, axis=0) - pv_full), 0, None)
        print(f"[{mode:8s}] VAL MAE={_mae(base[vm], av):.2f}  n_feat={len(feat_cols)}")

    # 最佳 + per-hour
    print("\n== rich + per-hour ==")
    times, X, pred_load, actual = build(mode="rich")
    feat_cols = list(X.columns)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    M = train_members(times, X, pred_load, actual, feat_cols, full_mask)
    base = np.clip(pv_full + LAM*(np.median(M, axis=0) - pv_full), 0, None)
    folds = C.TRAIN_CONFIG["best_it_folds"]
    oof_ens = np.full(len(times), np.nan)
    for te_s, vs_s, ve_s in folds:
        te, vs, ve = pd.Timestamp(te_s), pd.Timestamp(vs_s), pd.Timestamp(ve_s)
        ftr = full_mask & np.asarray(times <= te)
        fva = full_mask & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0: continue
        Mf = train_members(times, X, pred_load, actual, feat_cols, ftr)
        oof_ens[fva] = np.median(Mf, axis=0)[fva]
    oof_mask = full_mask & ~np.isnan(oof_ens)
    oof_resid = (pv_full + LAM*(oof_ens - pv_full)) - actual.values
    h_all = times.hour.values
    hb = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum(): hb[h] = np.average(oof_resid[m])
    corr_h = np.array([hb[h_all[i]] for i in range(len(times))])
    print(f"[rich + per-hour] VAL MAE={_mae(np.clip(base[vm]-corr_h[vm],0,None), av):.2f}")
    for mo in [3,4,5,6]:
        mm = (times.month.values[vm] == mo)
        if mm.sum():
            print(f"  mo={mo}: MAE={_mae(np.clip(base[vm][mm]-corr_h[vm][mm],0,None), av[mm]):.2f} bias={np.mean(np.clip(base[vm][mm]-corr_h[vm][mm],0,None)-av[mm]):.0f}")


if __name__ == "__main__":
    main()
