# -*- coding: utf-8 -*-
"""
exp51 — 测试 pl_wr 滞后特征 (pl_wr_lag_96, pl_wr_lag_672, pl_wr_lag_288) 是否补充方向信号。
模型已有 pl_wr_roll_mean_96 (含历史 pl_wr)，但原始滞后(精确 96/672 步前值)可能额外信息。
3 种子, nl255/mdl200/l24/λ1.0, BI=80, +per-hour +drift_corr(11-14)。对比 1513.80。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from load_pred import config as C, data_loader as dl, features as F


def build(extra_lags=False):
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)
    X = F.build_features(times, pred_load, weather)
    ts0 = pd.Timestamp(C.TRAIN_CONFIG["train_start"]); tr_end = pd.Timestamp(C.TRAIN_END)
    usable = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    mm = F.MismatchModel().fit(X, usable)
    X = mm.transform(X)
    if extra_lags:
        wr_s = pd.Series(X["pl_weather_residual"].values, index=X.index)
        X["pl_wr_lag_96"] = wr_s.shift(96).values
        X["pl_wr_lag_288"] = wr_s.shift(288).values
        X["pl_wr_lag_672"] = wr_s.shift(672).values
        X["pl_wr_x_lag_96"] = X["pl_weather_residual"].values * X["pl_wr_lag_96"].values
    return times, X, pred_load, actual, usable


def time_weights(times, mask, alpha):
    t = times[mask]; tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        return np.ones(len(t))
    return (1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()).values.astype(float)


def train_members(times, X, pred_load, actual, tr_mask, bi, cfg):
    feat_cols = list(X.columns)
    y_dir = actual; y_res = actual - pred_load
    Xtr = X[tr_mask][feat_cols]
    wtr = time_weights(times, tr_mask, cfg["alpha_w"])
    base = dict(metric="mae", learning_rate=cfg["learning_rate"], num_leaves=cfg["num_leaves"],
                min_data_in_leaf=cfg["min_data_in_leaf"], lambda_l2=cfg["lambda_l2"],
                feature_fraction=cfg["feature_fraction"], bagging_fraction=cfg["bagging_fraction"],
                bagging_freq=cfg["bagging_freq"], verbose=-1, force_col_wise=True)
    members, flags = [], []
    for residual in cfg["residual_modes"]:
        ytr = (y_res if residual else y_dir)[tr_mask]
        dtr = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj in cfg["objectives"]:
            alphas = cfg["quantile_alphas"] if obj == "quantile" else [None]
            for qa in alphas:
                for s in cfg["seeds"]:
                    p = dict(base, objective=obj, seed=s)
                    if obj == "quantile":
                        p["alpha"] = qa
                    members.append(lgb.train(p, dtr, num_boost_round=int(bi)))
                    flags.append(residual)
    return members, flags


def ens_raw(X, pred_load, members, flags, lam):
    pl = pred_load.reindex(X.index).values.astype(float)
    mp = np.empty((len(members), len(X)), dtype=float)
    for i, (bst, is_res) in enumerate(zip(members, flags)):
        raw = bst.predict(X)
        mp[i] = pl + raw if is_res else raw
    ens = np.median(mp, axis=0)
    return pl + lam * (ens - pl)


def run(times, X, pred_load, actual, usable, cfg, bi, label):
    val_mask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
                & actual.notna()).values
    h_all = pd.DatetimeIndex(times).hour.values
    hours_val = pd.DatetimeIndex(X[val_mask].index).hour.values.astype(int)
    av = actual[val_mask].values
    members, flags = train_members(times, X, pred_load, actual, usable, bi, cfg)
    pred_val = ens_raw(X[val_mask], pred_load, members, flags, cfg["shrinkage"])
    oof_pred = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        fm, ff = train_members(times, X, pred_load, actual, ftr, bi, cfg)
        oof_pred[fva] = ens_raw(X[fva], pred_load, fm, ff, cfg["shrinkage"])
    oof_mask = usable & oof_pred.notna().values
    oof_resid = (oof_pred - actual).values
    hb = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum():
            hb[h] = float(np.average(oof_resid[m]))
    pl_wr = X["pl_weather_residual"].values
    beta = np.zeros(24)
    for h in [11, 12, 13, 14]:
        m = oof_mask & (h_all == h)
        f = pl_wr[m]; e = oof_resid[m]
        good = np.isfinite(f) & np.isfinite(e)
        d = float(np.dot(f[good], f[good]))
        if d > 0:
            beta[h] = float(np.dot(f[good], e[good]) / d)
    pred = np.clip(pred_val - hb[hours_val] + beta[hours_val] * np.nan_to_num(pl_wr[val_mask]), 0.0, None)
    mae = float(np.mean(np.abs(pred - av)))
    print(f"  {label}: n_feat={X.shape[1]}  MAE={mae:.2f}", flush=True)
    return mae


def main():
    print("building baseline ...", flush=True)
    times, X, pred_load, actual, usable = build(extra_lags=False)
    cfg = dict(C.TRAIN_CONFIG)
    cfg["num_leaves"] = 255; cfg["min_data_in_leaf"] = 200; cfg["lambda_l2"] = 4.0
    cfg["shrinkage"] = 1.0; cfg["seeds"] = [42, 7, 123]
    mae0 = run(times, X, pred_load, actual, usable, cfg, 80, "baseline (无滞后)")
    print("building +pl_wr lags ...", flush=True)
    times, X, pred_load, actual, usable = build(extra_lags=True)
    mae1 = run(times, X, pred_load, actual, usable, cfg, 80, "+pl_wr_lag(96/288/672)+x_lag96")
    print(f"  delta={mae1-mae0:+.2f}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
