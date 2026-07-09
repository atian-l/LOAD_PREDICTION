# -*- coding: utf-8 -*-
"""
exp50 — 午间专用模型测试。午间(11-14) MAE≈3000 是最大误差来源(光伏/太阳能驱动)。
低 BI=80 的全局模型在午间欠拟合 pl_wr 方向信号(已用 drift_corr 线性补偿 -13 MW)。
测试：仅在午间(11-14)训练点上训练的高 BI 专用集成，是否能在午间击败全局模型+drift_corr。
若有效则混合(午间用专用模型，其余用全局)。3 种子，OOF 评估。不写产物。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from load_pred import config as C, data_loader as dl, features as F


def build():
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


def main():
    print("building ...", flush=True)
    times, X, pred_load, actual, usable = build()
    print(f"  n_feat={X.shape[1]}  usable={usable.sum()}", flush=True)
    cfg = dict(C.TRAIN_CONFIG)
    cfg["num_leaves"] = 255; cfg["min_data_in_leaf"] = 200; cfg["lambda_l2"] = 4.0
    cfg["shrinkage"] = 1.0; cfg["seeds"] = [42, 7, 123]
    bi = 80
    val_mask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
                & actual.notna()).values
    h_all = pd.DatetimeIndex(times).hour.values
    hours_val = pd.DatetimeIndex(X[val_mask].index).hour.values.astype(int)
    av = actual[val_mask].values
    midday_val = np.isin(hours_val, [11, 12, 13, 14])
    midday_train = np.isin(h_all, [11, 12, 13, 14])

    # --- 全局模型 + per-hour + drift_corr (基线) ---
    print("training global ensemble + OOF ...", flush=True)
    members, flags = train_members(times, X, pred_load, actual, usable, bi, cfg)
    pred_val_g = ens_raw(X[val_mask], pred_load, members, flags, cfg["shrinkage"])
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
    pred_g = np.clip(pred_val_g - hb[hours_val] + beta[hours_val] * np.nan_to_num(pl_wr[val_mask]), 0.0, None)
    mae_g = float(np.mean(np.abs(pred_g - av)))
    mae_g_mid = float(np.mean(np.abs(pred_g[midday_val] - av[midday_val])))
    print(f"  全局+per-hour+drift: 全体={mae_g:.2f}  午间(11-14)={mae_g_mid:.0f}", flush=True)

    # --- 午间专用模型 (高 BI, 仅午间训练点) ---
    print("training midday-only ensemble (高 BI) ...", flush=True)
    mid_usable = usable & midday_train
    for bi_mid in [150, 300, 600]:
        mm_members, mm_flags = train_members(times, X, pred_load, actual, mid_usable, bi_mid, cfg)
        pred_mid_raw = ens_raw(X[val_mask], pred_load, mm_members, mm_flags, cfg["shrinkage"])
        # 午间专用 per-hour (用全局 OOF 的午间残差近似，避免再算 OOF)
        pred_mid_corr = np.clip(pred_mid_raw - hb[hours_val], 0.0, None)
        mae_mid_only = float(np.mean(np.abs(pred_mid_corr[midday_val] - av[midday_val])))
        # 混合: 午间用专用, 其余用全局
        pred_blend = pred_g.copy()
        pred_blend[midday_val] = pred_mid_corr[midday_val]
        pred_blend = np.clip(pred_blend, 0.0, None)
        mae_blend = float(np.mean(np.abs(pred_blend - av)))
        print(f"  午间专用 BI={bi_mid}: 午间MAE={mae_mid_only:.0f}  混合全体MAE={mae_blend:.2f} "
              f"(vs 全局 {mae_g:.2f}, delta={mae_blend-mae_g:+.2f})", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
