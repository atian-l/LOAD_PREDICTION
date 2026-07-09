# -*- coding: utf-8 -*-
"""
exp49 — 精调午间 β·pl_wr 校正（exp47 最佳: 11-13 β·pl_wr = 1516.16）。
测试窗口(10-13/11-14/11-13)、多元(pl_wr+roll_mean_96+diff_672)、+solar_mismatch。
目标：压低午间 3102 MAE 的可校正部分。3 种子，OOF 算一次。不写产物。
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


def fit_beta_hours(feat_arr, oof_mask, h_all, oof_resid, hours_set):
    """对 hours_set 内各小时分别拟 β；其它小时 β=0。返回 [24]。"""
    beta = np.zeros(24)
    for h in hours_set:
        m = oof_mask & (h_all == h)
        f = feat_arr[m]; e = oof_resid[m]
        good = np.isfinite(f) & np.isfinite(e)
        d = float(np.dot(f[good], f[good]))
        if d > 0:
            beta[h] = float(np.dot(f[good], e[good]) / d)
    return beta


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

    print(f"training full ensemble (BI={bi}) + 3-fold OOF ...", flush=True)
    members, flags = train_members(times, X, pred_load, actual, usable, bi, cfg)
    pred_val_nb = ens_raw(X[val_mask], pred_load, members, flags, cfg["shrinkage"])
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
    h_all = pd.DatetimeIndex(times).hour.values
    hours_val = pd.DatetimeIndex(X[val_mask].index).hour.values.astype(int)
    av = actual[val_mask].values

    hb = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum():
            hb[h] = float(np.average(oof_resid[m]))
    pred_ph = np.clip(pred_val_nb - hb[hours_val], 0.0, None)
    mae_ph = float(np.mean(np.abs(pred_ph - av)))
    print(f"  no-bias={float(np.mean(np.abs(pred_val_nb-av))):.2f}  +per-hour={mae_ph:.2f}", flush=True)

    pl_wr = X["pl_weather_residual"].values
    pl_wr_rm96 = X["pl_wr_roll_mean_96"].values
    pl_wr_d672 = X["pl_wr_diff_672"].values
    sm = X["solar_mismatch"].values

    def eval_corr(name, betas, feats):
        """betas: list of [24] arrays; feats: list of val feature arrays."""
        c = np.zeros(len(av))
        for b, f in zip(betas, feats):
            c = c + b[hours_val] * np.nan_to_num(f[val_mask])
        p = np.clip(pred_ph + c, 0.0, None)
        mae = float(np.mean(np.abs(p - av)))
        print(f"  {name}: MAE={mae:.2f}  (delta={mae-mae_ph:+.2f})", flush=True)
        return mae, p

    print("== 午间窗口/特征精调 ==", flush=True)
    for win_name, hrs in [("10-13", range(10, 14)), ("11-14", range(11, 15)),
                          ("11-13", range(11, 14)), ("10-14", range(10, 15)),
                          ("09-15", range(9, 16))]:
        b = fit_beta_hours(pl_wr, oof_mask, h_all, oof_resid, set(hrs))
        eval_corr(f"pl_wr {win_name}", [b], [pl_wr])
    # 11-13 多元
    hrs = set(range(11, 14))
    b1 = fit_beta_hours(pl_wr, oof_mask, h_all, oof_resid, hrs)
    r1 = oof_resid - b1[h_all] * np.nan_to_num(pl_wr)
    b2 = fit_beta_hours(pl_wr_rm96, oof_mask, h_all, r1, hrs)
    r2 = r1 - b2[h_all] * np.nan_to_num(pl_wr_rm96)
    b3 = fit_beta_hours(pl_wr_d672, oof_mask, h_all, r2, hrs)
    eval_corr("11-13 多元(pl_wr+rm96+d672)", [b1, b2, b3], [pl_wr, pl_wr_rm96, pl_wr_d672])
    # 11-13 pl_wr + solar_mismatch
    b_sm = fit_beta_hours(sm, oof_mask, h_all, r1, hrs)
    eval_corr("11-13 pl_wr+solar_mismatch", [b1, b_sm], [pl_wr, sm])
    print("done.", flush=True)


if __name__ == "__main__":
    main()
