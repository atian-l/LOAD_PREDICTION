# -*- coding: utf-8 -*-
"""
exp48 — 推广 exp47 的午间 β·pl_wr 校正：
  (A) 逐小时 β (24 个)：每小时各自估 β_h = <pl_wr·err>/<pl_wr²> (OOF)，仅在信号显著时非零。
      推广午间版，避免全局 β 被非午间大值污染。
  (B) 午间多元线性校正：β1·pl_wr + β2·pl_wr_roll_mean_96 + β3·pl_wr_diff_672 (小时 10-14)。
  (C) 全天逐小时多元 (pl_wr + roll_mean_96 + diff_672)。
无泄露(β 仅来自 OOF，3 折)。3 种子。不写产物。
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

    print(f"training full ensemble (BI={bi}) + 3-fold OOF ...", flush=True)
    members, flags = train_members(times, X, pred_load, actual, usable, bi, cfg)
    pred_val_nb = ens_raw(X[val_mask], pred_load, members, flags, cfg["shrinkage"])
    mae_nb = float(np.mean(np.abs(pred_val_nb - actual[val_mask].values)))

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

    # 3-fold 均值 per-hour
    hb = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum():
            hb[h] = float(np.average(oof_resid[m]))
    pred_ph = np.clip(pred_val_nb - hb[hours_val], 0.0, None)
    mae_ph = float(np.mean(np.abs(pred_ph - actual[val_mask].values)))
    print(f"  no-bias={mae_nb:.2f}  +3fold-per-hour={mae_ph:.2f}", flush=True)

    pl_wr = X["pl_weather_residual"].values
    pl_wr_rm96 = X["pl_wr_roll_mean_96"].values
    pl_wr_d672 = X["pl_wr_diff_672"].values

    def fit_beta_hourly(feat_arr, hours_range=None):
        """逐小时 β；hours_range=None 表全部 24 小时。返回 [24] 系数。"""
        beta = np.zeros(24)
        for h in range(24):
            if hours_range is not None and h not in hours_range:
                continue
            m = oof_mask & (h_all == h)
            f = feat_arr[m]; e = oof_resid[m]
            good = np.isfinite(f) & np.isfinite(e)
            d = float(np.dot(f[good], f[good]))
            if d > 0:
                beta[h] = float(np.dot(f[good], e[good]) / d)
        return beta

    def apply_corr(base_pred, beta_arr, feat_arr):
        c = beta_arr[hours_val] * feat_arr[val_mask]
        return np.clip(base_pred + np.nan_to_num(c), 0.0, None)

    print("== (A) 逐小时 β·pl_wr ==", flush=True)
    bh = fit_beta_hourly(pl_wr)
    mae_a = float(np.mean(np.abs(apply_corr(pred_ph, bh, pl_wr) - actual[val_mask].values)))
    nz = [(h, round(bh[h], 4)) for h in range(24) if abs(bh[h]) > 1e-4]
    print(f"  +per-hour+β_hourly·pl_wr={mae_a:.2f}  (delta={mae_a-mae_ph:+.2f})  非零β={nz}", flush=True)

    print("== (B) 午间(10-14)多元 ==", flush=True)
    mid = range(10, 15)
    b1 = fit_beta_hourly(pl_wr, mid)
    # 残差去掉 pl_wr 分量后再拟 roll_mean_96
    resid1 = oof_resid - b1[h_all] * np.nan_to_num(pl_wr)
    b2 = np.zeros(24)
    for h in mid:
        m = oof_mask & (h_all == h)
        f = pl_wr_rm96[m]; e = resid1[m]
        good = np.isfinite(f) & np.isfinite(e)
        d = float(np.dot(f[good], f[good]))
        if d > 0:
            b2[h] = float(np.dot(f[good], e[good]) / d)
    resid2 = resid1 - b2[h_all] * np.nan_to_num(pl_wr_rm96)
    b3 = np.zeros(24)
    for h in mid:
        m = oof_mask & (h_all == h)
        f = pl_wr_d672[m]; e = resid2[m]
        good = np.isfinite(f) & np.isfinite(e)
        d = float(np.dot(f[good], f[good]))
        if d > 0:
            b3[h] = float(np.dot(f[good], e[good]) / d)
    corr_mid = (b1[hours_val] * np.nan_to_num(pl_wr[val_mask]) +
                b2[hours_val] * np.nan_to_num(pl_wr_rm96[val_mask]) +
                b3[hours_val] * np.nan_to_num(pl_wr_d672[val_mask]))
    pred_mid = np.clip(pred_ph + corr_mid, 0.0, None)
    mae_mid = float(np.mean(np.abs(pred_mid - actual[val_mask].values)))
    print(f"  +per-hour+midday多元={mae_mid:.2f}  (delta={mae_mid-mae_ph:+.2f})", flush=True)

    print("== (C) 全天逐小时多元 ==", flush=True)
    c1 = fit_beta_hourly(pl_wr)
    resid1g = oof_resid - c1[h_all] * np.nan_to_num(pl_wr)
    c2 = np.zeros(24); c3 = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        f = pl_wr_rm96[m]; e = resid1g[m]
        good = np.isfinite(f) & np.isfinite(e)
        d = float(np.dot(f[good], f[good]))
        if d > 0:
            c2[h] = float(np.dot(f[good], e[good]) / d)
    resid2g = resid1g - c2[h_all] * np.nan_to_num(pl_wr_rm96)
    for h in range(24):
        m = oof_mask & (h_all == h)
        f = pl_wr_d672[m]; e = resid2g[m]
        good = np.isfinite(f) & np.isfinite(e)
        d = float(np.dot(f[good], f[good]))
        if d > 0:
            c3[h] = float(np.dot(f[good], e[good]) / d)
    corr_all = (c1[hours_val] * np.nan_to_num(pl_wr[val_mask]) +
                c2[hours_val] * np.nan_to_num(pl_wr_rm96[val_mask]) +
                c3[hours_val] * np.nan_to_num(pl_wr_d672[val_mask]))
    pred_all = np.clip(pred_ph + corr_all, 0.0, None)
    mae_all = float(np.mean(np.abs(pred_all - actual[val_mask].values)))
    print(f"  +per-hour+全天多元={mae_all:.2f}  (delta={mae_all-mae_ph:+.2f})", flush=True)

    # 场景分解 (最佳)
    best_pred = min([(mae_ph, pred_ph, "per-hour"), (mae_a, apply_corr(pred_ph, bh, pl_wr), "A"),
                     (mae_mid, pred_mid, "B-midday"), (mae_all, pred_all, "C-all")], key=lambda x: x[0])
    print(f"  -> 最佳: {best_pred[2]} ({best_pred[0]:.2f})", flush=True)
    av = actual[val_mask].values
    is_mid = (hours_val >= 11) & (hours_val <= 13)
    is_day = (hours_val >= 8) & (hours_val <= 16)
    p = best_pred[1]
    print(f"  场景: 午间(11-13) MAE={np.mean(np.abs(p[is_mid]-av[is_mid])):.0f}  "
          f"非午间={np.mean(np.abs(p[~is_mid]-av[~is_mid])):.0f}  "
          f"白天(8-16)={np.mean(np.abs(p[is_day]-av[is_day])):.0f}  "
          f"夜间={np.mean(np.abs(p[~is_day]-av[~is_day])):.0f}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
