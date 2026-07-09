# -*- coding: utf-8 -*-
"""
exp46 — 为 +per-hour MAE 重新选超参（exp43 的 no-bias 最优 nl=255/λ=1.0 在 +per-hour 下
仅 1526.86，因深树+无收缩产生更平滑、可校正偏置更小的模型）。测试 nl=127 系配置
（exp41 的 nl=127/mdl=300/l2=2/λ=0.9 曾达 +per-hour 1519.31）在 BI=80 下的 +per-hour MAE。
3 种子、3 折 OOF 每小时偏置。无泄露。不写产物。
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


def ens_predict(X, pred_load, members, flags, lam, hour_bias=None):
    pl = pred_load.reindex(X.index).values.astype(float)
    mp = np.empty((len(members), len(X)), dtype=float)
    for i, (bst, is_res) in enumerate(zip(members, flags)):
        raw = bst.predict(X)
        mp[i] = pl + raw if is_res else raw
    ens = np.median(mp, axis=0)
    pred = pl + lam * (ens - pl)
    if hour_bias is not None:
        hours = pd.DatetimeIndex(X.index).hour.values.astype(int)
        pred = pred - hour_bias[hours]
    return np.clip(pred, 0.0, None)


def compute_hour_bias(times, X, pred_load, actual, usable, bi, cfg):
    oof_pred = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        members, flags = train_members(times, X, pred_load, actual, ftr, bi, cfg)
        oof_pred[fva] = ens_predict(X[fva], pred_load, members, flags, cfg["shrinkage"])
    oof_mask = usable & oof_pred.notna().values
    resid = (oof_pred - actual).values
    hb = np.zeros(24, dtype=float)
    h_all = pd.DatetimeIndex(times).hour.values
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum():
            hb[h] = float(np.average(resid[m]))
    return hb


def main():
    print("building ...", flush=True)
    times, X, pred_load, actual, usable = build()
    print(f"  n_feat={X.shape[1]}  usable={usable.sum()}", flush=True)
    base_cfg = dict(C.TRAIN_CONFIG)
    base_cfg["seeds"] = [42, 7, 123]
    val_mask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
                & actual.notna()).values
    configs = [
        dict(nl=127, mdl=300, l2=2.0, lam=0.9),   # exp41 配置（曾达 +per-hour 1519.31）
        dict(nl=127, mdl=200, l2=2.0, lam=1.0),
        dict(nl=127, mdl=200, l2=4.0, lam=0.9),
        dict(nl=255, mdl=200, l2=4.0, lam=1.0),   # exp43 no-bias 最优（已知 +per-hour 1526.86）
    ]
    bi = 80
    print(f"== +per-hour 配置扫描 (BI={bi}, 3 seeds) ==", flush=True)
    for c in configs:
        cfg = dict(base_cfg)
        cfg["num_leaves"] = c["nl"]; cfg["min_data_in_leaf"] = c["mdl"]
        cfg["lambda_l2"] = c["l2"]; cfg["shrinkage"] = c["lam"]
        members, flags = train_members(times, X, pred_load, actual, usable, bi, cfg)
        pred_nb = ens_predict(X[val_mask], pred_load, members, flags, cfg["shrinkage"])
        mae_nb = float(np.mean(np.abs(pred_nb - actual[val_mask].values)))
        hb = compute_hour_bias(times, X, pred_load, actual, usable, bi, cfg)
        pred_b = ens_predict(X[val_mask], pred_load, members, flags, cfg["shrinkage"], hb)
        mae_b = float(np.mean(np.abs(pred_b - actual[val_mask].values)))
        print(f"  nl={c['nl']} mdl={c['mdl']} l2={c['l2']} lam={c['lam']}: "
              f"no-bias={mae_nb:.2f}  +per-hour={mae_b:.2f}  (delta={mae_b-mae_nb:+.2f})", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
