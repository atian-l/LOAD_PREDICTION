# -*- coding: utf-8 -*-
"""
exp44 — best_iter 扫描（生产特征集 + exp43 最优超参 nl=255/mdl=200/l2=4.0/λ=1.0）。

背景：生产重训 best_iter=248（3 折均值，被春季折 485 抬高）→ 验证 MAE 1529（过拟合，
hour_bias 范围扩大到 [-887,303]）。exp43 用固定 BI≈147 得 no-bias 1522.80。本实验在
生产 126 特征求+残差特征集上扫描固定 best_iter，3 种子、无偏置，定位最优 BI。
无泄露：仅用 pred_load+weather+calendar；实际负荷仅作目标/评估。不写任何产物。
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


def ens_predict(X, pred_load, members, residual_flags, lam):
    pl = pred_load.reindex(X.index).values.astype(float)
    mp = np.empty((len(members), len(X)), dtype=float)
    for i, (bst, is_res) in enumerate(zip(members, residual_flags)):
        raw = bst.predict(X)
        mp[i] = pl + raw if is_res else raw
    ens = np.median(mp, axis=0)
    return pl + lam * (ens - pl)


def run_bi(times, X, pred_load, actual, usable, bi, cfg):
    feat_cols = list(X.columns)
    y_dir = actual
    y_res = actual - pred_load
    Xtr = X[usable][feat_cols]
    wtr = time_weights(times, usable, cfg["alpha_w"])
    base = dict(metric="mae", learning_rate=cfg["learning_rate"], num_leaves=cfg["num_leaves"],
                min_data_in_leaf=cfg["min_data_in_leaf"], lambda_l2=cfg["lambda_l2"],
                feature_fraction=cfg["feature_fraction"], bagging_fraction=cfg["bagging_fraction"],
                bagging_freq=cfg["bagging_freq"], verbose=-1, force_col_wise=True)
    members, flags = [], []
    for residual in cfg["residual_modes"]:
        ytr = (y_res if residual else y_dir)[usable]
        dtr = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj in cfg["objectives"]:
            alphas = cfg["quantile_alphas"] if obj == "quantile" else [None]
            for qa in alphas:
                for s in cfg["seeds"]:
                    p = dict(base, objective=obj, seed=s)
                    if obj == "quantile":
                        p["alpha"] = qa
                    bst = lgb.train(p, dtr, num_boost_round=int(bi))
                    members.append(bst); flags.append(residual)
    val_mask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
                & actual.notna()).values
    pred = ens_predict(X[val_mask], pred_load, members, flags, cfg["shrinkage"])
    a = actual[val_mask].values
    mae = float(np.mean(np.abs(pred - a)))
    return mae


def main():
    print("building ...", flush=True)
    times, X, pred_load, actual, usable = build()
    print(f"  n_feat={X.shape[1]}  usable={usable.sum()}", flush=True)
    cfg = dict(C.TRAIN_CONFIG)
    cfg["num_leaves"] = 255; cfg["min_data_in_leaf"] = 200; cfg["lambda_l2"] = 4.0
    cfg["shrinkage"] = 1.0
    cfg["seeds"] = [42, 7, 123]  # 3 种子加速
    print(f"  hyperparams: nl={cfg['num_leaves']} mdl={cfg['min_data_in_leaf']} "
          f"l2={cfg['lambda_l2']} lam={cfg['shrinkage']}  seeds=3  (no per-hour bias)", flush=True)
    print("== best_iter 扫描 (no-bias) ==", flush=True)
    for bi in [20, 40, 60, 80, 100]:
        mae = run_bi(times, X, pred_load, actual, usable, bi, cfg)
        print(f"  BI={bi:>4}: no-bias VAL MAE={mae:.2f}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
