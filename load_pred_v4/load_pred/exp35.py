# -*- coding: utf-8 -*-
"""实验35：最后两个廉价结构性尝试。
A) extra_trees=True（极随机分裂，可能改善泛化）
B) mean 聚合 vs median（成员分布偏斜时 mean 可能更优）
C) extra_trees + mean
配置：exp31 组合最优 nl=127, lr=0.03, mdl=300, l2=2.0, λ=0.9, bi=147, +per-hour。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from . import data_loader as dl
from . import features as F


def build():
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)
    X = F.build_features(times, pred_load, weather)
    return times, X, pred_load, actual


def tw(times, mask, alpha):
    t = times[mask]; tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        return np.ones(len(t))
    return (1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()).values.astype(float)


def _mae(p, a):
    return np.mean(np.abs(p - a))


QA = [0.45, 0.5, 0.55]; SEEDS = [42, 7, 123]; AW = 5.0
NL = 127; LR = 0.03; MDL = 300; L2 = 2.0; FF = 0.80; BF = 0.80; BI = 147
FOLDS = C.TRAIN_CONFIG["best_it_folds"]
LAM = 0.9


def train_members(times, X, pred_load, actual, feat_cols, train_mask, extra_trees=False):
    y_dir = actual; y_res = actual - pred_load
    Xtr = X[train_mask][feat_cols]; wtr = tw(times, train_mask, AW)
    PP = dict(learning_rate=LR, num_leaves=NL, min_data_in_leaf=MDL, lambda_l2=L2,
              feature_fraction=FF, bagging_fraction=BF, bagging_freq=1, verbose=-1, force_col_wise=True,
              extra_trees=extra_trees)
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


def oof_predict(times, X, pred_load, actual, feat_cols, usable, extra_trees=False, agg="median"):
    oof = pd.Series(np.nan, index=times)
    for te, vs, ve in FOLDS:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        M = train_members(times, X, pred_load, actual, feat_cols, ftr, extra_trees=extra_trees)
        ens = np.median(M, axis=0) if agg == "median" else np.mean(M, axis=0)
        oof[fva] = ens[fva]
    return oof


def hour_bias(times, oof, actual, usable, lam, pv):
    oof_mask = usable & oof.notna().values
    pred = np.clip(pv + lam * (oof.values - pv), 0, None)
    resid = pred - actual.values
    hb = np.zeros(24, dtype=float)
    h_all = pd.DatetimeIndex(times).hour.values
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum():
            hb[h] = float(np.mean(resid[m]))
    return hb


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]
    pv_full = pred_load.values
    h_all = pd.DatetimeIndex(times).hour.values.astype(int)

    for et, agg, label in [(False, "median", "baseline(median)"),
                            (False, "mean", "mean"),
                            (True, "median", "extra_trees+median"),
                            (True, "mean", "extra_trees+mean")]:
        print(f"\n== {label} (extra_trees={et}, agg={agg}) ==")
        M = train_members(times, X, pred_load, actual, feat_cols, full_mask, extra_trees=et)
        ens = np.median(M, axis=0) if agg == "median" else np.mean(M, axis=0)
        base = np.clip(pv_full + LAM * (ens - pv_full), 0, None)
        print(f"  no-bias:    VAL MAE={_mae(base[vm], av):.2f}")
        oof = oof_predict(times, X, pred_load, actual, feat_cols, full_mask, extra_trees=et, agg=agg)
        hb = hour_bias(times, oof, actual, full_mask, LAM, pv_full)
        corr = np.clip(base - hb[h_all], 0, None)
        print(f"  +per-hour:  VAL MAE={_mae(corr[vm], av):.2f}")


if __name__ == "__main__":
    main()
