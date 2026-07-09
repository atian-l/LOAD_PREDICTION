# -*- coding: utf-8 -*-
"""实验10：午间(10-15)样本加权 + aw2.5_q3 最优配置，spring25 单折 best_it。
牺牲 off-peak 富余精度换取午间(误差主源)精度。"""
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


def tw(times, mask, alpha, midday_w, mid_lo=10, mid_hi=15):
    """recency 权重 * 午间小时权重。"""
    t = times[mask]; tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        rec = np.ones(len(t))
    else:
        rec = (1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()).values.astype(float)
    h = t.hour.values
    hw = np.where((h >= mid_lo) & (h <= mid_hi), midday_w, 1.0)
    return rec * hw


def _m(p, a):
    err = p - a
    return (np.mean(np.abs(err)),
            1 - np.sum(err ** 2) / np.sum((a - a.mean()) ** 2),
            np.mean(err))


def run(times, X, pred_load, actual, feat_cols, usable, alpha_w, seeds, qalphas,
        lr, nl, md, midday_w, mid_lo=10, mid_hi=15):
    y_dir = actual; y_res = actual - pred_load
    base = dict(metric=["mae", "rmse"], learning_rate=lr, num_leaves=nl, min_data_in_leaf=md,
                lambda_l2=1.0, feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1,
                verbose=-1, force_col_wise=True, num_iterations=8000, early_stopping_rounds=300)
    pp = dict(base); pp.pop("num_iterations"); pp.pop("early_stopping_rounds")
    te = pd.Timestamp("2025-02-28"); vs = pd.Timestamp("2025-03-01"); ve = pd.Timestamp("2025-05-31")
    ftr = usable & (times <= te); fva = usable & (times >= vs) & (times <= ve)
    dtr = lgb.Dataset(X[ftr][feat_cols], label=y_dir[ftr].values,
                      weight=tw(times, ftr, alpha_w, midday_w, mid_lo, mid_hi))
    dva = lgb.Dataset(X[fva][feat_cols], label=y_dir[fva].values, reference=dtr)
    ev = {}
    b0 = lgb.train({**pp, "objective": "regression", "seed": 42}, dtr,
                   num_boost_round=base["num_iterations"], valid_sets=[dva], valid_names=["va"],
                   callbacks=[lgb.early_stopping(base["early_stopping_rounds"], verbose=False, first_metric_only=True),
                              lgb.record_evaluation(ev)])
    best_it = max(b0.best_iteration, 80)
    Xtr = X[usable][feat_cols]; wtr = tw(times, usable, alpha_w, midday_w, mid_lo, mid_hi)
    raw_sum = np.zeros(len(times)); n = 0
    objs = [("regression", None)] + [("quantile", q) for q in qalphas]
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[usable]
        d = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj, qa in objs:
            for s in seeds:
                p = dict(pp, objective=obj, seed=s)
                if obj == "quantile":
                    p["alpha"] = qa
                bst = lgb.train(p, d, num_boost_round=int(best_it))
                raw = bst.predict(X[feat_cols])
                raw_sum += (pred_load.values + raw) if residual else raw
                n += 1
    ens = np.clip(raw_sum / n, 0, None)
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    mae, r2, bias = _m(ens[vm], actual.values[vm])
    # 分时段 MAE
    hv = times[vm].hour.values
    mid_vm = (hv >= mid_lo) & (hv <= mid_hi)
    mid_mae = np.mean(np.abs(ens[vm][mid_vm] - actual.values[vm][mid_vm])) if mid_vm.any() else float("nan")
    off_mae = np.mean(np.abs(ens[vm][~mid_vm] - actual.values[vm][~mid_vm])) if (~mid_vm).any() else float("nan")
    return best_it, mae, r2, bias, n, mid_mae, off_mae


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2023-02-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    usable = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    s5 = [42, 7, 123, 2024, 99]
    aw = 2.5; qa = [0.45, 0.5, 0.55]
    cands = [
        ("mid1.0_base",   1.0, 10, 15),   # 无午间加权(对照)
        ("mid1.5",        1.5, 10, 15),
        ("mid2.0",        2.0, 10, 15),
        ("mid2.5",        2.5, 10, 15),
        ("mid3.0",        3.0, 10, 15),
        ("mid2.0_11-14",  2.0, 11, 14),   # 更聚焦峰值午间
        ("mid2.0_9-16",   2.0, 9, 16),    # 更宽
    ]
    for name, mw, lo, hi in cands:
        best_it, mae, r2, bias, n, mid_mae, off_mae = run(
            times, X, pred_load, actual, feat_cols, usable, aw, s5, qa, 0.02, 127, 300, mw, lo, hi)
        print(f"[{name:16s}] best_it={best_it} n={n}  VAL MAE={mae:.2f}  mid={mid_mae:.0f} off={off_mae:.0f}  Bias={bias:.0f}")


if __name__ == "__main__":
    main()
