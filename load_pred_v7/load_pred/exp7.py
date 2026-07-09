# -*- coding: utf-8 -*-
"""实验7：winning 集成超参 sweep（min_data/leaves/LR）。spring25 选 best_it。"""
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


def time_weights(times, mask, alpha):
    t = times[mask]; tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        return np.ones(len(t))
    return (1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()).values.astype(float)


def _m(p, a):
    err = p - a
    return np.mean(np.abs(err)), (1 - np.sum(err**2) / np.sum((a - a.mean())**2)), np.mean(err)


def run(times, X, pred_load, actual, feat_cols, usable, alpha_w, seeds, lr, nl, md, lam):
    y_dir = actual; y_res = actual - pred_load
    base = dict(metric=["mae", "rmse"], learning_rate=lr, num_leaves=nl,
                min_data_in_leaf=md, lambda_l2=1.0, feature_fraction=0.80,
                bagging_fraction=0.80, bagging_freq=1, verbose=-1, force_col_wise=True,
                num_iterations=8000, early_stopping_rounds=300)
    pp = dict(base); pp.pop("num_iterations"); pp.pop("early_stopping_rounds")
    # spring25 best_it
    te = pd.Timestamp("2025-02-28"); vs = pd.Timestamp("2025-03-01"); ve = pd.Timestamp("2025-05-31")
    ftr = usable & (times <= te); fva = usable & (times >= vs) & (times <= ve)
    dtr = lgb.Dataset(X[ftr][feat_cols], label=y_dir[ftr].values, weight=time_weights(times, ftr, alpha_w))
    dva = lgb.Dataset(X[fva][feat_cols], label=y_dir[fva].values, reference=dtr)
    ev = {}
    b0 = lgb.train({**pp, "objective": "regression", "seed": 42}, dtr,
                   num_boost_round=base["num_iterations"], valid_sets=[dva], valid_names=["va"],
                   callbacks=[lgb.early_stopping(base["early_stopping_rounds"], verbose=False,
                                                 first_metric_only=True), lgb.record_evaluation(ev)])
    best_it = max(b0.best_iteration, 80)
    Xtr = X[usable][feat_cols]; wtr = time_weights(times, usable, alpha_w)
    raw_sum = np.zeros(len(times)); n = 0
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[usable]
        for obj, qa in [("regression", None), ("quantile", 0.5)]:
            for s in seeds:
                p = dict(pp, objective=obj, seed=s)
                if obj == "quantile":
                    p["alpha"] = qa
                bst = lgb.train(p, lgb.Dataset(Xtr, label=ytr.values, weight=wtr), num_boost_round=int(best_it))
                raw = bst.predict(X[feat_cols])
                raw_sum += (pred_load.values + raw) if residual else raw
                n += 1
    ens = raw_sum / n
    pred = np.clip(pred_load.values + lam * (ens - pred_load.values), 0, None)
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    mae, r2, bias = _m(pred[vm], actual.values[vm])
    return best_it, mae, r2, bias


def main():
    print("building features ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2023-02-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    usable = (times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()
    seeds = [42, 7, 123, 2024, 99]

    grid = [
        (0.02, 127, 300),  # baseline
        (0.02, 159, 200),
        (0.02, 95, 400),
        (0.02, 255, 200),
        (0.02, 127, 500),
        (0.015, 127, 300),
        (0.03, 127, 300),
        (0.02, 191, 150),
        (0.02, 63, 600),
    ]
    for lam in (1.0, 0.9):
        print(f"\n--- λ={lam} ---")
        for lr, nl, md in grid:
            best_it, mae, r2, bias = run(times, X, pred_load, actual, feat_cols, usable,
                                         1.5, seeds, lr, nl, md, lam)
            print(f"  lr={lr} leaves={nl} md={md}  best_it={best_it}  VAL MAE={mae:.2f}  R2={r2:.4f}  Bias={bias:.0f}")


if __name__ == "__main__":
    main()
