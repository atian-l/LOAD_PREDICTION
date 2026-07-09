# -*- coding: utf-8 -*-
"""
实验5：精炼“瘦”集成（仅 regression + quantile0.5，direct+residual），
增加种子数，并在 spring25 fold 上选收缩 λ；同时对比 3-fold 平均 best_iter。
"""
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


FOLDS = [("2023-02-01", "2025-02-28", "2025-03-01", "2025-05-31"),
         ("2023-02-01", "2025-08-31", "2025-09-01", "2025-11-30"),
         ("2023-02-01", "2025-11-30", "2025-12-01", "2026-01-31")]


def best_iter_across_folds(times, X, y, usable, feat_cols, alpha_w, base):
    pp = dict(base); pp.pop("num_iterations"); pp.pop("early_stopping_rounds")
    its = []
    for ts, te, vs, ve in FOLDS:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & (times <= te); fva = usable & (times >= vs) & (times <= ve)
        dtr = lgb.Dataset(X[ftr][feat_cols], label=y[ftr].values, weight=time_weights(times, ftr, alpha_w))
        dva = lgb.Dataset(X[fva][feat_cols], label=y[fva].values, reference=dtr)
        ev = {}
        b = lgb.train({**pp, "objective": "regression", "num_leaves": 127, "seed": 42}, dtr,
                      num_boost_round=base["num_iterations"], valid_sets=[dva], valid_names=["va"],
                      callbacks=[lgb.early_stopping(base["early_stopping_rounds"], verbose=False,
                                                    first_metric_only=True), lgb.record_evaluation(ev)])
        its.append(b.best_iteration)
    return int(np.mean(its)), its


def ensemble_predict(times, X, pred_load, actual, feat_cols, usable, alpha_w, best_it, seeds, base):
    pp = dict(base); pp.pop("num_iterations"); pp.pop("early_stopping_rounds")
    y_dir = actual; y_res = actual - pred_load
    Xtr = X[usable][feat_cols]; wtr = time_weights(times, usable, alpha_w)
    raw_sum = np.zeros(len(times))
    n = 0
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[usable]
        for obj, qa in [("regression", None), ("quantile", 0.5)]:
            for s in seeds:
                p = dict(pp, objective=obj, num_leaves=127, feature_fraction=0.8,
                         bagging_fraction=0.8, seed=s)
                if obj == "quantile":
                    p["alpha"] = qa
                bst = lgb.train(p, lgb.Dataset(Xtr, label=ytr.values, weight=wtr), num_boost_round=int(best_it))
                raw = bst.predict(X[feat_cols])
                raw_sum += (pred_load.values + raw) if residual else raw
                n += 1
    return raw_sum / n, n


def main():
    print("building features ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2023-02-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    usable = (times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()
    y_dir = actual
    base = dict(metric=["mae", "rmse"], learning_rate=0.02, min_data_in_leaf=300,
                lambda_l2=1.0, bagging_freq=1, verbose=-1, force_col_wise=True,
                feature_fraction=0.80, bagging_fraction=0.80,
                num_iterations=8000, early_stopping_rounds=300)

    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    fva = (usable & (times >= pd.Timestamp("2025-03-01")) & (times <= pd.Timestamp("2025-05-31"))).values
    actual_arr = actual.values; pred_load_arr = pred_load.values

    for alpha_w in (1.0, 1.5):
        for n_seeds in (3, 5, 8):
            seeds = [42, 7, 123, 2024, 99, 31, 256, 555][:n_seeds]
            best_it, its = best_iter_across_folds(times, X, y_dir, usable, feat_cols, alpha_w, base)
            ens, n = ensemble_predict(times, X, pred_load, actual, feat_cols, usable, alpha_w, best_it, seeds, base)
            # λ on spring25 fold
            best_lam, best_spring = None, None
            for lam in np.arange(0.0, 1.26, 0.05):
                pred = np.clip(pred_load_arr + lam * (ens - pred_load_arr), 0, None)
                mae_s, _, _ = _m(pred[fva], actual_arr[fva])
                if best_spring is None or mae_s < best_spring:
                    best_spring = mae_s; best_lam = lam
            pred_lam = np.clip(pred_load_arr + best_lam * (ens - pred_load_arr), 0, None)
            mae1, r2_1, b1 = _m(ens[vm], actual_arr[vm])
            maeL, r2_L, bL = _m(pred_lam[vm], actual_arr[vm])
            print(f"[aw={alpha_w} seeds={n_seeds}] best_it={best_it}(folds={its}) members={n}  "
                  f"λ*={best_lam:.2f}  VAL@λ1={mae1:.1f}(R2={r2_1:.4f},b={b1:.0f})  "
                  f"VAL@λ*={maeL:.1f}(R2={r2_L:.4f},b={bL:.0f})")


if __name__ == "__main__":
    main()
