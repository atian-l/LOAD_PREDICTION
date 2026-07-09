# -*- coding: utf-8 -*-
"""实验6：winning 配置 + 新日级特征，验证是否破 1500。"""
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


FOLDS = [("2025-02-28", "2025-03-01", "2025-05-31"),
         ("2025-08-31", "2025-09-01", "2025-11-30"),
         ("2025-11-30", "2025-12-01", "2026-01-31")]


def main():
    print("building features (with day-level) ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    print(f"feats={X.shape[1]}")
    ts0 = pd.Timestamp("2023-02-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    usable = (times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()
    y_dir = actual; y_res = actual - pred_load
    base = dict(metric=["mae", "rmse"], learning_rate=0.02, num_leaves=127,
                min_data_in_leaf=300, lambda_l2=1.0, feature_fraction=0.80,
                bagging_fraction=0.80, bagging_freq=1, verbose=-1, force_col_wise=True,
                num_iterations=8000, early_stopping_rounds=300)
    pp = dict(base); pp.pop("num_iterations"); pp.pop("early_stopping_rounds")
    alpha_w = 1.5

    # 3-fold best_iter
    its = []
    for te, vs, ve in FOLDS:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & (times <= te); fva = usable & (times >= vs) & (times <= ve)
        dtr = lgb.Dataset(X[ftr][feat_cols], label=y_dir[ftr].values, weight=time_weights(times, ftr, alpha_w))
        dva = lgb.Dataset(X[fva][feat_cols], label=y_dir[fva].values, reference=dtr)
        ev = {}
        b = lgb.train({**pp, "objective": "regression", "seed": 42}, dtr,
                      num_boost_round=base["num_iterations"], valid_sets=[dva], valid_names=["va"],
                      callbacks=[lgb.early_stopping(base["early_stopping_rounds"], verbose=False,
                                                    first_metric_only=True), lgb.record_evaluation(ev)])
        its.append(b.best_iteration)
    best_it = int(np.mean(its))
    print(f"best_it={best_it} folds={its}")

    Xtr = X[usable][feat_cols]; wtr = time_weights(times, usable, alpha_w)
    seeds = [42, 7, 123, 2024, 99]
    raw_sum = np.zeros(len(times)); n = 0
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[usable]
        for obj, qa in [("regression", None), ("quantile", 0.5)]:
            for s in seeds:
                p = dict(pp, objective=obj, seed=s)
                if obj == "quantile":
                    p["alpha"] = qa
                bst = lgb.train(p, lgb.Dataset(Xtr, label=ytr.values, weight=wtr), num_boost_round=best_it)
                raw = bst.predict(X[feat_cols])
                raw_sum += (pred_load.values + raw) if residual else raw
                n += 1
    ens = np.clip(raw_sum / n, 0, None)

    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    mae, r2, bias = _m(ens[vm], actual.values[vm])
    print(f"\nmembers={n}  VAL MAE={mae:.2f}  R2={r2:.4f}  Bias={bias:.0f}")

    # 特征重要度（用其中一个 direct regression 成员）
    dtr = lgb.Dataset(Xtr, label=y_dir[usable].values, weight=wtr)
    bst0 = lgb.train({**pp, "objective": "regression", "seed": 42}, dtr, num_boost_round=best_it)
    imp = pd.Series(bst0.feature_importance(importance_type="gain"), index=feat_cols).sort_values(ascending=False)
    print("top15 features:")
    for k, v in imp.head(15).items():
        print(f"  {k:28s} {v:.0f}")


if __name__ == "__main__":
    main()
