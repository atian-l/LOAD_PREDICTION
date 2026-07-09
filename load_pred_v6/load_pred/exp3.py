# -*- coding: utf-8 -*-
"""
实验3：分位数回归 + 多样化集成。
  - quantile alpha=0.5（中位数）对 MAE 最优，对重尾稳健；
  - 集成 {L2 direct, L2 residual, quantile direct, quantile residual} × 多种子；
  - 叠加时间样本权重（近期加权）。
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
    t = times[mask]
    tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        return np.ones(len(t))
    w = 1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()
    return w.values.astype(float)


def _m(p, a):
    err = p - a
    mae = np.mean(np.abs(err))
    ss_res = np.sum(err**2); ss_tot = np.sum((a - a.mean())**2)
    return mae, (1 - ss_res / ss_tot if ss_tot > 0 else np.nan), np.mean(err)


def fit_one(Xtr, ytr, wtr, feat_cols, objective, alpha, params, num_round, seed):
    p = dict(params, objective=objective, seed=seed)
    if objective == "quantile":
        p["alpha"] = alpha
    dtr = lgb.Dataset(Xtr[feat_cols], label=ytr.values, weight=wtr)
    return lgb.train(p, dtr, num_boost_round=int(num_round))


def run(times, X, pred_load, actual, feat_cols, alpha_w, seeds, num_round):
    y_dir = actual
    y_res = actual - pred_load
    ts0 = pd.Timestamp("2023-02-01")
    tr_end = pd.Timestamp(C.TRAIN_END)
    usable = (times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()

    # spring25 fold 选 best_iter（用 L2 direct 估计）
    te = pd.Timestamp("2025-02-28"); vs = pd.Timestamp("2025-03-01"); ve = pd.Timestamp("2025-05-31")
    ftr = usable & (times <= te); fva = usable & (times >= vs) & (times <= ve)
    base = dict(metric=["mae", "rmse"], learning_rate=0.02, num_leaves=127,
                min_data_in_leaf=300, lambda_l2=1.0, feature_fraction=0.80,
                bagging_fraction=0.80, bagging_freq=1, verbose=-1, force_col_wise=True,
                num_iterations=8000, early_stopping_rounds=300)
    pp = dict(base); pp.pop("num_iterations"); pp.pop("early_stopping_rounds")
    dtr = lgb.Dataset(X[ftr][feat_cols], label=y_dir[ftr].values, weight=time_weights(times, ftr, alpha_w))
    dva = lgb.Dataset(X[fva][feat_cols], label=y_dir[fva].values, reference=dtr)
    ev = {}
    b0 = lgb.train({**pp, "objective": "regression", "seed": 42}, dtr,
                   num_boost_round=base["num_iterations"], valid_sets=[dva], valid_names=["va"],
                   callbacks=[lgb.early_stopping(base["early_stopping_rounds"], verbose=False,
                                                 first_metric_only=True), lgb.record_evaluation(ev)])
    best_it = b0.best_iteration
    best_it = max(best_it, 60)  # 给分位数模型足够轮次

    Xtr = X[usable][feat_cols]; wtr = time_weights(times, usable, alpha_w)

    # 集成：4 目标 × N 种子
    members = []
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[usable]
        for obj, qa in [("regression", None), ("quantile", 0.5)]:
            for s in seeds:
                bst = fit_one(Xtr, ytr, wtr, feat_cols, obj, qa, pp, best_it, s)
                members.append((bst, residual))
    raw_sum = np.zeros(len(times))
    for bst, residual in members:
        raw = bst.predict(X[feat_cols])
        raw_sum += (pred_load.values + raw) if residual else raw
    pred = np.clip(raw_sum / len(members), 0, None)

    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    mae_v, r2_v, bias_v = _m(pred[vm], actual.values[vm])
    mae_s, r2_s, _ = _m(pred[fva.values], actual.values[fva.values])
    return {"best_it": best_it, "n_members": len(members), "val_mae": mae_v,
            "val_r2": r2_v, "val_bias": bias_v, "spring_mae": mae_s}


def main():
    print("building features ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    seeds = [42, 7, 123]

    for alpha_w in (0.0, 1.0, 2.0):
        r = run(times, X, pred_load, actual, feat_cols, alpha_w, seeds, num_round=None)
        print(f"[ensemble alpha_w={alpha_w}] best_it={r['best_it']} members={r['n_members']}  "
              f"springMAE={r['spring_mae']:.1f}  VAL MAE={r['val_mae']:.2f}  R2={r['val_r2']:.4f}  Bias={r['val_bias']:.0f}")


if __name__ == "__main__":
    main()
