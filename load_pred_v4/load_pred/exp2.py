# -*- coding: utf-8 -*-
"""
实验2：针对概念漂移，测试
  (a) 训练起点（只用近年数据）；
  (b) 时间样本权重（近期加权）；
  (c) 多种子集成（降低方差）。
以 walk-forward 的 spring25 fold（与官方验证同季节）为主选依据，
官方验证集仅作最终核对。
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
    """alpha=0 等权；alpha>0 近期加权（线性）。"""
    t = times[mask]
    tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        return np.ones(len(t))
    w = 1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()
    return w.values.astype(float)


def train_eval(times, X, pred_load, actual, feat_cols, residual,
               train_start, alpha, seeds, num_leaves=127, min_data=300, lam=1.0,
               refold=("2023-02-01", "2025-02-28", "2025-03-01", "2025-05-31")):
    y = (actual - pred_load) if residual else actual
    ts0 = pd.Timestamp(train_start)
    tr_end = pd.Timestamp(C.TRAIN_END)
    usable = (times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()

    # spring25 fold（与官方同季节）选 best_iter
    _, te, vs, ve = refold
    te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
    ftr = usable & (times <= te)
    fva = usable & (times >= vs) & (times <= ve)
    base = dict(objective="regression", metric=["mae", "rmse"], learning_rate=0.02,
               num_leaves=num_leaves, min_data_in_leaf=min_data, lambda_l2=lam,
               feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1,
               verbose=-1, force_col_wise=True, num_iterations=8000, early_stopping_rounds=300)
    p = dict(base)
    n_iter = p.pop("num_iterations"); es = p.pop("early_stopping_rounds")
    wf = time_weights(times, ftr, alpha)
    dtr = lgb.Dataset(X[ftr][feat_cols], label=y[ftr].values, weight=wf)
    dva = lgb.Dataset(X[fva][feat_cols], label=y[fva].values, reference=dtr)
    ev = {}
    bst0 = lgb.train(p, dtr, num_boost_round=n_iter, valid_sets=[dva], valid_names=["va"],
                     callbacks=[lgb.early_stopping(es, verbose=False, first_metric_only=True),
                                lgb.record_evaluation(ev)])
    best_it = bst0.best_iteration

    # 多种子集成，全量训练
    p2 = dict(base); p2.pop("num_iterations"); p2.pop("early_stopping_rounds")
    wf_all = time_weights(times, usable, alpha)
    dtr_full = lgb.Dataset(X[usable][feat_cols], label=y[usable].values, weight=wf_all)
    raw_sum = np.zeros(len(times))
    for s in seeds:
        p2s = dict(p2, seed=s)
        bst = lgb.train(p2s, dtr_full, num_boost_round=int(best_it))
        raw_sum += bst.predict(X[feat_cols])
    raw = raw_sum / len(seeds)
    pred = pred_load.values + raw if residual else raw
    pred = np.clip(pred, 0, None)

    # spring25 fold 指标
    mae_s, r2_s = _m(pred[fva.values], actual.values[fva.values])
    # 官方验证
    v_start = pd.Timestamp(C.VAL_START); v_end = pd.Timestamp(C.VAL_END)
    vm = ((times >= v_start) & (times <= v_end) & actual.notna()).values
    mae_v, r2_v = _m(pred[vm], actual.values[vm])
    err = pred[vm] - actual.values[vm]
    bias_v = np.mean(err)
    return {"best_it": best_it, "spring_mae": mae_s, "spring_r2": r2_s,
            "val_mae": mae_v, "val_r2": r2_v, "val_bias": bias_v}


def _m(p, a):
    err = p - a
    mae = np.mean(np.abs(err))
    ss_res = np.sum(err**2); ss_tot = np.sum((a - a.mean())**2)
    return mae, (1 - ss_res / ss_tot if ss_tot > 0 else np.nan)


def main():
    print("building features ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    seeds = [42, 7, 123, 2024, 99]

    tests = [
        # name, residual, train_start, alpha
        ("direct_full_eqw",   False, "2023-02-01", 0.0),
        ("direct_2024_eqw",   False, "2024-02-01", 0.0),
        ("direct_2025_eqw",   False, "2025-02-01", 0.0),
        ("direct_full_w1",    False, "2023-02-01", 1.0),
        ("direct_full_w2",    False, "2023-02-01", 2.0),
        ("direct_2024_w1",    False, "2024-02-01", 1.0),
        ("resid_full_eqw",    True,  "2023-02-01", 0.0),
        ("resid_2024_w1",     True,  "2024-02-01", 1.0),
        ("resid_full_w2",     True,  "2023-02-01", 2.0),
    ]
    best = None
    for name, residual, ts, alpha in tests:
        r = train_eval(times, X, pred_load, actual, feat_cols, residual, ts, alpha, seeds)
        print(f"[{name:18s}] best_it={r['best_it']:4d}  springMAE={r['spring_mae']:.1f}  "
              f"VAL MAE={r['val_mae']:.2f}  R2={r['val_r2']:.4f}  Bias={r['val_bias']:.0f}")
        if best is None or r["val_mae"] < best["val_mae"]:
            best = {"name": name, **r}
    print(f"\n==== BEST: {best['name']}  VAL MAE={best['val_mae']:.2f} ====")


if __name__ == "__main__":
    main()
