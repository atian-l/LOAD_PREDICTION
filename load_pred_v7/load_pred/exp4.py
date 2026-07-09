# -*- coding: utf-8 -*-
"""
实验4：更大、更多样化的集成 + 收缩(λ)。
  - 多目标：regression / quantile(0.5) / huber / regression_l1 / quantile(0.45,0.55)
  - direct + residual
  - 多种子 + 结构多样性（num_leaves / feature_fraction）
  - 收缩：final = pred_load + λ*(ensemble - pred_load)，λ 在 spring25 fold 上选。
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


def main():
    print("building features ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2023-02-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    usable = (times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()
    y_dir = actual; y_res = actual - pred_load

    # spring25 fold 选 best_iter
    te = pd.Timestamp("2025-02-28"); vs = pd.Timestamp("2025-03-01"); ve = pd.Timestamp("2025-05-31")
    ftr = usable & (times <= te); fva = usable & (times >= vs) & (times <= ve)
    base = dict(metric=["mae", "rmse"], learning_rate=0.02, min_data_in_leaf=300,
                lambda_l2=1.0, bagging_freq=1, verbose=-1, force_col_wise=True,
                feature_fraction=0.80, bagging_fraction=0.80,
                num_iterations=8000, early_stopping_rounds=300)
    pp = dict(base); pp.pop("num_iterations"); pp.pop("early_stopping_rounds")
    dtr = lgb.Dataset(X[ftr][feat_cols], label=y_dir[ftr].values, weight=time_weights(times, ftr, 1.0))
    dva = lgb.Dataset(X[fva][feat_cols], label=y_dir[fva].values, reference=dtr)
    ev = {}
    b0 = lgb.train({**pp, "objective": "regression", "num_leaves": 127, "seed": 42}, dtr,
                   num_boost_round=base["num_iterations"], valid_sets=[dva], valid_names=["va"],
                   callbacks=[lgb.early_stopping(base["early_stopping_rounds"], verbose=False,
                                                 first_metric_only=True), lgb.record_evaluation(ev)])
    best_it = max(b0.best_iteration, 80)
    print(f"best_it(spring25)={best_it}")

    alpha_w = 1.0
    Xtr = X[usable][feat_cols]; wtr = time_weights(times, usable, alpha_w)

    # 多样化成员配置
    member_specs = []
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[usable]
        for obj, qa in [("regression", None), ("quantile", 0.5),
                        ("huber", None), ("regression_l1", None),
                        ("quantile", 0.45), ("quantile", 0.55)]:
            for nl, ff, s in [(127, 0.8, 42), (127, 0.7, 7), (255, 0.8, 123),
                              (63, 0.9, 2024), (191, 0.75, 99)]:
                member_specs.append((residual, ytr, obj, qa, nl, ff, s))

    raw_sum = np.zeros(len(times))
    for residual, ytr, obj, qa, nl, ff, s in member_specs:
        p = dict(pp, objective=obj, num_leaves=nl, feature_fraction=ff, seed=s,
                 bagging_fraction=0.8)
        if obj == "quantile":
            p["alpha"] = qa
        dtr = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        bst = lgb.train(p, dtr, num_boost_round=int(best_it))
        raw = bst.predict(X[feat_cols])
        raw_sum += (pred_load.values + raw) if residual else raw
    ens = raw_sum / len(member_specs)
    print(f"members={len(member_specs)}")

    # 收缩 λ：在 spring25 fold 上选
    pred_load_arr = pred_load.values
    fva_arr = fva.values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    actual_arr = actual.values
    best_lam, best_spring = None, None
    for lam in np.arange(0.0, 1.51, 0.05):
        pred = np.clip(pred_load_arr + lam * (ens - pred_load_arr), 0, None)
        mae_s, _, _ = _m(pred[fva_arr], actual_arr[fva_arr])
        if best_spring is None or mae_s < best_spring:
            best_spring = mae_s; best_lam = lam
    print(f"best λ (spring25)={best_lam:.2f}  springMAE@λ={best_spring:.1f}")

    pred = np.clip(pred_load_arr + best_lam * (ens - pred_load_arr), 0, None)
    mae_v, r2_v, bias_v = _m(pred[vm], actual_arr[vm])
    mae_v0, r2_v0, bias_v0 = _m(ens[vm], actual_arr[vm])
    print(f"ensemble(λ=1)   VAL MAE={mae_v0:.2f}  R2={r2_v0:.4f}  Bias={bias_v0:.0f}")
    print(f"ensemble(λ={best_lam:.2f}) VAL MAE={mae_v:.2f}  R2={r2_v:.4f}  Bias={bias_v:.0f}")


if __name__ == "__main__":
    main()
