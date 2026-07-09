# -*- coding: utf-8 -*-
"""实验13：最终配置搜索。每配置训练 40 成员(3fold best_it)，扫描 mean/median × λ。
目标：突破 1500。"""
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


def best_it_3fold(times, X, y_dir, feat_cols, usable, alpha_w, pp):
    folds = C.TRAIN_CONFIG["best_it_folds"]
    bests = []
    for te_s, vs_s, ve_s in folds:
        te, vs, ve = pd.Timestamp(te_s), pd.Timestamp(vs_s), pd.Timestamp(ve_s)
        ftr = usable & (times <= te); fva = usable & (times >= vs) & (times <= ve)
        if ftr.sum() < 1000 or fva.sum() < 500:
            continue
        dtr = lgb.Dataset(X[ftr][feat_cols], label=y_dir[ftr].values, weight=tw(times, ftr, alpha_w))
        dva = lgb.Dataset(X[fva][feat_cols], label=y_dir[fva].values, reference=dtr)
        bst = lgb.train({**pp, "objective": "regression", "seed": 42, "metric": ["mae", "rmse"]}, dtr,
                        num_boost_round=C.TRAIN_CONFIG["best_it_num_iterations"], valid_sets=[dva], valid_names=["va"],
                        callbacks=[lgb.early_stopping(C.TRAIN_CONFIG["best_it_early_stopping"], verbose=False,
                                                      first_metric_only=True), lgb.record_evaluation({})])
        bests.append(max(bst.best_iteration, 80))
    return int(np.mean(bests)) if bests else 290


def run_config(name, times, X, pred_load, actual, feat_cols, ts0_s, alpha_w, qa, seeds):
    ts0 = pd.Timestamp(ts0_s); tr_end = pd.Timestamp(C.TRAIN_END)
    usable = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    y_dir = actual; y_res = actual - pred_load
    pp = dict(learning_rate=0.02, num_leaves=127, min_data_in_leaf=300, lambda_l2=1.0,
              feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1, verbose=-1, force_col_wise=True)
    bi = best_it_3fold(times, X, y_dir, feat_cols, usable, alpha_w, pp)
    Xtr = X[usable][feat_cols]; wtr = tw(times, usable, alpha_w)
    member_preds = []
    objs = [("regression", None)] + [("quantile", q) for q in qa]
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[usable]
        d = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj, qa_ in objs:
            for s in seeds:
                p = dict(pp, objective=obj, seed=s)
                if obj == "quantile":
                    p["alpha"] = qa_
                bst = lgb.train(p, d, num_boost_round=int(bi))
                raw = bst.predict(X[feat_cols])
                member_preds.append((pred_load.values + raw) if residual else raw)
    M = np.array(member_preds)
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]; pv_full = pred_load.values
    ens_mean = M.mean(axis=0); ens_med = np.median(M, axis=0)
    print(f"\n[{name}] best_it={bi} n_members={M.shape[0]}")
    print(f"  mean  λ=1.0 : {_mae(np.clip(ens_mean[vm],0,None), av):.2f}")
    print(f"  median λ=1.0 : {_mae(np.clip(ens_med[vm],0,None), av):.2f}")
    best = (1e9, None)
    for lam in [0.70, 0.75, 0.80, 0.85, 0.90]:
        pred = np.clip(pv_full[vm] + lam * (ens_med[vm] - pv_full[vm]), 0, None)
        m = _mae(pred, av)
        if m < best[0]:
            best = (m, lam)
        print(f"  median λ={lam}: {m:.2f}")
    print(f"  >> BEST median λ={best[1]} MAE={best[0]:.2f}")
    return best[0]


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    s5 = [42, 7, 123, 2024, 99]
    # 配置：(name, train_start, alpha_w, quantile_alphas)
    configs = [
        ("cur+med",      "2023-02-01", 1.5, [0.5]),                 # 当前配置 + median/λ
        ("cur_q3_aw25",  "2023-02-01", 2.5, [0.45, 0.5, 0.55]),     # + q3 + aw2.5
        ("ts2024_q3_aw25","2024-01-01",2.5, [0.45, 0.5, 0.55]),     # 短窗口
    ]
    results = {}
    for name, ts, aw, qa in configs:
        results[name] = run_config(name, times, X, pred_load, actual, feat_cols, ts, aw, qa, s5)
    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k:16s}: {v:.2f}  {'PASS' if v < 1500 else 'FAIL'}")


if __name__ == "__main__":
    main()
