# -*- coding: utf-8 -*-
"""实验12：聚合/收缩扫描。训练一次(aw2.5_q3, ts2024-01, 3-fold best_it, 40 成员)，
保存每成员预测，然后扫描 mean/median/trimmed/λ收缩/pred_load 均值回归。"""
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


def determine_best_it(times, X, y_dir, feat_cols, usable, alpha_w, pp):
    """3-fold walk-forward avg best_it (复用 train.py 的折)。"""
    folds = C.TRAIN_CONFIG["best_it_folds"]
    bests = []
    for te_s, vs_s, ve_s in folds:
        te, vs, ve = pd.Timestamp(te_s), pd.Timestamp(vs_s), pd.Timestamp(ve_s)
        ftr = usable & (times <= te); fva = usable & (times >= vs) & (times <= ve)
        if ftr.sum() < 1000 or fva.sum() < 500:
            continue
        dtr = lgb.Dataset(X[ftr][feat_cols], label=y_dir[ftr].values, weight=tw(times, ftr, alpha_w))
        dva = lgb.Dataset(X[fva][feat_cols], label=y_dir[fva].values, reference=dtr)
        ev = {}
        bst = lgb.train({**pp, "objective": "regression", "seed": 42, "metric": ["mae", "rmse"]}, dtr,
                        num_boost_round=C.TRAIN_CONFIG["best_it_num_iterations"], valid_sets=[dva], valid_names=["va"],
                        callbacks=[lgb.early_stopping(C.TRAIN_CONFIG["best_it_early_stopping"], verbose=False,
                                                      first_metric_only=True), lgb.record_evaluation(ev)])
        bests.append(max(bst.best_iteration, 80))
    return int(np.mean(bests)) if bests else 290


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    usable = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    aw = 2.5; qa = [0.45, 0.5, 0.55]; seeds = [42, 7, 123, 2024, 99]
    lr = 0.02; nl = 127; md = 300
    y_dir = actual; y_res = actual - pred_load
    pp = dict(learning_rate=lr, num_leaves=nl, min_data_in_leaf=md, lambda_l2=1.0,
              feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1, verbose=-1, force_col_wise=True)

    print("determining best_it (3-fold) ...")
    best_it = determine_best_it(times, X, y_dir, feat_cols, usable, aw, pp)
    print(f"best_it={best_it}")

    Xtr = X[usable][feat_cols]; wtr = tw(times, usable, aw)
    member_preds = []  # 每成员对全时间索引的预测
    objs = [("regression", None)] + [("quantile", q) for q in qa]
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[usable]
        d = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj, qa_ in objs:
            for s in seeds:
                p = dict(pp, objective=obj, seed=s)
                if obj == "quantile":
                    p["alpha"] = qa_
                bst = lgb.train(p, d, num_boost_round=int(best_it))
                raw = bst.predict(X[feat_cols])
                member_preds.append((pred_load.values + raw) if residual else raw)
    M = np.array(member_preds)  # (n_members, T)
    print(f"trained {M.shape[0]} members")
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]; pv = pred_load.values[vm]; Mv = M[:, vm]

    pl_mean_672 = pred_load.rolling(672, min_periods=96).mean().values  # 近期均值(用作回归目标)
    pv_full = pred_load.values

    def report(name, pred):
        pred_v = np.clip(pred[vm], 0, None)
        print(f"[{name:22s}] VAL MAE={_mae(pred_v, av):.2f}")

    print("\n=== aggregation sweep ===")
    report("mean", M.mean(axis=0))
    report("median", np.median(M, axis=0))
    report("trim20%_mean", np.sort(M, axis=0)[M.shape[0]//5:-M.shape[0]//5].mean(axis=0))
    report("trim10%_mean", np.sort(M, axis=0)[M.shape[0]//10:-M.shape[0]//10].mean(axis=0))

    print("\n=== final shrinkage lambda (mean ens -> pred_load) ===")
    ens_mean = M.mean(axis=0)
    for lam in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        pred = pv_full + lam * (ens_mean - pv_full)
        report(f"lam={lam}", pred)

    print("\n=== final shrinkage lambda (median ens -> pred_load) ===")
    ens_med = np.median(M, axis=0)
    for lam in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        pred = pv_full + lam * (ens_med - pv_full)
        report(f"med_lam={lam}", pred)

    print("\n=== pred_load mean-regression on ensemble ===")
    # final = ens - k*(pred_load - pl_mean_672); 扫 k
    for k in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        pred = ens_mean - k * (pv_full - pl_mean_672)
        report(f"regress_k={k}", pred)

    print("\n=== median + pred_load mean-regression ===")
    for k in [0.10, 0.15, 0.20, 0.25]:
        pred = ens_med - k * (pv_full - pl_mean_672)
        report(f"med_regress_k={k}", pred)


if __name__ == "__main__":
    main()
