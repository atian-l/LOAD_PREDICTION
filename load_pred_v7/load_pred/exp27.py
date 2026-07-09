# -*- coding: utf-8 -*-
"""实验27：LightGBM linear_tree=True（叶节点线性回归，可外推）。
+ per-(hour,dow) 在干净 baseline 上测试。"""
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


QA = [0.45, 0.5, 0.55]; SEEDS = [42, 7, 123]; AW = 5.0; BI = 221; LAM = 0.8


def train_members(times, X, pred_load, actual, feat_cols, train_mask, linear_tree=False):
    y_dir = actual; y_res = actual - pred_load
    Xtr = X[train_mask][feat_cols]; wtr = tw(times, train_mask, AW)
    member_preds = []
    PP_base = dict(learning_rate=0.05, num_leaves=31, min_data_in_leaf=500, lambda_l2=1.0,
                   feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1, verbose=-1, force_col_wise=True)
    if linear_tree:
        PP_base["linear_tree"] = True
        PP_base["num_leaves"] = 15
        PP_base["min_data_in_leaf"] = 1000
    else:
        PP_base["learning_rate"] = 0.02
        PP_base["num_leaves"] = 127
        PP_base["min_data_in_leaf"] = 300
    objs = [("regression", None)] + [("quantile", q) for q in QA]
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[train_mask]
        d = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj, qa_ in objs:
            for s in SEEDS:
                p = dict(PP_base, objective=obj, seed=s)
                if obj == "quantile":
                    p["alpha"] = qa_
                bst = lgb.train(p, d, num_boost_round=int(BI if not linear_tree else 400))
                raw = bst.predict(X[feat_cols])
                member_preds.append((pred_load.values + raw) if residual else raw)
    return np.array(member_preds)


def ens_of(M):
    return np.median(M, axis=0)


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]; pv_full = pred_load.values
    mo_all = times.month.values; h_all = times.hour.values; dow_all = times.dayofweek.values

    # baseline (tree)
    print("training baseline (tree) ...")
    M_t = train_members(times, X, pred_load, actual, feat_cols, full_mask, linear_tree=False)
    base_t = np.clip(pv_full + LAM*(ens_of(M_t) - pv_full), 0, None)
    print(f"[tree baseline] VAL MAE={_mae(base_t[vm], av):.2f}")

    # linear_tree
    print("training linear_tree ...")
    try:
        M_l = train_members(times, X, pred_load, actual, feat_cols, full_mask, linear_tree=True)
        base_l = np.clip(pv_full + LAM*(ens_of(M_l) - pv_full), 0, None)
        print(f"[linear_tree λ0.8] VAL MAE={_mae(base_l[vm], av):.2f}")
        for lam in [0.5, 0.6, 0.7, 0.9, 1.0]:
            p = np.clip(pv_full + lam*(ens_of(M_l) - pv_full), 0, None)
            print(f"[linear_tree λ{lam}] VAL MAE={_mae(p[vm], av):.2f}")
        for mo in [3,4,5,6]:
            mm = (mo_all[vm] == mo)
            if mm.sum():
                print(f"  mo={mo}: tree={_mae(base_t[vm][mm], av[mm]):.2f} lin={_mae(base_l[vm][mm], av[mm]):.2f} lin_bias={np.mean(base_l[vm][mm]-av[mm]):.0f}")
        # blend tree & linear_tree
        for beta in [0.3, 0.5, 0.7]:
            blend = beta*base_l + (1-beta)*base_t
            print(f"[blend lin{beta}+tree{1-beta:.1f}] VAL MAE={_mae(blend[vm], av):.2f}")
    except Exception as e:
        print(f"linear_tree failed: {e}")

    # per-(hour,dow) on tree baseline (clean)
    print("\n== per-(hour,dow) on tree baseline ==")
    folds = C.TRAIN_CONFIG["best_it_folds"]
    oof_ens = np.full(len(times), np.nan)
    for te_s, vs_s, ve_s in folds:
        te, vs, ve = pd.Timestamp(te_s), pd.Timestamp(vs_s), pd.Timestamp(ve_s)
        ftr = full_mask & np.asarray(times <= te)
        fva = full_mask & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0: continue
        Mf = train_members(times, X, pred_load, actual, feat_cols, ftr, linear_tree=False)
        oof_ens[fva] = ens_of(Mf)[fva]
    oof_mask = full_mask & ~np.isnan(oof_ens)
    oof_resid = (pv_full + LAM*(oof_ens - pv_full)) - actual.values

    # per-hour
    hb = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum(): hb[h] = np.average(oof_resid[m])
    corr_h = np.array([hb[h_all[i]] for i in range(len(times))])
    print(f"[+ per-hour] VAL MAE={_mae(np.clip(base_t[vm]-corr_h[vm],0,None), av):.2f}")

    # per-(hour, dow)
    hdb = np.zeros((7, 24))
    for d in range(7):
        for h in range(24):
            m = oof_mask & (dow_all == d) & (h_all == h)
            if m.sum(): hdb[d, h] = np.average(oof_resid[m])
    corr_hd = np.array([hdb[dow_all[i], h_all[i]] for i in range(len(times))])
    print(f"[+ per-(h,dow)] VAL MAE={_mae(np.clip(base_t[vm]-corr_hd[vm],0,None), av):.2f}")

    # per-(hour, is_weekend)
    iwe = (dow_all >= 5).astype(int)
    hwb = np.zeros((2, 24))
    for we in range(2):
        for h in range(24):
            m = oof_mask & (iwe == we) & (h_all == h)
            if m.sum(): hwb[we, h] = np.average(oof_resid[m])
    corr_hwe = np.array([hwb[iwe[i], h_all[i]] for i in range(len(times))])
    print(f"[+ per-(h,weekend)] VAL MAE={_mae(np.clip(base_t[vm]-corr_hwe[vm],0,None), av):.2f}")


if __name__ == "__main__":
    main()
