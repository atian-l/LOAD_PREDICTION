# -*- coding: utf-8 -*-
"""实验28：pred_load × 气象交互特征。模型有 weather×calendar 交互，但无 pred_load×weather。
若 pred_load 偏置依赖气象（如冷天预测偏高），交互可捕获。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from . import data_loader as dl
from . import features as F


def build(add_interactions=False):
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)
    X = F.build_features(times, pred_load, weather)
    if add_interactions:
        # pred_load × 气象交互
        pl = X["pred_load"]
        X["pl_x_temp"] = pl * X["temp"]
        X["pl_x_hdd"] = pl * X["hdd"]
        X["pl_x_cdd"] = pl * X["cdd"]
        X["pl_x_irrad"] = pl * X["irrad"]
        X["pl_x_wind"] = pl * X["wind"]
        # pred_load 归一化后 × 气象（避免量纲）
        pl_norm = pl / (pl.rolling(672, min_periods=96).mean())
        X["plnorm_x_temp"] = pl_norm * X["temp"]
        X["plnorm_x_hdd"] = pl_norm * X["hdd"]
        X["plnorm_x_cdd"] = pl_norm * X["cdd"]
    return times, X, pred_load, actual


def tw(times, mask, alpha):
    t = times[mask]; tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        return np.ones(len(t))
    return (1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()).values.astype(float)


def _mae(p, a):
    return np.mean(np.abs(p - a))


PP = dict(learning_rate=0.02, num_leaves=127, min_data_in_leaf=300, lambda_l2=1.0,
          feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1, verbose=-1, force_col_wise=True)
QA = [0.45, 0.5, 0.55]; SEEDS = [42, 7, 123]; AW = 5.0; BI = 221; LAM = 0.8


def train_members(times, X, pred_load, actual, feat_cols, train_mask):
    y_dir = actual; y_res = actual - pred_load
    Xtr = X[train_mask][feat_cols]; wtr = tw(times, train_mask, AW)
    member_preds = []
    objs = [("regression", None)] + [("quantile", q) for q in QA]
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[train_mask]
        d = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj, qa_ in objs:
            for s in SEEDS:
                p = dict(PP, objective=obj, seed=s)
                if obj == "quantile":
                    p["alpha"] = qa_
                bst = lgb.train(p, d, num_boost_round=int(BI))
                raw = bst.predict(X[feat_cols])
                member_preds.append((pred_load.values + raw) if residual else raw)
    return np.array(member_preds)


def main():
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)

    print("== baseline (no interactions) ==")
    times, X, pred_load, actual = build(add_interactions=False)
    feat_cols = list(X.columns)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]; pv_full = pred_load.values
    M = train_members(times, X, pred_load, actual, feat_cols, full_mask)
    base = np.clip(pv_full + LAM*(np.median(M, axis=0) - pv_full), 0, None)
    print(f"[baseline] VAL MAE={_mae(base[vm], av):.2f}  n_feat={len(feat_cols)}")

    print("\n== + pred_load×weather interactions ==")
    times2, X2, pred_load2, actual2 = build(add_interactions=True)
    feat_cols2 = list(X2.columns)
    full_mask2 = ((times2 >= ts0) & (times2 <= tr_end) & pred_load2.notna() & actual2.notna()).values
    M2 = train_members(times2, X2, pred_load2, actual2, feat_cols2, full_mask2)
    base2 = np.clip(pv_full + LAM*(np.median(M2, axis=0) - pv_full), 0, None)
    print(f"[+ interactions] VAL MAE={_mae(base2[vm], av):.2f}  n_feat={len(feat_cols2)}")
    for mo in [3,4,5,6]:
        mm = (times.month.values[vm] == mo)
        if mm.sum():
            print(f"  mo={mo}: base={_mae(base[vm][mm], av[mm]):.2f} +int={_mae(base2[vm][mm], av[mm]):.2f}")

    # + per-hour correction on the better base
    print("\n== + per-hour on each ==")
    folds = C.TRAIN_CONFIG["best_it_folds"]
    for label, Xx, plx, axx, fcm, fm in [("baseline", X, pred_load, actual, feat_cols, full_mask),
                                          ("+interact", X2, pred_load2, actual2, feat_cols2, full_mask2)]:
        oof_ens = np.full(len(times), np.nan)
        for te_s, vs_s, ve_s in folds:
            te, vs, ve = pd.Timestamp(te_s), pd.Timestamp(vs_s), pd.Timestamp(ve_s)
            ftr = fm & np.asarray(times <= te)
            fva = fm & np.asarray(times >= vs) & np.asarray(times <= ve)
            if fva.sum() == 0: continue
            Mf = train_members(times, Xx, plx, axx, fcm, ftr)
            oof_ens[fva] = np.median(Mf, axis=0)[fva]
        oof_mask = fm & ~np.isnan(oof_ens)
        oof_resid = (pv_full + LAM*(oof_ens - pv_full)) - axx.values
        h_all = times.hour.values
        hb = np.zeros(24)
        for h in range(24):
            m = oof_mask & (h_all == h)
            if m.sum(): hb[h] = np.average(oof_resid[m])
        corr_h = np.array([hb[h_all[i]] for i in range(len(times))])
        b = (base if label=="baseline" else base2)
        print(f"[{label} + per-hour] VAL MAE={_mae(np.clip(b[vm]-corr_h[vm],0,None), av):.2f}")


if __name__ == "__main__":
    main()
