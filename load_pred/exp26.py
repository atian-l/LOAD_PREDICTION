# -*- coding: utf-8 -*-
"""实验26：(a) 加入 day_level_features（当前未启用！）；(b) per-(hour,dow) 校正；
(c) 预测截尾。features.py 里 day_level_features 定义了但 build_features 没调用。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from . import data_loader as dl
from . import features as F


def build(use_day_level=False):
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)
    X = F.build_features(times, pred_load, weather)
    if use_day_level:
        dlf = F.day_level_features(times, pred_load, weather)
        X = pd.concat([X, dlf], axis=1)
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


def ens_of(M):
    return np.median(M, axis=0)


def main():
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    vm = ((lambda t: ((t >= C.VAL_START) & (t <= C.VAL_END))))

    # (a) baseline vs +day_level
    print("== (a) day_level features ==")
    times, X, pred_load, actual = build(use_day_level=False)
    feat_cols = list(X.columns)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm_arr = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm_arr]; pv_full = pred_load.values
    M = train_members(times, X, pred_load, actual, feat_cols, full_mask)
    base = np.clip(pv_full + LAM*(ens_of(M) - pv_full), 0, None)
    print(f"[baseline          ] VAL MAE={_mae(base[vm_arr], av):.2f}  n_feat={len(feat_cols)}")

    times2, X2, pred_load2, actual2 = build(use_day_level=True)
    feat_cols2 = list(X2.columns)
    full_mask2 = ((times2 >= ts0) & (times2 <= tr_end) & pred_load2.notna() & actual2.notna()).values
    M2 = train_members(times2, X2, pred_load2, actual2, feat_cols2, full_mask2)
    base2 = np.clip(pv_full + LAM*(ens_of(M2) - pv_full), 0, None)
    print(f"[+ day_level feats ] VAL MAE={_mae(base2[vm_arr], av):.2f}  n_feat={len(feat_cols2)}")
    for mo in [3,4,5,6]:
        mm = (times.month.values[vm_arr] == mo)
        if mm.sum():
            print(f"  mo={mo}: base={_mae(base[vm_arr][mm], av[mm]):.2f} +dl={_mae(base2[vm_arr][mm], av[mm]):.2f}")

    # 用更好的基线（含 day_level）继续
    base_best = base2
    ens_best = ens_of(M2)

    # (b) per-(hour,dow) 校正 —— 需要 OOF
    print("\n== (b) per-(hour,dow) correction ==")
    folds = C.TRAIN_CONFIG["best_it_folds"]
    oof_ens = np.full(len(times2), np.nan)
    for te_s, vs_s, ve_s in folds:
        te, vs, ve = pd.Timestamp(te_s), pd.Timestamp(vs_s), pd.Timestamp(ve_s)
        ftr = full_mask2 & np.asarray(times2 <= te)
        fva = full_mask2 & np.asarray(times2 >= vs) & np.asarray(times2 <= ve)
        if fva.sum() == 0: continue
        Mf = train_members(times2, X2, pred_load2, actual2, feat_cols2, ftr)
        oof_ens[fva] = ens_of(Mf)[fva]
    oof_mask = full_mask2 & ~np.isnan(oof_ens)
    oof_resid = (pv_full + LAM*(oof_ens - pv_full)) - actual2.values
    h_all = times2.hour.values; dow_all = times2.dayofweek.values; mo_all = times2.month.values

    # per-hour
    hb = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum(): hb[h] = np.average(oof_resid[m])
    corr_h = np.array([hb[h_all[i]] for i in range(len(times2))])
    print(f"[+ per-hour        ] VAL MAE={_mae(np.clip(base_best[vm_arr]-corr_h[vm_arr],0,None), av):.2f}")

    # per-(hour, dow)
    hdb = np.zeros((7, 24))
    for d in range(7):
        for h in range(24):
            m = oof_mask & (dow_all == d) & (h_all == h)
            if m.sum(): hdb[d, h] = np.average(oof_resid[m])
    corr_hd = np.array([hdb[dow_all[i], h_all[i]] for i in range(len(times2))])
    print(f"[+ per-(h,dow)     ] VAL MAE={_mae(np.clip(base_best[vm_arr]-corr_hd[vm_arr],0,None), av):.2f}")

    # per-(hour, month) - 之前失败，再确认
    hmb = np.zeros((13, 24))
    for mo in range(1,13):
        for h in range(24):
            m = oof_mask & (mo_all == mo) & (h_all == h)
            if m.sum(): hmb[mo, h] = np.average(oof_resid[m])
    corr_hm = np.array([hmb[mo_all[i], h_all[i]] for i in range(len(times2))])
    print(f"[+ per-(h,mo)      ] VAL MAE={_mae(np.clip(base_best[vm_arr]-corr_hm[vm_arr],0,None), av):.2f}")

    # (c) 预测截尾
    print("\n== (c) prediction capping ==")
    err = base_best[vm_arr] - av
    print(f"  error: |e| p50={np.percentile(np.abs(err),50):.0f} p90={np.percentile(np.abs(err),90):.0f} p99={np.percentile(np.abs(err),99):.0f} max={np.abs(err).max():.0f}")
    # 截尾到 pred_load 的 [1-k, 1+k] 倍
    for k in [0.10, 0.15, 0.20, 0.30]:
        lo = pv_full * (1 - k); hi = pv_full * (1 + k)
        capped = np.clip(base_best, lo, hi)
        print(f"[cap pl±{int(k*100)}%] VAL MAE={_mae(capped[vm_arr], av):.2f}")
    # 截尾 + per-hour
    for k in [0.15, 0.20]:
        lo = pv_full * (1 - k); hi = pv_full * (1 + k)
        capped = np.clip(base_best - corr_h, lo, hi)
        print(f"[cap pl±{int(k*100)}% + per-hour] VAL MAE={_mae(capped[vm_arr], av):.2f}")

    # 组合最佳
    print("\n== best combo ==")
    # per-hour + 截尾
    lo = pv_full * 0.80; hi = pv_full * 1.20
    best = np.clip(base_best - corr_h, lo, hi)
    print(f"[day_level + per-hour + cap±20%] VAL MAE={_mae(best[vm_arr], av):.2f}")


if __name__ == "__main__":
    main()
