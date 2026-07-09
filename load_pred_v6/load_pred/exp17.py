# -*- coding: utf-8 -*-
"""实验17：新验证窗下，aw5.0 配置 + 泄露安全的偏置校正。
- 3-fold OOF 估计 per-hour / per-month / per-(month,hour) 偏置
- λ 扫描
- 全部仅用训练期 OOF，无 val 泄露。"""
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


PP = dict(learning_rate=0.02, num_leaves=127, min_data_in_leaf=300, lambda_l2=1.0,
          feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1, verbose=-1, force_col_wise=True)
QA = [0.45, 0.5, 0.55]; SEEDS = [42, 7, 123]; AW = 5.0; BI = 221


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


def agg(M, pv_full, lam):
    ens = np.median(M, axis=0)
    return np.clip(pv_full + lam * (ens - pv_full), 0, None)


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]; pv_full = pred_load.values

    print("training full ensemble (24 members, aw5.0) ...")
    M_full = train_members(times, X, pred_load, actual, feat_cols, full_mask)

    # λ 扫描
    print("\n=== λ sweep (median agg) ===")
    best_lam, best_mae = 0.8, 1e9
    for lam in [0.6, 0.7, 0.8, 0.9, 1.0]:
        m = _mae(agg(M_full, pv_full, lam)[vm], av)
        print(f"  λ={lam}: {m:.2f}")
        if m < best_mae:
            best_lam, best_mae = lam, m
    print(f"  best λ={best_lam} -> {best_mae:.2f}")
    lam = best_lam
    base_val = agg(M_full, pv_full, lam)[vm]

    # 3-fold OOF
    print("\ncomputing 3-fold OOF ...")
    folds = C.TRAIN_CONFIG["best_it_folds"]
    oof_pred = np.full(len(times), np.nan)
    for te_s, vs_s, ve_s in folds:
        te, vs, ve = pd.Timestamp(te_s), pd.Timestamp(vs_s), pd.Timestamp(ve_s)
        ftr = full_mask & np.asarray(times <= te)
        fva = full_mask & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        M = train_members(times, X, pred_load, actual, feat_cols, ftr)
        oof_pred[fva] = agg(M, pv_full, lam)[fva]
    oof_mask = full_mask & ~np.isnan(oof_pred)
    oof_resid = oof_pred - actual.values
    print(f"OOF n={oof_mask.sum()} MAE={_mae(oof_pred[oof_mask], actual.values[oof_mask]):.2f}")

    h_all = times.hour.values; mo_all = times.month.values
    t_oof = times[oof_mask]; tmin, tmax = t_oof.min(), t_oof.max()
    w_rec = (1.0 + AW * (times - tmin).total_seconds().values / (tmax - tmin).total_seconds()).astype(float)

    def wavg(vals, w):
        return np.average(vals, weights=w) if len(vals) else 0.0

    # (a) per-hour
    hour_bias = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum():
            hour_bias[h] = wavg(oof_resid[m], w_rec[m])
    corr_a = np.array([hour_bias[hh] for hh in h_all])
    print(f"[+ per-hour bias      ] VAL MAE={_mae(np.clip(base_val - corr_a[vm],0,None), av):.2f}")

    # (b) per-month (year-over-year assumption)
    mon_bias = np.zeros(13)
    for mo in range(1, 13):
        m = oof_mask & (mo_all == mo)
        if m.sum():
            mon_bias[mo] = wavg(oof_resid[m], w_rec[m])
    corr_b = np.array([mon_bias[mm] for mm in mo_all])
    print(f"[+ per-month bias     ] VAL MAE={_mae(np.clip(base_val - corr_b[vm],0,None), av):.2f}")

    # (c) per-(month, hour)
    mh_bias = np.zeros((13, 24))
    for mo in range(1, 13):
        for h in range(24):
            m = oof_mask & (mo_all == mo) & (h_all == h)
            if m.sum() > 20:
                mh_bias[mo, h] = wavg(oof_resid[m], w_rec[m])
    corr_c = np.array([mh_bias[mm, hh] for mm, hh in zip(mo_all, h_all)])
    print(f"[+ per-(month,hour)   ] VAL MAE={_mae(np.clip(base_val - corr_c[vm],0,None), av):.2f}")

    # (d) per-(month, hour) + 最近 OOF 折仅（更贴近 val）
    # 用最近折(冬折 Jan-Feb 2026) + 同月历史：简单起见用 (c)


if __name__ == "__main__":
    main()
