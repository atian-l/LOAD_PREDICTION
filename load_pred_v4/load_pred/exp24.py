# -*- coding: utf-8 -*-
"""实验24：最后手段——用 Jan-Feb 2025 vs 2026 OOF 估计同比漂移，外推到 Mar-Jun。
+ Ridge 不同模型类。所有 per-month 校正都被漂移挡住，试跨年漂移估计。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

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
QA = [0.45, 0.5, 0.55]; SEEDS = [42, 7, 123]; AW = 5.0; BI = 221; LAM = 0.8


def train_members(times, X, pred_load, actual, feat_cols, train_mask, aw=AW):
    y_dir = actual; y_res = actual - pred_load
    Xtr = X[train_mask][feat_cols]; wtr = tw(times, train_mask, aw)
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
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]; pv_full = pred_load.values
    mo_all = times.month.values; yr_all = times.year.values; h_all = times.hour.values

    # 全量集成 + 现有 3-fold OOF
    print("training full ensemble ...")
    M_full = train_members(times, X, pred_load, actual, feat_cols, full_mask)
    ens_full = ens_of(M_full)
    base_full = np.clip(pv_full + LAM * (ens_full - pv_full), 0, None)
    print(f"[baseline] VAL MAE={_mae(base_full[vm], av):.2f}")

    # 自定义 OOF：Jan-Feb 2025 (train ≤ 2024-12-31) 和 Jan-Feb 2026 (train ≤ 2025-12-31)
    print("\n== Jan-Feb year-over-year drift ==")
    # Jan-Feb 2025
    ftr25 = full_mask & np.asarray(times <= pd.Timestamp("2024-12-31"))
    fva25 = full_mask & np.asarray(times >= pd.Timestamp("2025-01-01")) & np.asarray(times <= pd.Timestamp("2025-02-28"))
    M25 = train_members(times, X, pred_load, actual, feat_cols, ftr25)
    oof25 = np.clip(pv_full + LAM * (ens_of(M25) - pv_full), 0, None)
    # Jan-Feb 2026 (fold 3 已有，重算)
    ftr26 = full_mask & np.asarray(times <= pd.Timestamp("2025-12-31"))
    fva26 = full_mask & np.asarray(times >= pd.Timestamp("2026-01-01")) & np.asarray(times <= pd.Timestamp("2026-02-28"))
    M26 = train_members(times, X, pred_load, actual, feat_cols, ftr26)
    oof26 = np.clip(pv_full + LAM * (ens_of(M26) - pv_full), 0, None)

    # per-hour year-over-year drift
    print("per-hour Jan-Feb 2025 vs 2026 bias:")
    drift_h = np.zeros(24)
    for h in range(24):
        m25 = fva25 & (h_all == h); m26 = fva26 & (h_all == h)
        if m25.sum() and m26.sum():
            b25 = np.mean(oof25[m25] - actual.values[m25])
            b26 = np.mean(oof26[m26] - actual.values[m26])
            drift_h[h] = b26 - b25
            print(f"  h={h:2d}: b25={b25:.0f} b26={b26:.0f} drift={b26-b25:.0f}")
    print(f"  mean drift = {np.mean(drift_h[drift_h!=0]):.0f}")

    # 应用漂移校正：val 预测 - drift_h (减去估计的同比漂移)
    # 假设 Mar-Jun 2026 的漂移 ≈ Jan-Feb 同比漂移
    corr_drift = np.array([drift_h[h_all[i]] for i in range(len(times))])
    pred_drift = np.clip(base_full - corr_drift, 0, None)
    print(f"[baseline - JanFeb drift] VAL MAE={_mae(pred_drift[vm], av):.2f}")
    for mo in [3,4,5,6]:
        mm = (mo_all[vm] == mo)
        if mm.sum():
            print(f"  mo={mo}: MAE={_mae(pred_drift[vm][mm], av[mm]):.2f} bias={np.mean(pred_drift[vm][mm]-av[mm]):.0f}")

    # 也试整体漂移
    b25_all = np.mean(oof25[fva25] - actual.values[fva25])
    b26_all = np.mean(oof26[fva26] - actual.values[fva26])
    drift_all = b26_all - b25_all
    print(f"overall Jan-Feb drift = {drift_all:.0f} (b25={b25_all:.0f} b26={b26_all:.0f})")
    pred_drift_all = np.clip(base_full - drift_all, 0, None)
    print(f"[baseline - overall JanFeb drift] VAL MAE={_mae(pred_drift_all[vm], av):.2f}")

    # Ridge 不同模型类
    print("\n== Ridge with all features ==")
    from sklearn.preprocessing import StandardScaler
    Xs = X.fillna(0).values
    for alpha in [1.0, 10.0, 100.0]:
        rg = Ridge(alpha=alpha).fit(Xs[full_mask], actual.values[full_mask], sample_weight=tw(times, full_mask, AW))
        pred_ridge = np.clip(rg.predict(Xs), 0, None)
        print(f"[Ridge a={alpha}] VAL MAE={_mae(pred_ridge[vm], av):.2f}")
        # blend with ens
        for lam in [0.5, 0.7]:
            p = np.clip(pred_ridge + lam*(ens_full - pred_ridge), 0, None)
            print(f"  [ridge+λ{lam}*(ens-ridge)] VAL MAE={_mae(p[vm], av):.2f}")

    # 组合最佳：baseline + per-hour corr + drift
    print("\n== best combo ==")
    # 现有 3-fold OOF per-hour
    folds = C.TRAIN_CONFIG["best_it_folds"]
    oof_ens = np.full(len(times), np.nan)
    for te_s, vs_s, ve_s in folds:
        te, vs, ve = pd.Timestamp(te_s), pd.Timestamp(vs_s), pd.Timestamp(ve_s)
        ftr = full_mask & np.asarray(times <= te)
        fva = full_mask & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        M = train_members(times, X, pred_load, actual, feat_cols, ftr)
        oof_ens[fva] = ens_of(M)[fva]
    oof_mask = full_mask & ~np.isnan(oof_ens)
    oof_resid = (pv_full + LAM*(oof_ens - pv_full)) - actual.values
    hour_bias = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum():
            hour_bias[h] = np.average(oof_resid[m])
    corr_h = np.array([hour_bias[h_all[i]] for i in range(len(times))])
    # per-hour + drift
    pred_combo = np.clip(base_full - corr_h - corr_drift, 0, None)
    print(f"[baseline - per-hour - JanFeb drift] VAL MAE={_mae(pred_combo[vm], av):.2f}")
    pred_combo2 = np.clip(base_full - corr_h - drift_all, 0, None)
    print(f"[baseline - per-hour - overall drift] VAL MAE={_mae(pred_combo2[vm], av):.2f}")


if __name__ == "__main__":
    main()
