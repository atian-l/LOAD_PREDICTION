# -*- coding: utf-8 -*-
"""实验25：最后想法——per-point λ 作为成员 spread 的函数。
若 ens 偏置与 spread 相关，则 spread 可作置信度信号。"""
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
    return np.array(member_preds)  # (n_members, n_times)


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]; pv_full = pred_load.values
    mo_all = times.month.values; h_all = times.hour.values

    print("training full ensemble ...")
    M_full = train_members(times, X, pred_load, actual, feat_cols, full_mask)  # (24, n)
    ens_full = np.median(M_full, axis=0)
    spread_full = np.std(M_full, axis=0)  # member spread per point
    base_full = np.clip(pv_full + 0.8 * (ens_full - pv_full), 0, None)
    print(f"[baseline] VAL MAE={_mae(base_full[vm], av):.2f}")
    print(f"  spread stats: mean={spread_full.mean():.0f} std={spread_full.std():.0f}")

    # val: spread vs |error| 相关性
    val_err = base_full[vm] - av
    print(f"  corr(spread, |err|) val = {np.corrcoef(spread_full[vm], np.abs(val_err))[0,1]:.3f}")
    print(f"  corr(spread, err) val = {np.corrcoef(spread_full[vm], val_err)[0,1]:.3f}")
    # per-month spread
    for mo in [3,4,5,6]:
        mm = (mo_all[vm] == mo)
        if mm.sum():
            print(f"  mo={mo}: spread_mean={spread_full[vm][mm].mean():.0f} |err|_mean={np.abs(val_err[mm]).mean():.0f}")

    # OOF spread
    print("\ncomputing 3-fold OOF ...")
    folds = C.TRAIN_CONFIG["best_it_folds"]
    oof_ens = np.full(len(times), np.nan); oof_spread = np.full(len(times), np.nan)
    for te_s, vs_s, ve_s in folds:
        te, vs, ve = pd.Timestamp(te_s), pd.Timestamp(vs_s), pd.Timestamp(ve_s)
        ftr = full_mask & np.asarray(times <= te)
        fva = full_mask & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        M = train_members(times, X, pred_load, actual, feat_cols, ftr)
        oof_ens[fva] = np.median(M, axis=0)[fva]
        oof_spread[fva] = np.std(M, axis=0)[fva]
    oof_mask = full_mask & ~np.isnan(oof_ens)
    oof_err = (pv_full + 0.8*(oof_ens - pv_full)) - actual.values
    print(f"  OOF corr(spread, |err|) = {np.corrcoef(oof_spread[oof_mask], np.abs(oof_err[oof_mask]))[0,1]:.3f}")

    # per-spread-bin optimal λ (OOF)
    print("\n== per-spread-bin λ (OOF) ==")
    sp_oof = oof_spread[oof_mask]
    bins = np.quantile(sp_oof, np.linspace(0, 1, 6))
    lam_by_bin = []
    for i in range(len(bins)-1):
        m = (sp_oof >= bins[i]) & (sp_oof < bins[i+1])
        if m.sum() < 50:
            lam_by_bin.append(0.8); continue
        a_pl = pv_full[oof_mask][m]; a_ens = oof_ens[oof_mask][m]; a_act = actual.values[oof_mask][m]
        best_lam, best_mae = 0.8, 1e18
        for lam in np.arange(0.0, 1.51, 0.1):
            mae = _mae(a_pl + lam*(a_ens - a_pl), a_act)
            if mae < best_mae: best_mae, best_lam = mae, lam
        lam_by_bin.append(best_lam)
        print(f"  spread[{bins[i]:.0f},{bins[i+1]:.0f}): n={m.sum()} λ*={best_lam:.1f} |err|={np.abs(oof_err[oof_mask][m]).mean():.0f}")

    # 应用到 val：按 spread 分箱选 λ
    sp_val = spread_full
    lam_per_point = np.full(len(times), 0.8)
    for i in range(len(bins)-1):
        m = (sp_val >= bins[i]) & (sp_val < bins[i+1])
        lam_per_point[m] = lam_by_bin[i]
    pred_spread = np.clip(pv_full + lam_per_point * (ens_full - pv_full), 0, None)
    print(f"[per-spread-bin λ] VAL MAE={_mae(pred_spread[vm], av):.2f}")

    # 连续：λ = 1 - k * normalized_spread
    print("\n== continuous λ(spread) ==")
    sp_norm = (sp_val - sp_val[full_mask].mean()) / (sp_val[full_mask].std() + 1e-9)
    for k in [0.1, 0.2, 0.3, 0.5]:
        lam_c = np.clip(0.9 - k * sp_norm, 0.0, 1.2)
        p = np.clip(pv_full + lam_c * (ens_full - pv_full), 0, None)
        print(f"[λ=0.9-{k}*spread_norm] VAL MAE={_mae(p[vm], av):.2f}")

    # 另一思路：用 |ens - pred_load| (校正幅度) 作信号——大校正可能更不准
    print("\n== λ based on |ens-pl| (correction magnitude) ==")
    corr_mag = np.abs(ens_full - pv_full)
    cm_oof = np.abs(oof_ens - pv_full)
    cm_oof_v = cm_oof[oof_mask]
    bins2 = np.quantile(cm_oof_v, np.linspace(0, 1, 6))
    lam_by_bin2 = []
    for i in range(len(bins2)-1):
        m = (cm_oof_v >= bins2[i]) & (cm_oof_v < bins2[i+1])
        if m.sum() < 50:
            lam_by_bin2.append(0.8); continue
        a_pl = pv_full[oof_mask][m]; a_ens = oof_ens[oof_mask][m]; a_act = actual.values[oof_mask][m]
        best_lam, best_mae = 0.8, 1e18
        for lam in np.arange(0.0, 1.51, 0.1):
            mae = _mae(a_pl + lam*(a_ens - a_pl), a_act)
            if mae < best_mae: best_mae, best_lam = mae, lam
        lam_by_bin2.append(best_lam)
        print(f"  |ens-pl|[{bins2[i]:.0f},{bins2[i+1]:.0f}): n={m.sum()} λ*={best_lam:.1f}")
    lam_pp2 = np.full(len(times), 0.8)
    for i in range(len(bins2)-1):
        m = (corr_mag >= bins2[i]) & (corr_mag < bins2[i+1])
        lam_pp2[m] = lam_by_bin2[i]
    pred_cm = np.clip(pv_full + lam_pp2 * (ens_full - pv_full), 0, None)
    print(f"[per-|ens-pl|-bin λ] VAL MAE={_mae(pred_cm[vm], av):.2f}")

    # 最佳组合 + per-hour
    oof_resid = (pv_full + 0.8*(oof_ens - pv_full)) - actual.values
    hour_bias = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum(): hour_bias[h] = np.average(oof_resid[m])
    corr_h = np.array([hour_bias[h_all[i]] for i in range(len(times))])
    pred_best = np.clip(pred_spread - corr_h, 0, None)
    print(f"\n[per-spread λ + per-hour corr] VAL MAE={_mae(pred_best[vm], av):.2f}")


if __name__ == "__main__":
    main()
