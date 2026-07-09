# -*- coding: utf-8 -*-
"""实验21：per-month 最优 λ。3/4月 ens 偏置大、pred_load 近无偏→应低 λ；
5月 pred_load 严重偏高、ens 修正好→应高 λ。λ 由 OOF 估计，测是否跨年稳定。"""
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
    mo_all = times.month.values
    h_all = times.hour.values

    print("training full ensemble ...")
    M_full = train_members(times, X, pred_load, actual, feat_cols, full_mask)
    ens_full = ens_of(M_full)
    base_full = np.clip(pv_full + 0.8 * (ens_full - pv_full), 0, None)
    print(f"[baseline λ0.8] VAL MAE={_mae(base_full[vm], av):.2f}")

    # 3-fold OOF
    print("computing 3-fold OOF ...")
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
    print(f"OOF n={oof_mask.sum()} ens MAE={_mae(oof_ens[oof_mask], actual.values[oof_mask]):.2f}")

    # OOF per-month λ* (回归 (actual-pred_load) on (ens-pred_load))
    pv = pv_full
    print("\n== OOF per-month λ* ==")
    lam_star_mo = np.ones(13) * 0.8
    for mo in range(1, 13):
        m = oof_mask & (mo_all == mo)
        if m.sum() < 50:
            continue
        a_act = actual.values[m]; a_pl = pv[m]; a_ens = oof_ens[m]
        y = a_act - a_pl; x = a_ens - a_pl
        denom = np.mean(x**2)
        lam = np.mean(y * x) / denom if denom > 1e-6 else 0.8
        lam = float(np.clip(lam, 0.0, 1.5))
        lam_star_mo[mo] = lam
        # 该月 OOF 用此 λ 的 MAE
        pred_mo = a_pl + lam * (a_ens - a_pl)
        print(f"  mo={mo}: n={m.sum()} λ*={lam:.3f}  OOF MAE(λ*)={_mae(pred_mo, a_act):.2f}  OOF MAE(0.8)={_mae(a_pl+0.8*(a_ens-a_pl), a_act):.2f}")

    # 应用 per-month λ* 到 val
    lam_per_mo = np.array([lam_star_mo[mo_all[i]] for i in range(len(times))])
    pred_pm = np.clip(pv + lam_per_mo * (ens_full - pv), 0, None)
    print(f"\n[per-month λ* from OOF] VAL MAE={_mae(pred_pm[vm], av):.2f}")
    for mo in [3,4,5,6]:
        mm = (mo_all[vm] == mo)
        if mm.sum():
            print(f"  mo={mo}: λ*={lam_star_mo[mo]:.3f} MAE={_mae(pred_pm[vm][mm], av[mm]):.2f} bias={np.mean(pred_pm[vm][mm]-av[mm]):.0f}")

    # 对比：val oracle per-month λ (泄漏，仅看上界)
    print("\n== val oracle per-month λ (上界, 泄漏) ==")
    pred_oracle = np.zeros(len(times))
    for mo in [3,4,5,6]:
        mm = vm & (mo_all == mo)
        if mm.sum() == 0:
            continue
        a_act = av[mo_all[vm] == mo]; a_pl = pv[mm]; a_ens = ens_full[mm]
        y = a_act - a_pl; x = a_ens - a_pl
        denom = np.mean(x**2)
        lam = float(np.clip(np.mean(y*x)/denom if denom>1e-6 else 0.8, 0.0, 1.5))
        pred_oracle[mm] = a_pl + lam * (a_ens - a_pl)
        print(f"  mo={mo}: oracle λ={lam:.3f} MAE={_mae(pred_oracle[mm], a_act):.2f}")
    print(f"[oracle per-month λ] VAL MAE={_mae(pred_oracle[vm], av):.2f}")

    # per-hour λ* from OOF
    print("\n== OOF per-hour λ* ==")
    lam_star_h = np.ones(24) * 0.8
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum() < 50:
            continue
        y = actual.values[m] - pv[m]; x = oof_ens[m] - pv[m]
        denom = np.mean(x**2)
        lam = float(np.clip(np.mean(y*x)/denom if denom>1e-6 else 0.8, 0.0, 1.5))
        lam_star_h[h] = lam
    lam_per_h = np.array([lam_star_h[h_all[i]] for i in range(len(times))])
    pred_ph = np.clip(pv + lam_per_h * (ens_full - pv), 0, None)
    print(f"[per-hour λ* from OOF] VAL MAE={_mae(pred_ph[vm], av):.2f}")

    # per-(month,hour) λ* from OOF
    print("\n== OOF per-(month,hour) λ* ==")
    lam_star_mh = {}
    for mo in range(1, 13):
        for h in range(24):
            m = oof_mask & (mo_all == mo) & (h_all == h)
            if m.sum() < 20:
                lam_star_mh[(mo,h)] = lam_star_mo[mo]
                continue
            y = actual.values[m] - pv[m]; x = oof_ens[m] - pv[m]
            denom = np.mean(x**2)
            lam = float(np.clip(np.mean(y*x)/denom if denom>1e-6 else lam_star_mo[mo], 0.0, 1.5))
            lam_star_mh[(mo,h)] = lam
    lam_per_mh = np.array([lam_star_mh[(mo_all[i], h_all[i])] for i in range(len(times))])
    pred_pmh = np.clip(pv + lam_per_mh * (ens_full - pv), 0, None)
    print(f"[per-(mo,h) λ* from OOF] VAL MAE={_mae(pred_pmh[vm], av):.2f}")

    # 组合：per-month λ* + per-hour 校正
    oof_resid = (pv + 0.8*(oof_ens - pv)) - actual.values
    hour_bias = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum():
            hour_bias[h] = np.average(oof_resid[m])
    corr_h = np.array([hour_bias[h_all[i]] for i in range(len(times))])
    pred_combo = np.clip(pred_pm - corr_h, 0, None)
    print(f"\n[per-month λ* + per-hour corr] VAL MAE={_mae(pred_combo[vm], av):.2f}")


if __name__ == "__main__":
    main()
