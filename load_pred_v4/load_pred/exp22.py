# -*- coding: utf-8 -*-
"""实验22：(a) 负荷增长诊断——按年/月均值；(b) MAE 最优 per-month λ（grid search，非 MSE）。
exp21 的 oracle per-month λ 用了 MSE 回归系数→反而更差。MAE 最优 λ 可能不同。"""
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
    mo_all = times.month.values; yr_all = times.year.values

    # (a) 负荷增长诊断
    print("== (a) mean actual load by (year, month) ==")
    df = pd.DataFrame({"t": times, "actual": actual.values, "pl": pv_full, "mo": mo_all, "yr": yr_all})
    piv = df.pivot_table(index="mo", columns="yr", values="actual", aggfunc="mean").round(0)
    print(piv)
    print("\nmean pred_load by (year, month):")
    piv2 = df.pivot_table(index="mo", columns="yr", values="pl", aggfunc="mean").round(0)
    print(piv2)

    # 训练
    print("\ntraining full ensemble ...")
    M_full = train_members(times, X, pred_load, actual, feat_cols, full_mask)
    ens_full = ens_of(M_full)

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

    # (b) MAE 最优 per-month λ (oracle, val) grid search
    print("\n== (b) MAE-optimal per-month λ ==")
    lams = np.arange(0.0, 1.51, 0.05)
    # oracle
    oracle_lam = np.ones(13) * 0.8
    pred_oracle = pv_full.copy()
    for mo in [3, 4, 5, 6]:
        mm = vm & (mo_all == mo)
        if mm.sum() == 0:
            continue
        a_act = av[mo_all[vm] == mo]; a_pl = pv_full[mm]; a_ens = ens_full[mm]
        best_lam, best_mae = 0.8, 1e18
        for lam in lams:
            mae = _mae(a_pl + lam * (a_ens - a_pl), a_act)
            if mae < best_mae:
                best_mae, best_lam = mae, lam
        oracle_lam[mo] = best_lam
        pred_oracle[mm] = a_pl + best_lam * (a_ens - a_pl)
        print(f"  mo={mo}: oracle MAE-λ={best_lam:.2f} MAE={best_mae:.2f}")
    print(f"[oracle per-month MAE-λ] VAL MAE={_mae(pred_oracle[vm], av):.2f}")

    # OOF MAE-optimal per-month λ
    oof_lam = np.ones(13) * 0.8
    for mo in range(1, 13):
        m = oof_mask & (mo_all == mo)
        if m.sum() < 50:
            continue
        a_act = actual.values[m]; a_pl = pv_full[m]; a_ens = oof_ens[m]
        best_lam, best_mae = 0.8, 1e18
        for lam in lams:
            mae = _mae(a_pl + lam * (a_ens - a_pl), a_act)
            if mae < best_mae:
                best_mae, best_lam = mae, lam
        oof_lam[mo] = best_lam
    pred_oof = np.clip(pv_full + np.array([oof_lam[mo_all[i]] for i in range(len(times))]) * (ens_full - pv_full), 0, None)
    print(f"[OOF per-month MAE-λ] VAL MAE={_mae(pred_oof[vm], av):.2f}")
    for mo in [3,4,5,6]:
        mm = (mo_all[vm] == mo)
        if mm.sum():
            print(f"  mo={mo}: OOF λ={oof_lam[mo]:.2f} oracle λ={oracle_lam[mo]:.2f} MAE={_mae(pred_oof[vm][mm], av[mm]):.2f}")

    # per-month MAE-λ + per-hour 校正
    oof_resid = (pv_full + 0.8*(oof_ens - pv_full)) - actual.values
    h_all = times.hour.values
    hour_bias = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum():
            hour_bias[h] = np.average(oof_resid[m])
    corr_h = np.array([hour_bias[h_all[i]] for i in range(len(times))])
    pred_combo = np.clip(pred_oof - corr_h, 0, None)
    print(f"[OOF per-month MAE-λ + per-hour corr] VAL MAE={_mae(pred_combo[vm], av):.2f}")

    # 连续 λ 模型：用 OOF 训练 λ 作为 pred_load level 的函数
    print("\n== continuous λ(pred_load level) ==")
    # 对每个 OOF 点，计算 MAE-optimal λ (粗略：sign-based)
    # 若 (ens-pl) 与 (actual-pl) 同号→λ=1，异号→λ=0
    oof_pl = pv_full[oof_mask]; oof_e = oof_ens[oof_mask]; oof_a = actual.values[oof_mask]
    sign_agree = ((oof_e - oof_pl) * (oof_a - oof_pl) > 0).astype(float)
    print(f"  OOF sign-agree rate = {sign_agree.mean():.3f}")
    # 按 pred_load level 分箱看 λ
    bins = np.quantile(oof_pl, np.linspace(0, 1, 6))
    for i in range(len(bins)-1):
        m = (oof_pl >= bins[i]) & (oof_pl < bins[i+1])
        if m.sum():
            print(f"  pl[{bins[i]:.0f},{bins[i+1]:.0f}): n={m.sum()} agree={sign_agree[m].mean():.3f} pl_bias={np.mean(oof_pl[m]-oof_a[m]):.0f} ens_bias={np.mean(oof_e[m]-oof_a[m]):.0f}")


if __name__ == "__main__":
    main()
