# -*- coding: utf-8 -*-
"""实验19：漂移感知 per-hour 校正 + λ 重扫 + 逐月分解。
核心问题：OOF(2025-03~2026-02) 与 val(2026-03~06) 存在漂移。
测试：per-hour 线性时间趋势外推、冬季折 per-hour、λ 扫描。"""
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

    print("training full ensemble ...")
    M_full = train_members(times, X, pred_load, actual, feat_cols, full_mask)

    # λ 扫描
    print("== λ sweep ==")
    for lam in [0.7, 0.8, 0.9, 1.0, 1.1]:
        base_full = agg(M_full, pv_full, lam)
        print(f"[λ={lam:.1f}] VAL MAE={_mae(base_full[vm], av):.2f}")

    LAM = 0.8
    base_full = agg(M_full, pv_full, LAM)
    print(f"\n[baseline λ={LAM}] VAL MAE={_mae(base_full[vm], av):.2f}")

    # 逐月分解
    print("== per-month breakdown ==")
    val_t = times[vm]
    for mo in [3, 4, 5, 6]:
        mm = (val_t.month == mo)
        if mm.sum():
            err = base_full[vm][mm] - av[mm]
            print(f"  mo={mo}: n={mm.sum()} MAE={_mae(base_full[vm][mm], av[mm]):.2f} bias={np.mean(err):.0f} std={np.std(err):.0f}")

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
        oof_pred[fva] = agg(M, pv_full, LAM)[fva]
    oof_mask = full_mask & ~np.isnan(oof_pred)
    oof_resid = oof_pred - actual.values
    oof_t = times[oof_mask]
    print(f"OOF n={oof_mask.sum()} MAE={_mae(oof_pred[oof_mask], actual.values[oof_mask]):.2f}")

    # OOF 逐月 bias
    print("== OOF per-month bias ==")
    for mo in range(1, 13):
        mm = (oof_t.month == mo)
        if mm.sum():
            print(f"  mo={mo}: n={mm.sum()} bias={np.mean(oof_resid[oof_mask][mm]):.0f}")

    h_all = times.hour.values
    t_num = (times - pd.Timestamp("2024-01-01")).total_seconds().values / 86400.0  # 天数

    # 1. per-hour 静态 (baseline)
    hour_bias = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum():
            hour_bias[h] = np.average(oof_resid[m])
    corr_h = np.array([hour_bias[hh] for hh in h_all])
    print(f"\n[+ per-hour static     ] VAL MAE={_mae(np.clip(base_full[vm]-corr_h[vm],0,None), av):.2f}")

    # 2. per-hour 缩放
    for sc in [1.3, 1.5, 2.0]:
        print(f"[+ per-hour ×{sc:.1f}      ] VAL MAE={_mae(np.clip(base_full[vm]-sc*corr_h[vm],0,None), av):.2f}")

    # 3. 冬季折 per-hour (Jan-Feb 2026, 最近)
    winter = oof_mask & (times >= pd.Timestamp("2026-01-01"))
    hour_bias_w = np.zeros(24)
    for h in range(24):
        m = winter & (h_all == h)
        if m.sum():
            hour_bias_w[h] = np.average(oof_resid[m])
    corr_hw = np.array([hour_bias_w[hh] for hh in h_all])
    print(f"[+ per-hour winter-fold] VAL MAE={_mae(np.clip(base_full[vm]-corr_hw[vm],0,None), av):.2f}")

    # 4. per-hour 线性时间趋势外推
    # 对每个 hour, 拟合 resid = a + b*t_num, 外推到 val
    oof_tnum = t_num[oof_mask]
    oof_h = h_all[oof_mask]
    oof_r = oof_resid[oof_mask]
    # val 时间范围
    val_tnum = t_num[vm]
    corr_trend = np.zeros(len(times))
    for h in range(24):
        m = (oof_h == h)
        if m.sum() >= 5:
            a, b = np.polyfit(oof_tnum[m], oof_r[m], 1)
            corr_trend += (a * t_num + b) * (h_all == h)
        else:
            corr_trend += hour_bias[h] * (h_all == h)
    print(f"[+ per-hour trend      ] VAL MAE={_mae(np.clip(base_full[vm]-corr_trend[vm],0,None), av):.2f}")

    # 5. 整体 bias 线性趋势 + per-hour 静态
    a_g, b_g = np.polyfit(oof_tnum, oof_r, 1)
    corr_g_trend = a_g * t_num + b_g
    print(f"[+ global trend + per-hr] VAL MAE={_mae(np.clip(base_full[vm]-corr_g_trend[vm]-corr_h[vm],0,None), av):.2f}")
    print(f"[+ global trend only   ] VAL MAE={_mae(np.clip(base_full[vm]-corr_g_trend[vm],0,None), av):.2f}")

    # 6. per-(hour,month) 仅用 shape (centered): 从 OOF 取 (h,mo) 减去月均值
    print("== per-(hour,month) centered (shape only) ==")
    oof_mo = times.month.values[oof_mask]
    month_mean = np.zeros(13)
    for mo in range(1, 13):
        m = oof_mo == mo
        if m.sum():
            month_mean[mo] = np.mean(oof_r[m])
    hm_bias = np.zeros((13, 24))
    for mo in range(1, 13):
        for h in range(24):
            m = (oof_mo == mo) & (oof_h == h)
            if m.sum():
                hm_bias[mo, h] = np.mean(oof_r[m]) - month_mean[mo]
    val_mo = times.month.values
    corr_hm_shape = np.array([hm_bias[val_mo[i], h_all[i]] for i in range(len(times))])
    # shape + per-hour 静态
    print(f"[+ (h,mo) shape + per-hr] VAL MAE={_mae(np.clip(base_full[vm]-corr_h[vm]-corr_hm_shape[vm],0,None), av):.2f}")
    # shape + global trend
    print(f"[+ (h,mo) shape + gtrend] VAL MAE={_mae(np.clip(base_full[vm]-corr_g_trend[vm]-corr_hm_shape[vm],0,None), av):.2f}")


if __name__ == "__main__":
    main()
