# -*- coding: utf-8 -*-
"""实验33：测试 recency-windowed hour_bias。
假设：compute_hour_bias 当前对全部 OOF 点(2024-2026)等权平均，被历史模式主导。
     若仅用最近 N 天的 OOF 残差估计 hour_bias，更能反映 2026 漂移。
     风险：Jan-Feb 2026 的偏置未必等于 March 2026 的偏置。
配置：exp31 组合最优 nl=127, lr=0.03, mdl=300, l2=2.0, λ=0.9, bi=147, +per-hour。"""
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
    X = F.build_features(times, pred_load, weather)  # 已含 14 交互特征
    return times, X, pred_load, actual


def tw(times, mask, alpha):
    t = times[mask]; tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        return np.ones(len(t))
    return (1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()).values.astype(float)


def _mae(p, a):
    return np.mean(np.abs(p - a))


QA = [0.45, 0.5, 0.55]; SEEDS = [42, 7, 123]; AW = 5.0
NL = 127; LR = 0.03; MDL = 300; L2 = 2.0; FF = 0.80; BF = 0.80; BI = 147
FOLDS = C.TRAIN_CONFIG["best_it_folds"]
LAM = 0.9


def train_members(times, X, pred_load, actual, feat_cols, train_mask):
    y_dir = actual; y_res = actual - pred_load
    Xtr = X[train_mask][feat_cols]; wtr = tw(times, train_mask, AW)
    PP = dict(learning_rate=LR, num_leaves=NL, min_data_in_leaf=MDL, lambda_l2=L2,
              feature_fraction=FF, bagging_fraction=BF, bagging_freq=1, verbose=-1, force_col_wise=True)
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


def oof_predict(times, X, pred_load, actual, feat_cols, usable):
    oof = pd.Series(np.nan, index=times)
    for te, vs, ve in FOLDS:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        M = train_members(times, X, pred_load, actual, feat_cols, ftr)
        ens = np.median(M, axis=0)
        oof[fva] = ens[fva]
    return oof


def hour_bias_windowed(times, oof, actual, usable, lam, pv, window_days=None):
    """window_days: 仅用距最大 OOF 时间最近 window_days 的同小时残差；None=全部等权。
    单个 24 维向量应用于所有时刻（与生产 hour_bias 一致）。"""
    oof_mask = usable & oof.notna().values
    pred = np.clip(pv + lam * (oof.values - pv), 0, None)
    resid = pred - actual.values
    hb = np.zeros(24, dtype=float)
    h_all = pd.DatetimeIndex(times).hour.values
    oof_idx = np.where(oof_mask)[0]
    if len(oof_idx) == 0:
        return hb
    max_t = times[oof_idx].max()
    for h in range(24):
        m = oof_mask & (h_all == h)
        if not m.sum():
            continue
        if window_days is not None:
            lo = max_t - pd.Timedelta(days=window_days)
            m = m & (times >= lo)
        if m.sum():
            hb[h] = float(np.mean(resid[m]))
    return hb


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]
    pv_full = pred_load.values

    print("training final ensemble ...")
    M = train_members(times, X, pred_load, actual, feat_cols, full_mask)
    ens = np.median(M, axis=0)
    base = np.clip(pv_full + LAM * (ens - pv_full), 0, None)
    print(f"  no-bias:        VAL MAE={_mae(base[vm], av):.2f}")

    print("computing OOF ...")
    oof = oof_predict(times, X, pred_load, actual, feat_cols, full_mask)
    h_all = pd.DatetimeIndex(times).hour.values.astype(int)

    # 1) 全窗口 hour_bias（baseline，等价生产）
    hb_all = hour_bias_windowed(times, oof, actual, full_mask, LAM, pv_full, window_days=None)
    corr_all = np.clip(base - hb_all[h_all], 0, None)
    print(f"  +per-hour(all): VAL MAE={_mae(corr_all[vm], av):.2f}")

    # 2) 最近 N 天窗口
    for wd in [30, 60, 90, 120, 180]:
        hb_w = hour_bias_windowed(times, oof, actual, full_mask, LAM, pv_full, window_days=wd)
        corr_w = np.clip(base - hb_w[h_all], 0, None)
        print(f"  +per-hour(win={wd}d): VAL MAE={_mae(corr_w[vm], av):.2f}")


if __name__ == "__main__":
    main()
