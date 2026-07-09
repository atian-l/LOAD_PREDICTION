# -*- coding: utf-8 -*-
"""实验34：per-(hour, dow) 偏置校正。Oracle per-(hour,dow)=1536（较 per-hour 1550 更低）。
     周结构(dow)比月结构更稳定，可能可从训练 OOF 估计。
配置：exp31 组合最优 nl=127, lr=0.03, mdl=300, l2=2.0, λ=0.9, bi=147。"""
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


def bias_lookup(hours, dows, oof, actual, usable, lam, pv, mode="hour_dow", window_days=None):
    """返回每个时刻的偏置估计（与时刻一一对应）。
    mode: "hour" / "dow" / "hour_dow"
    window_days: 仅用距最大 OOF 时间最近 window_days 的残差；None=全部。"""
    oof_mask = usable & oof.notna().values
    pred = np.clip(pv + lam * (oof.values - pv), 0, None)
    resid = pred - actual.values
    oof_idx = np.where(oof_mask)[0]
    if len(oof_idx) == 0:
        return np.zeros(len(hours))
    max_t = times_arr[oof_idx].max()
    time_mask = np.ones(len(oof_mask), dtype=bool)
    if window_days is not None:
        lo = max_t - pd.Timedelta(days=window_days)
        time_mask = np.asarray(times_arr >= lo)
    if mode == "hour":
        key = hours
        tbl = {}
        for h in range(24):
            m = oof_mask & time_mask & (hours == h)
            if m.sum():
                tbl[h] = float(np.mean(resid[m]))
        return np.array([tbl.get(h, 0.0) for h in hours])
    elif mode == "dow":
        key = dows
        tbl = {}
        for d in range(7):
            m = oof_mask & time_mask & (dows == d)
            if m.sum():
                tbl[d] = float(np.mean(resid[m]))
        return np.array([tbl.get(d, 0.0) for d in dows])
    else:  # hour_dow
        tbl = {}
        for h in range(24):
            for d in range(7):
                m = oof_mask & time_mask & (hours == h) & (dows == d)
                if m.sum():
                    tbl[(h, d)] = float(np.mean(resid[m]))
        return np.array([tbl.get((h, d), 0.0) for h, d in zip(hours, dows)])


times_arr = None


def main():
    global times_arr
    print("building ...")
    times, X, pred_load, actual = build()
    times_arr = times.values
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]
    pv_full = pred_load.values

    dt = pd.DatetimeIndex(times)
    hours = dt.hour.values.astype(int)
    dows = dt.dayofweek.values.astype(int)

    print("training final ensemble ...")
    M = train_members(times, X, pred_load, actual, feat_cols, full_mask)
    ens = np.median(M, axis=0)
    base = np.clip(pv_full + LAM * (ens - pv_full), 0, None)
    print(f"  no-bias:           VAL MAE={_mae(base[vm], av):.2f}")

    print("computing OOF ...")
    oof = oof_predict(times, X, pred_load, actual, feat_cols, full_mask)

    for mode in ["hour", "dow", "hour_dow"]:
        for wd in [None, 180, 365]:
            b = bias_lookup(hours, dows, oof, actual, full_mask, LAM, pv_full, mode=mode, window_days=wd)
            corr = np.clip(base - b, 0, None)
            tag = f"{mode} win={'all' if wd is None else wd}d"
            print(f"  +{tag:22s}: VAL MAE={_mae(corr[vm], av):.2f}")


if __name__ == "__main__":
    main()
