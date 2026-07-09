# -*- coding: utf-8 -*-
"""实验31：为交互特征集调超参。更多特征→最优 num_leaves/λ/min_data 可能变化。
base = exp30 "all" 交互集 (1554.40)。"""
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
    pl = X["pred_load"]
    pl_norm = pl / (pl.rolling(672, min_periods=96).mean())
    plvsm = X.get("pred_load_vs_mean_96", pl - pl.rolling(96, min_periods=24).mean())
    inter = pd.DataFrame(index=X.index)
    inter["pl_x_temp"] = pl * X["temp"]; inter["pl_x_hdd"] = pl * X["hdd"]; inter["pl_x_cdd"] = pl * X["cdd"]
    inter["pl_x_irrad"] = pl * X["irrad"]; inter["pl_x_wind"] = pl * X["wind"]
    inter["plnorm_x_temp"] = pl_norm * X["temp"]; inter["plnorm_x_hdd"] = pl_norm * X["hdd"]; inter["plnorm_x_cdd"] = pl_norm * X["cdd"]
    inter["pl_x_hour"] = pl * X["hour"]; inter["pl_x_dow"] = pl * X["dayofweek"]; inter["pl_x_month"] = pl * X["month"]
    inter["plnorm_x_hour"] = pl_norm * X["hour"]
    inter["plvsmean_x_temp"] = plvsm * X["temp"]; inter["plvsmean_x_hdd"] = plvsm * X["hdd"]
    X = pd.concat([X, inter], axis=1)
    return times, X, pred_load, actual


def tw(times, mask, alpha):
    t = times[mask]; tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        return np.ones(len(t))
    return (1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()).values.astype(float)


def _mae(p, a):
    return np.mean(np.abs(p - a))


QA = [0.45, 0.5, 0.55]; SEEDS = [42, 7, 123]; AW = 5.0; BI = 221


def train_members(times, X, pred_load, actual, feat_cols, train_mask, nl=127, lr=0.02, mdl=300, l2=1.0, ff=0.80, bf=0.80):
    y_dir = actual; y_res = actual - pred_load
    Xtr = X[train_mask][feat_cols]; wtr = tw(times, train_mask, AW)
    PP = dict(learning_rate=lr, num_leaves=nl, min_data_in_leaf=mdl, lambda_l2=l2,
              feature_fraction=ff, bagging_fraction=bf, bagging_freq=1, verbose=-1, force_col_wise=True)
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
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]; pv_full = pred_load.values

    print(f"n_feat={len(feat_cols)}")
    # num_leaves
    print("== num_leaves ==")
    for nl in [63, 127, 255, 511]:
        M = train_members(times, X, pred_load, actual, feat_cols, full_mask, nl=nl)
        for lam in [0.8]:
            b = np.clip(pv_full + lam*(np.median(M, axis=0) - pv_full), 0, None)
            print(f"  nl={nl} λ{lam}: VAL MAE={_mae(b[vm], av):.2f}")

    # λ sweep at nl=127
    print("== λ sweep (nl=127) ==")
    M127 = train_members(times, X, pred_load, actual, feat_cols, full_mask, nl=127)
    ens127 = np.median(M127, axis=0)
    for lam in [0.7, 0.8, 0.85, 0.9, 0.95, 1.0]:
        b = np.clip(pv_full + lam*(ens127 - pv_full), 0, None)
        print(f"  λ{lam}: VAL MAE={_mae(b[vm], av):.2f}")

    # min_data
    print("== min_data (nl=127, λ0.8) ==")
    for mdl in [200, 300, 500, 800]:
        M = train_members(times, X, pred_load, actual, feat_cols, full_mask, nl=127, mdl=mdl)
        b = np.clip(pv_full + 0.8*(np.median(M, axis=0) - pv_full), 0, None)
        print(f"  mdl={mdl}: VAL MAE={_mae(b[vm], av):.2f}")

    # lambda_l2
    print("== lambda_l2 (nl=127, λ0.8) ==")
    for l2 in [0.5, 1.0, 2.0, 5.0]:
        M = train_members(times, X, pred_load, actual, feat_cols, full_mask, nl=127, l2=l2)
        b = np.clip(pv_full + 0.8*(np.median(M, axis=0) - pv_full), 0, None)
        print(f"  l2={l2}: VAL MAE={_mae(b[vm], av):.2f}")

    # learning_rate
    print("== learning_rate (nl=127, λ0.8) ==")
    for lr, bi in [(0.01, 442), (0.02, 221), (0.03, 147), (0.05, 88)]:
        global BI; BI = bi
        M = train_members(times, X, pred_load, actual, feat_cols, full_mask, nl=127, lr=lr)
        b = np.clip(pv_full + 0.8*(np.median(M, axis=0) - pv_full), 0, None)
        print(f"  lr={lr} bi={bi}: VAL MAE={_mae(b[vm], av):.2f}")
    BI = 221


if __name__ == "__main__":
    main()
