# -*- coding: utf-8 -*-
"""实验42：继续强化 pl_weather_residual 系（all-day corr +0.29）。
新增：
  pl_wr_roll_mean_28d (2688)  月级持续漂移
  pl_wr_roll_mean_96_x_hour / _x_midday  持续漂移×时段
  pl_wr_x_hour_sin / _x_hour_cos          平滑时段交互
  pl_wr_roll_mean_672_x_midday
基于 exp41 特征集。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from .exp41 import build as build41, tw, _mae, hour_bias, FOLDS, LAM, AW, QA, SEEDS, NL, LR, MDL, L2, FF, BF, BI


def build():
    times, X, pred_load, actual = build41()
    pl_wr = pd.Series(X["pl_weather_residual"].values, index=X.index)
    dt = pd.DatetimeIndex(X.index)
    hour = dt.hour.values
    hour_sin = np.sin(2*np.pi*hour/24.0); hour_cos = np.cos(2*np.pi*hour/24.0)
    roll96 = pd.Series(X["pl_wr_roll_mean_96"].values, index=X.index)
    roll672 = pd.Series(X["pl_wr_roll_mean_672"].values, index=X.index)
    X["pl_wr_roll_mean_28d"] = pl_wr.rolling(2688, min_periods=672).mean().values
    X["pl_wr_roll_mean_96_x_hour"] = roll96.values * hour
    X["pl_wr_roll_mean_96_x_midday"] = roll96.values * X["is_midday"].values
    X["pl_wr_x_hour_sin"] = X["pl_weather_residual"].values * hour_sin
    X["pl_wr_x_hour_cos"] = X["pl_weather_residual"].values * hour_cos
    X["pl_wr_roll_mean_672_x_midday"] = roll672.values * X["is_midday"].values
    return times, X, pred_load, actual


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
        oof[fva] = np.median(M, axis=0)[fva]
    return oof


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    print(f"  n_feat={len(feat_cols)}")
    vm = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END)) & actual.notna()).values
    av = actual.values[vm]
    pv_full = pred_load.values
    dt = pd.DatetimeIndex(times); hour = dt.hour.values
    midday = (hour >= 11) & (hour <= 13)
    pl_err = pv_full - actual.values

    print("=== val 全天方向相关性 ===")
    for col in ["pl_wr_roll_mean_28d", "pl_wr_roll_mean_96", "pl_weather_residual"]:
        v = X[col].values[vm]
        m = np.isfinite(v) & np.isfinite(pl_err[vm])
        c = float(np.corrcoef(v[m], pl_err[vm][m])[0, 1]) if m.sum() > 10 else float("nan")
        print(f"  {col:24s} all-day corr={c:+.4f} (n={m.sum()})")

    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    h_all = hour.astype(int)
    print("\ntraining final ensemble ...")
    M = train_members(times, X, pred_load, actual, feat_cols, full_mask)
    ens = np.median(M, axis=0)
    base = np.clip(pv_full + LAM * (ens - pv_full), 0, None)
    print(f"  no-bias:   VAL MAE={_mae(base[vm], av):.2f}")
    oof = oof_predict(times, X, pred_load, actual, feat_cols, full_mask)
    hb = hour_bias(times, oof, actual, full_mask, LAM, pv_full)
    corr = np.clip(base - hb[h_all], 0, None)
    print(f"  +per-hour: VAL MAE={_mae(corr[vm], av):.2f}")
    mm = midday[vm]
    print(f"  午间 MAE={_mae(corr[vm][mm], av[mm]):.2f}  非午间 MAE={_mae(corr[vm][~mm], av[~mm]):.2f}")


if __name__ == "__main__":
    main()
