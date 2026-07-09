# -*- coding: utf-8 -*-
"""实验41：强化 pl_weather_residual（corr +0.34 最强方向信号）。
新增：
  pl_wr_x_hour, pl_wr_x_midday, pl_wr_x_is_daytime, pl_wr_x_clearness  显式交互
  pl_wr_roll_mean_96 / _672  —— 残差的滚动均值（持续性不匹配=无泄露的"近期偏置"代理）
  pl_wr_roll_std_672          —— 残差波动
基于 exp40 特征集。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from .exp40 import build as build40, tw, _mae, hour_bias, FOLDS, LAM, AW, QA, SEEDS, NL, LR, MDL, L2, FF, BF, BI


def build():
    times, X, pred_load, actual = build40()
    pl_wr = pd.Series(X["pl_weather_residual"].values, index=X.index)
    clearness = X["clearness"].values
    dt = pd.DatetimeIndex(X.index)
    hour = dt.hour.values
    X["pl_wr_x_hour"] = X["pl_weather_residual"].values * hour
    X["pl_wr_x_midday"] = X["pl_weather_residual"].values * X["is_midday"].values
    X["pl_wr_x_is_daytime"] = X["pl_weather_residual"].values * X["is_daytime"].values
    X["pl_wr_x_clearness"] = X["pl_weather_residual"].values * clearness
    X["pl_wr_roll_mean_96"] = pl_wr.rolling(96, min_periods=24).mean().values
    X["pl_wr_roll_mean_672"] = pl_wr.rolling(672, min_periods=96).mean().values
    X["pl_wr_roll_std_672"] = pl_wr.rolling(672, min_periods=96).std().values
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

    print("=== val 方向相关性（午间 / 全时段）===")
    for col in ["pl_wr_roll_mean_96", "pl_wr_roll_mean_672", "pl_weather_residual"]:
        v = X[col].values[vm]
        for name, mask in [("午间", midday[vm]), ("全天", np.ones(len(v), bool))]:
            m = mask & np.isfinite(v) & np.isfinite(pl_err[vm])
            c = float(np.corrcoef(v[m], pl_err[vm][m])[0, 1]) if m.sum() > 10 else float("nan")
            print(f"  {col:22s} {name}: corr={c:+.4f} (n={m.sum()})")

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
