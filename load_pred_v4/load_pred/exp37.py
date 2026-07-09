# -*- coding: utf-8 -*-
"""实验37（快速版）：CatBoost 竞争力评估。用户建议 CatBoost。
为控制时长：2 seeds × {RMSE, Q0.45, Q0.5, Q0.55} × {direct, residual} = 16 成员；
depth=8, iters=350；OOF 用单折(冬折, 最近)估 per-hour 偏置。
特征 = exp36 太阳能集。+per-hour +场景分解。"""
from __future__ import annotations
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from . import config as C
from .exp36 import build, tw, _mae, hour_bias, scenario_breakdown, LAM, AW, QA

SEEDS = [42, 7]
QA_CB = [0.45, 0.5, 0.55]
# 单折 OOF（冬折，最近，最接近验证集）
OOF_FOLD = ("2025-12-31", "2026-01-01", "2026-02-28")


def train_members_cb(times, X, pred_load, actual, feat_cols, train_mask,
                     depth=8, lr=0.05, l2=5.0, iters=350):
    y_dir = actual; y_res = actual - pred_load
    Xtr = X[train_mask][feat_cols]
    wtr = tw(times, train_mask, AW)
    base = dict(learning_rate=lr, depth=depth, l2_leaf_reg=l2, iterations=iters,
                verbose=False, allow_writing_files=False, random_strength=1.0,
                bagging_temperature=1.0, border_count=128, thread_count=-1)
    member_preds = []
    losses = ["RMSE"] + [f"Quantile:alpha={q}" for q in QA_CB]
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[train_mask]
        for loss in losses:
            for s in SEEDS:
                p = dict(base, loss_function=loss, random_seed=s)
                m = CatBoostRegressor(**p)
                m.fit(Xtr, ytr.values, sample_weight=wtr)
                raw = m.predict(X[feat_cols])
                member_preds.append((pred_load.values + raw) if residual else raw)
    return np.array(member_preds)


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    print(f"  n_feat={len(feat_cols)}")
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]
    pv_full = pred_load.values
    h_all = pd.DatetimeIndex(times).hour.values.astype(int)

    print("training final CatBoost ensemble (16 members) ...")
    M = train_members_cb(times, X, pred_load, actual, feat_cols, full_mask)
    ens = np.median(M, axis=0)
    base = np.clip(pv_full + LAM * (ens - pv_full), 0, None)
    print(f"  no-bias:   VAL MAE={_mae(base[vm], av):.2f}")

    print("training single-fold OOF (winter) for per-hour bias ...")
    te, vs, ve = OOF_FOLD
    te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
    ftr = full_mask & np.asarray(times <= te)
    fva = full_mask & np.asarray(times >= vs) & np.asarray(times <= ve)
    Mo = train_members_cb(times, X, pred_load, actual, feat_cols, ftr)
    oof = pd.Series(np.nan, index=times)
    oof[fva] = np.median(Mo, axis=0)[fva]
    oof_mask = full_mask & oof.notna().values
    oof_pred = np.clip(pv_full + LAM * (oof.values - pv_full), 0, None)
    resid = oof_pred - actual.values
    hb = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum():
            hb[h] = float(np.mean(resid[m]))
    corr = np.clip(base - hb[h_all], 0, None)
    print(f"  +per-hour: VAL MAE={_mae(corr[vm], av):.2f}")

    print("\n=== 场景分解（CatBoost +per-hour）===")
    scenario_breakdown(times[vm], corr[vm], av, vm, X.loc[times[vm]])


if __name__ == "__main__":
    main()
