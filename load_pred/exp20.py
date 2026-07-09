# -*- coding: utf-8 -*-
"""实验20：诊断 pred_load vs ens 偏置分解 + 月份特征消融 + 更多成员/更长历史。
核心：3月/4月 val 偏置与 OOF 符号反转。查 pred_load 是否同样偏置。"""
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
QA = [0.45, 0.5, 0.55]; BI = 221; LAM = 0.8


def train_members(times, X, pred_load, actual, feat_cols, train_mask, seeds, aw):
    y_dir = actual; y_res = actual - pred_load
    Xtr = X[train_mask][feat_cols]; wtr = tw(times, train_mask, aw)
    member_preds = []
    objs = [("regression", None)] + [("quantile", q) for q in QA]
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[train_mask]
        d = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj, qa_ in objs:
            for s in seeds:
                p = dict(PP, objective=obj, seed=s)
                if obj == "quantile":
                    p["alpha"] = qa_
                bst = lgb.train(p, d, num_boost_round=int(BI))
                raw = bst.predict(X[feat_cols])
                member_preds.append((pred_load.values + raw) if residual else raw)
    return np.array(member_preds)


def agg(M, pv_full, lam):
    ens = np.median(M, axis=0)
    return np.clip(pv_full + lam * (ens - pv_full), 0, None), ens


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]; pv_full = pred_load.values
    val_mo = times.month.values[vm]

    print("== pred_load vs model bias by month (val) ==")
    print(f"  pred_load VAL MAE (纯外部预测) = {_mae(pv_full[vm], av):.2f}")
    for mo in [3, 4, 5, 6]:
        mm = val_mo == mo
        if mm.sum():
            pl_b = np.mean(pv_full[vm][mm] - av[mm])
            print(f"  mo={mo}: pred_load_bias={pl_b:.0f}  pred_load_MAE={_mae(pv_full[vm][mm], av[mm]):.2f}")

    # 训练 baseline (24成员 aw5.0)
    print("\ntraining baseline (3 seeds, aw5.0) ...")
    M = train_members(times, X, pred_load, actual, feat_cols, full_mask, [42,7,123], 5.0)
    base_full, ens_full = agg(M, pv_full, LAM)
    print(f"[baseline] VAL MAE={_mae(base_full[vm], av):.2f}  ens MAE={_mae(ens_full[vm], av):.2f}")
    for mo in [3, 4, 5, 6]:
        mm = val_mo == mo
        if mm.sum():
            eb = np.mean(ens_full[vm][mm] - av[mm])
            mb = np.mean(base_full[vm][mm] - av[mm])
            print(f"  mo={mo}: ens_bias={eb:.0f} model_bias={mb:.0f}")

    # 5 seeds (40 成员)
    print("\ntraining 5 seeds (40 成员) ...")
    M5 = train_members(times, X, pred_load, actual, feat_cols, full_mask, [42,7,123,2024,99], 5.0)
    base5, ens5 = agg(M5, pv_full, LAM)
    print(f"[5 seeds λ0.8] VAL MAE={_mae(base5[vm], av):.2f}")
    for lam in [0.85, 0.9, 0.95]:
        b5, _ = agg(M5, pv_full, lam)
        print(f"[5 seeds λ{lam}] VAL MAE={_mae(b5[vm], av):.2f}")

    # 月份特征消融：去掉 month 相关特征
    print("\n== month-feature ablation ==")
    drop_cols = [c for c in feat_cols if c in ("mo", "month", "sin_mo", "cos_mo", "is_spring", "is_summer", "is_autumn", "is_winter")]
    print(f"  dropping: {drop_cols}")
    fc2 = [c for c in feat_cols if c not in drop_cols]
    M_nm = train_members(times, X, pred_load, actual, fc2, full_mask, [42,7,123], 5.0)
    base_nm, _ = agg(M_nm, pv_full, LAM)
    print(f"[no-month λ0.8] VAL MAE={_mae(base_nm[vm], av):.2f}")
    for mo in [3,4,5,6]:
        mm = val_mo == mo
        if mm.sum():
            print(f"  mo={mo}: MAE={_mae(base_nm[vm][mm], av[mm]):.2f} bias={np.mean(base_nm[vm][mm]-av[mm]):.0f}")

    # 更长历史 train_start=2023-02
    print("\n== train_start=2023-02-01 ==")
    ts0b = pd.Timestamp("2023-02-01")
    fmb = ((times >= ts0b) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    M_b = train_members(times, X, pred_load, actual, feat_cols, fmb, [42,7,123], 5.0)
    base_b, _ = agg(M_b, pv_full, LAM)
    print(f"[ts2023 λ0.8] VAL MAE={_mae(base_b[vm], av):.2f}")


if __name__ == "__main__":
    main()
