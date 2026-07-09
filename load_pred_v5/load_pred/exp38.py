# -*- coding: utf-8 -*-
"""实验38：场景条件偏置校正。针对午间&多云(-615 bias, MAE 4472)。
物理稳定：多云午间→光伏少→净负荷高→actual>pred（欠估）。该关系跨年应稳定（不像月漂移）。
方案：per-hour 偏置后，额外按场景(cell)从 OOF 估计偏置并扣除。
场景：{夜, 白天晴, 白天多云, 午间晴, 午间多云} 5 cell；及连续 cloud_deficit 回归校正。
特征 = exp36 太阳能集。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from .exp36 import (build, tw, _mae, hour_bias, FOLDS, LAM, AW,
                    QA, SEEDS, NL, LR, MDL, L2, FF, BF, BI)


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


def scenario_id(times, X):
    """返回每个时刻的场景 id: 0=夜, 1=白天晴, 2=白天多云, 3=午间晴, 4=午间多云。"""
    dt = pd.DatetimeIndex(times)
    hour = dt.hour.values
    midday = (hour >= 11) & (hour <= 13)
    day = (hour >= 8) & (hour <= 16)
    irrad = X["irrad_clean"].values
    clearness = X["clearness"].values
    cloudy = (clearness < 0.5) & day  # clearness<0.5 视为多云
    sid = np.zeros(len(times), dtype=int)  # 0=夜
    sid[day & ~midday & ~cloudy] = 1
    sid[day & ~midday & cloudy] = 2
    sid[midday & ~cloudy] = 3
    sid[midday & cloudy] = 4
    return sid


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]
    pv_full = pred_load.values
    h_all = pd.DatetimeIndex(times).hour.values.astype(int)

    print("training final ensemble ...")
    M = train_members(times, X, pred_load, actual, feat_cols, full_mask)
    ens = np.median(M, axis=0)
    base = np.clip(pv_full + LAM * (ens - pv_full), 0, None)
    print(f"  no-bias:        VAL MAE={_mae(base[vm], av):.2f}")
    oof = oof_predict(times, X, pred_load, actual, feat_cols, full_mask)
    hb = hour_bias(times, oof, actual, full_mask, LAM, pv_full)
    after_hour = np.clip(base - hb[h_all], 0, None)
    print(f"  +per-hour:      VAL MAE={_mae(after_hour[vm], av):.2f}")

    # OOF 残差（per-hour 校正后）
    oof_mask = full_mask & oof.notna().values
    oof_pred = np.clip(pv_full + LAM * (oof.values - pv_full), 0, None)
    oof_corr = np.clip(oof_pred - hb[h_all], 0, None)
    resid = oof_corr - actual.values  # OOF 残差

    sid = scenario_id(times, X)
    sid_val = sid[vm]

    print("\n  --- 各场景 OOF 残差均值(训练期) vs VAL 偏置 ---")
    for s, name in [(0,"夜"),(1,"白天晴"),(2,"白天多云"),(3,"午间晴"),(4,"午间多云")]:
        m_tr = oof_mask & (sid == s)
        m_va = (sid_val == s)
        tr_bias = np.mean(resid[m_tr]) if m_tr.sum() else 0.0
        va_bias = np.mean(after_hour[vm][m_va] - av[m_va]) if m_va.sum() else 0.0
        print(f"    {name:8s} tr_n={m_tr.sum():5d} tr_bias={tr_bias:+8.1f}  val_n={m_va.sum():4d} val_bias={va_bias:+8.1f}")

    # 方案A：场景 cell 偏置（从 OOF 估计）
    sc_bias = np.zeros(5)
    for s in range(5):
        m = oof_mask & (sid == s)
        if m.sum():
            sc_bias[s] = float(np.mean(resid[m]))
    corrA = np.clip(after_hour - sc_bias[sid], 0, None)
    print(f"\n  +scenario-bias: VAL MAE={_mae(corrA[vm], av):.2f}")

    # 方案B：仅午间多云 cell 偏置（最保守，只改 1 cell）
    sc_biasB = np.zeros(5)
    m = oof_mask & (sid == 4)
    if m.sum():
        sc_biasB[4] = float(np.mean(resid[m]))
    corrB = np.clip(after_hour - sc_biasB[sid], 0, None)
    print(f"  +midday-cloudy-bias: VAL MAE={_mae(corrB[vm], av):.2f}")

    # 方案C：连续 cloud_deficit 回归（白天）。resid ~ b * cloud_deficit * is_daytime
    cd = X["cloud_deficit"].values
    is_day = ((pd.DatetimeIndex(times).hour.values >= 8) & (pd.DatetimeIndex(times).hour.values <= 16)).astype(float)
    feat = cd * is_day
    m = oof_mask & (feat > 0)
    if m.sum():
        num = np.sum(feat[m] * resid[m]); den = np.sum(feat[m]**2)
        b = num / den if den > 0 else 0.0
        corrC = np.clip(after_hour - b * feat, 0, None)
        print(f"  +cloud-deficit-reg(b={b:.4f}): VAL MAE={_mae(corrC[vm], av):.2f}")

    # 方案D：场景 cell + 连续 cloud_deficit（午间）
    feat_mid = cd * ((pd.DatetimeIndex(times).hour.values >= 10) & (pd.DatetimeIndex(times).hour.values <= 14)).astype(float)
    m = oof_mask & (feat_mid > 0)
    b_mid = np.sum(feat_mid[m]*resid[m]) / (np.sum(feat_mid[m]**2) or 1)
    corrD = np.clip(corrA - b_mid * feat_mid, 0, None)
    print(f"  +scenario+mid-cloud-reg: VAL MAE={_mae(corrD[vm], av):.2f}")


if __name__ == "__main__":
    main()
