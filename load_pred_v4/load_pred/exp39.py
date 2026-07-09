# -*- coding: utf-8 -*-
"""实验39：pred_load 隐含光伏 vs 天气辐照 的"不匹配"信号（无泄露新思路）。
假设：pred_load 反映外部预报员的光伏假设（午间低谷深度）；我们的 irrad 独立。
     若 pred_load 午间深谷但天气报多云(irrad低)→不匹配→pred_load 午间可能偏低(欠估)。
     该方向信号此前未测（不同于集成离散度，那只预测幅度不预测方向）。
步骤：1) 构造不匹配特征；2) 分析 val 午间 corr(特征, 误差方向)；3) 若有信号则加特征重训。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from . import data_loader as dl
from . import features as F
from .exp36 import build as build_base, tw, _mae, hour_bias, FOLDS, LAM, AW, QA, SEEDS, NL, LR, MDL, L2, FF, BF, BI


def build():
    times, X, pred_load, actual = build_base()
    pl = X["pred_load"]
    irrad = X["irrad"].values  # 已 clip
    clearness = X["clearness"].values
    # 午间低谷深度：pl 相对其 24h(96点)均值的下凹
    pl_dip = (pl.rolling(96, min_periods=24).mean() - pl).values
    # irrad 异常：相对 7 天同点滚动均值
    irrad_s = pd.Series(irrad, index=X.index)
    irrad_anom = (irrad_s - irrad_s.rolling(672, min_periods=96).mean()).values
    X["pl_dip_96"] = pl_dip
    X["irrad_anom_672"] = irrad_anom
    X["pl_dip_x_irrad"] = pl_dip * irrad
    X["pl_dip_x_clearness"] = pl_dip * clearness
    X["pl_dip_x_midday"] = pl_dip * X["is_midday"].values
    X["pl_x_irrad_anom"] = pl.values * irrad_anom
    X["irrad_anom_x_midday"] = irrad_anom * X["is_midday"].values
    # 不匹配：pl_dip 与 irrad 的不一致。用训练期回归 pl_dip ~ irrad(daytime) 求 b
    dt = pd.DatetimeIndex(X.index)
    hr = dt.hour.values
    day = (hr >= 8) & (hr <= 16)
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    tr = ((X.index >= ts0) & (X.index <= tr_end)) & day & np.isfinite(pl_dip) & np.isfinite(irrad)
    b = np.sum(irrad[tr] * pl_dip[tr]) / (np.sum(irrad[tr]**2) + 1e-9)
    mismatch = pl_dip - b * irrad
    X["solar_mismatch"] = mismatch
    X["solar_mismatch_x_midday"] = mismatch * X["is_midday"].values
    print(f"  (mismatch b={b:.4f})")
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
    dt = pd.DatetimeIndex(times)
    hour = dt.hour.values
    midday = (hour >= 11) & (hour <= 13)

    # --- 方向信号分析（val 午间）---
    pl_err = pv_full - actual.values  # 正=高估
    print("\n=== val 午间(11-13) 方向相关性 corr(feature, pred_load_err) ===")
    for col in ["pl_dip_96", "irrad_anom_672", "solar_mismatch", "irrad", "clearness"]:
        v = X[col].values[vm]
        m = midday[vm] & np.isfinite(v) & np.isfinite(pl_err[vm])
        c = float(np.corrcoef(v[m], pl_err[vm][m])[0, 1]) if m.sum() > 10 else float("nan")
        print(f"  {col:18s} corr={c:+.4f}  (n={m.sum()})")
    print("  (负相关=该特征大时 pred_load 偏高估；正相关=偏低估)")

    # --- 重训评估 ---
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
    # 午间子集
    mm = midday[vm]
    print(f"  午间 MAE={_mae(corr[vm][mm], av[mm]):.2f}  非午间 MAE={_mae(corr[vm][~mm], av[~mm]):.2f}")


if __name__ == "__main__":
    main()
