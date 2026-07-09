# -*- coding: utf-8 -*-
"""实验40：扩展不匹配信号（exp39 突破 -10MW，solar_mismatch corr -0.178 有方向性）。
新增：
  pl_weather_residual: Ridge(pl ~ irrad,temp,hdd,cdd, hour_sin/cos, month_sin/cos, doy_sin/cos) 残差
                       —— pred_load 偏离"天气应有水平"的广义不匹配（捕捉光伏+温感综合漂移）
  solar_mismatch_cs:   pl_dip - b*clear_sky（用天文晴空辐照）
  solar_mismatch_hour: 逐小时 b 的不匹配
  pl_dip_x_clear_sky, pl_dip_ratio (= pl_dip / (clear_sky+1))
基于 exp39 特征集。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

from . import config as C
from . import data_loader as dl
from . import features as F
from .exp39 import build as build39, tw, _mae, hour_bias, FOLDS, LAM, AW, QA, SEEDS, NL, LR, MDL, L2, FF, BF, BI


def build():
    times, X, pred_load, actual = build39()
    pl = X["pred_load"].values
    irrad = X["irrad"].values
    clear_sky = X["clear_sky"].values
    pl_dip = X["pl_dip_96"].values
    dt = pd.DatetimeIndex(X.index)
    hr = dt.hour.values

    # 1) 广义天气残差：Ridge(pl ~ 天气+日历) on 训练期，残差应用于全期
    hour_sin = np.sin(2*np.pi*hr/24.0); hour_cos = np.cos(2*np.pi*hr/24.0)
    mo = dt.month.values; mo_sin = np.sin(2*np.pi*mo/12.0); mo_cos = np.cos(2*np.pi*mo/12.0)
    doy = dt.dayofyear.values; doy_sin = np.sin(2*np.pi*doy/365.25); doy_cos = np.cos(2*np.pi*doy/365.25)
    feat_mat = np.column_stack([irrad, X["temp"].values, X["hdd"].values, X["cdd"].values,
                                hour_sin, hour_cos, mo_sin, mo_cos, doy_sin, doy_cos])
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    tr = ((X.index >= ts0) & (X.index <= tr_end))
    fin = np.isfinite(feat_mat).all(axis=1) & np.isfinite(pl)
    trf = tr & fin
    rg = Ridge(alpha=1.0)
    rg.fit(feat_mat[trf], pl[trf])
    # 用训练期列均值填补 NaN（仅用于残差特征，不引入泄露）
    fm = feat_mat.copy()
    col_mean = np.nanmean(feat_mat[trf], axis=0)
    nan_mask = ~np.isfinite(fm)
    fm[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
    pl_wr = pl - rg.predict(fm)  # 残差：pl 偏离天气应有水平
    X["pl_weather_residual"] = pl_wr
    X["pl_wr_x_midday"] = pl_wr * X["is_midday"].values

    # 2) solar_mismatch_cs（用 clear_sky）
    day = (hr >= 8) & (hr <= 16)
    trd = tr & day & np.isfinite(pl_dip) & np.isfinite(clear_sky)
    b_cs = np.sum(clear_sky[trd]*pl_dip[trd]) / (np.sum(clear_sky[trd]**2) + 1e-9)
    X["solar_mismatch_cs"] = pl_dip - b_cs * clear_sky
    X["solar_mismatch_cs_x_midday"] = X["solar_mismatch_cs"].values * X["is_midday"].values

    # 3) 逐小时 solar_mismatch（b 按 hour 求）
    sm_hour = np.zeros(len(pl))
    for h in range(24):
        mh = (hr == h) & day
        trh = tr & mh & np.isfinite(pl_dip) & np.isfinite(irrad)
        if trh.sum() > 20:
            bh = np.sum(irrad[trh]*pl_dip[trh]) / (np.sum(irrad[trh]**2) + 1e-9)
            sm_hour[mh] = pl_dip[mh] - bh * irrad[mh]
    X["solar_mismatch_hour"] = sm_hour
    X["solar_mismatch_hour_x_midday"] = sm_hour * X["is_midday"].values

    # 4) pl_dip 与晴空比
    X["pl_dip_x_clear_sky"] = pl_dip * clear_sky
    X["pl_dip_ratio"] = pl_dip / (clear_sky + 1.0)
    print(f"  (b_cs={b_cs:.4f})")
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
    print("\n=== val 午间 方向相关性 ===")
    for col in ["pl_weather_residual", "solar_mismatch_cs", "solar_mismatch_hour", "pl_dip_ratio", "solar_mismatch"]:
        v = X[col].values[vm]
        m = midday[vm] & np.isfinite(v) & np.isfinite(pl_err[vm])
        c = float(np.corrcoef(v[m], pl_err[vm][m])[0, 1]) if m.sum() > 10 else float("nan")
        print(f"  {col:22s} corr={c:+.4f}  (n={m.sum()})")

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
