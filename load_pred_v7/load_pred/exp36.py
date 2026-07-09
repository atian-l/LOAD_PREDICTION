# -*- coding: utf-8 -*-
"""实验36：太阳能/晴空特征 + irrad 清洗。针对午间(11-13)高误差(3449 vs 1405)。
新增特征（均无泄露，晴空为天文确定量）：
  irrad_clean    = irrad.clip(0, 1200)  （原始 irrad 有 ±22 万离群值）
  clear_sky      = 0.8*1367*sin(elevation)  （山东 36°N 天文晴空辐照）
  clearness      = irrad_clean / (clear_sky+1)  （云量逆指标）
  cloud_deficit  = clear_sky - irrad_clean       （被云遮挡的辐照）
  is_midday      = hour in [11,13]
  is_daytime     = hour in [8,16]
  pl_x_clearness, pl_x_cloud_deficit, pl_x_irrad_clean
  irrad_x_midday, pl_x_irrad_x_midday, clearness_x_midday
配置：exp31 组合最优 nl=127, lr=0.03, mdl=300, l2=2.0, λ=0.9, bi=147, +per-hour。
输出：整体/按小时/午间/白天多云 的 MAE 分解。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from . import data_loader as dl
from . import features as F


LAT = np.radians(36.0)


def solar_features(times, irrad, hour, doy):
    decl = np.radians(23.45 * np.sin(np.radians(360.0 * (284 + doy) / 365.25)))
    ha = np.radians((hour * 60 + 0 - 720) * 0.25)  # 15min 网格，分钟=0；时角
    sin_elev = np.sin(LAT) * np.sin(decl) + np.cos(LAT) * np.cos(decl) * np.cos(ha)
    sin_elev = np.clip(sin_elev, 0.0, None)
    clear_sky = 0.8 * 1367.0 * sin_elev
    irrad_clean = np.clip(irrad, 0.0, 1200.0)
    clearness = irrad_clean / (clear_sky + 1.0)
    cloud_deficit = clear_sky - irrad_clean
    return clear_sky, irrad_clean, clearness, cloud_deficit


def build():
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)
    X = F.build_features(times, pred_load, weather)
    dt = pd.DatetimeIndex(times)
    hour = dt.hour.values
    doy = dt.dayofyear.values
    irrad = X["irrad"].values
    cs, ir_clean, clearness, cdef = solar_features(times, irrad, hour, doy)
    X["irrad_clean"] = ir_clean
    X["clear_sky"] = cs
    X["clearness"] = clearness
    X["cloud_deficit"] = cdef
    X["is_midday"] = ((hour >= 11) & (hour <= 13)).astype(int)
    X["is_daytime"] = ((hour >= 8) & (hour <= 16)).astype(int)
    pl = X["pred_load"].values
    X["pl_x_clearness"] = pl * clearness
    X["pl_x_cloud_deficit"] = pl * cdef
    X["pl_x_irrad_clean"] = pl * ir_clean
    X["irrad_x_midday"] = ir_clean * X["is_midday"].values
    X["pl_x_irrad_x_midday"] = pl * ir_clean * X["is_midday"].values
    X["clearness_x_midday"] = clearness * X["is_midday"].values
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


def hour_bias(times, oof, actual, usable, lam, pv):
    oof_mask = usable & oof.notna().values
    pred = np.clip(pv + lam * (oof.values - pv), 0, None)
    resid = pred - actual.values
    hb = np.zeros(24, dtype=float)
    h_all = pd.DatetimeIndex(times).hour.values
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum():
            hb[h] = float(np.mean(resid[m]))
    return hb


def scenario_breakdown(times, pred, actual, vm, X_val):
    err = pred - actual
    dt = pd.DatetimeIndex(times)
    hour = dt.hour.values
    midday = (hour >= 11) & (hour <= 13)
    day = (hour >= 8) & (hour <= 16)
    irrad = X_val["irrad_clean"].values
    cloudy = np.zeros(len(err), bool)
    for h in range(8, 17):
        mh = (hour == h) & day
        if mh.sum() == 0:
            continue
        med = np.nanmedian(irrad[mh])
        cloudy = cloudy | (mh & (irrad < med * 0.6))
    def s(m, n):
        e = err[m]
        if m.sum() == 0:
            print(f"  {n:24s}: n=0"); return
        print(f"  {n:24s}: n={m.sum():5d} MAE={np.mean(np.abs(e)):.1f} bias={np.mean(e):+.1f}")
    s(np.ones(len(err), bool), "all")
    s(midday, "午间11-13")
    s(~midday, "非午间")
    s(day, "白天8-16")
    s(day & cloudy, "白天多云")
    s(day & ~cloudy, "白天晴")
    s(midday & cloudy, "午间&多云")
    print("  --- 按小时 MAE top8 ---")
    rows = [(h, (hour == h)) for h in range(24)]
    rows = [(h, m.sum(), np.mean(np.abs(err[m]))) for h, m in rows if m.sum()]
    for h, n, mae in sorted(rows, key=lambda r: -r[2])[:8]:
        print(f"    h={h:2d} n={n:4d} MAE={mae:.1f}")


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

    print("training final ensemble ...")
    M = train_members(times, X, pred_load, actual, feat_cols, full_mask)
    ens = np.median(M, axis=0)
    base = np.clip(pv_full + LAM * (ens - pv_full), 0, None)
    print(f"  no-bias:   VAL MAE={_mae(base[vm], av):.2f}")
    oof = oof_predict(times, X, pred_load, actual, feat_cols, full_mask)
    hb = hour_bias(times, oof, actual, full_mask, LAM, pv_full)
    corr = np.clip(base - hb[h_all], 0, None)
    print(f"  +per-hour: VAL MAE={_mae(corr[vm], av):.2f}")

    print("\n=== 场景分解（模型预测，+per-hour）===")
    scenario_breakdown(times[vm], corr[vm], av, vm, X.loc[times[vm]])


if __name__ == "__main__":
    main()
