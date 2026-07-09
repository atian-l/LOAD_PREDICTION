# -*- coding: utf-8 -*-
"""exp59: clearness×midday 漂移校正（新维度，未被 pl_wr 捕获）。

exp58 发现：午间晴天(clearness>0.7, 11-13h)有 +688 MW 系统高估，且此时 pl_wr≈0
（高辐照下 Ridge 与 pred_load 都低），故现有 drift_corr(pl_wr) 抓不到。这是 clearness 驱动
的物理（光伏）偏置，可能跨年迁移。
测试：在午间加一个 clearness 线性校正 β_clear[h] = <clearness, oof_resid>/<clearness²>，
与现有 per-hour + pl_wr drift 并存。对比生产 1512.63。
合规：β 仅从训练期 3-fold OOF 估计，无泄露。仅诊断。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from load_pred import config as C, features as F, train as T


def main():
    times, X, pred_load, actual = T.build_dataset()
    usable = T.usable_mask(times, pred_load, actual)
    mm = F.MismatchModel().fit(X, usable); X = mm.transform(X)
    val = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END)) & actual.notna()).values
    cfg = dict(C.TRAIN_CONFIG); cfg["best_it_fixed"] = 80

    # 全量集成 raw
    print("full ensemble ...", flush=True)
    model = T.train_ensemble(times, X, pred_load, actual, usable, cfg, 80)
    raw_v = model.predict_load(X[val], pred_load[val])
    act_v = actual[val].values
    print(f"  raw val MAE = {np.abs(raw_v - act_v).mean():.2f}", flush=True)

    # 3-fold OOF
    print("3-fold OOF ...", flush=True)
    oof = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0: continue
        fm = T.train_ensemble(times, X, pred_load, actual, ftr, cfg, 80)
        oof[fva] = fm.predict_load(X[fva], pred_load[fva])
    oof_mask = usable & oof.notna().values
    resid = (oof - actual).values
    h_all = pd.DatetimeIndex(times).hour.values
    plwr_all = X["pl_weather_residual"].values.astype(float)
    clear_all = X["clearness"].values.astype(float)

    # per-hour bias (mean)
    hb = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum(): hb[h] = resid[m].mean()

    # pl_wr drift β (11-14) —— 生产同款
    bp = np.zeros(24)
    for h in (11, 12, 13, 14):
        m = oof_mask & (h_all == h)
        f = plwr_all[m]; e = resid[m]; g = np.isfinite(f) & np.isfinite(e)
        d = float(np.dot(f[g], f[g]))
        if d > 0: bp[h] = float(np.dot(f[g], e[g]) / d)

    # clearness drift β (11-14) —— 新维度
    bc = np.zeros(24)
    for h in (11, 12, 13, 14):
        m = oof_mask & (h_all == h)
        f = clear_all[m]; e = resid[m]; g = np.isfinite(f) & np.isfinite(e)
        d = float(np.dot(f[g], f[g]))
        if d > 0: bc[h] = float(np.dot(f[g], e[g]) / d)
    print(f"  OOF clearness β(11-14) = {[round(bc[h],1) for h in (11,12,13,14)]}", flush=True)

    hours_v = pd.DatetimeIndex(times[val]).hour.values.astype(int)
    plwr_v = plwr_all[val]; clear_v = clear_all[val]

    # oracle clearness β (val 残差) —— 看迁移性
    val_resid = raw_v - act_v
    bco = np.zeros(24)
    for h in (11, 12, 13, 14):
        sel = (hours_v == h)
        f = clear_v[sel]; e = val_resid[sel]; g = np.isfinite(f) & np.isfinite(e)
        d = float(np.dot(f[g], f[g]))
        if d > 0: bco[h] = float(np.dot(f[g], e[g]) / d)
    print(f"  ORACLE clearness β(11-14) = {[round(bco[h],1) for h in (11,12,13,14)]}  (迁移性参考)", flush=True)
    print(flush=True)

    def mae_of(pred):
        return np.abs(pred - act_v).mean()

    # A: 生产 = per-hour + pl_wr drift
    A = raw_v - hb[hours_v] + bp[hours_v] * plwr_v
    print(f"A  per-hour + pl_wr drift        = {mae_of(A):.2f}  (生产=1512.63)", flush=True)
    # B: + clearness drift
    B = A + bc[hours_v] * clear_v
    print(f"B  + clearness drift             = {mae_of(B):.2f}  (ΔA {mae_of(B)-mae_of(A):+.2f})", flush=True)
    # C: 仅 clearness drift（替换 pl_wr）
    Cpred = raw_v - hb[hours_v] + bc[hours_v] * clear_v
    print(f"C  per-hour + clearness drift    = {mae_of(Cpred):.2f}  (替换pl_wr)", flush=True)
    # D: 阈值版 —— 午间 clearness>0.7 减固定偏置（OOF 估）
    cloudy_midday_oof = oof_mask & np.isin(h_all, [11,12,13,14]) & (clear_all > 0.7)
    if cloudy_midday_oof.sum() > 20:
        shift = resid[cloudy_midday_oof].mean()
        D = A.copy()
        sel = np.isin(hours_v, [11,12,13,14]) & (clear_v > 0.7)
        D[sel] = D[sel] - shift
        print(f"D  + 午间clear>0.7 减 {shift:+.0f}     = {mae_of(D):.2f}  (ΔA {mae_of(D)-mae_of(A):+.2f}, n={sel.sum()})", flush=True)
    # E: 阈值 oracle（看天花板）
    sel_v = np.isin(hours_v, [11,12,13,14]) & (clear_v > 0.7)
    if sel_v.sum() > 20:
        shift_o = val_resid[sel_v].mean()
        E = A.copy(); E[sel_v] = E[sel_v] - shift_o
        print(f"E  oracle 午间clear>0.7 减 {shift_o:+.0f}  = {mae_of(E):.2f}  (天花板, ΔA {mae_of(E)-mae_of(A):+.2f})", flush=True)


if __name__ == "__main__":
    main()
