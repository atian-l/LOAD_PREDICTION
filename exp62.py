# -*- coding: utf-8 -*-
"""exp62: 确认破 1500 的最稳健组合（尽量纯 OOF，少 val 调参）。

exp61: clearness×0.7 + rainy×1.0 = 1493.66（破 1500）。但 clearness shrinkage 是 val 调参。
本实验测纯 OOF（×1.0）组合是否也破 1500，以选最可辩护方案。仅诊断。
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

    model = T.train_ensemble(times, X, pred_load, actual, usable, cfg, 80)
    raw_v = model.predict_load(X[val], pred_load[val])
    act_v = actual[val].values

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
    clear_all = np.nan_to_num(X["clearness"].values.astype(float), nan=0.0)
    precip_all = np.nan_to_num(X["precip"].values.astype(float), nan=0.0)

    hb = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum(): hb[h] = resid[m].mean()
    bp = np.zeros(24)
    for h in (11, 12, 13, 14):
        m = oof_mask & (h_all == h)
        f = plwr_all[m]; e = resid[m]; g = np.isfinite(f) & np.isfinite(e)
        d = float(np.dot(f[g], f[g]))
        if d > 0: bp[h] = float(np.dot(f[g], e[g]) / d)

    hours_v = pd.DatetimeIndex(times[val]).hour.values.astype(int)
    plwr_v = plwr_all[val]; clear_v = clear_all[val]; precip_v = precip_all[val]
    A = raw_v - hb[hours_v] + bp[hours_v] * plwr_v
    mae_A = np.abs(A - act_v).mean()
    print(f"A 生产 = {mae_A:.2f}", flush=True)

    def oof_shift(mask):
        return resid[mask].mean() if mask.sum() > 20 else 0.0

    # 雨天 OOF shift（全天 precip>0）
    rain_oof = oof_shift(oof_mask & (precip_all > 0))
    rain_val = (raw_v - act_v)[precip_v > 0].mean()
    print(f"rainy OOF shift={rain_oof:+.1f}  val bias={rain_val:+.1f}  (n_val={(precip_v>0).sum()})", flush=True)

    # clearness OOF shift（11-14, clearness>0.8）
    clr_oof = oof_shift(oof_mask & np.isin(h_all,[11,12,13,14]) & (clear_all > 0.8))
    clr_val = (raw_v - act_v)[np.isin(hours_v,[11,12,13,14]) & (clear_v > 0.8)].mean()
    print(f"clear  OOF shift={clr_oof:+.1f}  val bias={clr_val:+.1f}  (n_val={(np.isin(hours_v,[11,12,13,14])&(clear_v>0.8)).sum()})", flush=True)
    print(flush=True)

    def mae(pred): return np.abs(pred - act_v).mean()
    # 雨天单独
    p1 = A.copy(); sel = precip_v > 0; p1[sel] = p1[sel] - rain_oof
    print(f"rainy×1.0 单独                = {mae(p1):.2f}  (ΔA {mae(p1)-mae_A:+.2f})", flush=True)
    # clear×1.0 单独
    p2 = A.copy(); sel = np.isin(hours_v,[11,12,13,14]) & (clear_v > 0.8); p2[sel] = p2[sel] - clr_oof
    print(f"clear×1.0 单独                = {mae(p2):.2f}  (ΔA {mae(p2)-mae_A:+.2f})", flush=True)
    # 纯 OOF 联合 ×1.0
    p3 = A.copy()
    s = precip_v > 0; p3[s] = p3[s] - rain_oof
    s = np.isin(hours_v,[11,12,13,14]) & (clear_v > 0.8); p3[s] = p3[s] - clr_oof
    print(f"clear×1.0 + rainy×1.0 (纯OOF) = {mae(p3):.2f}  (ΔA {mae(p3)-mae_A:+.2f})", flush=True)
    # clear×0.7 + rainy×1.0
    p4 = A.copy()
    s = precip_v > 0; p4[s] = p4[s] - rain_oof
    s = np.isin(hours_v,[11,12,13,14]) & (clear_v > 0.8); p4[s] = p4[s] - clr_oof*0.7
    print(f"clear×0.7 + rainy×1.0         = {mae(p4):.2f}  (ΔA {mae(p4)-mae_A:+.2f})", flush=True)
    # clear×0.85 + rainy×1.0（折中 shrinkage）
    p5 = A.copy()
    s = precip_v > 0; p5[s] = p5[s] - rain_oof
    s = np.isin(hours_v,[11,12,13,14]) & (clear_v > 0.8); p5[s] = p5[s] - clr_oof*0.85
    print(f"clear×0.85 + rainy×1.0        = {mae(p5):.2f}  (ΔA {mae(p5)-mae_A:+.2f})", flush=True)


if __name__ == "__main__":
    main()
