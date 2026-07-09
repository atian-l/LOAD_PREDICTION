# -*- coding: utf-8 -*-
"""exp60: 优化 clearness×midday 校正（exp59 给 -5 MW，oracle 天花板 1502）。

exp59: 阈值 clearness>0.7 @11-14，OOF shift=+1407（过校，oracle=+792）。
本实验测：
  - mean vs median shift（median 抗右偏尾，更接近 MAE 最优）
  - 收缩因子 0.5/0.7（OOF 2025 漂移到 2026，过校）
  - 小时窗 11-14 / 11-13 / 12-13
  - 阈值 0.6/0.7/0.8
  - 线性 β（修 NaN）
目标：把 OOF 版尽量逼近 oracle 1502。合规：仅 OOF，不写产物。
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
    plwr_v = plwr_all[val]; clear_v = clear_all[val]
    A = raw_v - hb[hours_v] + bp[hours_v] * plwr_v
    mae_A = np.abs(A - act_v).mean()
    print(f"A  生产 = {mae_A:.2f}", flush=True)
    print(flush=True)

    def try_threshold(hours, thr, agg, shrink, label):
        hset = set(hours)
        m_oof = oof_mask & np.isin(h_all, list(hset)) & (clear_all > thr)
        if m_oof.sum() < 20: return
        if agg == "mean": shift = resid[m_oof].mean()
        elif agg == "median": shift = np.median(resid[m_oof])
        shift *= shrink
        sel = np.isin(hours_v, list(hset)) & (clear_v > thr)
        pred = A.copy()
        pred[sel] = pred[sel] - shift
        mae = np.abs(pred - act_v).mean()
        print(f"  {label:42s} shift={shift:+7.0f} n={sel.sum():4d}  MAE={mae:.2f}  (ΔA {mae-mae_A:+.2f})", flush=True)

    print("=== 阈值 mean/median/shrink @11-14 ===", flush=True)
    for thr in (0.6, 0.7, 0.8):
        for agg in ("mean", "median"):
            for sh in (1.0, 0.7, 0.5):
                try_threshold([11,12,13,14], thr, agg, sh, f"thr>{thr} {agg} ×{sh}")
    print(flush=True)
    print("=== 小时窗 12-13 / 11-13 ===", flush=True)
    for hrs, lab in (([12,13],"12-13"), ([11,12,13],"11-13")):
        for agg in ("mean","median"):
            for sh in (1.0, 0.7):
                try_threshold(hrs, 0.7, agg, sh, f"{lab} thr>0.7 {agg} ×{sh}")
    print(flush=True)

    # 线性 β（修 NaN）—— 连续 clearness
    print("=== 线性 β×clearness @午间 ===", flush=True)
    for hrs in ([11,12,13,14], [12,13]):
        bc = np.zeros(24)
        for h in hrs:
            m = oof_mask & (h_all == h)
            f = clear_all[m]; e = resid[m]; g = np.isfinite(f) & np.isfinite(e)
            d = float(np.dot(f[g], f[g]))
            if d > 0: bc[h] = float(np.dot(f[g], e[g]) / d)
        pred = A + bc[hours_v] * clear_v
        mae = np.abs(pred - act_v).mean()
        print(f"  线性 β×clearness @ {hrs}  MAE={mae:.2f}  (ΔA {mae-mae_A:+.2f})", flush=True)


if __name__ == "__main__":
    main()
