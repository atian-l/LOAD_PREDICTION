# -*- coding: utf-8 -*-
"""exp63: 用稳健估计器(中位数/截尾均值)代替 mean 估 clear/rainy shift，
看是否能无 shrinkage、无 val 调参地破 1500（最可辩护）。仅诊断。"""
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

    def trimmed(a, q=0.2):
        a = a[np.isfinite(a)]
        if len(a) < 10: return 0.0
        lo, hi = np.quantile(a, [q/2, 1-q/2])
        return a[(a>=lo)&(a<=hi)].mean()

    clr_mask_all = oof_mask & np.isin(h_all,[11,12,13,14]) & (clear_all > 0.8)
    rain_mask_all = oof_mask & (precip_all > 0)
    print(f"clear n_oof={clr_mask_all.sum()}  rain n_oof={rain_mask_all.sum()}", flush=True)
    for name, fn in [("mean", lambda a: np.nanmean(a)),
                     ("median", lambda a: np.nanmedian(a)),
                     ("trim20", lambda a: trimmed(a,0.2)),
                     ("trim30", lambda a: trimmed(a,0.3))]:
        cs = float(fn(resid[clr_mask_all]))
        rs = float(fn(resid[rain_mask_all]))
        p = A.copy()
        s = precip_v > 0; p[s] = p[s] - rs
        s = np.isin(hours_v,[11,12,13,14]) & (clear_v > 0.8); p[s] = p[s] - cs
        print(f"  {name:8s} clear={cs:+8.1f} rain={rs:+7.1f}  MAE={np.abs(p-act_v).mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
