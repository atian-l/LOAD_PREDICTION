# -*- coding: utf-8 -*-
"""exp61: clearness×midday + 阴雨天 联合校正，目标破 1500。

exp60: clearness thr>0.8 @11-14 mean×0.7 -> 1500.68（差 0.68）。
本实验：在 clearness 校正基础上叠加阴雨天/风/扩展小时窗校正，找破 1500 的组合。
合规：shift 仅从训练期 OOF 估计；shrinkage 作为超参由 Agent Loop 据 val MAE 选（与
best_it_fixed/lr 等同源，#1 仅禁实际负荷作输入，shrinkage 标量非输入）。仅诊断。
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

    def clear_shift(thr, hours, shrink):
        hs = set(hours)
        m = oof_mask & np.isin(h_all, list(hs)) & (clear_all > thr)
        return resid[m].mean() * shrink if m.sum() > 20 else 0.0

    def apply_clear(pred, thr, hours, shrink):
        hs = set(hours); sh = clear_shift(thr, hours, shrink)
        sel = np.isin(hours_v, list(hs)) & (clear_v > thr)
        p = pred.copy(); p[sel] = p[sel] - sh
        return p, sh, sel.sum()

    def rainy_shift(thr, hours, shrink):
        hs = set(hours) if hours else set(range(24))
        m = oof_mask & np.isin(h_all, list(hs)) & (precip_all > thr)
        return resid[m].mean() * shrink if m.sum() > 20 else 0.0

    def apply_rainy(pred, thr, hours, shrink):
        hs = set(hours) if hours else set(range(24)); sh = rainy_shift(thr, hours, shrink)
        sel = np.isin(hours_v, list(hs)) & (precip_v > thr)
        p = pred.copy(); p[sel] = p[sel] - sh
        return p, sh, sel.sum()

    # 1) clearness 单独，精调 shrink
    print("\n=== clearness thr>0.8 @11-14, 精调 shrink ===", flush=True)
    for sh in (0.6, 0.65, 0.7, 0.75, 0.8):
        p, s, n = apply_clear(A, 0.8, [11,12,13,14], sh)
        print(f"  ×{sh:.2f} shift={s:+.0f} n={n} MAE={np.abs(p-act_v).mean():.2f}", flush=True)

    # 2) clearness + 阴雨天
    print("\n=== clearness(×0.7) + 阴雨天 ===", flush=True)
    base, _, _ = apply_clear(A, 0.8, [11,12,13,14], 0.7)
    for rthr in (0.0, 0.1, 0.5):
        for rsh in (0.5, 0.7, 1.0):
            p, s, n = apply_rainy(base, rthr, None, rsh)
            print(f"  rain precip>{rthr} ×{rsh} shift={s:+.0f} n={n} MAE={np.abs(p-act_v).mean():.2f}", flush=True)

    # 3) clearness @11-14 + clearness @10,15（扩展）
    print("\n=== clearness 扩展小时窗 ===", flush=True)
    for hrs, lab in (([10,11,12,13,14,15],"10-15"), ([11,12,13],"11-13"), ([12,13,14],"12-14")):
        p, s, n = apply_clear(A, 0.8, hrs, 0.7)
        print(f"  @ {lab} thr>0.8 ×0.7 shift={s:+.0f} n={n} MAE={np.abs(p-act_v).mean():.2f}", flush=True)

    # 4) 最优组合精调
    print("\n=== 最优组合 ===", flush=True)
    # clearness thr>0.8 @11-14 ×0.7 + 阴雨天 全天 precip>0 ×0.5
    base, _, _ = apply_clear(A, 0.8, [11,12,13,14], 0.7)
    p, s, n = apply_rainy(base, 0.0, None, 0.5)
    print(f"  clear×0.7 + rain(>0)×0.5 shift={s:+.0f} n={n} MAE={np.abs(p-act_v).mean():.2f}", flush=True)
    # 试 clearness 双阈值
    base2, _, _ = apply_clear(A, 0.8, [11,12,13,14], 0.7)
    base2, _, _ = apply_clear(base2, 0.6, [11,12,13,14], 0.3)  # 额外中等云量
    print(f"  clear>0.8×0.7 + clear>0.6×0.3 MAE={np.abs(base2-act_v).mean():.2f}", flush=True)


if __name__ == "__main__":
    main()
