# -*- coding: utf-8 -*-
"""exp53: per-hour 偏置的聚合统计量选择（mean / median / trimmed_mean）。

发现（exp52）：oracle(val-tuned) per-hour 用 mean 反而比 raw 差（1532 vs 1528），
因为负荷误差右偏 -> mean > median -> 减 mean 把主体往负移。MAE 最优是 median。
本实验：用同一份 3-fold OOF 残差，对比 mean/median/trimmed(10%) 作为 hour_bias。
合规：仅 OOF（训练期），不写产物。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd

from load_pred import config as C, data_loader as dl, features as F, model as M, train as T


def trimmed_mean(x, trim=0.1):
    if len(x) == 0:
        return 0.0
    x = np.sort(x)
    k = int(len(x) * trim)
    if k > 0 and len(x) > 2 * k:
        x = x[k:-k]
    return float(np.mean(x))


def main():
    times, X, pred_load, actual = T.build_dataset()
    usable = T.usable_mask(times, pred_load, actual)
    mm = F.MismatchModel().fit(X, usable)
    X = mm.transform(X)
    val = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
           & actual.notna()).values

    cfg = dict(C.TRAIN_CONFIG)
    cfg["best_it_fixed"] = 80
    best_it = 80

    # 3-fold OOF（手动，收集 oof_pred）
    oof_pred = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        fm = T.train_ensemble(times, X, pred_load, actual, ftr, cfg, best_it)
        oof_pred[fva] = fm.predict_load(X[fva], pred_load[fva])
    oof_mask = usable & oof_pred.notna().values
    oof_resid = (oof_pred - actual).values  # 全长度
    h_all = pd.DatetimeIndex(times).hour.values

    # drift β（OOF，午间 11-14）—— 与生产一致
    plwr_all = X["pl_weather_residual"].values.astype(float)
    beta = np.zeros(24)
    for h in (11, 12, 13, 14):
        m = oof_mask & (h_all == h)
        f = plwr_all[m]; e = oof_resid[m]
        good = np.isfinite(f) & np.isfinite(e)
        d = float(np.dot(f[good], f[good]))
        if d > 0:
            beta[h] = float(np.dot(f[good], e[good]) / d)

    # 全量集成 raw val pred
    print("training full ensemble ...", flush=True)
    model = T.train_ensemble(times, X, pred_load, actual, usable, cfg, best_it)
    raw_v = model.predict_load(X[val], pred_load[val])
    actual_v = actual[val].values
    hours_v = pd.DatetimeIndex(times[val]).hour.values.astype(int)
    plwr_v = plwr_all[val]
    mae_raw = np.abs(raw_v - actual_v).mean()
    print(f"raw val MAE = {mae_raw:.2f}", flush=True)
    print(flush=True)

    # 三种聚合的 OOF hour_bias
    for name, agg in [("mean", lambda x: float(np.mean(x)) if len(x) else 0.0),
                      ("median", lambda x: float(np.median(x)) if len(x) else 0.0),
                      ("trimmed10", lambda x: trimmed_mean(x, 0.1))]:
        hb = np.zeros(24)
        for h in range(24):
            m = oof_mask & (h_all == h)
            if m.sum():
                hb[h] = agg(oof_resid[m])
        pred = raw_v - hb[hours_v] + beta[hours_v] * plwr_v
        mae = np.abs(pred - actual_v).mean()
        print(f"OOF hour_bias[{name:10s}] + drift  val MAE = {mae:.2f}  (Δraw {mae-mae_raw:+.2f})", flush=True)

    print(flush=True)
    # 同样测 oracle（val 残差）三种聚合 —— 看天花板
    val_resid = raw_v - actual_v
    for name, agg in [("mean", lambda x: float(np.mean(x)) if len(x) else 0.0),
                      ("median", lambda x: float(np.median(x)) if len(x) else 0.0),
                      ("trimmed10", lambda x: trimmed_mean(x, 0.1))]:
        hb = np.zeros(24)
        for h in range(24):
            sel = (hours_v == h)
            if sel.any():
                hb[h] = agg(val_resid[sel])
        # oracle β
        bo = np.zeros(24)
        for h in (11, 12, 13, 14):
            sel = (hours_v == h)
            xh = plwr_v[sel]; yh = val_resid[sel]
            good = np.isfinite(xh) & np.isfinite(yh)
            d = float(np.dot(xh[good], xh[good]))
            if d > 0:
                bo[h] = float(np.dot(xh[good], yh[good]) / d)
        pred = raw_v - hb[hours_v] + bo[hours_v] * plwr_v
        mae = np.abs(pred - actual_v).mean()
        print(f"ORACLE hour_bias[{name:10s}] + drift  val MAE = {mae:.2f}  (Δraw {mae-mae_raw:+.2f})", flush=True)


if __name__ == "__main__":
    main()
