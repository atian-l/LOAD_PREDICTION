# -*- coding: utf-8 -*-
"""exp52: 诊断 per-hour + drift_corr 修正的“天花板”。

三种 val MAE：
  (1) raw ensemble（无 per-hour、无 drift_corr）
  (2) OOF per-hour + OOF drift_corr（=生产 ~1512.63）
  (3) oracle per-hour + oracle drift_corr（val 残差本身，val-tuning，仅测天花板）

若 oracle 仍 >1500，说明这两类修正已榨干，需要新信号。
合规：仅诊断，不写产物；oracle 绝不进入生产。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd

from load_pred import config as C, data_loader as dl, features as F, model as M, train as T


def main():
    times, X, pred_load, actual = T.build_dataset()
    usable = T.usable_mask(times, pred_load, actual)
    mm = F.MismatchModel().fit(X, usable)
    X = mm.transform(X)
    print(f"features={X.shape[1]} usable={usable.sum()}", flush=True)

    val = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
           & actual.notna()).values
    print(f"val={val.sum()}", flush=True)

    cfg = dict(C.TRAIN_CONFIG)
    cfg["best_it_fixed"] = 80
    best_it, _ = T.determine_best_iteration(times, X, actual, usable, cfg)

    # ---- 全量集成（不设 hour_bias/drift_corr -> predict_load 返回 raw）----
    print("training full ensemble ...", flush=True)
    model = T.train_ensemble(times, X, pred_load, actual, usable, cfg, best_it)
    # raw val pred
    raw_pred = pd.Series(model.predict_load(X[val], pred_load[val]), index=times[val])
    actual_v = actual[val]
    mae_raw = (raw_pred - actual_v).abs().mean()
    print(f"(1) raw ensemble val MAE = {mae_raw:.2f}", flush=True)

    # ---- (2) OOF per-hour + OOF drift_corr（生产同款）----
    hour_bias, drift_corr = T.compute_hour_bias(times, X, pred_load, actual, usable, cfg, best_it)
    hours = pd.DatetimeIndex(times[val]).hour.values.astype(int)
    plwr = X["pl_weather_residual"].values.astype(float)[val]
    beta = drift_corr[0][1] if drift_corr else np.zeros(24)
    # 模型应用: corrected = raw - hour_bias + beta*plwr  (见 model.predict_load)
    pred_oof = raw_pred.values - hour_bias[hours] + beta[hours] * plwr
    mae_oof = np.abs(pred_oof - actual_v.values).mean()
    print(f"(2) OOF per-hour + OOF drift_corr val MAE = {mae_oof:.2f}  (应≈生产)", flush=True)

    # ---- (3) oracle：用 val 残差本身估 per-hour + β ----
    val_resid = raw_pred.values - actual_v.values
    hour_bias_o = np.zeros(24)
    for h in range(24):
        sel = (hours == h)
        if sel.any():
            hour_bias_o[h] = val_resid[sel].mean()
    beta_o = np.zeros(24)
    for h in (11, 12, 13, 14):
        sel = (hours == h)
        if sel.any():
            xh = plwr[sel]; yh = val_resid[sel]
            good = np.isfinite(xh) & np.isfinite(yh)
            d = float(np.dot(xh[good], xh[good]))
            if d > 0:
                beta_o[h] = float(np.dot(xh[good], yh[good]) / d)
    pred_or = raw_pred.values - hour_bias_o[hours] + beta_o[hours] * plwr
    mae_or = np.abs(pred_or - actual_v.values).mean()
    print(f"(3) ORACLE per-hour + ORACLE drift_corr val MAE = {mae_or:.2f}  <- 天花板", flush=True)
    print(flush=True)
    print(f"headroom (raw->oracle): {mae_raw - mae_or:.2f} MW", flush=True)
    print(f"gap to 1500 (oracle): {mae_or - 1500:.2f} MW", flush=True)


if __name__ == "__main__":
    main()
