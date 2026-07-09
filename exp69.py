# -*- coding: utf-8 -*-
"""exp69: 可达性下界诊断。

1. OOF MAE（含校正）—— 模型在 2025 训练期的表现（模型地板）。
2. Oracle per-day / per-(day,hour)（val 调参，泄露，仅界定可校正信号）。
3. 残差自相关（是否有持续性/persistence 信号可利用）。
判定 1300 是否无泄露可达。仅诊断。
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
    model.hour_bias, model.drift_corr, model.threshold_corr = T.compute_hour_bias(
        times, X, pred_load, actual, usable, cfg, 80)
    pred_v = model.predict_load(X[val], pred_load[val])
    act_v = actual[val].values
    resid_v = pred_v - act_v
    base = np.abs(resid_v).mean()
    print(f"val MAE(含校正) = {base:.2f}", flush=True)

    # OOF（含校正）
    oof = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0: continue
        fm = T.train_ensemble(times, X, pred_load, actual, ftr, cfg, 80)
        oof[fva] = fm.predict_load(X[fva], pred_load[fva])
    oof_mask = usable & oof.notna().values
    h_all = pd.DatetimeIndex(times).hour.values.astype(int)
    oof_cor = oof.values - model.hour_bias[h_all]
    for fn, beta in model.drift_corr:
        oof_cor = oof_cor + np.asarray(beta, dtype=float)[h_all] * X[fn].values.astype(float)
    for fn, thr, hrs, sh in model.threshold_corr:
        fv = X[fn].values.astype(float); sel = fv > thr
        if hrs is not None: sel = sel & np.isin(h_all, hrs)
        oof_cor[sel] = oof_cor[sel] - sh
    oof_resid = oof_cor - actual.values
    oof_mae = np.abs(oof_resid[oof_mask]).mean()
    print(f"OOF MAE(含校正, 2025训练期) = {oof_mae:.2f}  (val-OOF 漂移 = {base-oof_mae:.1f})", flush=True)

    # Oracle per-day（val 调参，泄露）
    t_idx = pd.DatetimeIndex(times[val])
    df = pd.DataFrame({"d": t_idx.date, "h": t_idx.hour, "r": resid_v})
    day_mean = df.groupby("d")["r"].mean()
    df["day_corr"] = df["d"].map(day_mean)
    oracle_day = np.abs(resid_v - df["day_corr"].values).mean()
    print(f"Oracle per-day MAE = {oracle_day:.2f}  (日级可校正信号 ≈ {base-oracle_day:.1f})", flush=True)

    # Oracle per-(day,hour)
    dh_mean = df.groupby(["d","h"])["r"].mean()
    df["dh_corr"] = df.apply(lambda r: dh_mean.get((r["d"], r["h"]), 0), axis=1)
    oracle_dh = np.abs(resid_v - df["dh_corr"].values).mean()
    print(f"Oracle per-(day,hour) MAE = {oracle_dh:.2f}  (日×时级可校正信号 ≈ {base-oracle_dh:.1f})", flush=True)

    # Oracle per-hour（已由 hour_bias 部分校正，看残余）
    h_mean = df.groupby("h")["r"].mean()
    df["h_corr"] = df["h"].map(h_mean)
    oracle_h = np.abs(resid_v - df["h_corr"].values).mean()
    print(f"Oracle per-hour MAE = {oracle_h:.2f}  (残余小时偏置 ≈ {base-oracle_h:.1f})", flush=True)

    # 残差自相关（lag 1点=15min, 96点=1天）：是否有 persistence
    print("\n=== 残差自相关（val，按 lag）===", flush=True)
    r = pd.Series(resid_v, index=t_idx)
    for lag in [1, 4, 8, 96, 192]:
        if lag < len(r):
            ac = r.autocorr(lag=lag)
            print(f"  lag={lag:>3} ({'15min' if lag==1 else str(lag//96)+'天' if lag>=96 else str(lag//4)+'h'}): autocorr={ac:+.3f}", flush=True)

    # 1天持续性校正：用昨天的日均值残差修正今天（模拟运行时：D日已知D-1的残差）
    # val 内：每天的日残差 = 前一天日残差（shift 1 day）
    day_r = pd.Series(resid_v, index=t_idx).groupby(t_idx.date).mean()
    day_r_prev = day_r.shift(1)
    persist_corr = t_idx.date.map(day_r_prev)
    persist_pred = pred_v - np.where(np.isfinite(persist_corr), persist_corr, 0)
    persist_mae = np.abs(persist_pred - act_v).mean()
    print(f"\n1天持续性校正(用前日残差) MAE = {persist_mae:.2f}  Δ={persist_mae-base:+.2f}", flush=True)
    # 2天平均持续性
    day_r_roll2 = day_r.shift(1).rolling(2, min_periods=1).mean()
    persist_corr2 = t_idx.date.map(day_r_roll2)
    p2 = pred_v - np.where(np.isfinite(persist_corr2), persist_corr2, 0)
    print(f"2天滚动持续性 MAE = {np.abs(p2-act_v).mean():.2f}  Δ={np.abs(p2-act_v).mean()-base:+.2f}", flush=True)

    print(f"\n=== 结论 ===", flush=True)
    print(f"  val={base:.0f} OOF={oof_mae:.0f} oracle_day={oracle_day:.0f} oracle_dh={oracle_dh:.0f}", flush=True)
    print(f"  日级可校正信号(oracle)={base-oracle_day:.0f} MW；其中 OOF 已捕获的≈{base-oof_mae:.0f}？(OOF是2025,不完全可比)", flush=True)


if __name__ == "__main__":
    main()
