# -*- coding: utf-8 -*-
"""exp75 — 评估 96 维 Quarter Bias（用户风险优化建议）。

当前 hour_bias 为 24 维（逐小时）。测 96 维（逐 15min slot）/48 维（逐半小时）是否更优。
方法：训练一次集成 + 一次 OOF，从同一 OOF 残差估计三种粒度偏置 + drift_corr + threshold_corr，
手动应用到 val 比较。避免重复训练。仅诊断。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from load_pred import config as C, features as F, train as T


def main():
    times, X0, pred_load, actual = T.build_dataset()
    usable = T.usable_mask(times, pred_load, actual)
    mm = F.MismatchModel().fit(X0, usable); X = mm.transform(X0)
    cfg = dict(C.TRAIN_CONFIG); cfg["best_it_fixed"] = 80
    val = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END)) & actual.notna()).values

    # 训练完整集成（无 hour_bias），取 val 上的原始集成预测（中位数）
    model = T.train_ensemble(times, X, pred_load, actual, usable, cfg, 80)
    pl_v = pred_load[val].values.astype(float)
    member_preds = np.empty((len(model.members), val.sum()), dtype=float)
    for i, (booster, is_res) in enumerate(zip(model.members, model.member_residual)):
        raw = booster.predict(X[val][model.feature_cols])
        member_preds[i] = pl_v + raw if is_res else raw
    ens_v = np.median(member_preds, axis=0)
    pred_v = pl_v + model.shrinkage * (ens_v - pl_v)  # 收缩后、未校正
    a = actual[val].values

    # OOF（3 折）原始预测 → 残差
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
    h_all = pd.DatetimeIndex(times).hour.values.astype(int)
    mod_all = pd.DatetimeIndex(times).hour.values * 60 + pd.DatetimeIndex(times).minute.values
    quarter_all = (mod_all // 15).astype(int)
    half_all = (mod_all // 30).astype(int)

    # 三种粒度偏置（OOF 残差均值）
    hb24 = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum(): hb24[h] = float(np.average(resid[m]))
    hb48 = np.zeros(48)
    for q in range(48):
        m = oof_mask & (half_all == q)
        if m.sum(): hb48[q] = float(np.average(resid[m]))
    hb96 = np.zeros(96)
    for q in range(96):
        m = oof_mask & (quarter_all == q)
        if m.sum(): hb96[q] = float(np.average(resid[m]))

    # drift_corr（与生产同：pl_wr 午间 11-14 逐小时 β，从 resid 估计）
    feat = X["pl_weather_residual"].values.astype(float)
    beta = np.zeros(24)
    for h in [11, 12, 13, 14]:
        m = oof_mask & (h_all == h)
        f = feat[m]; e = resid[m]
        good = np.isfinite(f) & np.isfinite(e)
        d = float(np.dot(f[good], f[good]))
        if d > 0: beta[h] = float(np.dot(f[good], e[good]) / d)

    # threshold_corr（生产 4 项，从 resid 估计）
    tc_list = []
    for tc in cfg.get("threshold_corr", []):
        fn = tc["feature"]; op = tc.get("op", ">"); thr = tc["thr"]; hrs = tc["hours"]
        fv = X[fn].values.astype(float)
        if op == "range":
            lo, hi = thr; m = oof_mask & (fv >= lo) & (fv < hi)
        elif op == "<": m = oof_mask & (fv < thr)
        else: m = oof_mask & (fv > thr)
        if hrs is not None: m = m & np.isin(h_all, list(hrs))
        sh = float(np.average(resid[m])) * float(tc["shrinkage"]) if m.sum() else 0.0
        tc_list.append((fn, op, thr, hrs, sh))

    # val 上的小时/quarter/half 索引 + drift feat
    h_v = pd.DatetimeIndex(times[val]).hour.values.astype(int)
    mod_v = pd.DatetimeIndex(times[val]).hour.values * 60 + pd.DatetimeIndex(times[val]).minute.values
    q_v = (mod_v // 15).astype(int); hf_v = (mod_v // 30).astype(int)
    feat_v = X["pl_weather_residual"].values[val].astype(float)

    def apply(hb, idx_v, tag):
        p = pred_v.copy()
        p = p - hb[idx_v]
        p = p + beta[h_v] * feat_v
        Xv = X[val]
        for fn, op, thr, hrs, sh in tc_list:
            fv = Xv[fn].values.astype(float)
            if op == "range":
                lo, hi = thr; sel = (fv >= lo) & (fv < hi)
            elif op == "<": sel = fv < thr
            else: sel = fv > thr
            if hrs is not None: sel = sel & np.isin(h_v, list(hrs))
            if sh != 0.0: p[sel] = p[sel] - sh
        p = np.clip(p, 0.0, None)
        mae = np.abs(p - a).mean()
        print(f"  [{tag}] val MAE={mae:.2f}  Δ={mae-1461.63:+.2f}", flush=True)
        return mae

    print("=== Quarter Bias 粒度对比（基线 v4=1461.63）===", flush=True)
    apply(hb24, h_v, "24维 逐小时(生产)")
    apply(hb48, hf_v, "48维 逐半小时")
    apply(hb96, q_v, "96维 逐15min")


if __name__ == "__main__":
    main()
