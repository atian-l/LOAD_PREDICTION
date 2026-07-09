# -*- coding: utf-8 -*-
"""exp68: 自动阈值校正挖掘（贪心）。

系统扫描所有 (特征, 阈值, 时段) 候选，用 OOF 已校正残差估 shift，贪心叠加能持续降 MAE 的校正。
避免手动选 pocket 的重叠/重复计数。仅诊断。
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
    pred_v0 = model.predict_load(X[val], pred_load[val]).copy()
    act_v = actual[val].values
    base = np.abs(pred_v0 - act_v).mean()
    print(f"基线(生产, 含现有 threshold_corr) = {base:.2f}", flush=True)

    # OOF 已校正残差（现有校正后）
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
    oof_resid0 = oof_cor - actual.values  # 初始 OOF 残差（现有校正后）

    hours_v = pd.DatetimeIndex(times[val]).hour.values.astype(int)
    # 候选特征
    feat_vals = {
        "clearness": np.nan_to_num(X["clearness"].values.astype(float), nan=0.0),
        "precip": np.nan_to_num(X["precip"].values.astype(float), nan=0.0),
        "temp": np.nan_to_num(X["temp"].values.astype(float), nan=0.0),
        "irrad": np.nan_to_num((X["irrad"].values if "irrad" in X.columns else X["光伏_辐照度"].values).astype(float), nan=0.0),
        "pl_wr": np.nan_to_num(X["pl_weather_residual"].values.astype(float), nan=0.0),
        "pl": np.nan_to_num(pred_load.values.astype(float), nan=0.0),
    }
    wind_col = "风电_风速" if "风电_风速" in X.columns else ("w_风电_风速" if "w_风电_风速" in X.columns else None)
    if wind_col: feat_vals["wind"] = np.nan_to_num(X[wind_col].values.astype(float), nan=0.0)

    # 候选时段
    hour_bands = {"全天": None, "11-14": [11,12,13,14], "9-15": [9,10,11,12,13,14,15], "8-20": list(range(8,21))}

    # 构建候选列表：(name, feat_val_all, op, thr, hours_list)
    cands = []
    for fname, fv in feat_vals.items():
        qs = np.quantile(fv[usable & np.isfinite(fv)], [0.1,0.25,0.5,0.75,0.9])
        thrs = sorted(set([0.0] + list(qs)))
        for thr in thrs:
            for op in [">", "<="]:
                for hb, hrs in hour_bands.items():
                    cands.append((f"{fname}{op}{thr:.2f}@{hb}", fv, op, thr, hrs))
    print(f"候选数: {len(cands)}", flush=True)

    def make_mask(fv, op, thr, hrs, hh):
        if op == ">": m = fv > thr
        else: m = fv <= thr
        if hrs is not None:
            m = m & np.isin(hh, hrs)
        return m

    # 贪心
    pred_v = pred_v0.copy()
    oof_resid = oof_resid0.copy()
    applied = []
    for it in range(8):
        best = None
        for name, fv, op, thr, hrs in cands:
            m_all = oof_mask & make_mask(fv, op, thr, hrs, h_all)
            if m_all.sum() < 100: continue
            sh = float(np.nanmean(oof_resid[m_all]))
            if abs(sh) < 100: continue
            m_v = make_mask(fv[val], op, thr, hrs, hours_v)
            if m_v.sum() < 30: continue
            p = pred_v.copy()
            p[m_v] = p[m_v] - sh
            d = np.abs(p - act_v).mean() - np.abs(pred_v - act_v).mean()
            if best is None or d < best[0]:
                best = (d, name, op, thr, hrs, fv, sh, m_v, m_all)
        if best is None or best[0] >= -0.5:
            print(f"  [贪心 {it}] 无更多改善 (best Δ={best[0] if best else None})", flush=True)
            break
        d, name, op, thr, hrs, fv, sh, m_v, m_all = best
        pred_v = pred_v.copy(); pred_v[m_v] = pred_v[m_v] - sh
        oof_resid = oof_resid.copy(); oof_resid[m_all] = oof_resid[m_all] - sh
        applied.append((name, sh, int(m_v.sum()), d))
        cur = np.abs(pred_v - act_v).mean()
        print(f"  [贪心 {it}] +{name}  shift={sh:+.1f} n_v={m_v.sum()}  Δ={d:+.2f}  累计MAE={cur:.2f}", flush=True)

    print(f"\n=== 最终: {np.abs(pred_v-act_v).mean():.2f}  (Δ基线 {np.abs(pred_v-act_v).mean()-base:+.2f}) ===", flush=True)
    print("已应用:", flush=True)
    for a in applied: print(f"  {a}", flush=True)


if __name__ == "__main__":
    main()
