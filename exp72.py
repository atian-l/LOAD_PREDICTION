# -*- coding: utf-8 -*-
"""exp72: 时段×温度带 校正 + 天气不确定性信号 + 最佳清洁组合。

温度响应可能随时段不同（冷晨供暖爬坡 vs 冷午间）。测试 (时段×温度带) 校正。
天气集合离散度(_std)可能信号化预报不确定性→偏置。仅诊断。
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
    cfg = dict(C.TRAIN_CONFIG); cfg["best_it_fixed"]=80
    model = T.train_ensemble(times, X, pred_load, actual, usable, cfg, 80)
    model.hour_bias, model.drift_corr, model.threshold_corr = T.compute_hour_bias(times, X, pred_load, actual, usable, cfg, 80)
    pred_v = model.predict_load(X[val], pred_load[val]); a = actual[val].values
    base = np.abs(pred_v-a).mean()
    print(f"基线={base:.2f}", flush=True)

    oof = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te); fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum()==0: continue
        fm = T.train_ensemble(times, X, pred_load, actual, ftr, cfg, 80)
        oof[fva] = fm.predict_load(X[fva], pred_load[fva])
    oof_mask = usable & oof.notna().values
    h_all = pd.DatetimeIndex(times).hour.values.astype(int)
    oof_cor = oof.values - model.hour_bias[h_all]
    for fn, beta in model.drift_corr:
        oof_cor = oof_cor + np.asarray(beta,float)[h_all]*X[fn].values.astype(float)
    for fn, thr, hrs, sh in model.threshold_corr:
        fv = X[fn].values.astype(float); sel = fv > thr
        if hrs is not None: sel = sel & np.isin(h_all, hrs)
        oof_cor[sel] = oof_cor[sel] - sh
    oof_resid = oof_cor - actual.values

    hours_v = pd.DatetimeIndex(times[val]).hour.values.astype(int)
    temp_all = np.nan_to_num(X["temp"].values.astype(float), nan=0.0); temp_v = temp_all[val]
    clear = np.nan_to_num(X["clearness"].values.astype(float), nan=0.0); clear_v = clear[val]
    irrad_all = np.nan_to_num((X["irrad"].values if "irrad" in X.columns else X["光伏_辐照度"].values).astype(float), nan=0.0); irrad_v = irrad_all[val]

    def shift_of(m_all, shk=1.0):
        r = oof_resid[m_all]; return float(np.nanmean(r))*shk if np.isfinite(r).sum()>20 else 0.0
    def mae(p): return np.abs(p-a).mean()

    # 时段定义
    bands = {"晨6-10":[6,7,8,9,10], "午11-14":[11,12,13,14], "晚15-20":[15,16,17,18,19,20], "夜21-5":[21,22,23,0,1,2,3,4,5]}
    print("\n=== 时段×温度带（冷<8 / 热>28）校正 ===", flush=True)
    p = pred_v.copy()
    for bn, hrs in bands.items():
        for name, mraw in [("冷<8", temp_all<8), ("热>28", temp_all>28)]:
            m_all = oof_mask & mraw & np.isin(h_all, hrs)
            sh = shift_of(m_all)
            m_v = np.isin(hours_v, hrs) & mraw[val]
            if m_v.sum()<30 or abs(sh)<100: continue
            pp = p.copy(); pp[m_v] = pp[m_v]-sh
            print(f"  {bn} {name}: shift={sh:+.0f} n_v={m_v.sum()} Δ={mae(pp)-mae(p):+.2f} (累计{mae(pp):.2f})", flush=True)
            # 若改善则保留
            if mae(pp) < mae(p): p = pp
    print(f"  → 时段×温度组合 MAE={mae(p):.2f} Δ={mae(p)-base:+.2f}", flush=True)

    # 清洁最佳组合（现有 + temp<8 + clr0.2-0.5@11-14 + irrad低@9-15）
    print("\n=== 清洁组合 ===", flush=True)
    p2 = pred_v.copy()
    cands = [
        ("temp<8@全天", temp_all<8, None),
        ("clr0.2-0.5@11-14", (clear>=0.2)&(clear<0.5), [11,12,13,14]),
        ("irrad<400@9-15", irrad_all<400, [9,10,11,12,13,14,15]),
        ("temp<5@全天", temp_all<5, None),
        ("temp>30@全天", temp_all>30, None),
    ]
    for name, mraw, hrs in cands:
        m_all = oof_mask & mraw & (np.isin(h_all,hrs) if hrs else True)
        sh = shift_of(m_all)
        m_v = mraw[val] & (np.isin(hours_v,hrs) if hrs else True)
        pp = p2.copy(); pp[m_v]=pp[m_v]-sh
        print(f"  +{name}: shift={sh:+.0f} n_v={m_v.sum()} → MAE={mae(pp):.2f} Δ={mae(pp)-mae(p2):+.2f}", flush=True)
    # 贪心清洁
    print("  贪心清洁:", flush=True)
    p3 = pred_v.copy(); used=[]
    for _ in range(6):
        best=None
        for name, mraw, hrs in cands:
            m_all = oof_mask & mraw & (np.isin(h_all,hrs) if hrs else True)
            sh = shift_of(m_all)
            m_v = mraw[val] & (np.isin(hours_v,hrs) if hrs else True)
            if m_v.sum()<30 or abs(sh)<100: continue
            pp = p3.copy(); pp[m_v]=pp[m_v]-sh
            d = mae(pp)-mae(p3)
            if best is None or d<best[0]: best=(d,name,sh,m_v,m_all)
        if best is None or best[0]>=-0.3: break
        d,name,sh,m_v,m_all = best
        p3 = p3.copy(); p3[m_v]=p3[m_v]-sh
        print(f"    +{name} shift={sh:+.0f} n_v={m_v.sum()} Δ={d:+.2f} 累计={mae(p3):.2f}", flush=True)
    print(f"  → 清洁贪心 MAE={mae(p3):.2f} Δ={mae(p3)-base:+.2f}", flush=True)

    # 天气不确定性(_std)作为偏置信号
    print("\n=== 天气不确定性(_std) ===", flush=True)
    std_cols = [c for c in X.columns if "_std" in c and "温度" in c]
    if std_cols:
        sc = std_cols[0]
        std_all = np.nan_to_num(X[sc].values.astype(float), nan=0.0); std_v = std_all[val]
        print(f"  使用 {sc}: 范围[{std_all[oof_mask].min():.2f},{std_all[oof_mask].max():.2f}]", flush=True)
        for thr in np.quantile(std_all[oof_mask], [0.75, 0.9]):
            m_all = oof_mask & (std_all>thr); sh=shift_of(m_all)
            m_v = std_v>thr
            pp=pred_v.copy(); pp[m_v]=pp[m_v]-sh
            print(f"  {sc}>{thr:.2f}: shift={sh:+.0f} n_v={m_v.sum()} MAE={mae(pp):.2f} Δ={mae(pp)-base:+.2f}", flush=True)


if __name__=="__main__":
    main()
