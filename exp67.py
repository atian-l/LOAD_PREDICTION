# -*- coding: utf-8 -*-
"""exp67: 温度响应校正测试（exp66 发现冷天欠预测/热天过预测）。

模型虽有 temp/temp_sq，但冷天(<5C)仍欠预测 -1164、热天(>30C)过预测 +1200。
测试：阈值校正(temp<5C, temp>30C) + 二次回归校正，检查 OOF 迁移性。
仅诊断。
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
    base = np.abs(pred_v - act_v).mean()
    print(f"基线 = {base:.2f}", flush=True)

    # OOF 已校正残差
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

    temp_all = X["temp"].values.astype(float)
    temp_v = temp_all[val]
    print(f"  temp 范围: OOF[{np.nanmin(temp_all[oof_mask]):.1f}, {np.nanmax(temp_all[oof_mask]):.1f}]  val[{temp_v.min():.1f}, {temp_v.max():.1f}]", flush=True)

    def mae(p): return np.abs(p - act_v).mean()
    def shift_of(mask_all, shk):
        r = oof_resid[mask_all]
        return float(np.nanmean(r)) * shk if np.isfinite(r).sum() > 20 else 0.0

    print("\n=== temp 阈值校正（全天 / 仅白天 8-20）===", flush=True)
    print(f"  {'cond':>14} {'hrs':>6} {'shk':>5} {'oof_sh':>8} {'n_v':>5} {'MAE':>9} {'Δ':>8}", flush=True)
    for name, mraw in [("temp<0", temp_all<0), ("temp<5", temp_all<5), ("temp<8", temp_all<8),
                       ("temp>28", temp_all>28), ("temp>30", temp_all>30), ("temp>32", temp_all>32)]:
        for hrs_name, hrs in [("全天", None), ("8-20", [8,9,10,11,12,13,14,15,16,17,18,19,20])]:
            mask_all = oof_mask & mraw
            if hrs is not None: mask_all = mask_all & np.isin(h_all, hrs)
            mask_v = mraw[val]
            if hrs is not None: mask_v = mask_v & np.isin(pd.DatetimeIndex(times[val]).hour.values, hrs)
            for shk in [1.0, 0.7]:
                sh = shift_of(mask_all, shk)
                p = pred_v.copy()
                if sh != 0: p[mask_v] = p[mask_v] - sh
                print(f"  {name:>14} {hrs_name:>6} {shk:>5.1f} {sh:>8.1f} {mask_v.sum():>5} {mae(p):>9.2f} {mae(p)-base:>+8.2f}", flush=True)

    # 二次 temp 校正（polyfit，稳健）
    print("\n=== 二次 temp 回归校正（OOF 拟合，val 应用）===", flush=True)
    o_t = temp_all[oof_mask]; o_r = oof_resid[oof_mask]
    good = np.isfinite(o_t) & np.isfinite(o_r)
    o_t, o_r = o_t[good], o_r[good]
    # 分小时拟合？先全局
    for deg in [1, 2, 3]:
        coef = np.polyfit(o_t, o_r, deg)
        poly = np.poly1d(coef)
        corr_v = poly(temp_v)
        corr_v = np.nan_to_num(corr_v, nan=0.0)
        # 去均值（避免整体偏移）
        p = pred_v - (corr_v - corr_v.mean())
        print(f"  deg={deg} coef={coef.round(3)}  MAE(去均值)={mae(p):.2f}  Δ={mae(p)-base:+.2f}", flush=True)
        p2 = pred_v - corr_v
        print(f"         MAE(不去均值)={mae(p2):.2f}  Δ={mae(p2)-base:+.2f}", flush=True)

    # 分小时二次 temp（白天/夜间分别）
    print("\n=== 分时段二次 temp（白天8-20 / 夜间）===", flush=True)
    hours_v = pd.DatetimeIndex(times[val]).hour.values.astype(int)
    is_day = np.isin(h_all, list(range(8,21)))
    is_day_v = np.isin(hours_v, list(range(8,21)))
    for name, m_all, m_v in [("白天", oof_mask&is_day, is_day_v), ("夜间", oof_mask&~is_day, ~is_day_v)]:
        o_t = temp_all[m_all]; o_r = oof_resid[m_all]
        g = np.isfinite(o_t)&np.isfinite(o_r); o_t,o_r=o_t[g],o_r[g]
        if len(o_t)<50: continue
        coef = np.polyfit(o_t, o_r, 2); poly = np.poly1d(coef)
        c_v = np.nan_to_num(poly(temp_v), nan=0.0)
        # 仅在该时段应用
        p = pred_v.copy()
        corr = c_v - c_v[m_v].mean() if m_v.sum() else c_v
        p[m_v] = p[m_v] - corr[m_v]
        print(f"  {name}: coef={coef.round(3)}  MAE={mae(p):.2f}  Δ={mae(p)-base:+.2f}  (n_v={m_v.sum()})", flush=True)

    # 组合：temp<5 + temp>30 + clr0.2-0.5(最佳)
    print("\n=== 组合 ===", flush=True)
    clear = np.nan_to_num(X["clearness"].values.astype(float), nan=0.0)
    clear_v = clear[val]
    def combo(corrs):
        p = pred_v.copy()
        for mask_v, sh in corrs:
            if sh != 0: p[mask_v] = p[mask_v] - sh
        return p
    # 各 shift
    sh_cold = shift_of(oof_mask & (temp_all<5), 1.0)
    sh_hot = shift_of(oof_mask & (temp_all>30), 1.0)
    sh_clr = shift_of(oof_mask & (clear>=0.2)&(clear<0.5) & np.isin(h_all,[11,12,13,14]), 1.0)
    print(f"  shifts: cold(<5)={sh_cold:.1f} hot(>30)={sh_hot:.1f} clr0.2-0.5@11-14={sh_clr:.1f}", flush=True)
    c1 = combo([(temp_v<5, sh_cold), (temp_v>30, sh_hot)])
    print(f"  cold+hot                   MAE={mae(c1):.2f}  Δ={mae(c1)-base:+.2f}", flush=True)
    c2 = combo([(temp_v<5, sh_cold), (temp_v>30, sh_hot),
                (np.isin(hours_v,[11,12,13,14])&(clear_v>=0.2)&(clear_v<0.5), sh_clr)])
    print(f"  cold+hot+clr0.2-0.5@11-14  MAE={mae(c2):.2f}  Δ={mae(c2)-base:+.2f}", flush=True)
    for shk in [0.7, 1.0]:
        sc=shift_of(oof_mask&(temp_all<5),shk); sh=shift_of(oof_mask&(temp_all>30),shk)
        scl=shift_of(oof_mask&(clear>=0.2)&(clear<0.5)&np.isin(h_all,[11,12,13,14]),shk)
        c=combo([(temp_v<5,sc),(temp_v>30,sh),(np.isin(hours_v,[11,12,13,14])&(clear_v>=0.2)&(clear_v<0.5),scl)])
        print(f"  cold+hot+clr ×{shk}            MAE={mae(c):.2f}  Δ={mae(c)-base:+.2f}", flush=True)


if __name__ == "__main__":
    main()
