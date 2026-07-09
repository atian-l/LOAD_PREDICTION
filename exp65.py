# -*- coding: utf-8 -*-
"""exp65: 测试多波段 clearness + irrad 阈值校正（exp64 发现的可迁移 pocket）。

exp64 发现（val_bias / oof_bias，后者为已校正OOF残差均值）：
  clearness 0.2-0.5: -458 / -836  (n=1086)
  clearness 0.5-0.8: +150 / +793  (n=2275)
  clearness >=0.8  : -168 / +347  (n=1591, 已被现有 >0.8 校正部分覆盖)
  irrad >=800      : -802 / -1779 (n=468)
  irrad 400-800    : +270 / +1191 (n=2260)
新校正 shift 从"已校正OOF残差"估计（避免与现有 hour_bias/drift/threshold 重复计数）。
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
    base_mae = np.abs(pred_v - act_v).mean()
    print(f"基线(生产) = {base_mae:.2f}", flush=True)

    # OOF 已校正残差（新校正的估计基础）
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

    clear = np.nan_to_num(X["clearness"].values.astype(float), nan=0.0)
    irrad = np.nan_to_num((X["irrad"].values if "irrad" in X.columns else X["光伏_辐照度"].values).astype(float), nan=0.0)
    hours_v = pd.DatetimeIndex(times[val]).hour.values.astype(int)
    clear_v = clear[val]; irrad_v = irrad[val]

    def shift_of(mask_all, shrink):
        r = oof_resid[mask_all]
        return float(r.mean()) * shrink if len(r) > 20 else 0.0

    def apply(pred, mask_v, sh):
        p = pred.copy()
        if sh != 0.0: p[mask_v] = p[mask_v] - sh
        return p

    def mae(p): return np.abs(p - act_v).mean()

    MID = [9,10,11,12,13,14,15]
    MID1411 = [11,12,13,14]
    print(flush=True)

    # 单独测试每个新 pocket（@9-15 和 @11-14），shrinkage 1.0 和 0.7
    pockets = [
        ("clr 0.2-0.5", clear, (clear>=0.2)&(clear<0.5)),
        ("clr 0.5-0.8", clear, (clear>=0.5)&(clear<0.8)),
        ("clr >=0.8",   clear, (clear>=0.8)),
        ("irrad>=800",  irrad, (irrad>=800)),
        ("irrad 400-800", irrad, (irrad>=400)&(irrad<800)),
        ("irrad 100-400", irrad, (irrad>=100)&(irrad<400)),
    ]
    print("=== 单 pocket（@9-15 / @11-14, ×1.0 与 ×0.7）===", flush=True)
    print(f"  {'pocket':>16} {'hrs':>6} {'shrink':>7} {'oof_sh':>8} {'n_v':>5} {'MAE':>9} {'Δ':>8}", flush=True)
    for name, feat, mask_all_raw in pockets:
        for hrs_name, hrs in [("9-15", MID), ("11-14", MID1411)]:
            mask_all = oof_mask & mask_all_raw & np.isin(h_all, hrs)
            mask_v = np.isin(hours_v, hrs) & mask_all_raw[val]
            for shk in [1.0, 0.7]:
                sh = shift_of(mask_all, shk)
                p = apply(pred_v, mask_v, sh)
                print(f"  {name:>16} {hrs_name:>6} {shk:>7.1f} {sh:>8.1f} {mask_v.sum():>5} {mae(p):>9.2f} {mae(p)-base_mae:>+8.2f}", flush=True)
    print(flush=True)

    # 组合：现有(>0.8×0.7) + 新 pocket
    print("=== 组合（基于现有 >0.8×0.7 + 新增）===", flush=True)
    def combo(new_pockets, hrs=MID):
        p = pred_v.copy()
        for name, feat, mraw, shk in new_pockets:
            mask_all = oof_mask & mraw & np.isin(h_all, hrs)
            mask_v = np.isin(hours_v, hrs) & mraw[val]
            sh = shift_of(mask_all, shk)
            p = apply(p, mask_v, sh)
        return p
    cands = [
        ("+clr0.2-0.5×0.7", [("c025", clear, (clear>=0.2)&(clear<0.5), 0.7)]),
        ("+clr0.5-0.8×0.7", [("c058", clear, (clear>=0.5)&(clear<0.8), 0.7)]),
        ("+irrad>=800×0.7", [("i800", irrad, (irrad>=800), 0.7)]),
        ("+irrad400-800×0.7", [("i48", irrad, (irrad>=400)&(irrad<800), 0.7)]),
        ("+clr0.2-0.5+0.5-0.8×0.7", [("c025",clear,(clear>=0.2)&(clear<0.5),0.7),("c058",clear,(clear>=0.5)&(clear<0.8),0.7)]),
        ("+all4×0.7", [("c025",clear,(clear>=0.2)&(clear<0.5),0.7),("c058",clear,(clear>=0.5)&(clear<0.8),0.7),("i800",irrad,(irrad>=800),0.7),("i48",irrad,(irrad>=400)&(irrad<800),0.7)]),
        ("+all4×1.0", [("c025",clear,(clear>=0.2)&(clear<0.5),1.0),("c058",clear,(clear>=0.5)&(clear<0.8),1.0),("i800",irrad,(irrad>=800),1.0),("i48",irrad,(irrad>=400)&(irrad<800),1.0)]),
    ]
    for name, pcs in cands:
        p = combo(pcs)
        print(f"  {name:>28} MAE={mae(p):.2f}  Δ={mae(p)-base_mae:+.2f}", flush=True)


if __name__ == "__main__":
    main()
