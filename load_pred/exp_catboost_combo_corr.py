# -*- coding: utf-8 -*-
"""
CatBoost P1-2: dw_regonly 基础上 F1/F2 重估校正扫描

基础配置 = Depthwise + reg_only（P1-1 的 dw_regonly；若 P1-1 发现更优组合，改下方 HP_DW/cfg_ro）。
在 official 10 成员 + OOF 3 折训练一次后，对 {agg × λ} 网格每个组合：
  - 用该 agg/λ 重新聚合 OOF 残差 -> 重估 hour_bias/drift/threshold（严格，非复用 median 校正）
  - val 用 official members + 该 agg/λ + 重估校正 -> MAE/q99/Bias

AGGS = {median, trimmed:0.1, trimmed:0.2}  ×  LAMS = {1.0, 1.2, 1.5}  = 9 组合
（Tier1 F1/F2 的 caveat"校正不重估"在此修正：每组合重估 OOF 校正，Bias 应被 hour_bias 吸收）

合规：不修改生产脚本；仅 import 复用 hp._train_ensemble + tier1_diag._member_preds/_aggregate/
_predict_param + ab/train/features；6 条泄露不变量全保持。OOF 3 折，不接触官方验证集。
运行：python -m load_pred.exp_catboost_combo_corr   （4090 上约 1-3 min，OOF 训练一次复用）
"""
from __future__ import annotations
import sys
import time
import io
import contextlib
import copy
import warnings

import numpy as np
import pandas as pd

from . import config as C
from .train import build_dataset, usable_mask
from .features import MismatchModel, MosModel
from .exp_catboost_ab import V6_VAL_MAE
from . import exp_catboost_ab as ab
from . import exp_catboost_hp as hp
from .exp_catboost_tier1_diag import _member_preds, _aggregate, _predict_param

# 基础配置 = dw_regonly（Depthwise + reg_only + d8 + l2_8）。P1-1 若更优则改此处。
HP_DW = {"depth": 8, "lr": 0.03, "l2": 8.0, "bagging_temp": 1.0,
         "grow_policy": "Depthwise", "max_leaves": None}
BEST_IT = 80
L2_8_VAL_MAE = 1477.67
DW_VAL_MAE = 1463.88
REGONLY_VAL_MAE = 1466.02

AGGS = ["median", "trimmed:0.1", "trimmed:0.2"]
LAMS = [1.0, 1.2, 1.5]


def _agg_ens(mp, anchor_vals, agg, shrinkage):
    """聚合 + λ 收缩（无校正，用于 OOF pred）。"""
    ens = _aggregate(mp, agg)
    return anchor_vals + shrinkage * (ens - anchor_vals)


def _estimate_corr(resid, oof_mask, times, X, cfg):
    """在 OOF 残差上重估 hour_bias/drift_corr/threshold_corr（复刻 hp._compute_oof 校正段）。"""
    n_slots = int(cfg.get("hour_bias_slots", 96))
    step = 1440 // n_slots
    dt_all = pd.DatetimeIndex(times)
    mod_all = dt_all.hour.values * 60 + dt_all.minute.values
    slot_all = (mod_all // step).astype(int)
    h_all = dt_all.hour.values
    hour_bias = np.zeros(n_slots)
    for q in range(n_slots):
        m = oof_mask & (slot_all == q)
        if m.sum():
            hour_bias[q] = float(np.average(resid[m]))

    drift_corr = []
    dc = cfg.get("drift_corr")
    if dc:
        fn = dc["feature"]; hs = set(dc["hours"])
        feat = X[fn].values.astype(float)
        beta = np.zeros(24)
        for h in range(24):
            if h not in hs:
                continue
            m = oof_mask & (h_all == h)
            f = feat[m]; e = resid[m]
            good = np.isfinite(f) & np.isfinite(e)
            d = float(np.dot(f[good], f[good]))
            if d > 0:
                beta[h] = float(np.dot(f[good], e[good]) / d)
        drift_corr.append((fn, beta))

    threshold_corr = []
    for tc in cfg.get("threshold_corr", []):
        fn = tc["feature"]; op = tc.get("op", ">"); thr = tc["thr"]
        hl = tc["hours"]; shrink = float(tc["shrinkage"])
        feat = X[fn].values.astype(float)
        if op == "range":
            m = oof_mask & (feat >= thr[0]) & (feat < thr[1])
        elif op == ">=":
            m = oof_mask & (feat >= thr)
        elif op == "<":
            m = oof_mask & (feat < thr)
        elif op == "<=":
            m = oof_mask & (feat <= thr)
        else:
            m = oof_mask & (feat > thr)
        if hl is not None:
            m = m & np.isin(h_all, list(hl))
        shift = float(np.average(resid[m])) * shrink if m.sum() else 0.0
        threshold_corr.append({"feature": fn, "op": op, "thr": thr, "hours": hl, "shift": shift})
    return hour_bias, drift_corr, threshold_corr


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print(f"CatBoost P1-2: dw_regonly 上 F1/F2 重估校正 "
          f"({len(AGGS)}agg × {len(LAMS)}λ = {len(AGGS) * len(LAMS)}组合)")
    print(f"  基础: dw_regonly(Depthwise+reg_only)  对照: dw={DW_VAL_MAE} regonly={REGONLY_VAL_MAE} v6={V6_VAL_MAE}")
    print("=" * 74)

    print("[1] 构建数据集...")
    times, X, pred_load, actual = build_dataset()
    ab.times_global, ab.pred_load_global = times, pred_load
    usable = usable_mask(times, pred_load, actual)
    mm = MismatchModel().fit(X, usable)
    X = mm.transform(X)
    mos_model = MosModel(cols=C.TRAIN_CONFIG["mos"]["cols"],
                         alpha=C.TRAIN_CONFIG["mos"]["alpha"]).fit(X, actual, usable)
    anchor = pd.Series(mos_model.transform(X), index=X.index)
    feat_cols = list(X.columns)
    cfg = C.TRAIN_CONFIG
    cfg_ro = copy.deepcopy(cfg)
    cfg_ro["objectives"] = ["regression"]
    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    Xv = X[val_m]
    av = actual[val_m].to_numpy(np.float64)
    anchor_v = anchor[val_m].to_numpy(np.float64)
    print(f"    特征数={len(feat_cols)}  val点={int(val_m.sum())}")

    print("[2] 训练 official 10 成员 + OOF 3 折（dw_regonly，训练一次复用）...")
    with contextlib.redirect_stdout(io.StringIO()):
        official_members = hp._train_ensemble(X, actual, anchor, usable, cfg_ro,
                                              BEST_IT, feat_cols, HP_DW)
        oof_folds = []
        for te, vs, ve in cfg["best_it_folds"]:
            te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
            ftr = usable & np.asarray(times <= te)
            members = hp._train_ensemble(X, actual, anchor, ftr, cfg_ro,
                                         BEST_IT, feat_cols, HP_DW)
            oof_folds.append((vs, ve, members))
    print(f"    official成员={len(official_members)}  OOF折={len(oof_folds)}")

    print(f"\n[3] 逐组合重估校正 + val 评估 ...")
    print(f"  {'agg':14} {'λ':>5} {'val MAE':>10} {'q99':>8} {'Bias':>8} {'Δmedianλ1':>10}")
    rows = []
    baseline_mae = None
    for agg in AGGS:
        for lam in LAMS:
            # OOF pred（各折 members + agg/lam 聚合）
            oof_pred = pd.Series(np.nan, index=times)
            for vs, ve, members in oof_folds:
                fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
                if fva.sum() == 0:
                    continue
                mp = _member_preds(members, X[fva], anchor[fva].values, feat_cols)
                oof_pred[fva] = _agg_ens(mp, anchor[fva].values, agg, lam)
            oof_mask = usable & oof_pred.notna().values
            resid = (oof_pred - actual).values
            hour_bias, drift_corr, threshold_corr = _estimate_corr(
                resid, oof_mask, times, X, cfg_ro)
            # val
            mp_val = _member_preds(official_members, Xv, anchor_v, feat_cols)
            pred_val = _predict_param(mp_val, anchor_v, lam, agg, hour_bias,
                                      drift_corr, threshold_corr, Xv, None)
            e = pred_val - av
            mae = float(np.mean(np.abs(e)))
            if baseline_mae is None:
                baseline_mae = mae
            rows.append({"agg": agg, "lam": lam, "MAE": mae,
                         "q99": float(np.percentile(np.abs(e), 99)),
                         "Bias": float(np.mean(e))})
            print(f"  {agg:14} {lam:>5.1f} {mae:>10.2f} "
                  f"{float(np.percentile(np.abs(e), 99)):>8.0f} {float(np.mean(e)):>+8.1f} "
                  f"{mae - baseline_mae:>+10.2f}")

    print("\n" + "=" * 74)
    print(f"P1-2 dw_regonly F1/F2 重估校正对比（v6={V6_VAL_MAE}）")
    print("=" * 74)
    print(f"{'agg':14} {'λ':>5} {'MAE':>8} {'Δv6':>8} {'Δmedianλ1':>10} {'q99':>8} {'Bias':>8}")
    for r in rows:
        print(f"{r['agg']:14} {r['lam']:>5.1f} {r['MAE']:>8.2f} "
              f"{r['MAE'] - V6_VAL_MAE:>+8.2f} {r['MAE'] - baseline_mae:>+10.2f} "
              f"{r['q99']:>8.0f} {r['Bias']:>+8.1f}")
    best = min(rows, key=lambda r: r["MAE"])
    print(f"\n最优: agg={best['agg']} λ={best['lam']}  MAE={best['MAE']:.2f} "
          f"(Δv6 {best['MAE'] - V6_VAL_MAE:+.2f})")
    if best["MAE"] < V6_VAL_MAE:
        print(f"  -> 破 v6！dw_regonly + {best['agg']} + λ={best['lam']} "
              f"MAE={best['MAE']:.2f} < v6 {V6_VAL_MAE}")
    else:
        print(f"  -> 仍 > v6 {V6_VAL_MAE}，距 {best['MAE'] - V6_VAL_MAE:+.2f}MW")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
