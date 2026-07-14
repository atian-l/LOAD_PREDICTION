# -*- coding: utf-8 -*-
"""
CatBoost P2-7: OOF 场景权重选择 + 更细场景门控（验证 P2-6 的 1438.30 是否稳健）

背景：P2-6 场景门控融合 1438.30（-7.32MW），clear_noon w=1.0（CatBoost 晴天午间 -58MW）。
但场景 w 在 val 上选，有过拟合风险。本脚本用 OOF 选场景 w，val 验证稳健性。

流程：
  1. build_dataset + MismatchModel + MOS
  2. v6 LightGBM official + OOF 3 折（含校正）-> pred_lgb_val, oof_pred_lgb
  3. CatBoost d10 official + OOF 3 折（含校正）-> pred_cat_val, oof_pred_cat
  4. 更细场景定义（7 场景，互斥）：
     clear_noon_high(clr>=0.9@11-14) / clear_noon_mid(clr[0.8,0.9)@11-14) /
     halfclear_noon(clr[0.5,0.8)@11-14) / cloudy_noon(clr[0.2,0.5)@11-14) /
     rainy(precip>0) / cold(temp<8) / baseline
  5. OOF 选 w：各场景在 OOF 上选最优 w（粗步长 0.1，防过拟合）
  6. val 验证：用 OOF w 在 val 算 MAE；对比 val 自选 w
  7. OOF/val 一致性 -> 判断 1438 是否稳健

合规: 不修改生产脚本; 仅 import train/hp/ab/features; 6 条泄露不变量全保持; val eval-only;
      OOF 3 折全在训练期内; actual 仅 target/MOS/评估。
运行: python -m load_pred.exp_catboost_gate_oof  (本地 3060 约 8 min)
"""
from __future__ import annotations
import sys
import time
import copy
import io
import contextlib
import warnings

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from . import config as C
from .train import build_dataset, usable_mask, train_ensemble, compute_hour_bias
from .features import MismatchModel, MosModel
from .exp_catboost_ab import V6_VAL_MAE, _predict_load
from . import exp_catboost_ab as ab
from . import exp_catboost_hp as hp

HP_DW = {"depth": 10, "lr": 0.03, "l2": 8.0, "bagging_temp": 1.0,
         "grow_policy": "Depthwise", "max_leaves": None}
BEST_IT = 80

SCENES = ["clear_noon_high", "clear_noon_mid", "halfclear_noon",
          "cloudy_noon", "rainy", "cold", "baseline"]


def _scene_masks(clearness, precip, temp, hours):
    """返回互斥场景 mask dict（优先级：午间 clear/cloudy > rainy > cold > baseline）。"""
    mid = (hours >= 11) & (hours <= 14)
    raw = {
        "clear_noon_high": (clearness >= 0.9) & mid,
        "clear_noon_mid": ((clearness >= 0.8) & (clearness < 0.9)) & mid,
        "halfclear_noon": ((clearness >= 0.5) & (clearness < 0.8)) & mid,
        "cloudy_noon": ((clearness >= 0.2) & (clearness < 0.5)) & mid,
        "rainy": (precip > 0),
        "cold": (temp < 8),
    }
    assigned = np.zeros(len(hours), dtype=bool)
    excl = {}
    for name in ["clear_noon_high", "clear_noon_mid", "halfclear_noon", "cloudy_noon"]:
        excl[name] = raw[name] & ~assigned
        assigned |= excl[name]
    for name in ["rainy", "cold"]:
        excl[name] = raw[name] & ~assigned
        assigned |= excl[name]
    excl["baseline"] = ~assigned
    return excl


def _mae(pred, actual):
    return float(np.mean(np.abs(pred - actual)))


def _best_w(pl, pc, av, m, ws):
    """在 mask m 上选最优 w（最小 MAE）。"""
    if m.sum() == 0:
        return 0.0, float('nan')
    best_w, best_mae = 0.0, _mae(pl[m], av[m])
    for w in ws:
        mae = _mae(w * pc[m] + (1 - w) * pl[m], av[m])
        if mae < best_mae:
            best_w, best_mae = float(w), mae
    return best_w, best_mae


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print(f"CatBoost P2-7: OOF 场景权重 + 更细门控验证 (v6={V6_VAL_MAE})")
    print("=" * 74)

    print("[1] 构建数据集 ...")
    times, X, pred_load, actual = build_dataset()
    ab.times_global, ab.pred_load_global = times, pred_load
    usable = usable_mask(times, pred_load, actual)
    mismatch_model = MismatchModel().fit(X, usable)
    X = mismatch_model.transform(X)
    mc = C.TRAIN_CONFIG["mos"]
    mos_model = MosModel(cols=mc["cols"], alpha=mc["alpha"]).fit(X, actual, usable)
    anchor = pd.Series(mos_model.transform(X), index=X.index)
    feat_cols = list(X.columns)
    cfg = C.TRAIN_CONFIG
    cfg_ro = copy.deepcopy(cfg)
    cfg_ro["objectives"] = ["regression"]
    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    av = actual[val_m].to_numpy(np.float64)
    print(f"    特征数={len(feat_cols)}  val点={int(val_m.sum())}")

    WS = np.arange(0.0, 1.01, 0.1)

    # ---- v6 LightGBM: official + OOF 含校正 ----
    print("\n[2] 训练 v6 LightGBM official + OOF 3 折（含校正）...")
    ts = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        lgb_model = train_ensemble(times, X, pred_load, actual, usable, cfg,
                                   BEST_IT, mos_model=mos_model)
        lgb_model.mismatch_model = mismatch_model
        hb, dc, tc = compute_hour_bias(times, X, pred_load, actual, usable, cfg,
                                       BEST_IT, mos_model=mos_model)
        lgb_model.hour_bias, lgb_model.drift_corr, lgb_model.threshold_corr = hb, dc, tc
        oof_pred_lgb = pd.Series(np.nan, index=times)
        for te, vs, ve in cfg["best_it_folds"]:
            te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
            ftr = usable & np.asarray(times <= te)
            fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
            if fva.sum() == 0:
                continue
            fm = train_ensemble(times, X, pred_load, actual, ftr, cfg, BEST_IT, mos_model=mos_model)
            fm.mismatch_model = mismatch_model
            fm.hour_bias, fm.drift_corr, fm.threshold_corr = hb, dc, tc
            oof_pred_lgb[fva] = fm.predict_load(X[fva], pred_load[fva])
    pred_lgb_val = lgb_model.predict_load(X[val_m], pred_load[val_m])
    print(f"    v6 LightGBM  val MAE={_mae(pred_lgb_val, av):.2f}  ({time.perf_counter()-ts:.0f}s)")

    # ---- CatBoost d10: official + OOF 含校正 ----
    print("\n[3] 训练 CatBoost d10 official + OOF 3 折（含校正）...")
    ts = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        cat_members = hp._train_ensemble(X, actual, anchor, usable, cfg_ro,
                                         BEST_IT, feat_cols, HP_DW)
        hb_c, dc_c, tc_c, _, _ = hp._compute_oof(times, X, pred_load, actual, usable,
                                                 anchor, cfg_ro, BEST_IT, feat_cols, HP_DW)
        oof_pred_cat = pd.Series(np.nan, index=times)
        for te, vs, ve in cfg["best_it_folds"]:
            te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
            ftr = usable & np.asarray(times <= te)
            fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
            if fva.sum() == 0:
                continue
            fm = hp._train_ensemble(X, actual, anchor, ftr, cfg_ro, BEST_IT, feat_cols, HP_DW)
            oof_pred_cat[fva] = _predict_load(fm, X[fva], anchor[fva].to_numpy(),
                                              feat_cols, cfg["shrinkage"], hb_c, dc_c, tc_c)
    pred_cat_val = _predict_load(cat_members, X[val_m], anchor[val_m].to_numpy(),
                                 feat_cols, cfg["shrinkage"], hb_c, dc_c, tc_c)
    print(f"    CatBoost d10  val MAE={_mae(pred_cat_val, av):.2f}  ({time.perf_counter()-ts:.0f}s)")

    # ---- OOF + val 场景分析 ----
    oof_mask = usable & oof_pred_lgb.notna().values & oof_pred_cat.notna().values
    oof_lgb = oof_pred_lgb[oof_mask].to_numpy()
    oof_cat = oof_pred_cat[oof_mask].to_numpy()
    oof_act = actual[oof_mask].to_numpy()
    oof_clearness = X[oof_mask]["clearness"].values.astype(float)
    oof_precip = X[oof_mask]["precip"].values.astype(float)
    oof_temp = X[oof_mask]["temp"].values.astype(float)
    oof_hours = pd.DatetimeIndex(times[oof_mask]).hour.values
    oof_scenes = _scene_masks(oof_clearness, oof_precip, oof_temp, oof_hours)

    val_clearness = X[val_m]["clearness"].values.astype(float)
    val_precip = X[val_m]["precip"].values.astype(float)
    val_temp = X[val_m]["temp"].values.astype(float)
    val_hours = pd.DatetimeIndex(times[val_m]).hour.values
    val_scenes = _scene_masks(val_clearness, val_precip, val_temp, val_hours)

    print(f"\n[4] 场景分析（OOF 选 w，val 验证）  OOF点={int(oof_mask.sum())}")
    print(f"  {'场景':>18} {'OOFn':>6} {'OOF_lgb':>8} {'OOF_cat':>8} {'OOFw':>5} "
          f"{'valn':>5} {'val_lgb':>8} {'val_cat':>8} {'valw':>5}")
    oof_w = {}
    val_w = {}
    for name in SCENES:
        mo = oof_scenes[name]; mv = val_scenes[name]
        oof_l = _mae(oof_lgb[mo], oof_act[mo]) if mo.sum() else float('nan')
        oof_c = _mae(oof_cat[mo], oof_act[mo]) if mo.sum() else float('nan')
        wo, _ = _best_w(oof_lgb, oof_cat, oof_act, mo, WS) if mo.sum() else (0.0, float('nan'))
        val_l = _mae(pred_lgb_val[mv], av[mv]) if mv.sum() else float('nan')
        val_c = _mae(pred_cat_val[mv], av[mv]) if mv.sum() else float('nan')
        wv, _ = _best_w(pred_lgb_val, pred_cat_val, av, mv, WS) if mv.sum() else (0.0, float('nan'))
        oof_w[name] = wo
        val_w[name] = wv
        print(f"  {name:>18} {int(mo.sum()):>6} {oof_l:>8.1f} {oof_c:>8.1f} {wo:>5.1f} "
              f"{int(mv.sum()):>5} {val_l:>8.1f} {val_c:>8.1f} {wv:>5.1f}")

    # ---- 用 OOF w 融合 val（稳健）----
    print(f"\n[5] 用 OOF 选的 w 融合 val（稳健，防过拟合）:")
    pred_oofw = pred_lgb_val.copy()
    for name in SCENES:
        mv = val_scenes[name]
        if mv.sum():
            w = oof_w[name]
            pred_oofw[mv] = w * pred_cat_val[mv] + (1 - w) * pred_lgb_val[mv]
    mae_oofw = _mae(pred_oofw, av)
    print(f"  OOF-w 融合 val MAE={mae_oofw:.2f} (Δv6 {mae_oofw-V6_VAL_MAE:+.2f})")

    # ---- 用 val 自选 w 融合（参考上界，可能过拟合）----
    pred_valw = pred_lgb_val.copy()
    for name in SCENES:
        mv = val_scenes[name]
        if mv.sum():
            w = val_w[name]
            pred_valw[mv] = w * pred_cat_val[mv] + (1 - w) * pred_lgb_val[mv]
    mae_valw = _mae(pred_valw, av)
    print(f"  val-w 融合 val MAE={mae_valw:.2f} (Δv6 {mae_valw-V6_VAL_MAE:+.2f})  [上界参考]")

    # ---- OOF/val 一致性 ----
    print(f"\n[6] OOF/val 场景 w 一致性:")
    print(f"  {'场景':>18} {'OOFw':>5} {'valw':>5} {'一致':>5}")
    n_consistent = 0
    for name in SCENES:
        wo = oof_w[name]; wv = val_w[name]
        cons = abs(wo - wv) <= 0.2
        n_consistent += int(cons)
        print(f"  {name:>18} {wo:>5.1f} {wv:>5.1f} {'✓' if cons else '✗':>5}")
    print(f"  一致场景: {n_consistent}/{len(SCENES)}")

    # ---- 汇总 ----
    print("\n" + "=" * 74)
    print("P2-7 汇总")
    print("=" * 74)
    print(f"  v6 LightGBM            MAE={_mae(pred_lgb_val, av):.2f}")
    print(f"  CatBoost d10           MAE={_mae(pred_cat_val, av):.2f}")
    print(f"  P2-6 场景门控(val选w)  MAE≈1438.30 (Δv6 -7.32)  [P2-6 参考]")
    print(f"  本轮 val-w 融合(7场景) MAE={mae_valw:.2f} (Δv6 {mae_valw-V6_VAL_MAE:+.2f})")
    print(f"  本轮 OOF-w 融合(稳健)  MAE={mae_oofw:.2f} (Δv6 {mae_oofw-V6_VAL_MAE:+.2f})")
    print(f"\n  OOF-w 是稳健估计（防 val 过拟合）。若 OOF-w ≈ val-w -> 场景门控稳健可迁移。")
    print(f"  OOF w: {oof_w}")
    print(f"  val w: {val_w}")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
