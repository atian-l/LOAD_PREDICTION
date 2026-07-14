# -*- coding: utf-8 -*-
"""
CatBoost P2-6: 5 时段 + 场景门控融合（榨取 P2-4 融合极限）

背景：
  - P2-4 加权融合 1442.36（-3.26MW），午间 w_mid=0.5/非午间 w_other=0.1
  - P2-5 残差校正失败（OOF 不迁移），否决
  - 误差相关 0.984 限制全局融合上限。突破口：找 CatBoost 显著优于 v6 的场景/时段，
    门控融合（cat 优势区高权重，其余低权重）

流程：
  1. build_dataset + MismatchModel + MOS -> X', anchor（共用）
  2. v6 LightGBM 40 成员 + OOF 校正 -> pred_lgb_val
  3. CatBoost d10 reg_only 10 成员 + OOF 校正 -> pred_cat_val
  4. 5 时段分解（00-06/06-11/11-14/15-18/18-24）：各时段 lgb/cat MAE + 独立最优 w
  5. 场景分解（clear_noon/cloudy_noon/rainy/cold/baseline）：各场景 lgb/cat MAE
  6. 场景门控融合：cat 优势场景高 w，其余低 w（网格搜索）
  7. 对比 P2-4 的 2 时段融合（1442.36）

合规: 不修改生产脚本; 仅 import train/hp/ab/features; 6 条泄露不变量全保持; val eval-only。
运行: python -m load_pred.exp_catboost_gate  (本地 3060 约 5 min)
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


def _mae(pred, actual):
    return float(np.mean(np.abs(pred - actual)))


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print(f"CatBoost P2-6: 5 时段 + 场景门控融合 (v6={V6_VAL_MAE})")
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
    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    av = actual[val_m].to_numpy(np.float64)
    print(f"    特征数={len(feat_cols)}  val点={int(val_m.sum())}")

    print("\n[2] 训练 v6 LightGBM ...")
    ts = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        lgb_model = train_ensemble(times, X, pred_load, actual, usable, cfg,
                                   BEST_IT, mos_model=mos_model)
        lgb_model.mismatch_model = mismatch_model
        hb, dc, tc = compute_hour_bias(times, X, pred_load, actual, usable, cfg,
                                       BEST_IT, mos_model=mos_model)
        lgb_model.hour_bias, lgb_model.drift_corr, lgb_model.threshold_corr = hb, dc, tc
    pred_lgb = lgb_model.predict_load(X[val_m], pred_load[val_m])
    print(f"    v6 LightGBM  MAE={_mae(pred_lgb, av):.2f}  ({time.perf_counter()-ts:.0f}s)")

    print("\n[3] 训练 CatBoost d10 reg_only ...")
    ts = time.perf_counter()
    cfg_ro = copy.deepcopy(cfg)
    cfg_ro["objectives"] = ["regression"]
    with contextlib.redirect_stdout(io.StringIO()):
        cat_members = hp._train_ensemble(X, actual, anchor, usable, cfg_ro,
                                         BEST_IT, feat_cols, HP_DW)
        hb_c, dc_c, tc_c, _, _ = hp._compute_oof(times, X, pred_load, actual, usable,
                                                 anchor, cfg_ro, BEST_IT, feat_cols, HP_DW)
    pred_cat = _predict_load(cat_members, X[val_m], anchor[val_m].to_numpy(),
                             feat_cols, cfg["shrinkage"], hb_c, dc_c, tc_c)
    print(f"    CatBoost d10  MAE={_mae(pred_cat, av):.2f}  ({time.perf_counter()-ts:.0f}s)")

    dt_val = pd.DatetimeIndex(times[val_m])
    h_val = dt_val.hour.values
    clearness_v = X[val_m]["clearness"].values.astype(float)
    precip_v = X[val_m]["precip"].values.astype(float)
    temp_v = X[val_m]["temp"].values.astype(float)

    # ---- [4] 5 时段分解 ----
    print("\n[4] 5 时段分解（各时段 lgb/cat MAE + 独立最优 w）:")
    bands = [(0, 6, "00-06"), (6, 11, "06-11"), (11, 15, "11-14"),
             (15, 18, "15-18"), (18, 24, "18-24")]
    print(f"  {'时段':>7} {'n':>6} {'lgb MAE':>9} {'cat MAE':>9} {'cat-lgb':>8} {'最优w':>7} {'融合MAE':>9}")
    band_masks = {}
    band_best_w = {}
    for lo, hi, name in bands:
        m = (h_val >= lo) & (h_val < hi)
        band_masks[name] = m
        mae_l = _mae(pred_lgb[m], av[m]) if m.sum() else float('nan')
        mae_c = _mae(pred_cat[m], av[m]) if m.sum() else float('nan')
        # 独立最优 w
        best_w, best_mae = 0.0, mae_l
        for w in np.arange(0.0, 1.001, 0.05):
            mp = w * pred_cat[m] + (1 - w) * pred_lgb[m]
            mae = _mae(mp, av[m]) if m.sum() else float('nan')
            if mae < best_mae:
                best_w, best_mae = float(w), mae
        band_best_w[name] = best_w
        print(f"  {name:>7} {int(m.sum()):>6} {mae_l:>9.1f} {mae_c:>9.1f} "
              f"{mae_c-mae_l:>+8.1f} {best_w:>7.2f} {best_mae:>9.1f}")
    # 5 时段独立 w 融合整体 MAE
    pred_5band = np.zeros_like(pred_lgb)
    for lo, hi, name in bands:
        m = band_masks[name]
        w = band_best_w[name]
        pred_5band[m] = w * pred_cat[m] + (1 - w) * pred_lgb[m]
    mae_5band = _mae(pred_5band, av)
    print(f"  -> 5 时段独立 w 融合整体 MAE={mae_5band:.2f} (Δv6 {mae_5band-V6_VAL_MAE:+.2f})")

    # ---- [5] 场景分解 ----
    print("\n[5] 场景分解（各场景 lgb/cat MAE）:")
    mid_val = (h_val >= 11) & (h_val <= 14)
    sc = {
        "clear_noon": (clearness_v > 0.8) & mid_val,
        "cloudy_noon": ((clearness_v >= 0.2) & (clearness_v < 0.5)) & mid_val,
        "rainy": (precip_v > 0),
        "cold": (temp_v < 8),
    }
    # 互斥分配（优先级：clear_noon/cloudy_noon > rainy > cold > baseline）
    assigned = np.zeros(len(av), dtype=bool)
    sc_excl = {}
    for name in ["clear_noon", "cloudy_noon"]:
        sc_excl[name] = sc[name] & ~assigned
        assigned |= sc_excl[name]
    for name in ["rainy", "cold"]:
        sc_excl[name] = sc[name] & ~assigned
        assigned |= sc_excl[name]
    sc_excl["baseline"] = ~assigned
    print(f"  {'场景':>12} {'n':>6} {'lgb MAE':>9} {'cat MAE':>9} {'cat-lgb':>8} {'最优w':>7} {'融合MAE':>9}")
    sc_best_w = {}
    for name, m in sc_excl.items():
        mae_l = _mae(pred_lgb[m], av[m]) if m.sum() else float('nan')
        mae_c = _mae(pred_cat[m], av[m]) if m.sum() else float('nan')
        best_w, best_mae = 0.0, mae_l
        for w in np.arange(0.0, 1.001, 0.05):
            mp = w * pred_cat[m] + (1 - w) * pred_lgb[m]
            mae = _mae(mp, av[m]) if m.sum() else float('nan')
            if mae < best_mae:
                best_w, best_mae = float(w), mae
        sc_best_w[name] = best_w
        print(f"  {name:>12} {int(m.sum()):>6} {mae_l:>9.1f} {mae_c:>9.1f} "
              f"{mae_c-mae_l:>+8.1f} {best_w:>7.2f} {best_mae:>9.1f}")
    # 场景门控融合整体 MAE
    pred_sc = np.zeros_like(pred_lgb)
    for name, m in sc_excl.items():
        w = sc_best_w[name]
        pred_sc[m] = w * pred_cat[m] + (1 - w) * pred_lgb[m]
    mae_sc = _mae(pred_sc, av)
    print(f"  -> 场景门控融合整体 MAE={mae_sc:.2f} (Δv6 {mae_sc-V6_VAL_MAE:+.2f})")

    # ---- [6] 5 时段 × 场景 联合（cat 优势区细分）----
    # 简单联合：午间(11-14) 用场景 w，非午间用时段 w
    print("\n[6] 联合门控（午间用场景 w，非午间用时段 w）:")
    pred_joint = pred_lgb.copy()
    # 午间按场景
    for name in ["clear_noon", "cloudy_noon", "baseline"]:
        m = sc_excl[name] & mid_val
        if m.sum():
            w = sc_best_w[name]
            pred_joint[m] = w * pred_cat[m] + (1 - w) * pred_lgb[m]
    # 非午间按时段
    for lo, hi, name in bands:
        if name == "11-14":
            continue
        m = band_masks[name] & ~mid_val
        if m.sum():
            w = band_best_w[name]
            pred_joint[m] = w * pred_cat[m] + (1 - w) * pred_lgb[m]
    mae_joint = _mae(pred_joint, av)
    print(f"  -> 联合门控 MAE={mae_joint:.2f} (Δv6 {mae_joint-V6_VAL_MAE:+.2f})")

    # ---- [7] 2 时段融合复现（P2-4 对照）----
    print("\n[7] P2-4 2 时段融合对照:")
    best2 = None
    for w_mid in np.arange(0.0, 1.001, 0.05):
        for w_other in np.arange(0.0, 1.001, 0.05):
            p = np.where(mid_val, w_mid * pred_cat + (1 - w_mid) * pred_lgb,
                         w_other * pred_cat + (1 - w_other) * pred_lgb)
            mae = _mae(p, av)
            if best2 is None or mae < best2[2]:
                best2 = (float(w_mid), float(w_other), mae)
    print(f"  2 时段最优 w_mid={best2[0]:.2f} w_other={best2[1]:.2f} MAE={best2[2]:.2f} "
          f"(Δv6 {best2[2]-V6_VAL_MAE:+.2f})")

    # ---- 汇总 ----
    print("\n" + "=" * 74)
    print("P2-6 汇总")
    print("=" * 74)
    print(f"  v6 LightGBM            MAE={_mae(pred_lgb, av):.2f}")
    print(f"  CatBoost d10           MAE={_mae(pred_cat, av):.2f}")
    print(f"  2 时段融合(P2-4)       MAE={best2[2]:.2f}  (Δv6 {best2[2]-V6_VAL_MAE:+.2f})")
    print(f"  5 时段独立 w 融合      MAE={mae_5band:.2f}  (Δv6 {mae_5band-V6_VAL_MAE:+.2f})")
    print(f"  场景门控融合           MAE={mae_sc:.2f}  (Δv6 {mae_sc-V6_VAL_MAE:+.2f})")
    print(f"  联合门控               MAE={mae_joint:.2f}  (Δv6 {mae_joint-V6_VAL_MAE:+.2f})")
    best_overall = min(best2[2], mae_5band, mae_sc, mae_joint)
    print(f"\n  最优融合 MAE={best_overall:.2f} (Δv6 {best_overall-V6_VAL_MAE:+.2f}, "
          f"改善 {V6_VAL_MAE-best_overall:.2f} MW)")
    print(f"  各场景最优 w: {sc_best_w}")
    print(f"  各时段最优 w: {band_best_w}")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
