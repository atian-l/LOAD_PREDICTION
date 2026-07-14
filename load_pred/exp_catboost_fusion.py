# -*- coding: utf-8 -*-
"""
CatBoost P2-4: CatBoost d10 × v6 LightGBM 异质融合评估

背景：CatBoost 单独已到顶 dw_regonly_d10=1455.01（Δv6 +9.39），P2-1 d10 重估校正退化至
1460.88（重估过拟合 OOF 折）。CatBoost 单独破 v6 无望。本脚本评估异质融合能否破 v6。

流程：
  1. build_dataset + MismatchModel + MOS -> X', anchor（v6 与 CatBoost 共用，公平对比）
  2. v6 LightGBM: train.train_ensemble(40成员, best_it=80) + compute_hour_bias(OOF 3折校正)
     -> pred_lgb_val
  3. CatBoost d10 reg_only: hp._train_ensemble(10成员) + hp._compute_oof(OOF 3折校正)
     -> pred_cat_val
  4. 融合分析:
     - 两者 val MAE / Bias
     - 误差相关性 corr(err_lgb, err_cat)  (<0.95 -> 融合有理论增益)
     - 加权融合 w 扫描: pred = w*cat + (1-w)*lgb, w in [0,1] step 0.05
     - 时段分解: 午间(11-14) vs 非午间 独立 w
     - inverse-variance 理论最优 w（参考）

合规: 不修改生产脚本; 仅 import train/hp/ab/features 复用; 6 条泄露不变量全保持;
      v6 与 CatBoost 均仅用 usable(<=TRAIN_END) 训练, OOF 3 折全在训练期内, val eval-only;
      actual 仅作 target/MOS目标/评估。
运行: python -m load_pred.exp_catboost_fusion  (本地 3060 约 3-5 min)
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

# CatBoost d10 reg_only 配置（P1-1 最优 dw_regonly_d10=1455.01）
HP_DW = {"depth": 10, "lr": 0.03, "l2": 8.0, "bagging_temp": 1.0,
         "grow_policy": "Depthwise", "max_leaves": None}
BEST_IT = 80


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print(f"CatBoost P2-4: CatBoost d10 x v6 LightGBM 异质融合评估 (v6={V6_VAL_MAE})")
    print("=" * 74)

    print("[1] 构建数据集（v6 与 CatBoost 共用同一 X/anchor）...")
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
    print(f"    特征数={len(feat_cols)}  训练点={int(usable.sum())}  val点={int(val_m.sum())}")

    # ---- v6 LightGBM ----
    print("\n[2] 训练 v6 LightGBM 40 成员 + OOF 3 折校正 ...")
    ts = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        lgb_model = train_ensemble(times, X, pred_load, actual, usable, cfg,
                                   BEST_IT, mos_model=mos_model)
        lgb_model.mismatch_model = mismatch_model
        hb, dc, tc = compute_hour_bias(times, X, pred_load, actual, usable, cfg,
                                       BEST_IT, mos_model=mos_model)
        lgb_model.hour_bias, lgb_model.drift_corr, lgb_model.threshold_corr = hb, dc, tc
    pred_lgb = lgb_model.predict_load(X[val_m], pred_load[val_m])
    err_lgb = pred_lgb - av
    mae_lgb = float(np.mean(np.abs(err_lgb)))
    print(f"    v6 LightGBM  val MAE={mae_lgb:.2f}  Bias={float(np.mean(err_lgb)):+.1f}  "
          f"({time.perf_counter()-ts:.0f}s)")

    # ---- CatBoost d10 reg_only ----
    print("\n[3] 训练 CatBoost d10 reg_only 10 成员 + OOF 3 折校正 ...")
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
    err_cat = pred_cat - av
    mae_cat = float(np.mean(np.abs(err_cat)))
    print(f"    CatBoost d10  val MAE={mae_cat:.2f}  Bias={float(np.mean(err_cat)):+.1f}  "
          f"({time.perf_counter()-ts:.0f}s)")

    # ---- 融合分析 ----
    print("\n[4] 融合分析 ...")
    corr_err = float(np.corrcoef(err_lgb, err_cat)[0, 1])
    corr_pred = float(np.corrcoef(pred_lgb, pred_cat)[0, 1])
    var_lgb = float(np.var(err_lgb))
    var_cat = float(np.var(err_cat))
    # inverse-variance 理论最优 w（假设误差不相关；相关时为近似参考）
    w_iv = var_lgb / (var_lgb + var_cat) if (var_lgb + var_cat) > 0 else 0.0
    print(f"    误差相关性 corr(err_lgb, err_cat) = {corr_err:.4f}")
    print(f"    预测相关性 corr(pred_lgb, pred_cat) = {corr_pred:.4f}")
    print(f"    var(err_lgb)={var_lgb:.0f}  var(err_cat)={var_cat:.0f}  "
          f"-> inverse-variance w_cat={w_iv:.3f}")
    print(f"    (误差相关性 <0.95 -> 融合有理论增益；越低增益越大)")

    # 加权融合扫描
    print(f"\n  [4a] 全局加权融合  pred = w*cat + (1-w)*lgb:")
    print(f"  {'w':>5} {'MAE':>10} {'Δv6':>9} {'Bias':>8}")
    rows = []
    for w in np.arange(0.0, 1.001, 0.05):
        pred = w * pred_cat + (1 - w) * pred_lgb
        mae = float(np.mean(np.abs(pred - av)))
        bias = float(np.mean(pred - av))
        rows.append((float(w), mae, mae - V6_VAL_MAE, bias))
        print(f"  {w:>5.2f} {mae:>10.2f} {mae - V6_VAL_MAE:>+9.2f} {bias:>+8.1f}")
    best = min(rows, key=lambda r: r[1])
    print(f"\n  最优 w_cat={best[0]:.2f}  MAE={best[1]:.2f} (Δv6 {best[2]:+.2f})  Bias={best[3]:+.1f}")
    if best[1] < V6_VAL_MAE:
        print(f"  >>> 破 v6！全局融合 w_cat={best[0]:.2f} MAE={best[1]:.2f} < v6 {V6_VAL_MAE}")

    # 时段分解：午间(11-14) vs 非午间 独立 w
    print(f"\n  [4b] 时段分解（午间 11-14 独立 w_mid / 非午间 w_other）:")
    dt_val = pd.DatetimeIndex(times[val_m])
    h_val = dt_val.hour.values
    mid = (h_val >= 11) & (h_val <= 14)
    print(f"    午间点数={int(mid.sum())}  非午间={int((~mid).sum())}")
    # 粗网格搜索最优
    best_seg = None
    for w_mid in np.arange(0.0, 1.001, 0.1):
        for w_other in np.arange(0.0, 1.001, 0.1):
            pred = np.where(mid,
                            w_mid * pred_cat + (1 - w_mid) * pred_lgb,
                            w_other * pred_cat + (1 - w_other) * pred_lgb)
            mae = float(np.mean(np.abs(pred - av)))
            if best_seg is None or mae < best_seg[2]:
                best_seg = (float(w_mid), float(w_other), mae)
    # 打印关键组合
    print(f"  {'w_mid':>6} {'w_other':>8} {'MAE':>10} {'Δv6':>9}")
    for w_mid in [0.0, 0.2, 0.3, 0.4, 0.5]:
        for w_other in [0.0, 0.2, 0.3, 0.4, 0.5]:
            pred = np.where(mid,
                            w_mid * pred_cat + (1 - w_mid) * pred_lgb,
                            w_other * pred_cat + (1 - w_other) * pred_lgb)
            mae = float(np.mean(np.abs(pred - av)))
            print(f"  {w_mid:>6.1f} {w_other:>8.1f} {mae:>10.2f} {mae - V6_VAL_MAE:>+9.2f}")
    print(f"\n  最优: w_mid={best_seg[0]:.1f} w_other={best_seg[1]:.1f}  "
          f"MAE={best_seg[2]:.2f} (Δv6 {best_seg[2]-V6_VAL_MAE:+.2f})")
    if best_seg[2] < V6_VAL_MAE:
        print(f"  >>> 破 v6！时段融合 w_mid={best_seg[0]:.1f} w_other={best_seg[1]:.1f} "
              f"MAE={best_seg[2]:.2f} < v6 {V6_VAL_MAE}")

    # ---- 汇总 ----
    print("\n" + "=" * 74)
    print("P2-4 汇总")
    print("=" * 74)
    print(f"  v6 LightGBM        MAE={mae_lgb:.2f}  (Δv6 {mae_lgb-V6_VAL_MAE:+.2f})")
    print(f"  CatBoost d10       MAE={mae_cat:.2f}  (Δv6 {mae_cat-V6_VAL_MAE:+.2f})")
    print(f"  误差相关性         {corr_err:.4f}")
    print(f"  全局融合最优       w={best[0]:.2f}  MAE={best[1]:.2f}  (Δv6 {best[2]:+.2f})")
    print(f"  时段融合最优       w_mid={best_seg[0]:.1f}/w_other={best_seg[1]:.1f}  "
          f"MAE={best_seg[2]:.2f}  (Δv6 {best_seg[2]-V6_VAL_MAE:+.2f})")
    fuse_best = min(best[1], best_seg[2])
    if fuse_best < V6_VAL_MAE:
        print(f"\n  >>> 异质融合破 v6！最优融合 MAE={fuse_best:.2f} < v6 {V6_VAL_MAE} "
              f"(改善 {V6_VAL_MAE-fuse_best:.2f} MW)")
    else:
        print(f"\n  融合仍未破 v6，距 {fuse_best-V6_VAL_MAE:+.2f}MW")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
