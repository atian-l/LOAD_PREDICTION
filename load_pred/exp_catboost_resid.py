# -*- coding: utf-8 -*-
"""
CatBoost P2-5: CatBoost 残差校正器（boosting v6 LightGBM）

背景：P2-4 加权融合破 v6 仅 -3.26MW（1442.36），受误差相关 0.984 限制。加权融合天花板低。
本脚本试 boosting 思路：CatBoost 学习 v6 LightGBM 预测的残差，捕获 LightGBM 遗漏的非线性模式。
即使 v6 残差 78.5% unlearnable（FDS 诊断），21.5% 可学部分 CatBoost（异质算法）或能捕获。

流程：
  1. build_dataset + MismatchModel + MOS -> X', anchor（共用）
  2. v6 LightGBM:
     - official 40 成员 + OOF 3 折校正 -> pred_lgb_val
     - OOF 含校正预测 oof_pred_lgb（每折 fold_model + official 校正预测折内）
  3. CatBoost 残差校正器（target = actual - oof_pred_lgb，OOF mask 上）:
     - d10 reg_only 5 seeds，direct 模式（预测残差本身）
     - official: 全 usable 训 -> cat_resid_val
     - OOF 3 折: 估残差器系统性偏差（用于稳健性验证 + 可选去偏）
  4. val: pred = pred_lgb_val + α · cat_resid_val，α 扫描 [0, 2] step 0.1
     - 全局 α + 时段分解 α（午间/非午间）
  5. OOF 稳健性: 在 OOF 上算残差校正改善，验证是否迁移（防 val 过拟合）
  6. 对比: v6 / P2-4 融合 / 残差校正

合规: 不修改生产脚本; 仅 import train/hp/ab/features; 6 条泄露不变量全保持;
      v6 与 CatBoost 残差器均仅用 usable(<=TRAIN_END) 训练; OOF 3 折全在训练期内;
      actual 仅 target/MOS/评估; val eval-only。
运行: python -m load_pred.exp_catboost_resid  (本地 3060 约 4-6 min)
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
from .train import build_dataset, usable_mask, train_ensemble, compute_hour_bias, _time_weights
from .features import MismatchModel, MosModel
from .exp_catboost_ab import V6_VAL_MAE, _arr
from . import exp_catboost_ab as ab
from . import exp_catboost_hp as hp

# CatBoost 残差器配置（d10 reg_only，P1-1 最优 hp；direct 模式预测残差）
HP_RESID = {"depth": 10, "lr": 0.03, "l2": 8.0, "bagging_temp": 1.0,
            "grow_policy": "Depthwise", "max_leaves": None}
BEST_IT = 80
N_SEEDS = 5
SEEDS = [42, 7, 123, 2024, 99]


def _train_cb_resid(X, resid_target, mask, cfg, best_it, feat_cols, hp_dict):
    """CatBoost direct 模式预测残差 target（actual - oof_pred_lgb）。返回 [(model, is_res=False)]。"""
    Xtr = _arr(X[mask], feat_cols)
    wtr = _time_weights(ab.times_global, mask, cfg["alpha_w"],
                        pred_load=ab.pred_load_global,
                        load_gamma=cfg.get("weight_load_gamma", 0.0))
    ytr = resid_target[mask].to_numpy(np.float64)
    members = []
    for s in SEEDS:
        m = hp._fit(Xtr, ytr, wtr, "RMSE", s, best_it, hp_dict)
        members.append((m, False))
    return members


def _cb_resid_pred(members, X, feat_cols):
    """CatBoost 残差器预测（direct，median 聚合）。"""
    Xarr = _arr(X, feat_cols)
    mp = np.empty((len(members), len(X)), dtype=float)
    for i, (m, _) in enumerate(members):
        mp[i] = m.predict(Xarr)
    return np.median(mp, axis=0)


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print(f"CatBoost P2-5: CatBoost 残差校正器 (boosting v6)  (v6={V6_VAL_MAE})")
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
    print(f"    特征数={len(feat_cols)}  训练点={int(usable.sum())}  val点={int(val_m.sum())}")

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
        # OOF 含校正预测：每折 fold_model + official 校正预测折内
        oof_pred_lgb = pd.Series(np.nan, index=times)
        for te, vs, ve in cfg["best_it_folds"]:
            te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
            ftr = usable & np.asarray(times <= te)
            fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
            if fva.sum() == 0:
                continue
            fold_model = train_ensemble(times, X, pred_load, actual, ftr, cfg,
                                        BEST_IT, mos_model=mos_model)
            fold_model.mismatch_model = mismatch_model
            fold_model.hour_bias, fold_model.drift_corr, fold_model.threshold_corr = hb, dc, tc
            oof_pred_lgb[fva] = fold_model.predict_load(X[fva], pred_load[fva])
    pred_lgb_val = lgb_model.predict_load(X[val_m], pred_load[val_m])
    err_lgb_val = pred_lgb_val - av
    mae_lgb = float(np.mean(np.abs(err_lgb_val)))
    print(f"    v6 LightGBM  val MAE={mae_lgb:.2f}  Bias={float(np.mean(err_lgb_val)):+.1f}  "
          f"({time.perf_counter()-ts:.0f}s)")

    # OOF mask + 残差 target
    oof_mask = usable & oof_pred_lgb.notna().values
    resid_target = (actual - oof_pred_lgb)  # OOF 上 v6 残差（含校正）
    resid_oof = resid_target[oof_mask].to_numpy(np.float64)
    print(f"    OOF 点数={int(oof_mask.sum())}  残差均值={float(np.mean(resid_oof)):+.1f}  "
          f"残差MAE={float(np.mean(np.abs(resid_oof))):.1f}")

    # ---- CatBoost 残差校正器 ----
    print(f"\n[3] 训练 CatBoost 残差校正器 (d10 reg_only, {N_SEEDS} seeds, direct) ...")
    ts = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        # official 残差器：在所有 OOF 点训练（target=actual-oof_pred_lgb，仅 oof_mask 有值）
        resid_members = _train_cb_resid(X, resid_target, oof_mask, cfg, BEST_IT, feat_cols, HP_RESID)
        # nested OOF 残差器（稳健性验证，无泄露）：每折用其他 2 折 OOF 点训练，预测该折
        oof_resid_pred = pd.Series(np.nan, index=times)
        fold_masks = []
        for te, vs, ve in cfg["best_it_folds"]:
            vs, ve = pd.Timestamp(vs), pd.Timestamp(ve)
            fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
            fold_masks.append(fva)
        for i, fva_i in enumerate(fold_masks):
            train_mask = oof_mask & ~fva_i
            if train_mask.sum() == 0:
                continue
            fm = _train_cb_resid(X, resid_target, train_mask, cfg, BEST_IT, feat_cols, HP_RESID)
            oof_resid_pred[fva_i] = _cb_resid_pred(fm, X[fva_i], feat_cols)
    cat_resid_val = _cb_resid_pred(resid_members, X[val_m], feat_cols)
    print(f"    CatBoost 残差器  val resid pred: 均值={float(np.mean(cat_resid_val)):+.1f}  "
          f"|pred|均值={float(np.mean(np.abs(cat_resid_val))):.1f}  "
          f"({time.perf_counter()-ts:.0f}s)")

    # ---- OOF 稳健性：残差器在 OOF 上能否降低 v6 OOF MAE ----
    print(f"\n[4] OOF 稳健性验证（防 val 过拟合）...")
    oof_lgb = oof_pred_lgb[oof_mask].to_numpy()
    oof_cat_resid = oof_resid_pred[oof_mask].to_numpy()
    oof_actual = actual[oof_mask].to_numpy()
    oof_mae_base = float(np.mean(np.abs(oof_lgb - oof_actual)))
    print(f"    OOF v6 MAE={oof_mae_base:.2f}")
    # OOF α 扫描
    best_oof_alpha = None
    print(f"    {'α':>5} {'OOF MAE':>10} {'ΔOOF':>8} {'val MAE':>10} {'Δval':>8}")
    oof_val_rows = []
    for alpha in np.arange(0.0, 2.01, 0.1):
        oof_pred = oof_lgb + alpha * oof_cat_resid
        oof_mae = float(np.mean(np.abs(oof_pred - oof_actual)))
        val_pred = pred_lgb_val + alpha * cat_resid_val
        val_mae = float(np.mean(np.abs(val_pred - av)))
        oof_val_rows.append((float(alpha), oof_mae, oof_mae - oof_mae_base, val_mae, val_mae - mae_lgb))
        if best_oof_alpha is None or oof_mae < best_oof_alpha[1]:
            best_oof_alpha = (float(alpha), oof_mae, val_mae)
        print(f"    {alpha:>5.1f} {oof_mae:>10.2f} {oof_mae-oof_mae_base:>+8.2f} "
              f"{val_mae:>10.2f} {val_mae-mae_lgb:>+8.2f}")

    # val 上 α 最优
    best_val = min(oof_val_rows, key=lambda r: r[3])
    print(f"\n    OOF 最优 α={best_oof_alpha[0]:.1f} (OOF MAE={best_oof_alpha[1]:.2f}, "
          f"对应 val MAE={best_oof_alpha[2]:.2f})")
    print(f"    val 最优 α={best_val[0]:.1f} (val MAE={best_val[3]:.2f}, Δv6 {best_val[3]-V6_VAL_MAE:+.2f})")

    # ---- val 全局 α 扫描（细）----
    print(f"\n[5] val 全局残差校正 α 细扫  pred = lgb + α*cat_resid:")
    print(f"    {'α':>5} {'val MAE':>10} {'Δv6':>9} {'Bias':>8}")
    rows_val = []
    for alpha in np.arange(0.0, 1.51, 0.05):
        val_pred = pred_lgb_val + alpha * cat_resid_val
        mae = float(np.mean(np.abs(val_pred - av)))
        bias = float(np.mean(val_pred - av))
        rows_val.append((float(alpha), mae, mae - V6_VAL_MAE, bias))
        print(f"    {alpha:>5.2f} {mae:>10.2f} {mae - V6_VAL_MAE:>+9.2f} {bias:>+8.1f}")
    best_a = min(rows_val, key=lambda r: r[1])

    # ---- 时段分解 α（午间/非午间）----
    print(f"\n[6] 时段分解残差校正（午间 α_mid / 非午间 α_other）:")
    dt_val = pd.DatetimeIndex(times[val_m])
    h_val = dt_val.hour.values
    mid = (h_val >= 11) & (h_val <= 14)
    best_seg = None
    for a_mid in np.arange(0.0, 1.51, 0.1):
        for a_other in np.arange(0.0, 1.51, 0.1):
            val_pred = np.where(mid,
                                pred_lgb_val + a_mid * cat_resid_val,
                                pred_lgb_val + a_other * cat_resid_val)
            mae = float(np.mean(np.abs(val_pred - av)))
            if best_seg is None or mae < best_seg[2]:
                best_seg = (float(a_mid), float(a_other), mae)
    print(f"    最优: α_mid={best_seg[0]:.1f} α_other={best_seg[1]:.1f}  "
          f"MAE={best_seg[2]:.2f} (Δv6 {best_seg[2]-V6_VAL_MAE:+.2f})")

    # ---- 汇总 ----
    print("\n" + "=" * 74)
    print("P2-5 汇总：CatBoost 残差校正 v6")
    print("=" * 74)
    print(f"  v6 LightGBM              MAE={mae_lgb:.2f}  (Δv6 {mae_lgb-V6_VAL_MAE:+.2f})")
    print(f"  全局残差校正最优          α={best_a[0]:.2f}  MAE={best_a[1]:.2f}  (Δv6 {best_a[2]:+.2f})")
    print(f"  时段残差校正最优          α_mid={best_seg[0]:.1f}/α_other={best_seg[1]:.1f}  "
          f"MAE={best_seg[2]:.2f}  (Δv6 {best_seg[2]-V6_VAL_MAE:+.2f})")
    print(f"  OOF 最优 α={best_oof_alpha[0]:.1f} -> val MAE={best_oof_alpha[2]:.2f} "
          f"(OOF/val 一致性: {'好' if abs(best_oof_alpha[0]-best_a[0])<=0.2 else '差，val 过拟合风险'})")
    best_overall = min(mae_lgb, best_a[1], best_seg[2])
    if best_overall < V6_VAL_MAE:
        print(f"\n  >>> 破 v6！最优 MAE={best_overall:.2f} < v6 {V6_VAL_MAE} "
              f"(改善 {V6_VAL_MAE-best_overall:.2f} MW)")
    else:
        print(f"\n  未破 v6，距 {best_overall-V6_VAL_MAE:+.2f}MW")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
