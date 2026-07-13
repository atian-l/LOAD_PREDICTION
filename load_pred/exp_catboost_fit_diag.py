# -*- coding: utf-8 -*-
"""
临时诊断：CatBoost l2_8（3-B1 最优配置）在 训练集 / OOF / val 三处的拟合，
判断欠拟合 vs 过拟合，以及 模型问题 vs 数据问题。

三类预测：
  - train_raw : 模型在自己训练的数据上预测（含已见，乐观下界）
  - OOF       : 3 折 walk-forward 无泄露预测（训练数据的真实可预测性）
  - val_raw   : 验证集预测（泛化能力）

判定：
  train_raw − OOF  大(>400) -> 过拟合；小(<150) -> 欠拟合
  OOF vs val_raw   |差|<200 -> 泛化一致(数据信号弱)；OOF<<val -> 漂移；OOF>>val -> 训练期更难
  CatBoost train_raw vs LightGBM train_raw -> 拟合能力差异(模型 vs 数据)

可选：若 lightgbm 可用，跑 v6 配置同条件对比。
运行：python -m load_pred.exp_catboost_fit_diag   （4090 上约 3~5 min）
"""
from __future__ import annotations
import sys
import time
import warnings

import numpy as np
import pandas as pd

from . import config as C
from .train import build_dataset, usable_mask
from .features import MismatchModel, MosModel
from .exp_catboost_hp import _train_ensemble, _ensemble_raw, _compute_oof
from .exp_catboost_ab import _predict_load, _arr, V6_VAL_MAE
from . import exp_catboost_ab as ab

L2_8 = {"depth": 8, "lr": 0.03, "l2": 8.0, "bagging_temp": 1.0,
        "grow_policy": "SymmetricTree", "max_leaves": None}
BEST_IT = 80


def _mae(pred, actual):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(a) & np.isfinite(p)
    return float(np.mean(np.abs(p[m] - a[m]))) if m.sum() else float("nan")


def _r2(pred, actual):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    ss_res = float(np.sum((p - a) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 72)
    print(f"拟合诊断: CatBoost l2_8 (best_it={BEST_IT})  train / OOF / val")
    print("=" * 72)

    times, X, pred_load, actual = build_dataset()
    ab.times_global, ab.pred_load_global = times, pred_load
    usable = usable_mask(times, pred_load, actual)
    mm = MismatchModel().fit(X, usable)
    X = mm.transform(X)
    mos_model = MosModel(cols=C.TRAIN_CONFIG["mos"]["cols"],
                         alpha=C.TRAIN_CONFIG["mos"]["alpha"]).fit(X, actual, usable)
    anchor = pd.Series(mos_model.transform(X), index=X.index)
    feat_cols = list(X.columns)
    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    cfg = C.TRAIN_CONFIG
    print(f"特征数={len(feat_cols)}  训练点={int(usable.sum())}  val点={int(val_m.sum())}")

    print("\n[1] 训练 CatBoost 40 成员 + 3 折 OOF 校正估计 ...")
    members = _train_ensemble(X, actual, anchor, usable, cfg, BEST_IT, feat_cols, L2_8)
    hour_bias, drift_corr, threshold_corr, oof_pred, oof_mask = _compute_oof(
        times, X, pred_load, actual, usable, anchor, cfg, BEST_IT, feat_cols, L2_8)
    print(f"   成员数={len(members)}  OOF点={int(np.asarray(oof_mask).sum())}")

    print("\n[2] 三类预测 (raw 无校正 / full 含校正 / OOF) ...")
    train_raw = _ensemble_raw(members, X[usable], anchor[usable].values, feat_cols, cfg["shrinkage"])
    val_raw = _ensemble_raw(members, X[val_m], anchor[val_m].values, feat_cols, cfg["shrinkage"])
    train_full = _predict_load(members, X[usable], anchor[usable].values, feat_cols,
                               cfg["shrinkage"], hour_bias, drift_corr, threshold_corr)
    val_full = _predict_load(members, X[val_m], anchor[val_m].values, feat_cols,
                             cfg["shrinkage"], hour_bias, drift_corr, threshold_corr)

    oof_mask_arr = np.asarray(oof_mask)
    oof_p = oof_pred.values[oof_mask_arr]
    oof_a = actual.values[oof_mask_arr]
    tr_a = actual[usable].values
    va_a = actual[val_m].values

    m_train_raw = _mae(train_raw, tr_a)
    m_oof = _mae(oof_p, oof_a)
    m_val_raw = _mae(val_raw, va_a)
    m_train_full = _mae(train_full, tr_a)
    m_val_full = _mae(val_full, va_a)
    r2_train = _r2(train_raw, tr_a)
    r2_val = _r2(val_raw, va_a)

    print("\n" + "=" * 72)
    print("拟合诊断结果（CatBoost l2_8）")
    print("=" * 72)
    print(f"{'':28} {'MAE':>9} {'R²':>8}")
    print(f"{'train (raw, 含已见乐观)':28} {m_train_raw:>9.2f} {r2_train:>8.4f}")
    print(f"{'OOF (无泄露训练误差)':28} {m_oof:>9.2f} {'':>8}")
    print(f"{'val (raw)':28} {m_val_raw:>9.2f} {r2_val:>8.4f}")
    print(f"{'train (full, 含校正)':28} {m_train_full:>9.2f}")
    print(f"{'val (full, 含校正)':28} {m_val_full:>9.2f}   (vs v6 {V6_VAL_MAE})")
    print("-" * 72)
    gap_to = m_train_raw - m_oof
    ov = m_oof - m_val_raw
    print(f"train_raw − OOF   = {gap_to:+.1f}   (>400 过拟合; <150 欠拟合)")
    print(f"OOF − val_raw     = {ov:+.1f}   (|<200| 泛化一致/数据信号弱; >200 训练期更难; <-200 漂移)")

    # 分时段 train vs val (raw)
    print("\n分时段 MAE (raw):")
    h_tr = pd.DatetimeIndex(times[usable]).hour.values
    h_va = pd.DatetimeIndex(times[val_m]).hour.values
    err_tr = train_raw - tr_a
    err_va = val_raw - va_a
    print(f"{'时段':>8} {'train':>9} {'val':>9} {'Δ(val-tr)':>10}")
    for lo, hi, n in [(0, 6, "00-06"), (6, 11, "06-11"), (11, 15, "11-14"),
                      (15, 18, "15-18"), (18, 24, "18-24")]:
        mt = (h_tr >= lo) & (h_tr < hi)
        mv = (h_va >= lo) & (h_va < hi)
        tmae = float(np.mean(np.abs(err_tr[mt]))) if mt.sum() else float("nan")
        vmae = float(np.mean(np.abs(err_va[mv]))) if mv.sum() else float("nan")
        print(f"{n:>8} {tmae:>9.0f} {vmae:>9.0f} {vmae - tmae:>+10.0f}")

    # ---- 可选 LightGBM v6 对比 ----
    print("\n[3] LightGBM v6 配置同条件对比（若 lightgbm 可用）...")
    try:
        import lightgbm as lgb  # noqa: F401
        from .train import train_ensemble as lgb_train_ens
        lgb_model = lgb_train_ens(times, X, pred_load, actual, usable, cfg,
                                  cfg["best_it_fixed"], mos_model)

        def _lgb_raw(X_sub):
            Xarr = X_sub[feat_cols]
            av = anchor[X_sub.index].values
            mp = np.empty((len(lgb_model.members), len(X_sub)), dtype=float)
            for i, (booster, is_res) in enumerate(zip(lgb_model.members, lgb_model.member_residual)):
                raw = booster.predict(Xarr)
                mp[i] = av + raw if is_res else raw
            return av + cfg["shrinkage"] * (np.median(mp, axis=0) - av)

        lgb_train = _lgb_raw(X[usable])
        lgb_val = _lgb_raw(X[val_m])
        m_lgb_tr = _mae(lgb_train, tr_a)
        m_lgb_va = _mae(lgb_val, va_a)
        r2_lgb_tr = _r2(lgb_train, tr_a)
        r2_lgb_va = _r2(lgb_val, va_a)
        print(f"  {'':16} {'train_raw':>10} {'val_raw':>9}  {'R²_tr':>7} {'R²_val':>7}")
        print(f"  {'CatBoost l2_8':16} {m_train_raw:>10.2f} {m_val_raw:>9.2f}  {r2_train:>7.4f} {r2_val:>7.4f}")
        print(f"  {'LightGBM v6':16} {m_lgb_tr:>10.2f} {m_lgb_va:>9.2f}  {r2_lgb_tr:>7.4f} {r2_lgb_va:>7.4f}")
        d_tr = m_train_raw - m_lgb_tr
        d_va = m_val_raw - m_lgb_va
        print(f"  差(CB−LGB): train_raw {d_tr:+.1f}, val_raw {d_va:+.1f}")
        if d_tr > 50:
            print(f"  -> CatBoost train_raw 显著高于 LightGBM：拟合能力更弱 -> 模型问题(算法拟合能力差)")
        else:
            print(f"  -> 两者 train_raw 相近：拟合能力相当，val 差异来自泛化/数据，非 CatBoost 特有拟合不足")
    except ImportError:
        print("  (lightgbm 未安装，跳过对比；仅 CatBoost 自对比)")
    except Exception as e:
        ename = type(e).__name__
        print(f"  (LightGBM 对比失败: {ename}: {str(e)[:120]})")

    # ---- 判定 ----
    print("\n" + "=" * 72)
    print("判定：")
    if gap_to > 400:
        print(f"  train_raw({m_train_raw:.0f}) << OOF({m_oof:.0f})：能拟合训练数据但泛化差 -> 过拟合倾向")
    elif gap_to < 150:
        print(f"  train_raw({m_train_raw:.0f}) ≈ OOF({m_oof:.0f})：连训练数据都拟合不好 -> 欠拟合(能力不足/噪声大)")
    else:
        print(f"  train_raw({m_train_raw:.0f}) vs OOF({m_oof:.0f}) gap={gap_to:.0f}：轻度过拟合")
    if abs(ov) < 200:
        print(f"  OOF({m_oof:.0f}) ≈ val_raw({m_val_raw:.0f})：训练期与验证期误差一致 -> 泛化已一致，误差来自数据(信号弱/噪声)，非泛化失败")
    elif ov > 200:
        print(f"  OOF({m_oof:.0f}) > val_raw({m_val_raw:.0f})：训练期更难预测(数据更噪/季节差异)")
    else:
        print(f"  OOF({m_oof:.0f}) < val_raw({m_val_raw:.0f})：验证期显著更差 -> 漂移/泛化失败")
    print("=" * 72)
    print(f"\n耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
