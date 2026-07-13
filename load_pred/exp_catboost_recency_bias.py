# -*- coding: utf-8 -*-
"""
CatBoost G2: recency-weighted hour_bias

OOF 估计 96-slot hour_bias 时，给近期折（冬2026，最接近验证集）更大权重，
让 slot bias 更贴近 val，改善 bias 迁移（Phase 0 指出的 debiased 漂移问题）。

流程：
  1. 训练 l2_8 official 40 成员 + OOF 3 折（复用 hp._train_ensemble / _compute_oof）
  2. 取 OOF 残差 resid = oof_pred - actual（仅 oof_mask 上）
  3. 等权 hour_bias（基线，= hp._compute_oof 输出）
  4. 多组 recency 权重下的 hour_bias：
       - 连续衰减 0.5 + 1.5*tnorm（tnorm = 时间归一化 [0,1]）
       - 折阶跃 [1,1.5,2.5] / [1,2,4] / [1,1,2]（春/秋/冬）
       - winter_only x3（仅冬折加权）
  5. 各 hour_bias 套到 val（official members + 同 drift/threshold），算 MAE/q99/Bias
  6. 折间残差均值漂移诊断（看冬折 vs 春/秋折残差偏置差）

合规：
  - 不修改任何生产脚本；仅 import train.build_dataset / features.{MismatchModel,MosModel}
    / exp_catboost_ab.{_predict_load,_arr,V6_VAL_MAE} / exp_catboost_hp.{_train_ensemble,_compute_oof}。
  - OOF 残差仅在 3 折(best_it_folds)上估计，不接触官方验证集；actual 仅作 target/MOS/评估。
  - 6 条泄露不变量全保持。recency 权重仅作用于 OOF 残差的 slot 估计（训练期数据），不引入 val 信息。

Caveat：
  - drift_corr/threshold_corr 沿用等权 OOF 估的值（不变），仅换 hour_bias。
    若某 recency 方案显著优，可在 Tier2 把 drift/threshold 也纳入 recency 重估。
  - 天花板受 Phase 0 debiased~1483 限制，预期降 3-10MW，难追 v6 1445。

运行：python -m load_pred.exp_catboost_recency_bias   （4090 上约 4-6 min）
"""
from __future__ import annotations
import sys
import time
import io
import contextlib
import warnings

import numpy as np
import pandas as pd

from . import config as C
from .train import build_dataset, usable_mask
from .features import MismatchModel, MosModel
from .exp_catboost_ab import _predict_load, _arr, V6_VAL_MAE
from . import exp_catboost_ab as ab
from . import exp_catboost_hp as hp

HP_L2_8 = {"depth": 8, "lr": 0.03, "l2": 8.0, "bagging_temp": 1.0,
           "grow_policy": "SymmetricTree", "max_leaves": None}
BEST_IT = 80
L2_8_VAL_MAE = 1477.67


def _weighted_hour_bias(resid, slot_all, oof_mask, weights, n_slots=96):
    """加权 slot 残差均值（仅在 oof_mask 上）。weights 与 resid 同长度。"""
    hb = np.zeros(n_slots)
    for q in range(n_slots):
        m = oof_mask & (slot_all == q)
        if m.sum():
            w = weights[m]
            if np.sum(w) > 0:
                hb[q] = float(np.average(resid[m], weights=w))
            else:
                hb[q] = float(np.mean(resid[m]))
    return hb


def _fold_of(t, folds):
    """返回 t 所属 OOF 折索引（0=春,1=秋,2=冬），不在任何折返回 -1。"""
    for k, (_te, vs, ve) in enumerate(folds):
        if pd.Timestamp(vs) <= t <= pd.Timestamp(ve):
            return k
    return -1


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print(f"CatBoost G2: recency-weighted hour_bias (l2_8, best_it={BEST_IT})")
    print(f"  基线: l2_8 val={L2_8_VAL_MAE}  v6={V6_VAL_MAE}")
    print("=" * 74)

    print("[0] 构建数据集...")
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
    folds = cfg["best_it_folds"]
    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    Xv = X[val_m]
    av = actual[val_m].to_numpy(np.float64)
    anchor_v = anchor[val_m].to_numpy(np.float64)
    print(f"    特征数={len(feat_cols)}  val点={int(val_m.sum())}")

    print("[1] 训练 official 40 成员 + OOF 3 折校正...")
    with contextlib.redirect_stdout(io.StringIO()):
        members = hp._train_ensemble(X, actual, anchor, usable, cfg, BEST_IT, feat_cols, HP_L2_8)
        hour_bias_eq, drift_corr, threshold_corr, oof_pred, oof_mask = hp._compute_oof(
            times, X, pred_load, actual, usable, anchor, cfg, BEST_IT, feat_cols, HP_L2_8)
    resid = (oof_pred - actual).to_numpy(np.float64)
    times_arr = pd.DatetimeIndex(times)
    slot_all = (times_arr.hour.values * 60 + times_arr.minute.values) // 15
    slot_all = slot_all.astype(int)
    print(f"    成员数={len(members)}  OOF点={int(oof_mask.sum())}")

    # 基线 val_full（等权 hour_bias）
    val_full_eq = _predict_load(members, Xv, anchor_v, feat_cols, cfg["shrinkage"],
                                hour_bias_eq, drift_corr, threshold_corr)
    mae_eq = float(np.mean(np.abs(val_full_eq - av)))
    print(f"    等权 hour_bias val MAE={mae_eq:.2f}")

    # ---- 构造 recency 权重 ----
    # 连续衰减：tnorm = (t - tmin_oof) / (tmax_oof - tmin_oof) in [0,1] on oof_mask
    ts_sec = times_arr.values.astype("datetime64[s]").astype(np.float64)
    tmin_s = ts_sec[oof_mask].min()
    tmax_s = ts_sec[oof_mask].max()
    tnorm = (ts_sec - tmin_s) / max(1.0, tmax_s - tmin_s)
    # 折索引（向量化）
    fold_idx = np.full(len(times), -1, dtype=int)
    for k, (_te, vs, ve) in enumerate(folds):
        vs_ts = pd.Timestamp(vs)
        ve_ts = pd.Timestamp(ve)
        in_fold = (times_arr >= vs_ts) & (times_arr <= ve_ts)
        fold_idx[in_fold] = k

    weight_schemes = [
        ("equal(基线)",          np.ones(len(times))),
        ("cont_decay 0.5+1.5tn", 0.5 + 1.5 * tnorm),
        ("fold[1,1.5,2.5]",      np.where(fold_idx == 0, 1.0, np.where(fold_idx == 1, 1.5, np.where(fold_idx == 2, 2.5, 1.0)))),
        ("fold[1,2,4]",          np.where(fold_idx == 0, 1.0, np.where(fold_idx == 1, 2.0, np.where(fold_idx == 2, 4.0, 1.0)))),
        ("fold[1,1,2]",          np.where(fold_idx == 0, 1.0, np.where(fold_idx == 1, 1.0, np.where(fold_idx == 2, 2.0, 1.0)))),
        ("winter_only x3",       np.where(fold_idx == 2, 3.0, 1.0)),
    ]

    print("\n" + "=" * 74)
    print("[2] recency-weighted hour_bias 对比（drift/threshold 不变）")
    print("=" * 74)
    print(f"  {'方案':22} {'val MAE':>10} {'Δ基线':>8} {'q99':>8} {'Bias':>8} {'bias范围':>16}")
    for name, weights in weight_schemes:
        hb = _weighted_hour_bias(resid, slot_all, oof_mask, weights) if name != "equal(基线)" else hour_bias_eq
        pred = _predict_load(members, Xv, anchor_v, feat_cols, cfg["shrinkage"],
                             hb, drift_corr, threshold_corr)
        e = pred - av
        mae = float(np.mean(np.abs(e)))
        q99 = float(np.percentile(np.abs(e), 99))
        bias = float(np.mean(e))
        print(f"  {name:22} {mae:>10.2f} {mae - mae_eq:>+8.2f} {q99:>8.0f} "
              f"{bias:>+8.1f} [{hb.min():.0f},{hb.max():.0f}]")

    # ---- 折间残差漂移诊断 ----
    print("\n[3] OOF 残差折间漂移诊断")
    fold_names = ["春2025", "秋2025", "冬2026"]
    for fk in range(3):
        m = oof_mask & (fold_idx == fk)
        if m.sum():
            r = resid[m]
            print(f"  {fold_names[fk]}: n={int(m.sum())}  残差均值={float(np.mean(r)):+.1f}  "
                  f"|残差|均值={float(np.mean(np.abs(r))):.1f}  残差q99={float(np.percentile(np.abs(r), 99)):.0f}")

    print("\n" + "=" * 74)
    print(f"G2 recency-hour_bias 完成  耗时 {time.perf_counter() - t0:.0f}s")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
