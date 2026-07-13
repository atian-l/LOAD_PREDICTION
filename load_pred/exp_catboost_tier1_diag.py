# -*- coding: utf-8 -*-
"""
CatBoost Tier1 诊断 + 后处理（A2/A3/A4/F1/F2/G5）

训练 l2_8 配置一次（40 成员 + OOF 校正），复用同一组成员做 6 个诊断/后处理实验：
  A2 成员级误差分解   - 定位哪类成员(direct/residual × reg/quantile)拖后腿/极端
  A3 时段/场景误差分布 - 定位极端误差集中的时段/场景（指导 G4 增场景）
  A4 残差可校正性上界 - oracle 96-slot 去偏后的 MAE，判定校正天花板（指导 G2 是否值得）
  F1 聚合方式扫描     - median/mean/trimmed 哪个降 val MAE/q99
  F2 shrinkage λ 扫描 - λ∈{0.5,0.7,1.0,1.2,1.5}
  G5 上界 clip 扫描   - 防 CatBoost 重尾高预测拖高 MAE

合规：
  - 不修改任何生产脚本；仅 import train.build_dataset / features.{MismatchModel,MosModel}
    / exp_catboost_ab.{_predict_load,_arr,V6_VAL_MAE} / exp_catboost_hp.{_train_ensemble,_compute_oof}。
  - actual 仅作 target/MOS 目标/评估；训练仅 usable mask(<=TRAIN_END)；val eval-only。
  - 6 条泄露不变量全保持。OOF 校正在 3 折(best_it_folds)上估计，不接触官方验证集。

Caveat：
  - F1 改聚合后 hour_bias/drift/threshold 未重估（沿用 median 聚合估的校正），仅为快速筛选；
    若 F1 某聚合显著优，需在 Tier2 单独重估校正确认。F2(λ)/G5(clip) 不改校正，结果严格。
  - A2 "去某类"用 raw median（无校正），仅看成员结构趋势。

运行：python -m load_pred.exp_catboost_tier1_diag   （4090 上约 4-6 min）
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

# l2_8 = hp 实验最优配置（depth8/lr0.03/l2_8/bt1.0/SymmetricTree, best_it=80）
HP_L2_8 = {"depth": 8, "lr": 0.03, "l2": 8.0, "bagging_temp": 1.0,
           "grow_policy": "SymmetricTree", "max_leaves": None}
BEST_IT = 80
L2_8_VAL_MAE = 1477.67  # hp 实验 l2_8 基线（含 OOF 校正，+32MW vs v6）


# --------------------------------------------------------------------------- #
# 参数化预测原语（支持聚合/λ/clip 扫描）
# --------------------------------------------------------------------------- #
def _member_preds(members, X, anchor_vals, feat_cols):
    """逐成员 raw 预测 -> (n_members, n_points)。"""
    Xarr = _arr(X, feat_cols)
    mp = np.empty((len(members), len(X)), dtype=float)
    for i, (m, is_res) in enumerate(members):
        raw = m.predict(Xarr)
        mp[i] = anchor_vals + raw if is_res else raw
    return mp


def _trim_mean(mp, p):
    """去首尾 p 比例后均值（axis=0）。"""
    n = mp.shape[0]
    k = int(n * p)
    if 2 * k >= n:
        return np.median(mp, axis=0)
    mp_s = np.sort(mp, axis=0)
    return np.mean(mp_s[k:n - k], axis=0)


def _aggregate(mp, agg):
    if agg == "median":
        return np.median(mp, axis=0)
    if agg == "mean":
        return np.mean(mp, axis=0)
    if agg.startswith("trimmed"):
        p = float(agg.split(":")[1])
        return _trim_mean(mp, p)
    raise ValueError(agg)


def _predict_param(mp, anchor_vals, shrinkage, agg, hour_bias,
                   drift_corr, threshold_corr, X, clip_upper):
    """参数化完整预测：aggregate -> λ收缩 -> hour_bias -> drift -> threshold -> clip。"""
    ens = _aggregate(mp, agg)
    pred = anchor_vals + shrinkage * (ens - anchor_vals)
    dt = pd.DatetimeIndex(X.index)
    hours = dt.hour.values.astype(int)
    if hour_bias is not None:
        n = len(hour_bias)
        mod = hours * 60 + dt.minute.values
        idx = ((mod * n) // 1440).astype(int)
        pred = pred - hour_bias[idx]
    for feat_name, beta in drift_corr:
        beta = np.asarray(beta, dtype=float)
        pred = pred + beta[hours] * X[feat_name].values.astype(float)
    for tc in threshold_corr:
        fv = X[tc["feature"]].values.astype(float)
        op = tc.get("op", ">")
        thr = tc["thr"]
        if op == "range":
            sel = (fv >= thr[0]) & (fv < thr[1])
        elif op == ">=":
            sel = fv >= thr
        elif op == "<":
            sel = fv < thr
        elif op == "<=":
            sel = fv <= thr
        else:
            sel = fv > thr
        hl = tc.get("hours")
        if hl is not None:
            sel = sel & np.isin(hours, list(hl))
        shift = tc.get("shift", 0.0)
        if shift != 0.0:
            pred[sel] = pred[sel] - shift
    if clip_upper is None:
        return np.clip(pred, 0.0, None)
    return np.clip(pred, 0.0, float(clip_upper))


def _member_labels(cfg):
    """重建 40 成员标签（顺序同 _train_ensemble: residual×obj×alpha×seed）。"""
    labels = []
    for residual in cfg["residual_modes"]:
        for obj in cfg["objectives"]:
            alphas = cfg["quantile_alphas"] if obj == "quantile" else [None]
            for qa in alphas:
                for s in cfg["seeds"]:
                    mode = "res" if residual else "dir"
                    objn = "reg" if obj == "regression" else f"q{qa}"
                    labels.append({"mode": mode, "obj": objn, "seed": s})
    return labels


# --------------------------------------------------------------------------- #
def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print(f"CatBoost Tier1 诊断+后处理 (l2_8, best_it={BEST_IT})")
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
    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    Xv = X[val_m]
    av = actual[val_m].to_numpy(np.float64)
    anchor_v = anchor[val_m].to_numpy(np.float64)
    dt_val = pd.DatetimeIndex(times[val_m])
    hours_val = dt_val.hour.values
    print(f"    特征数={len(feat_cols)}  val点={int(val_m.sum())}")

    print("[1] 训练 l2_8 official 40 成员 + OOF 3 折校正...")
    with contextlib.redirect_stdout(io.StringIO()):
        members = hp._train_ensemble(X, actual, anchor, usable, cfg, BEST_IT, feat_cols, HP_L2_8)
        hour_bias, drift_corr, threshold_corr, oof_pred, oof_mask = hp._compute_oof(
            times, X, pred_load, actual, usable, anchor, cfg, BEST_IT, feat_cols, HP_L2_8)
    print(f"    成员数={len(members)}  OOF点={int(oof_mask.sum())}")

    # 基线 val_raw(无校正) / val_full(当前校正)
    mp_val = _member_preds(members, Xv, anchor_v, feat_cols)  # (40, n_val)
    val_raw = anchor_v + cfg["shrinkage"] * (np.median(mp_val, axis=0) - anchor_v)
    val_full = _predict_load(members, Xv, anchor_v, feat_cols, cfg["shrinkage"],
                             hour_bias, drift_corr, threshold_corr)
    err_full = val_full - av
    mae_full = float(np.mean(np.abs(err_full)))
    mae_raw = float(np.mean(np.abs(val_raw - av)))
    print(f"    val_raw(无校正) MAE={mae_raw:.2f}  val_full(当前校正) MAE={mae_full:.2f}")

    # ===================== A2 成员级误差分解 ===================== #
    print("\n" + "=" * 74)
    print("[A2] 成员级误差分解（逐成员 raw val 误差）")
    print("=" * 74)
    labels = _member_labels(cfg)
    keys = [f"{l['mode']}-{'q' if l['obj'] != 'reg' else 'reg'}" for l in labels]
    member_mae = np.array([float(np.mean(np.abs(mp_val[i] - av))) for i in range(len(members))])
    member_max = np.array([float(np.max(np.abs(mp_val[i] - av))) for i in range(len(members))])
    member_q99 = np.array([float(np.percentile(np.abs(mp_val[i] - av), 99)) for i in range(len(members))])
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    print(f"  {'类':12} {'n':>3} {'MAE均':>8} {'MAE min/max':>16} {'max均':>8} {'q99均':>8}")
    for k in sorted(groups):
        idx = groups[k]
        print(f"  {k:12} {len(idx):>3} {member_mae[idx].mean():>8.1f} "
              f"{member_mae[idx].min():>6.1f}/{member_mae[idx].max():<6.1f} "
              f"{member_max[idx].mean():>8.0f} {member_q99[idx].mean():>8.0f}")
    print("  去某类后集成 raw MAE（无校正重估，仅趋势）:")
    for k in sorted(groups):
        sel = [i for i in range(len(members)) if keys[i] != k]
        ens = np.median(mp_val[sel], axis=0)
        mae = float(np.mean(np.abs(ens - av)))
        print(f"    去 {k:10} (剩{len(sel):2}): raw MAE={mae:.2f}")

    # ===================== A3 时段/场景误差分布 ===================== #
    print("\n" + "=" * 74)
    print("[A3] 时段/场景误差分布（val_full）")
    print("=" * 74)
    print(f"  {'时段':10} {'n':>5} {'MAE':>8} {'q99':>8} {'max':>8}")
    for name, lo, hi in [("00-06", 0, 6), ("06-11", 6, 11), ("11-14", 11, 14),
                         ("15-18", 15, 18), ("18-24", 18, 24)]:
        m = (hours_val >= lo) & (hours_val < hi)
        if m.sum():
            e = np.abs(err_full[m])
            print(f"  {name:10} {int(m.sum()):>5} {e.mean():>8.1f} "
                  f"{float(np.percentile(e, 99)):>8.0f} {e.max():>8.0f}")
    clearness = Xv["clearness"].to_numpy(float) if "clearness" in Xv else None
    precip = Xv["precip"].to_numpy(float) if "precip" in Xv else None
    temp = Xv["temp"].to_numpy(float) if "temp" in Xv else None
    mid = (hours_val >= 11) & (hours_val <= 14)
    scen = []
    if clearness is not None:
        scen.append(("clear_noon clr>0.8@11-14", mid & (clearness > 0.8)))
        scen.append(("cloudy_noon clr[0.2,0.5)@11-14", mid & (clearness >= 0.2) & (clearness < 0.5)))
    if precip is not None:
        scen.append(("rainy precip>0", precip > 0))
    if temp is not None:
        scen.append(("cold temp<8", temp < 8))
    any_scen = np.zeros(len(av), dtype=bool)
    for _, s in scen:
        any_scen |= s
    base_mae = float(np.mean(np.abs(err_full[~any_scen]))) if (~any_scen).sum() else float("nan")
    print(f"  {'场景':30} {'n':>5} {'MAE':>8} {'基线MAE':>8} {'比值':>6}")
    for name, s in scen:
        if s.sum():
            e = np.abs(err_full[s])
            r = e.mean() / base_mae if base_mae == base_mae else float("nan")
            print(f"  {name:30} {int(s.sum()):>5} {e.mean():>8.1f} {base_mae:>8.1f} {r:>6.2f}")
    if (~any_scen).sum():
        print(f"  {'baseline(非场景)':30} {int((~any_scen).sum()):>5} "
              f"{base_mae:>8.1f} {base_mae:>8.1f} {1.00:>6.2f}")
    abserr = np.abs(err_full)
    top = np.argsort(-abserr)[:20]
    print("  Top20 误差点:")
    for r, i in enumerate(top):
        print(f"    {r + 1:2}. {times[val_m][i]}  actual={av[i]:.0f} "
              f"pred={val_full[i]:.0f} err={err_full[i]:+.0f}")

    # ===================== A4 残差可校正性上界 ===================== #
    print("\n" + "=" * 74)
    print("[A4] 残差可校正性上界（oracle 96-slot 去偏）")
    print("=" * 74)
    err_raw = val_raw - av
    mod_val = hours_val * 60 + dt_val.minute.values
    slot_val = (mod_val // 15).astype(int)
    oracle_bias = np.zeros(96)
    for q in range(96):
        m = slot_val == q
        if m.sum():
            oracle_bias[q] = float(np.mean(err_raw[m]))
    debiased_oracle = val_raw - oracle_bias[slot_val]
    mae_oracle = float(np.mean(np.abs(debiased_oracle - av)))
    gain_current = mae_raw - mae_full
    gain_residual = mae_full - mae_oracle
    print(f"  val_raw(无校正)     MAE = {mae_raw:.2f}")
    print(f"  val_full(当前校正)  MAE = {mae_full:.2f}   (当前校正收益 {gain_current:+.2f})")
    print(f"  debiased_oracle     MAE = {mae_oracle:.2f}   (剩余校正空间 {gain_residual:+.2f})")
    print(f"  v6={V6_VAL_MAE}  l2_8基线={L2_8_VAL_MAE}")
    if mae_oracle > V6_VAL_MAE:
        print(f"  -> oracle 天花板 {mae_oracle:.0f} > v6 {V6_VAL_MAE}："
              f"校正无望追平 v6，G2 recency 收益有限")
    elif mae_oracle < V6_VAL_MAE + 10:
        print(f"  -> oracle 天花板 {mae_oracle:.0f} 接近 v6：G2 recency 值得做")
    else:
        print(f"  -> oracle 天花板 {mae_oracle:.0f} 介于 v6 与 l2_8 之间：G2 或有部分收益")

    # ===================== F1 聚合方式扫描 ===================== #
    print("\n" + "=" * 74)
    print("[F1] 聚合方式扫描（复用 official members，校正不重估 -> 仅快速筛选）")
    print("=" * 74)
    print(f"  {'聚合':16} {'val MAE':>10} {'q99':>8} {'max':>8} {'Bias':>8}")
    for agg in ["median", "mean", "trimmed:0.1", "trimmed:0.2", "trimmed:0.3"]:
        pred = _predict_param(mp_val, anchor_v, cfg["shrinkage"], agg,
                              hour_bias, drift_corr, threshold_corr, Xv, None)
        e = pred - av
        print(f"  {agg:16} {float(np.mean(np.abs(e))):>10.2f} "
              f"{float(np.percentile(np.abs(e), 99)):>8.0f} "
              f"{float(np.max(np.abs(e))):>8.0f} {float(np.mean(e)):>+8.1f}")

    # ===================== F2 shrinkage λ 扫描 ===================== #
    print("\n" + "=" * 74)
    print("[F2] shrinkage λ 扫描（median 聚合，校正不重估）")
    print("=" * 74)
    print(f"  {'λ':>5} {'val MAE':>10} {'q99':>8} {'Bias':>8}")
    for lam in [0.5, 0.7, 1.0, 1.2, 1.5]:
        pred = _predict_param(mp_val, anchor_v, lam, "median",
                              hour_bias, drift_corr, threshold_corr, Xv, None)
        e = pred - av
        print(f"  {lam:>5.1f} {float(np.mean(np.abs(e))):>10.2f} "
              f"{float(np.percentile(np.abs(e), 99)):>8.0f} {float(np.mean(e)):>+8.1f}")

    # ===================== G5 上界 clip 扫描 ===================== #
    print("\n" + "=" * 74)
    print("[G5] 上界 clip 扫描（median, λ=1.0）")
    print("=" * 74)
    amax = float(av.max())
    ap99 = float(np.percentile(av, 99))
    print(f"  actual: max={amax:.0f} p99={ap99:.0f}")
    print(f"  {'上界':16} {'val MAE':>10} {'q99':>8} {'max(pred)':>10}")
    uppers = [(None, "None(当前)"), (80000.0, "80000"), (75000.0, "75000"),
              (70000.0, "70000"), (ap99, "actual_p99"),
              (amax * 1.2, "max*1.2"), (amax * 1.3, "max*1.3")]
    for upper, label in uppers:
        pred = _predict_param(mp_val, anchor_v, cfg["shrinkage"], "median",
                              hour_bias, drift_corr, threshold_corr, Xv, upper)
        e = pred - av
        print(f"  {label:16} {float(np.mean(np.abs(e))):>10.2f} "
              f"{float(np.percentile(np.abs(e), 99)):>8.0f} {float(np.max(pred)):>10.0f}")

    print("\n" + "=" * 74)
    print(f"Tier1 诊断+后处理完成  耗时 {time.perf_counter() - t0:.0f}s")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
