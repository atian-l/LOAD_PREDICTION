# -*- coding: utf-8 -*-
"""
Phase 0B：TCN 重尾诊断（tcn tuning plan Phase 0B）

加载已存模型（不重训），在官方 val 上分析误差分布，定位重尾根因 + 试 cheap 修复。

诊断项：
  1. 基础指标（确认 MAE/q50/q99，对照 2027/1362/10223）
  2. Top 20 误差点（时段/场景/特征值）
  3. 分时段 MAE/q99（重尾集中点）
  4. 各成员 raw 预测分布（找极端成员；direct/residual × regression/quantile）
  5. clip 上界敏感性（现 model.py:166 只 clip 下界 0；试上界，无重训）
  6. 聚合敏感性（median/mean/trimmed/direct_only/residual_only/regression_only）
  7. 校正贡献（raw ens vs +校正，看校正是否在极端点过度）
  8. 特征标准化后极值（|z|>10 边缘样本致爆炸？）

合规：只读模型 + 推理；不训练、不写模型；actual 仅评估；6 不变量保持。
运行：python -m load_pred_tcn.exp_tcn_tail_diag  （云端，需已训练模型，~5min）
"""
from __future__ import annotations
import sys
import warnings

import numpy as np
import pandas as pd

from . import config as C
from .train import build_dataset, usable_mask
from .model import EnsembleModel
from .tcn import predict_tcn

V6_VAL_MAE = 1445.62


def _mae(p, a):
    m = np.isfinite(a) & np.isfinite(p)
    return float(np.mean(np.abs(p[m] - a[m]))) if m.sum() else float("nan")


def _r2(p, a):
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    ss_res = float(np.sum((p - a) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _member_preds(model, X_sub, pred_load_sub):
    """复现 predict_load 的成员预测部分，返回 (member_preds[N,T], anchor[T])。"""
    pl = pred_load_sub.reindex(X_sub.index).values.astype(float)
    anchor = model.mos_model.transform(X_sub) if model.mos_model is not None else pl
    X_arr = X_sub[model.feature_cols].to_numpy(dtype=np.float32)
    mp = np.empty((len(model.members), len(X_sub)), dtype=float)
    for i, (tcn, is_res) in enumerate(zip(model.members, model.member_residual)):
        raw = predict_tcn(tcn, X_arr, model.feat_mean, model.feat_std, model.device)
        mp[i] = anchor + raw if is_res else raw
    return mp, np.asarray(anchor, dtype=float)


def main() -> int:
    print("=" * 76)
    print("Phase 0B: TCN 重尾诊断 (加载已存模型, 不重训)")
    print(f"  参考: val MAE=2027.47 / q50=1362 / q99=10223 ; v6={V6_VAL_MAE}")
    print("=" * 76)

    print("[1] 加载模型 + 构建数据 ...")
    model = EnsembleModel.load(C.MODEL_BUNDLE)
    times, X, pred_load, actual = build_dataset()
    usable = usable_mask(times, pred_load, actual)
    if model.mismatch_model is not None:
        X = model.mismatch_model.transform(X)
    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    X_val = X[val_m]
    pl_val = pred_load[val_m]
    a_val = actual[val_m].values.astype(float)
    print(f"    成员数={len(model.members)}  val点={int(val_m.sum())}  特征数={len(model.feature_cols)}")

    print("\n[2] 完整 predict_load (含校正) ...")
    pred_val = model.predict_load(X_val, pl_val)
    err = pred_val - a_val
    ae = np.abs(err)
    print(f"    MAE={_mae(pred_val, a_val):.2f}  R2={_r2(pred_val, a_val):.4f}  Bias={np.mean(err):.2f}")
    print(f"    |err| 分位: q50={np.quantile(ae,0.5):.0f}  q90={np.quantile(ae,0.9):.0f}  "
          f"q95={np.quantile(ae,0.95):.0f}  q99={np.quantile(ae,0.99):.0f}  max={np.nanmax(ae):.0f}")

    # ---- 3. Top 20 误差点 ----
    print("\n[3] Top 20 误差点:")
    top_idx = np.argsort(-ae)[:20]
    feat_show = [f for f in ["pred_load", "clearness", "temp", "precip", "pl_weather_residual"]
                 if f in X_val.columns]
    hdr = f"    {'时间':19} {'actual':>8} {'pred':>8} {'err':>8}"
    for f in feat_show:
        hdr += f" {f[:9]:>10}"
    print(hdr)
    dt_val = pd.DatetimeIndex(X_val.index)
    for i in top_idx:
        line = f"    {dt_val[i].strftime('%Y-%m-%d %H:%M')} {a_val[i]:>8.0f} {pred_val[i]:>8.0f} {err[i]:>+8.0f}"
        for f in feat_show:
            v = float(X_val[f].values[i])
            line += f" {v:>10.2f}"
        print(line)

    # ---- 4. 分时段 ----
    print("\n[4] 分时段 MAE / q99 (重尾集中点):")
    h = dt_val.hour.values
    print(f"    {'时段':>8} {'MAE':>8} {'q50':>8} {'q99':>8} {'max':>8} {'n':>6}")
    for lo, hi, n in [(0,6,"00-06"),(6,11,"06-11"),(11,15,"11-14"),(15,18,"15-18"),(18,24,"18-24")]:
        m = (h >= lo) & (h < hi)
        if m.sum():
            e = ae[m]
            print(f"    {n:>8} {_mae(pred_val[m], a_val[m]):>8.0f} {np.quantile(e,0.5):>8.0f} "
                  f"{np.quantile(e,0.99):>8.0f} {np.nanmax(e):>8.0f} {m.sum():>6}")

    # ---- 5. 成员 raw 预测分布 ----
    print("\n[5] 各成员 raw 预测分布 (找极端成员; 按 max 降序 Top10):")
    mp, anchor = _member_preds(model, X_val, pl_val)
    print(f"    {'成员':>4} {'mode':>5} {'MAE':>8} {'max':>9} {'q99':>9} {'min':>9}")
    rows = []
    for i, is_res in enumerate(model.member_residual):
        p = mp[i]
        pf = p[np.isfinite(p)]
        rows.append((i, "res" if is_res else "dir", _mae(p, a_val),
                      float(np.nanmax(p)), float(np.quantile(pf, 0.99)), float(np.nanmin(p))))
    rows.sort(key=lambda r: -r[3])
    for i, mode, mae_i, mx, q9, mn in rows[:10]:
        print(f"    {i:>4} {mode:>5} {mae_i:>8.0f} {mx:>9.0f} {q9:>9.0f} {mn:>9.0f}")
    all_max = [r[3] for r in rows]
    print(f"    成员 max 范围: [{min(all_max):.0f}, {max(all_max):.0f}]  "
          f"(actual max={np.nanmax(a_val):.0f})")

    # ---- 6. clip 上界敏感性（关键 cheap 修复）----
    print("\n[6] clip 上界敏感性 (无重训, 在最终 pred 上试上界):")
    a_max = float(np.nanmax(a_val))
    a_p99 = float(np.quantile(a_val, 0.99))
    print(f"    actual: max={a_max:.0f} p99={a_p99:.0f} p95={np.quantile(a_val,0.95):.0f}")
    uppers = [None, 80000, 75000, 70000, a_p99, a_max * 1.2, a_max * 1.3]
    print(f"    {'上界':>12} {'MAE':>8} {'q99|err|':>10} {'max|err|':>10}")
    for up in uppers:
        p = np.clip(pred_val, 0.0, up) if up is not None else np.clip(pred_val, 0.0, None)
        e = np.abs(p - a_val)
        label = "None(当前)" if up is None else f"{up:.0f}"
        print(f"    {label:>12} {_mae(p, a_val):>8.0f} {np.quantile(e,0.99):>10.0f} {np.nanmax(e):>10.0f}")

    # ---- 7. 聚合敏感性 ----
    print("\n[7] 聚合敏感性 (无重训, 用已算 member_preds; raw 无校正):")
    n_mem = mp.shape[0]
    ens_median = np.median(mp, axis=0)
    ens_mean = np.mean(mp, axis=0)
    k = int(np.floor(n_mem * 0.2 / 2))
    ens_trim = np.mean(np.sort(mp, axis=0)[k:n_mem - k], axis=0) if k > 0 else ens_mean
    dir_idx = [i for i, r in enumerate(model.member_residual) if not r]
    res_idx = [i for i, r in enumerate(model.member_residual) if r]
    # regression 成员: 每 20 块的前 5（train_ensemble 顺序: residual->obj->alpha->seed）
    reg_idx = [i for i in range(n_mem) if (i % 20) < 5]
    ens_dir = np.median(mp[dir_idx], axis=0) if dir_idx else ens_median
    ens_res = np.median(mp[res_idx], axis=0) if res_idx else ens_median
    ens_reg = np.median(mp[reg_idx], axis=0) if reg_idx else ens_median
    print(f"    {'聚合':>16} {'raw_MAE':>9} {'raw_q99':>9} {'raw_max':>9}")
    for name, ens in [("median(当前)", ens_median), ("mean", ens_mean), ("trimmed0.2", ens_trim),
                       ("direct_only", ens_dir), ("residual_only", ens_res),
                       ("regression_only", ens_reg)]:
        e = np.abs(ens - a_val)
        print(f"    {name:>16} {_mae(ens, a_val):>9.0f} {np.quantile(e,0.99):>9.0f} {np.nanmax(e):>9.0f}")

    # ---- 8. 校正贡献 ----
    print("\n[8] 校正贡献 (raw ens median -> 完整校正):")
    base = anchor + model.shrinkage * (ens_median - anchor)
    e0 = np.abs(base - a_val)
    e1 = np.abs(err)
    print(f"    raw ens(无校正):  MAE={_mae(base, a_val):.0f}  q99={np.quantile(e0,0.99):.0f}  max={np.nanmax(e0):.0f}")
    print(f"    +完整校正:        MAE={_mae(pred_val, a_val):.0f}  q99={np.quantile(e1,0.99):.0f}  max={np.nanmax(e1):.0f}")
    print(f"    校正降 MAE {_mae(base, a_val) - _mae(pred_val, a_val):.0f} MW  "
          f"(q99 {np.quantile(e0,0.99) - np.quantile(e1,0.99):.0f})")

    # ---- 9. 特征标准化后极值 ----
    print("\n[9] 特征标准化后极值 (|z|>10 边缘样本?):")
    Xa = X_val[model.feature_cols].to_numpy(dtype=np.float64)
    Xf = np.where(np.isnan(Xa), model.feat_mean[None, :], Xa)
    Xn = (Xf - model.feat_mean[None, :]) / model.feat_std[None, :]
    zmax = float(np.nanmax(np.abs(Xn)))
    n_extreme = int(np.sum(np.abs(Xn) > 10))
    print(f"    max|z|={zmax:.1f}  |z|>10 的样本-特征数={n_extreme} (总 {Xn.size})")
    col_max = np.nanmax(np.abs(Xn), axis=0)
    top_cols = np.argsort(-col_max)[:5]
    for c in top_cols:
        print(f"    {model.feature_cols[c]:>30}  max|z|={col_max[c]:.1f}")

    print("\n" + "=" * 76)
    print("诊断完成。重点看:")
    print("  [6] clip 上界 - 哪个上界最优（无重训 cheap 修复）")
    print("  [7] 聚合 - 哪种聚合降 q99/max 最多")
    print("  [3][4] 重尾是否集中在特定时段/场景")
    print("  [5] 极端成员 - 是否某类成员（residual/quantile）系统性爆炸")
    print("  [9] 特征极值 - 是否边缘样本致爆炸")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
