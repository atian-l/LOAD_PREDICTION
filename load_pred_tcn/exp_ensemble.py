# -*- coding: utf-8 -*-
"""
exp_ensemble：集成后处理扫（λ 收缩 / 聚合方式 / trim / 成员数）。

【合规提示】λ、aggregation、trim_frac 属"其他东西"（生产 λ=1.0 / median / trim0.2，v6 最优）。
本脚本对它们作**诊断探针**：若 λ<1 或 mean/trimmed 显著更优，说明 TCN 弱于锚或成员冗余，
**记录但不改生产配置**（忠实端口约束）。成员数扫用全目标结构（仅变 seeds 数）。

三块：
  Block 1（post-hoc，无重训）：λ ∈ {0.3,0.5,0.7,1.0}，median 聚合。RAW（无 OOF 校正）val MAE。
  Block 2（post-hoc）：aggregation ∈ {median,mean,trimmed} × trim ∈ {0.1,0.2,0.3}，λ=1。RAW val MAE。
  Block 3（重训，WF-CV 选）：成员数 {8,16}（seeds=[42]/[42,7]，全目标），+ 参考 40 成员=TCN 基线。

Block 1/2 在同一 16 成员模型上做（训练一次，post-hoc 即时）；RAW 以隔离 λ/聚合效应
（OOF 校正的边际贡献见 exp_oof_ablation）。选择信号 Block 3 用 WF-CV；Block 1/2 用 RAW val
读数（post-hoc 无 WF-CV 概念，仅看相对排序）。
运行：python -m load_pred_tcn.exp_ensemble
"""
from __future__ import annotations
import numpy as np

from . import exp_common as E
from .tcn import predict_tcn


# Block 1/2 用的模型：2 seeds × 全目标(regression+quantile×3) × {direct,residual} = 16 成员
POSTHOC_OVERRIDE = {"seeds": [42, 7]}


def _member_preds(model, X_sub, pred_load_sub, mos_model):
    """计算各成员在 X_sub 上的 raw 预测（anchor+raw if residual else raw）。返回 [n_mem, T] 与 anchor。"""
    X_arr = X_sub[model.feature_cols].to_numpy(dtype=np.float32)
    pl = pred_load_sub.reindex(X_sub.index).values.astype(float)
    anchor = mos_model.transform(X_sub) if mos_model is not None else pl
    mp = np.empty((len(model.members), len(X_sub)), dtype=float)
    for i, (tcn, is_res) in enumerate(zip(model.members, model.member_residual)):
        raw = predict_tcn(tcn, X_arr, model.feat_mean, model.feat_std, model.device)
        mp[i] = anchor + raw if is_res else raw
    return mp, anchor


def _aggregate(mp, agg, trim):
    if agg == "mean":
        return mp.mean(axis=0)
    if agg == "trimmed":
        n = mp.shape[0]
        k = int(np.floor(n * trim / 2))
        if k > 0 and (n - 2 * k) > 0:
            Ms = np.sort(mp, axis=0)
            return Ms[k:n - k].mean(axis=0)
        return mp.mean(axis=0)
    return np.median(mp, axis=0)  # median（默认）


def _lam_pred(mp, anchor, lam, agg, trim):
    """anchor + λ·(agg(mp) − anchor)，含 clip（与 predict_load 一致）。"""
    ens = _aggregate(mp, agg, trim)
    pred = anchor + lam * (ens - anchor)
    return np.clip(pred, 0.0, None)


def main():
    d = E.build_cached()
    times, X, pred_load, actual, val_m, mos_model = (d["times"], d["X"], d["pred_load"],
                                                      d["actual"], d["val_m"], d["mos_model"])
    va = actual[val_m].values
    print(f"\n[exp_ensemble] 数据: 特征{X.shape[1]} 可用{d['usable'].sum()} val{val_m.sum()}")
    print("[Block 1/2] 训练 16 成员模型（seeds=[42,7] 全目标）用于 post-hoc 扫 ...")
    model = E.train_ens(POSTHOC_OVERRIDE)
    mp, anchor = _member_preds(model, X[val_m], pred_load[val_m], mos_model)
    n_mem = mp.shape[0]

    # ---- Block 1：λ 扫（median 聚合，RAW）----
    print("\n" + "=" * 70)
    print(f"[Block 1] λ 扫 (median 聚合, RAW, {n_mem} 成员)   生产 λ=1.0")
    print("=" * 70)
    print(f"{'λ':>6} {'val_MAE':>9} {'Δvsλ1':>8}")
    base1 = None
    for lam in (0.3, 0.5, 0.7, 1.0):
        mae = E._mae(_lam_pred(mp, anchor, lam, "median", 0.2), va)
        if lam == 1.0:
            base1 = mae
        dlt = "" if base1 is None else f"{mae - base1:+8.1f}"
        print(f"{lam:>6.1f} {mae:>9.1f} {dlt}")
    print(f"  注：λ<1 更优 => TCN 弱于锚（诊断，不改生产 λ=1.0）")

    # ---- Block 2：聚合 × trim 扫（λ=1，RAW）----
    print("\n" + "=" * 70)
    print(f"[Block 2] 聚合 × trim 扫 (λ=1, RAW, {n_mem} 成员)   生产 median/trim0.2")
    print("=" * 70)
    print(f"{'agg':>9} {'trim':>5} {'val_MAE':>9} {'Δvs_med':>9}")
    base2 = E._mae(_lam_pred(mp, anchor, 1.0, "median", 0.2), va)
    for agg in ("median", "mean", "trimmed"):
        trims = (0.2,) if agg != "trimmed" else (0.1, 0.2, 0.3)
        for tr in trims:
            mae = E._mae(_lam_pred(mp, anchor, 1.0, agg, tr), va)
            print(f"{agg:>9} {tr:>5.1f} {mae:>9.1f} {mae - base2:+9.1f}")
    print(f"  注：mean/trimmed 更优 => 成员冗余/离群（诊断，不改生产 median）")

    # ---- Block 3：成员数扫（全目标，变 seeds；WF-CV 选）----
    print("\n" + "=" * 70)
    print(f"[Block 3] 成员数扫 (全目标, 变 seeds; WF-CV)   40 成员生产基线 MAE={E.TCN_BASE_MAE}")
    print("=" * 70)
    mc_configs = [
        ("mem8_s1",  {"seeds": [42]}),       # 1×(1+3)×2 = 8
        ("mem16_s2", {"seeds": [42, 7]}),    # 2×4×2 = 16
    ]
    rows, stage_a = E.hp_sweep(mc_configs, "exp_ensemble 成员数扫", topk=2)
    print(f"\n  参考: 生产 40 成员(5 seeds) TCN 基线 val MAE={E.TCN_BASE_MAE} "
          f"(v6 LGB={E.V6_VAL_MAE})；成员数↑通常边际递减，须权衡训练成本)")


if __name__ == "__main__":
    main()
