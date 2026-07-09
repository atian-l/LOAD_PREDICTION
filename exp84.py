# -*- coding: utf-8 -*-
"""exp84 - CS1 特征剪枝独立验证（walk-forward 跨折稳定性）。

FDS/report.md 候选建议 CS1：移除在 v6 验证集上置换重要性为负的特征（置换后 MAE 反降，
提示过拟合噪声）。本实验独立验证该剪枝是否在多折上稳定有益，还是仅官方 val 偶然。

候选剪枝特征（a11 置换重要性为负；已排除 lag_192——CLAUDE.md 规定其为 mandatory minimum
lag，不得移除）：
  pred_load_vs_mean_672 (-6.3), pred_load_roll_mean_96 (-2.4), pred_load_roll_mean_672 (-2.1),
  pl_x_temp (-1.2), dayofyear (-0.6)

实验设计（三部分，互为印证）：
  Part A  官方验证集【全管线】（集成 + OOF 校正；校正用 cfg 3 折，均在 val 之前，无泄露）
          基线应复现 1445.62；剪枝版给出生产意义上的 Δ。
  Part B  4 折 walk-forward【原始集成】稳定性（每折 mismatch/MOS 就地按 train_mask 重拟合，
          严格无泄露）。关注 Δ=prune-base 在各折的符号与幅度，判稳定性。
  Part C  官方验证集【原始集成】逐特征 LOO，归属哪个特征驱动效应。

合规：仅训练期数据训练；actual 仅作目标/评估；不修改生产代码/模型/训练流程；不写任何产物。
判定：剪枝稳定有益 = Part A Δ≤0 且 Part B 各折 Δ 均无显著恶化（≤+5 MW）。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from load_pred import config as C, data_loader as dl, features as F, train as T

# 候选剪枝特征集（不含 lag_192：CLAUDE.md 强制保留的最短滞后）
PRUNE_SET = ["pred_load_vs_mean_672", "pred_load_roll_mean_96",
             "pred_load_roll_mean_672", "pl_x_temp", "dayofyear"]

# (折名, 训练截止, 验证起, 验证止) —— 前 3 折在训练期内，第 4 折为官方验证窗
FOLDS = [
    ("spring25", "2025-02-28 23:45:00", "2025-03-01 00:00:00", "2025-05-31 23:45:00"),
    ("autumn25", "2025-08-31 23:45:00", "2025-09-01 00:00:00", "2025-11-30 23:45:00"),
    ("winter26", "2025-12-31 23:45:00", "2026-01-01 00:00:00", "2026-02-28 23:45:00"),
    ("official", C.TRAIN_END,            C.VAL_START,            C.VAL_END),
]
DEC = int(C.TRAIN_CONFIG.get("round_decimals", 2))


def mae(pred, act):
    p = np.round(np.asarray(pred, dtype=float), DEC)
    return float(np.abs(p - act).mean())


def build_full():
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)
    X_base = F.build_features(times, pred_load, weather)
    usable = T.usable_mask(times, pred_load, actual)
    return times, X_base, pred_load, actual, usable


def full_pipeline_val(times, X_base, pred_load, actual, usable, cfg, best_it, val_mask, drop_cols=None):
    """官方 val 全管线：mismatch+MOS 在 usable 上拟合，集成 + OOF 校正（cfg 3 折，val 前无泄露）。"""
    mm = F.MismatchModel().fit(X_base, usable); X = mm.transform(X_base)
    mos = F.MosModel().fit(X, actual, usable)
    if drop_cols:
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    model = T.train_ensemble(times, X, pred_load, actual, usable, cfg, best_it, mos_model=mos)
    model.mismatch_model = mm
    model.hour_bias, model.drift_corr, model.threshold_corr = T.compute_hour_bias(
        times, X, pred_load, actual, usable, cfg, best_it, mos_model=mos)
    return model.predict_load(X[val_mask], pred_load[val_mask]), X.shape[1]


def raw_ensemble_fold(times, X_base, pred_load, actual, cfg, best_it, train_mask, val_mask, drop_cols=None):
    """单折原始集成（无校正；mismatch/MOS 就地按 train_mask 拟合，严格无泄露）。返回 val MAE。"""
    mm = F.MismatchModel().fit(X_base, train_mask); X = mm.transform(X_base)
    mos = F.MosModel().fit(X, actual, train_mask)
    if drop_cols:
        X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    model = T.train_ensemble(times, X, pred_load, actual, train_mask, cfg, best_it, mos_model=mos)
    model.mismatch_model = mm
    pred = model.predict_load(X[val_mask], pred_load[val_mask])
    return mae(pred, actual[val_mask].values)


def main():
    cfg = C.TRAIN_CONFIG; best_it = cfg["best_it_fixed"]
    times, X_base, pred_load, actual, usable = build_full()
    missing = [c for c in PRUNE_SET if c not in X_base.columns]
    assert not missing, f"剪枝候选特征不存在于特征矩阵: {missing}"
    print(f"基础特征数={X_base.shape[1]}  剪枝候选({len(PRUNE_SET)})={PRUNE_SET}", flush=True)

    val_mask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
                & actual.notna() & pred_load.notna()).values
    act_val = actual[val_mask].values

    # ================ Part A: 官方验证集全管线 ================
    print("\n==== Part A: 官方验证集全管线（集成 + OOF 校正）====", flush=True)
    pred_A0, nf0 = full_pipeline_val(times, X_base, pred_load, actual, usable, cfg, best_it, val_mask)
    mae_A0 = mae(pred_A0, act_val)
    pred_A1, nf1 = full_pipeline_val(times, X_base, pred_load, actual, usable, cfg, best_it,
                                     val_mask, drop_cols=PRUNE_SET)
    mae_A1 = mae(pred_A1, act_val)
    dA = mae_A1 - mae_A0
    print(f"A0 基线(全特征 n={nf0})        val MAE={mae_A0:.2f}  (生产 v6≈1445.62)", flush=True)
    print(f"A1 剪枝(去{len(PRUNE_SET)}特征 n={nf1})    val MAE={mae_A1:.2f}  Δ={dA:+.2f}", flush=True)

    # ================ Part B: 4 折 walk-forward 原始集成稳定性 ================
    print("\n==== Part B: 4 折 walk-forward 原始集成稳定性（无校正，就地重拟合）====", flush=True)
    hdr = f"{'折':<10}{'val_range':<26}{'base_MAE':>10}{'prune_MAE':>11}{'Δ':>9}"
    print(hdr, flush=True)
    deltas_B = []; fold_base = {}
    for name, te, vs, ve in FOLDS:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        tr_mask = usable if name == "official" else (usable & (times <= te))
        vmask = ((times >= vs) & (times <= ve) & actual.notna() & pred_load.notna()).values
        m_base = raw_ensemble_fold(times, X_base, pred_load, actual, cfg, best_it, tr_mask, vmask)
        m_prun = raw_ensemble_fold(times, X_base, pred_load, actual, cfg, best_it, tr_mask, vmask,
                                   drop_cols=PRUNE_SET)
        fold_base[name] = m_base
        d = m_prun - m_base
        deltas_B.append(d)
        vr = f"{vs.strftime('%m-%d')}~{ve.strftime('%m-%d')}"
        print(f"{name:<10}{vr:<26}{m_base:>10.1f}{m_prun:>11.1f}{d:>+9.1f}", flush=True)
    n_help = sum(1 for d in deltas_B if d < 0)
    max_hurt = max(deltas_B)
    print(f"Part B: {n_help}/4 折剪枝有益(Δ<0)  最大恶化={max_hurt:+.1f} MW", flush=True)

    # ================ Part C: 官方验证集原始集成逐特征 LOO ================
    print("\n==== Part C: 官方验证集原始集成逐特征 LOO（归属驱动特征）====", flush=True)
    m_baseC = fold_base["official"]
    print(f"基线(原始集成) val MAE={m_baseC:.2f}", flush=True)
    print(f"{'移除特征':<26}{'MAE':>10}{'Δ':>9}", flush=True)
    for f in PRUNE_SET:
        m_f = raw_ensemble_fold(times, X_base, pred_load, actual, cfg, best_it, usable, val_mask,
                                drop_cols=[f])
        print(f"{f:<26}{m_f:>10.1f}{m_f - m_baseC:>+9.1f}", flush=True)

    # ================ 判定 ================
    print("\n================ CS1 判定 ================", flush=True)
    stable = max_hurt <= 5.0
    print(f"Part A 全管线 Δ = {dA:+.2f} MW", flush=True)
    print(f"Part B 各折 Δ = {[round(d, 1) for d in deltas_B]}  稳定={stable}", flush=True)
    if dA <= 0 and stable:
        verdict = "VALIDATED 稳定有益（或持平无害）—— 可作为候选生产变更，仍需正式 train.py 复测确认"
    elif abs(dA) < 3 and stable:
        verdict = "NEUTRAL 持平无害——不降 MAE，仅可降复杂度，价值有限"
    else:
        verdict = "REJECTED 不稳定或有害——官方 val 置换重要性为负系偶然/过拟合，不建议采纳"
    print(f"结论: {verdict}", flush=True)


if __name__ == "__main__":
    main()
