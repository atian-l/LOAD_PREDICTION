# -*- coding: utf-8 -*-
"""
路线A Phase 0: TFT raw gate（快速验证 raw val 是否突破 v6 raw 天花板 1515）。

设计：用 2 成员（regression × {direct, residual} × seed 42）快速训练，避免 40 成员全量训练
耗时，先看 raw val 是否 < v6 raw 天花板 1515（fit_diag）。若 raw 已 < 1515 -> TFT 有潜力，
进 Phase 1（40 成员 + 调优）；若 raw >= 1515 -> 同 TCN(raw 1640) 无法突破 raw 地板，NO-GO。

报告：
  - TFT raw val（无 OOF 校正）vs v6 raw 1512
  - TFT OOF 校正后 val vs v6 1445.62
  - OOF/val slot 方向一致率（P2-8 纪律，<50% 判过拟合）

合规: 不修改生产脚本; 复用 build_dataset/train_ensemble/compute_hour_bias(不变量#5);
actual 仅 target/MOS-target/eval(#1); pred_load lags 不变(#2); OOF 3 折全在训练期内;
throwaway 纯 stdout。运行: python -m load_pred_tft.exp_tft_phase0  (云端 GPU 约 10-20 min)
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

V6_VAL_MAE = 1445.62      # v6 LightGBM 校正后 val（生产基线）
V6_RAW_MAE = 1512.0       # v6 raw val（无 OOF 校正，fit_diag 估计）
RAW_GATE = 1515.0         # raw gate 阈值（v6 raw 天花板）


def _mae(p, a):
    p = np.asarray(p, dtype=float); a = np.asarray(a, dtype=float)
    m = np.isfinite(p) & np.isfinite(a)
    return float(np.mean(np.abs(p[m] - a[m]))) if m.sum() else float('nan')


def _slot_idx(times, n_slots=96):
    dt = pd.DatetimeIndex(times)
    mod = dt.hour.values * 60 + dt.minute.values
    return ((mod * n_slots) // 1440).astype(int)


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print("路线A Phase 0: TFT raw gate（2 成员快速验证）")
    print(f"  v6 baseline={V6_VAL_MAE}  v6 raw≈{V6_RAW_MAE}  raw gate<{RAW_GATE}")
    print("=" * 74)

    print("[1] 构建数据集 ...")
    times, X, pred_load, actual = build_dataset()
    usable = usable_mask(times, pred_load, actual)
    mismatch_model = MismatchModel().fit(X, usable)
    X = mismatch_model.transform(X)
    mc = C.TRAIN_CONFIG["mos"]
    mos_model = MosModel(cols=mc["cols"], alpha=mc["alpha"]).fit(X, actual, usable)
    print(f"    特征数={X.shape[1]}  训练点={int(usable.sum())}")

    # Phase 0 小集成：2 成员（regression × {direct,residual} × seed 42）
    cfg = copy.deepcopy(C.TRAIN_CONFIG)
    cfg["objectives"] = ["regression"]
    cfg["residual_modes"] = [False, True]
    cfg["seeds"] = [42]
    epochs = int(cfg["best_it_fixed"])
    print(f"[2] 训练 TFT 2 成员 (regression × direct+residual × seed42, {epochs} epochs) ...")
    ts = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        model = train_ensemble(times, X, pred_load, actual, usable, cfg, epochs, mos_model=mos_model)
    model.mismatch_model = mismatch_model
    model.hour_bias = None
    model.drift_corr = []
    model.threshold_corr = []
    print(f"    训练完成 ({time.perf_counter()-ts:.0f}s)")

    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    av = actual[val_m].to_numpy(np.float64)

    # ---- [3] raw val（无 OOF 校正）----
    print("[3] raw val 预测（无 OOF 校正，TFT 按 predict_load full 前向）...")
    ts = time.perf_counter()
    pred_full_raw = model.predict_load(X, pred_load)
    raw_val = np.asarray(pred_full_raw)[val_m]
    print(f"    raw val MAE={_mae(raw_val, av):.2f}  (v6 raw≈{V6_RAW_MAE})  ({time.perf_counter()-ts:.0f}s)")

    # ---- [4] OOF 3 折 hour_bias + 校正后 val ----
    print(f"[4] OOF 3 折估 hour_bias（每折 2 成员）...")
    ts = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        hb, dc, tc = compute_hour_bias(times, X, pred_load, actual, usable, cfg, epochs, mos_model=mos_model)
    print(f"    OOF 完成 ({time.perf_counter()-ts:.0f}s)  hour_bias 范围=[{hb.min():.0f},{hb.max():.0f}]")
    model.hour_bias = hb
    model.drift_corr = dc
    model.threshold_corr = tc
    pred_full_corr = model.predict_load(X, pred_load)
    corr_val = np.asarray(pred_full_corr)[val_m]
    print(f"    OOF校正后 val MAE={_mae(corr_val, av):.2f}  (Δv6 {_mae(corr_val, av)-V6_VAL_MAE:+.2f})")

    # ---- [5] OOF/val slot 一致性 ----
    print("[5] OOF/val slot 方向一致性 ...")
    # 重建 OOF 残差（用 compute_hour_bias 内部逻辑再算一次太贵；用 raw 残差 vs OOF bias 方向）
    raw_resid_val = raw_val - av
    val_slot = _slot_idx(times[val_m], 96)
    slot_all = _slot_idx(times, 96)
    # OOF bias 来自训练期 OOF；val raw 残差方向应与 OOF bias 同号才迁移
    consistent = 0; total = 0
    for q in range(96):
        mv = val_slot == q
        if mv.sum() < 5:
            continue
        total += 1
        val_b = np.mean(raw_resid_val[mv])
        if np.sign(hb[q]) == np.sign(val_b):
            consistent += 1
    consist_pct = 100.0 * consistent / total if total else 0.0
    print(f"    slot 方向一致: {consistent}/{total} = {consist_pct:.1f}%  (<50% 判过拟合)")

    # ---- 汇总 + gate ----
    raw_m = _mae(raw_val, av)
    corr_m = _mae(corr_val, av)
    print("\n" + "=" * 74)
    print("Phase 0 gate 判定")
    print("=" * 74)
    print(f"  TFT raw val        = {raw_m:.2f}  (v6 raw≈{V6_RAW_MAE}, gate<{RAW_GATE})")
    print(f"  TFT OOF校正后 val  = {corr_m:.2f}  (v6={V6_VAL_MAE}, Δv6 {corr_m-V6_VAL_MAE:+.2f})")
    print(f"  OOF/val 一致率     = {consist_pct:.1f}%")
    if raw_m < RAW_GATE:
        print(f"  -> RAW GATE 通过: raw {raw_m:.2f} < {RAW_GATE}，TFT 有潜力突破 raw 地板。")
        print(f"     建议 Phase 1: 40 成员全量 + encoder_len/lr/hidden 调优 + OOF gate。")
    else:
        print(f"  -> RAW GATE 未通过: raw {raw_m:.2f} >= {RAW_GATE}，TFT 同 TCN 无法突破 raw 地板。")
        print(f"     建议: 路线A NO-GO，回 v6 / 转路线B。")
    if corr_m < V6_VAL_MAE and consist_pct >= 50:
        print(f"  -> 校正后已破 v6 且 OOF/val 一致，TFT 路线强信号。")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
