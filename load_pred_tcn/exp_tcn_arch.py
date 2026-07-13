# -*- coding: utf-8 -*-
"""
Phase 2：TCN 架构超参调优（tcn tuning plan Phase 2）

每配置完整 40 成员 + 3 折 OOF 校正 + val 评估，6 指标 gate。
配置维度：epochs / num_channels / seq_len / dropout / kernel_size

FAST_MODE（默认 True）：减 seeds(2) + 1 折 OOF，快速筛选趋势（~16 成员/配置）。
  最优配置须 FAST_MODE=False 全量（40 成员 + 3 折）确认后再落地 config.py。

合规：仅复用 train.py 函数；actual 仅 target/eval；6 不变量保持；不写生产模型。
运行：python -m load_pred_tcn.exp_tcn_arch  （云端，FAST_MODE ~2h / 全量 ~9h）
"""
from __future__ import annotations
import sys
import time
import copy
import warnings

import numpy as np
import pandas as pd

from . import config as C
from .train import build_dataset, usable_mask, train_ensemble, compute_hour_bias
from .features import MismatchModel, MosModel

V6_VAL_MAE = 1445.62

# (tag, epochs, num_channels, seq_len, dropout, kernel_size)
CONFIGS = [
    ("baseline", 120, [64, 64, 64, 64],         480, 0.1, 7),
    ("ep200",    200, [64, 64, 64, 64],         480, 0.1, 7),
    ("ep300",    300, [64, 64, 64, 64],         480, 0.1, 7),
    ("ch128",    120, [128, 128, 128, 128],     480, 0.1, 7),
    ("deep6",    120, [64, 64, 64, 64, 64, 64], 480, 0.1, 7),
    ("seq672",   120, [64, 64, 64, 64],         672, 0.1, 7),
    ("drop02",   120, [64, 64, 64, 64],         480, 0.2, 7),
    ("kern9",    120, [64, 64, 64, 64],         480, 0.1, 9),
]

FAST_MODE = True  # True=快速筛选(2 seeds/1 折); False=全量(40 成员/3 折)确认


def _make_cfg(epochs, channels, seq_len, dropout, kernel):
    cfg = copy.deepcopy(C.TRAIN_CONFIG)
    cfg["best_it_fixed"] = epochs
    cfg["num_channels"] = list(channels)
    cfg["seq_len"] = seq_len
    cfg["dropout"] = dropout
    cfg["kernel_size"] = kernel
    if FAST_MODE:
        cfg["seeds"] = [42, 7]                       # 2 seeds -> 16 成员
        cfg["best_it_folds"] = [cfg["best_it_folds"][0]]  # 1 折 OOF
    return cfg


def _eval_metrics(pred, actual):
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    ae = np.abs(err)
    return {"MAE": mae, "R2": r2, "RMSE": rmse, "Bias": bias,
            "q50": float(np.quantile(ae, 0.5)), "q99": float(np.quantile(ae, 0.99)),
            "max": float(np.nanmax(ae))}


def _run_config(tag, epochs, channels, seq_len, dropout, kernel,
                times, X, pred_load, actual, usable, mos_model, val_m):
    cfg = _make_cfg(epochs, channels, seq_len, dropout, kernel)
    rf = 1 + (kernel - 1) * sum(2 ** i for i in range(len(channels)))
    best_it = cfg["best_it_fixed"]
    print(f"\n  [{tag}] epochs={epochs} ch={channels} seq={seq_len} drop={dropout} k={kernel} RF={rf}")
    ts = time.perf_counter()
    model = train_ensemble(times, X, pred_load, actual, usable, cfg, best_it, mos_model=mos_model)
    model.hour_bias, model.drift_corr, model.threshold_corr = compute_hour_bias(
        times, X, pred_load, actual, usable, cfg, best_it, mos_model=mos_model)
    pred_val = model.predict_load(X[val_m], pred_load[val_m])
    mt = _eval_metrics(pred_val, actual[val_m].values.astype(float))
    dt = time.perf_counter() - ts
    print(f"  [{tag}] MAE={mt['MAE']:.0f} q50={mt['q50']:.0f} q99={mt['q99']:.0f} "
          f"max={mt['max']:.0f} Bias={mt['Bias']:.0f} R2={mt['R2']:.4f} ({dt:.0f}s)")
    mt["tag"] = tag
    mt["dt"] = dt
    return mt


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 78)
    print(f"Phase 2: TCN 架构超参调优  (FAST_MODE={FAST_MODE}, v6={V6_VAL_MAE})")
    print(f"  FAST_MODE={'ON(16成员/1折, 筛选)' if FAST_MODE else 'OFF(40成员/3折, 确认)'}")
    print("=" * 78)

    print("[1] 构建数据集（共享，仅一次）...")
    times, X, pred_load, actual = build_dataset()
    usable = usable_mask(times, pred_load, actual)
    mismatch_model = MismatchModel().fit(X, usable)
    X = mismatch_model.transform(X)
    mos_model = None
    if C.TRAIN_CONFIG.get("mos"):
        mc = C.TRAIN_CONFIG["mos"]
        mos_model = MosModel(cols=mc.get("cols"), alpha=mc.get("alpha", 1.0)).fit(X, actual, usable)
    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    print(f"    特征数={X.shape[1]}  训练点={int(usable.sum())}  val点={int(val_m.sum())}  配置数={len(CONFIGS)}")

    print(f"\n[2] 逐配置训练 + 评估 ...")
    rows = []
    for tag, epochs, channels, seq_len, dropout, kernel in CONFIGS:
        try:
            mt = _run_config(tag, epochs, channels, seq_len, dropout, kernel,
                             times, X, pred_load, actual, usable, mos_model, val_m)
            rows.append(mt)
        except Exception as e:
            ename = type(e).__name__
            msg = str(e).splitlines()[0][:90]
            print(f"  [{tag}] FAIL ({ename}: {msg})")

    if not rows:
        print("\n所有配置失败。")
        return 1

    # ---- 对比表 ----
    print("\n" + "=" * 78)
    print(f"架构调优对比（FAST_MODE={FAST_MODE}; v6={V6_VAL_MAE}; 当前 TCN~2027/q99~10223）")
    print("=" * 78)
    print(f"{'tag':>10} {'MAE':>7} {'Δv6':>7} {'q50':>6} {'q99':>6} {'max':>7} {'Bias':>6} {'R2':>7} {'time':>6}")
    for r in rows:
        print(f"{r['tag']:>10} {r['MAE']:>7.0f} {r['MAE']-V6_VAL_MAE:>+7.0f} {r['q50']:>6.0f} "
              f"{r['q99']:>6.0f} {r['max']:>7.0f} {r['Bias']:>+6.0f} {r['R2']:>7.4f} {r['dt']:>5.0f}s")

    best = min(rows, key=lambda r: r["MAE"])
    print(f"\n最优: {best['tag']}  MAE={best['MAE']:.0f} (Δv6 {best['MAE']-V6_VAL_MAE:+.0f})  "
          f"q99={best['q99']:.0f}  q50={best['q50']:.0f}")
    print("-" * 78)
    if best["MAE"] < V6_VAL_MAE:
        print(f"  最优 MAE < v6！须 FAST_MODE=False 全量确认 + 折间稳定后才落地 config.py。")
    elif best["MAE"] < 1500:
        print(f"  最优 MAE < 1500 gate。建议全量确认 + 重尾修复（Phase 1）后看能否追 v6。")
    else:
        print(f"  最优 MAE 仍 >1500。结合 Phase 0B 重尾诊断 + Phase 1 修复（clip/聚合）再评估。")
    if FAST_MODE:
        print("  (FAST_MODE=ON: 结果为筛选趋势，非定论；最优配置须 OFF 全量确认)")
    print("=" * 78)

    try:
        with open("exp_tcn_arch_result.txt", "w", encoding="utf-8") as f:
            f.write(f"v6={V6_VAL_MAE} FAST_MODE={FAST_MODE}\n")
            f.write("tag\tMAE\tDelta_v6\tq50\tq99\tmax\tBias\tR2\ttime_s\n")
            for r in rows:
                f.write(f"{r['tag']}\t{r['MAE']:.4f}\t{r['MAE']-V6_VAL_MAE:+.4f}\t{r['q50']:.4f}\t"
                        f"{r['q99']:.4f}\t{r['max']:.4f}\t{r['Bias']:.4f}\t{r['R2']:.4f}\t{r['dt']:.0f}\n")
            f.write(f"best={best['tag']}\n")
        print("(已写 exp_tcn_arch_result.txt)")
    except Exception as e:
        print(f"(写结果失败: {e})")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
