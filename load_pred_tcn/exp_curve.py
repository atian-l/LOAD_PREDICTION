# -*- coding: utf-8 -*-
"""
TCN 逐 epoch 学习曲线（exp_curve）-- 调优门控脚本（配合 exp_fit_diag）。

训练一个小集成，按不同 epoch 数从零重训，在每个 epoch 点上评估：
  - train_mae  : 训练折内 MAE（含已见，随 epoch 下降）
  - foldval_mae: 一折 held-out MAE（先降后升，触底处 = 过拟合 onset）
  - gap        : train_mae − foldval_mae（随 epoch 扩大 = 过拟合加剧）

直接看 foldval_mae 第几个 epoch 点触底回升 -> 客观定 best_it_fixed 最优点
（替代当前"凭经验 60"）。并对比不同 dropout 下曲线差异，验证正则效果。

用冬季折（最接近 val）作 held-out：train on <=2025-12-31，eval on 2026-01/02。
小集成（2 成员：regression × {direct,residual} × seed42）保速度。
运行：python -m load_pred_tcn.exp_curve   （4090 上约 3~8 min）
"""
from __future__ import annotations
import sys
import time
import warnings

import numpy as np
import pandas as pd

from . import config as C
from . import exp_common as ec

# 冬季折（best_it_folds 第三折，最接近 val）
FOLD = ("2025-12-31", "2026-01-01", "2026-02-28")
EPOCHS = [10, 20, 30, 40, 50, 60, 80, 100, 120]
DROPOUTS = [0.1, 0.2, 0.3, 0.4]   # 正则对比
QUICK_MEMBERS = {"seeds": [42], "objectives": ["regression"], "residual_modes": [False, True]}


def _fold_masks(times, usable):
    te, vs, ve = (pd.Timestamp(FOLD[0]), pd.Timestamp(FOLD[1]), pd.Timestamp(FOLD[2]))
    ftr = usable & np.asarray(times <= te)
    fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
    return ftr, fva


def _curve_one(dropout, weight_decay, d, ftr, fva):
    """单条曲线：epoch 扫描，返回 [(ep, train_mae, foldval_mae, gap)]。"""
    times, X, pred_load, actual = d["times"], d["X"], d["pred_load"], d["actual"]
    a_tr = actual[ftr].values
    a_va = actual[fva].values
    rows = []
    for ep in EPOCHS:
        ov = dict(QUICK_MEMBERS)
        ov["dropout"] = dropout
        ov["weight_decay"] = weight_decay
        m = ec.train_ens(ov, usable=ftr, best_it=ep, verbose=False)
        tr = ec.ensemble_raw(m, X[ftr], pred_load[ftr])
        va = ec.ensemble_raw(m, X[fva], pred_load[fva])
        tm = ec._mae(tr, a_tr); vm = ec._mae(va, a_va)
        rows.append((ep, tm, vm, tm - vm))
    return rows


def main() -> int:
    t0 = time.perf_counter()
    d = ec.build_cached()
    times, X, pred_load, actual, usable = (d["times"], d["X"], d["pred_load"],
                                           d["actual"], d["usable"])
    ftr, fva = _fold_masks(times, usable)
    cur_dropout = C.TRAIN_CONFIG["dropout"]
    cur_wd = C.TRAIN_CONFIG["weight_decay"]
    print("=" * 78)
    print(f"TCN 学习曲线  (held-out 折={FOLD[1]}~{FOLD[2]}; 小集成 2 成员)")
    print(f"  当前生产: dropout={cur_dropout} weight_decay={cur_wd}")
    print("=" * 78)
    print(f"  训练折点={int(ftr.sum())}  held-out点={int(fva.sum())}")

    # ---- [1] 当前配置 epoch 扫描曲线 ----
    print(f"\n[1] 当前配置 epoch 扫描 (dropout={cur_dropout}, wd={cur_wd}) ...")
    rows = _curve_one(cur_dropout, cur_wd, d, ftr, fva)
    print(f"\n  {'epoch':>6} {'train_mae':>10} {'foldval_mae':>12} {'gap':>8}")
    for ep, tm, vm, g in rows:
        print(f"  {ep:>6} {tm:>10.1f} {vm:>12.1f} {g:>+8.1f}")
    best = min(rows, key=lambda r: r[2])
    onset = next((r for r in rows if r[2] > best[2] and r[0] > best[0]), None)
    print(f"\n  foldval_mae 最小 @ epoch={best[0]} -> {best[2]:.1f} (train={best[1]:.1f})")
    if onset:
        print(f"  首次回升 @ epoch={onset[0]} -> {onset[2]:.1f} (过拟合 onset，best_it_fixed 宜 ≤{best[0]})")
    else:
        print(f"  未见回升（扫描范围内未过拟合）-> best_it_fixed 可取上限 {rows[-1][0]}")

    # ---- [2] dropout 正则对比（在 onset 附近 epoch 与上限 epoch 两点）----
    print(f"\n[2] dropout 正则对比 @ epoch={EPOCHS[-1]} (看 gap 宽窄) ...")
    print(f"  {'dropout':>8} {'train_mae':>10} {'foldval_mae':>12} {'gap':>8}")
    for dp in DROPOUTS:
        ov = dict(QUICK_MEMBERS); ov["dropout"] = dp; ov["weight_decay"] = cur_wd
        m = ec.train_ens(ov, usable=ftr, best_it=EPOCHS[-1], verbose=False)
        tr = ec.ensemble_raw(m, X[ftr], pred_load[ftr])
        va = ec.ensemble_raw(m, X[fva], pred_load[fva])
        tm = ec._mae(tr, actual[ftr].values)
        vm = ec._mae(va, actual[fva].values)
        mark = "  <- 当前" if abs(dp - cur_dropout) < 1e-9 else ""
        print(f"  {dp:>8} {tm:>10.1f} {vm:>12.1f} {tm - vm:>+8.1f}{mark}")
    print("  (gap 越窄=过拟合越轻；foldval 越低=泛化越好)")

    print("\n" + "=" * 78)
    print("判定：")
    print(f"  best_it_fixed 客观最优点 ≈ epoch {best[0]} (foldval 最低 {best[2]:.0f})")
    print("  若当前 60ep 已过 onset -> 减 epoch；若未到 onset -> 可增 epoch 或先看正则对比。")
    print("=" * 78)
    print(f"\n耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
