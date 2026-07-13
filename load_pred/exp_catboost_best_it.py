# -*- coding: utf-8 -*-
"""
Phase 3-A：best_it 单变量扫描

目的：验证 Phase 1 的 bias=-152 / 折间不稳是否由 best_iter=216 过大所致
      （复刻 v6 当年 walk-forward 248 过拟合 2026 的坑，v6 改 fixed 80 解决）。

做法：best_it ∈ {40,80,120,160,216}，每个跑完整管线（40 成员 + OOF 校正重估 + val 评估），
      其余超参与 Phase 1 完全一致（depth=8 / lr=0.03 / l2=4 / bagging_temp=1 / Plain / rsm=1）。
      额外报告 debiased_MAE = mean(|err - bias|) —— 去整体偏置后的 MAE，
      作为"若 bias 完美校正"的上限诊断（oracle，不进生产）。

预期：U 型曲线，最优 best_it 在 60~120；bias 随 best_it 减小而改善 -> 确认过拟合假设。

运行：python -m load_pred.exp_catboost_best_it   （4090 上约 8~12 min）
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
from . import exp_catboost_ab as ab
from .exp_catboost_ab import (
    _cb_train_ensemble, _cb_compute_hour_bias, _predict_load, _metrics, V6_VAL_MAE,
)

BEST_ITS = [40, 80, 120, 160, 216]


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 70)
    print(f"Phase 3-A: best_it 单变量扫描   (val vs v6={V6_VAL_MAE})")
    print("=" * 70)

    print("[1] 构建数据集（复用 Phase 1 同一特征矩阵）...")
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
    print(f"    特征数={len(feat_cols)}  val点={int(val_m.sum())}")

    print(f"\n[2] 扫描 best_it = {BEST_ITS} ...")
    rows = []
    for best_it in BEST_ITS:
        ts = time.perf_counter()
        # 静默 OOF 估计的逐折/逐校正量打印，只留本层汇总
        with contextlib.redirect_stdout(io.StringIO()):
            members = _cb_train_ensemble(X, actual, anchor, usable, cfg, best_it, feat_cols)
            hour_bias, drift_corr, threshold_corr, oof_pred, oof_mask = _cb_compute_hour_bias(
                times, X, pred_load, actual, usable, anchor, cfg, best_it, feat_cols)
        pred_val = _predict_load(members, X[val_m], anchor[val_m].values, feat_cols,
                                 cfg["shrinkage"], hour_bias, drift_corr, threshold_corr)
        actual_val = actual[val_m]
        mt = _metrics(pd.Series(pred_val, index=times[val_m]), actual_val, times[val_m])
        # debiased MAE：去整体 bias 后的 MAE（oracle 上限诊断）
        err = pred_val - actual_val.values
        debiased = float(np.mean(np.abs(err - err.mean())))
        # 折 MAE + CV
        fmaes = []
        for te, vs, ve in cfg["best_it_folds"]:
            vs, ve = pd.Timestamp(vs), pd.Timestamp(ve)
            fm = usable & (times >= vs) & (times <= ve) & oof_pred.notna().values
            if fm.sum():
                fmaes.append(float(np.mean(np.abs(oof_pred[fm].values - actual[fm].values))))
        farr = np.array(fmaes)
        fcv = float(farr.std() / farr.mean()) if len(farr) >= 2 and farr.mean() > 0 else float("nan")
        dt = time.perf_counter() - ts
        rows.append({
            "best_it": best_it, "MAE": mt["MAE"], "Bias": mt["Bias"], "R2": mt["R2"],
            "midday": mt["midday_MAE"], "debiased": debiased, "fcv": fcv,
            "fmaes": fmaes, "dt": dt,
        })
        print(f"  best_it={best_it:3d}  MAE={mt['MAE']:.2f}  Bias={mt['Bias']:+7.1f}  "
              f"debiased={debiased:.2f}  折CV={fcv:.3f}  折MAE={[f'{m:.0f}' for m in fmaes]}  ({dt:.0f}s)")

    # ---- 汇总对比表 ----
    print("\n" + "=" * 70)
    print("best_it 扫描对比（vs v6 1445.62）")
    print("=" * 70)
    print(f"{'best_it':>7} {'MAE':>8} {'Δv6':>8} {'Bias':>8} {'debiased':>9} {'午间':>7} {'折CV':>6}")
    for r in rows:
        print(f"{r['best_it']:>7} {r['MAE']:>8.2f} {r['MAE']-V6_VAL_MAE:>+8.2f} "
              f"{r['Bias']:>+8.1f} {r['debiased']:>9.2f} {r['midday']:>7.0f} {r['fcv']:>6.3f}")

    best = min(rows, key=lambda r: r["MAE"])
    print(f"\n最优(val MAE): best_it={best['best_it']}  MAE={best['MAE']:.2f}  "
          f"Bias={best['Bias']:+.1f}  debiased={best['debiased']:.2f}  折CV={best['fcv']:.3f}")
    print(f"v6 基线:        MAE={V6_VAL_MAE:.2f}  Bias≈-17")
    print("-" * 70)

    # ---- 诊断结论 ----
    bias_trend = [(r["best_it"], r["Bias"]) for r in rows]
    bi_min = min(rows, key=lambda r: r["best_it"])
    bi_max = max(rows, key=lambda r: r["best_it"])
    print("\n诊断：")
    print(f"  bias 随 best_it: {bi_min['best_it']}->{bi_min['Bias']:+.1f}  ...  "
          f"{bi_max['best_it']}->{bi_max['Bias']:+.1f}")
    if abs(bi_min["Bias"]) < abs(bi_max["Bias"]) - 30:
        print("  -> bias 随 best_it 减小而改善：确认 best_iter 过大 -> 过拟合 2025 -> 偏置漂移。")
        print(f"  -> debiased 上限 {best['debiased']:.0f} MW（若 bias 完美校正）；")
        print(f"     最优 best_it={best['best_it']} 下 debiased={best['debiased']:.0f}，")
        if best["debiased"] < V6_VAL_MAE:
            print(f"     已低于 v6 {V6_VAL_MAE} -> 存在无泄露校正手段（recency-weighted hour_bias /")
            print("     近期折加权）可让 CatBoost 追平或略超 v6。建议 Phase 3-B：在最优 best_it 上做 bias 校正增强。")
        else:
            print("     仍高于 v6 -> CatBoost 纯波动误差也偏大，建议 Phase 3-B 超参搜索(depth/lr/l2)。")
    else:
        print("  -> bias 不随 best_it 改善：bias 非 best_it 所致，可能 CatBoost 算法本身偏置迁移差。")
        print("     建议：Phase 3-B 试 recency-weighted hour_bias；若仍无效 -> CatBoost 作异质集成成员(Phase 4)或放弃。")

    if best["fcv"] < 0.06 and best["MAE"] <= V6_VAL_MAE + 30:
        print(f"\n  最优 best_it={best['best_it']} 折间已稳({best['fcv']:.3f}<0.06) 且 MAE 接近 v6 -> 可进 Phase 3-B 超参搜索。")
    else:
        print(f"\n  最优 best_it={best['best_it']} 折CV={best['fcv']:.3f} 仍不稳 -> Phase 3-B 先稳折（降 depth / recency 加权）。")
    print("=" * 70)

    # ---- 写结果 ----
    try:
        with open("exp_catboost_best_it_result.txt", "w", encoding="utf-8") as f:
            f.write(f"v6={V6_VAL_MAE}\n")
            f.write("best_it\tMAE\tDelta_v6\tBias\tdebiased\tmidday\tfold_cv\tfold_maes\n")
            for r in rows:
                f.write(f"{r['best_it']}\t{r['MAE']:.4f}\t{r['MAE']-V6_VAL_MAE:+.4f}\t"
                        f"{r['Bias']:.4f}\t{r['debiased']:.4f}\t{r['midday']:.4f}\t"
                        f"{r['fcv']:.4f}\t{r['fmaes']}\n")
            f.write(f"best={best['best_it']}\n")
        print("(已写 exp_catboost_best_it_result.txt)")
    except Exception as e:
        print(f"(写结果失败: {e})")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
