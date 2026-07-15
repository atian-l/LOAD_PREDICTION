# -*- coding: utf-8 -*-
"""TFT 数值稳定性诊断（exp_tft_diag.py）。

定位 phase0 raw=1868 + All-NaN 警告的根因：
  [1] 打印 feat_std 极端列（哪列尺度爆炸致 LSTM 溢出）
  [2] 配置 A: feat_clip=None（复现原 bug）+ verbose loss 轨迹 + NaN% + raw val
  [3] 配置 B: feat_clip=10.0（标准化空间 clip 修复）+ 同上
判定：
  - NaN 由极端 feat_std 放大致 LSTM 溢出 -> clip 修复有效（根因确认）
  - B raw < v6 raw 1512 -> TFT 有潜力，进 Phase 1（生产 train.py 已默认 feat_clip=10）
  - B raw >= v6 raw 1512 -> clip 修了 NaN 但 raw 仍不达标 -> TFT NO-GO 回 v6

合规: 不修改生产脚本; 复用 build_dataset/train_ensemble（不变量#5）; throwaway 纯 stdout。
"""
from __future__ import annotations
import copy
import time
import numpy as np
import pandas as pd

from . import config as C
from .train import build_dataset, usable_mask, train_ensemble
from .features import MismatchModel, MosModel
from .tft import _standardize_fit

V6_RAW_MAE = 1512.0       # v6 raw val（无 OOF 校正，fit_diag 估计）


def _val_metrics(times, pred, actual, usable):
    """val 期（times > TRAIN_END）MAE + NaN 比例。
    pred 前段 NaN（无 encoder 历史，在训练期内）不计入 val。"""
    av = actual.reindex(times).values
    val = np.asarray((times > pd.Timestamp(C.TRAIN_END)) & np.isfinite(av))
    nan_n = int(np.sum(val & ~np.isfinite(pred)))
    tot = int(np.sum(val))
    m = val & np.isfinite(pred)
    mae = float(np.mean(np.abs(pred[m] - av[m]))) if m.sum() > 0 else float("inf")
    return mae, nan_n, tot


def run_config(label, feat_clip, times, X, pred_load, actual, usable,
               mismatch_model, mos_model):
    """训练 2 成员（regression × direct+residual × seed42）并报 raw val + NaN%。"""
    cfg = copy.deepcopy(C.TRAIN_CONFIG)
    cfg["objectives"] = ["regression"]
    cfg["residual_modes"] = [False, True]
    cfg["seeds"] = [42]
    cfg["feat_clip"] = feat_clip        # None=复现bug / 10.0=修复
    epochs = int(cfg["best_it_fixed"])
    print(f"\n=== 配置 {label}: feat_clip={feat_clip}  ({epochs} ep, 2 成员, verbose loss) ===")
    ts = time.perf_counter()
    # 不吞 stdout：让 train_tft verbose 每 epoch 打印 loss（定位发散点）
    model = train_ensemble(times, X, pred_load, actual, usable, cfg, epochs,
                           mos_model=mos_model)
    print(f"  训练完成 ({time.perf_counter()-ts:.0f}s)")
    model.mismatch_model = mismatch_model
    model.hour_bias = None
    model.drift_corr = []
    model.threshold_corr = []
    pred = model.predict_load(X, pred_load)
    mae, nan_n, tot = _val_metrics(times, pred, actual, usable)
    print(f"  >> raw val MAE={mae:.2f}  NaN={nan_n}/{tot} ({100*nan_n/max(tot,1):.1f}%)")
    return mae, nan_n, tot


def main() -> int:
    print("=" * 72)
    print("TFT 数值稳定性诊断：定位 phase0 raw=1868 + All-NaN 根因")
    print(f"  v6 raw≈{V6_RAW_MAE}  (gate<1515)")
    print("=" * 72)

    print("\n[1] 构建数据集 ...")
    times, X, pred_load, actual = build_dataset()
    usable = usable_mask(times, pred_load, actual)
    mismatch_model = MismatchModel().fit(X, usable)
    X = mismatch_model.transform(X)
    mc = C.TRAIN_CONFIG["mos"]
    mos_model = MosModel(cols=mc["cols"], alpha=mc["alpha"]).fit(X, actual, usable)
    print(f"    特征数={X.shape[1]}  训练点={int(usable.sum())}")

    print("\n[2] feat_std 极端列定位（usable 行估计）...")
    feat_cols = list(X.columns)
    X_arr = X[feat_cols].to_numpy(dtype=np.float64)
    usable_rows = np.asarray(usable, dtype=bool)
    _, feat_std = _standardize_fit(X_arr, usable_rows)
    order = np.argsort(feat_std)[::-1]
    span = feat_std.max() / max(feat_std.min(), 1e-12)
    print(f"    feat_std 范围: [{feat_std.min():.4e}, {feat_std.max():.4e}]  跨度={span:.2e}x")
    print("    top-5 极端 std 列（疑似溢出源）:")
    for i in order[:5]:
        print(f"      {feat_cols[i]:32s} std={feat_std[i]:.4e}")
    print(f"    最小 std 列: {feat_cols[order[-1]]}  std={feat_std[order[-1]]:.4e}")

    print("\n[3] 配置 A：feat_clip=None（复现 phase0 bug，看 loss 是否发散）...")
    mae_A, nan_A, tot = run_config("A(复现bug)", None,
                                   times, X, pred_load, actual, usable,
                                   mismatch_model, mos_model)

    print("\n[4] 配置 B：feat_clip=10.0（标准化空间 clip 修复）...")
    mae_B, nan_B, _ = run_config("B(+clip=10修复)", 10.0,
                                 times, X, pred_load, actual, usable,
                                 mismatch_model, mos_model)

    print("\n" + "=" * 72)
    print("诊断结论")
    print("=" * 72)
    print(f"  A feat_clip=None : raw val={mae_A:.2f}  NaN={nan_A}/{tot} ({100*nan_A/max(tot,1):.1f}%)")
    print(f"  B feat_clip=10   : raw val={mae_B:.2f}  NaN={nan_B}/{tot} ({100*nan_B/max(tot,1):.1f}%)")
    print(f"  v6 raw≈{V6_RAW_MAE}")
    print("-" * 72)
    if nan_A > 0 and nan_B == 0:
        print("  根因确认: NaN 由极端 feat_std 放大致 LSTM 溢出，clip=10 修复有效。")
    elif nan_A > 0 and nan_B > 0:
        print(f"  clip 未完全消 NaN（A={nan_A} -> B={nan_B}），仍有其他溢出源，需进一步排查。")
    else:
        print("  A 无 NaN：phase0 的 All-NaN 非训练发散，可能是 OOF 折特定问题（见 loss 轨迹）。")
    print("-" * 72)
    if np.isfinite(mae_B) and mae_B < V6_RAW_MAE:
        print(f"  判定: B raw {mae_B:.2f} < v6 raw {V6_RAW_MAE} -> TFT 有潜力，进 Phase 1")
        print("        (生产 train.py 已默认 feat_clip=10；可直接 python -m load_pred_tft.train)")
        return 0
    else:
        b_str = f"{mae_B:.2f}" if np.isfinite(mae_B) else "inf"
        print(f"  判定: B raw {b_str} >= v6 raw {V6_RAW_MAE} -> clip 修了 NaN 但 raw 仍不达标")
        print("        TFT NO-GO，回 v6（深度学习第 3 次 NO-GO，转新数据方向）")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
