# -*- coding: utf-8 -*-
"""exp_drift_sign.py - 只读诊断：检验 drift_corr 应用符号（+ vs -）。

背景（bug 报告 #2）：
  train.py:221 估计 beta = <feat, (oof_pred - actual)> / <feat²>  （OOF 误差对 feat 的 OLS）
  model.py:125 应用 pred = pred + beta[hour] * feat                （"叠加"）
  数学上：误差 = pred - actual；要削减误差应 pred -= OLS(误差 ~ feat)，即 pred - beta·feat。
  故 model.py:125 的 "+" 疑似符号错误，应为 "-"。但 config 注释称 exp49 实测 -13 MW（+），
  存在矛盾。本脚本用已保存 v6 模型的 beta，在验证集上直接比较 "+" 与 "-" 的 MAE，一锤定音。

方法（只读，不修改任何生产代码/模型）：
  diag_val.csv 的 pred 已含 drift_corr(+) 应用。翻转符号等价于 pred - 2·drift_contrib：
    drift_contrib = beta[hour] * feat  （仅 hours 11-14 非零，余为 0）
    pred_minus = clip(pred - 2·drift_contrib, 0)   等价于应用 "-" 而非 "+"
  比较 MAE(pred_plus=当前) vs MAE(pred_minus=翻转)。若 minus 更低 -> 符号 bug 确认。

合规：仅读取 diag_val.csv 与 model bundle；不训练、不保存模型、不修改生产代码。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd

from load_pred import config as C
from load_pred.model import EnsembleModel

DIAG = Path(__file__).parent / "FDS" / "output" / "diag_val.csv"


def main():
    d = pd.read_csv(DIAG, encoding="utf-8-sig")
    for col in ("pred", "actual", "pl_weather_residual", "hour"):
        assert col in d.columns, f"缺少列 {col}（diag_val.csv 列: {list(d.columns)})"

    model = EnsembleModel.load(C.MODEL_BUNDLE)
    feat_name, beta = model.drift_corr[0]
    beta = np.asarray(beta, dtype=float)
    hours = pd.DatetimeIndex(d["hour"]).values if False else d["hour"].values.astype(int)
    feat = d["pl_weather_residual"].values.astype(float)

    # 当前（生产 "+"）：diag_val.pred 已含 +drift_contrib
    pred_plus = d["pred"].values.astype(float)
    actual = d["actual"].values.astype(float)

    # drift_contrib = beta[hour]*feat（生产当前已 + 之）
    contrib = beta[hours] * feat
    # 翻转符号：+contrib -> -contrib，差值 -2*contrib
    pred_minus = np.clip(pred_plus - 2.0 * contrib, 0.0, None)

    mid = (hours >= 11) & (hours <= 14)

    def mae(p, mask=None):
        m = mask if mask is not None else np.ones(len(p), dtype=bool)
        return float(np.mean(np.abs(p[m] - actual[m])))

    print("=" * 64)
    print("drift_corr 符号诊断（feat=pl_weather_residual, hours 11-14）")
    print("=" * 64)
    print(f"beta[11..14] = {[round(beta[h], 4) for h in (11, 12, 13, 14)]}")
    print(f"beta 其余小时 = 0")
    # 午间 feat 与（当前 pred 误差）的相关：正=feat 高对应高估
    err = pred_plus - actual
    mm = mid & np.isfinite(feat) & np.isfinite(err)
    if mm.sum() > 1 and np.std(feat[mm]) > 0:
        corr = float(np.corrcoef(feat[mm], err[mm])[0, 1])
    else:
        corr = float("nan")
    print(f"午间 corr(feat, error=pred-actual) = {corr:+.4f}   (正=feat高->高估)")
    print(f"午间 mean(feat) = {np.nanmean(feat[mm]):+.2f}  mean(contrib) = {np.nanmean(contrib[mm]):+.2f}")
    print("-" * 64)
    print(f"全员 MAE  当前(+) = {mae(pred_plus):.4f}")
    print(f"全员 MAE  翻转(-) = {mae(pred_minus):.4f}")
    print(f"全员 Δ(翻转-当前) = {mae(pred_minus) - mae(pred_plus):+.4f}   (负=翻转更优=符号bug确认)")
    print("-" * 64)
    print(f"午间 MAE  当前(+) = {mae(pred_plus, mid):.4f}")
    print(f"午间 MAE  翻转(-) = {mae(pred_minus, mid):.4f}")
    print(f"午间 Δ(翻转-当前) = {mae(pred_minus, mid) - mae(pred_plus, mid):+.4f}")
    n_clip = int((pred_plus <= 0).sum())
    print(f"(注: pred<=0 的点数 = {n_clip}（裁剪近似影响可忽略）)")
    print("=" * 64)
    if mae(pred_minus) < mae(pred_plus):
        print("结论: 翻转(-) MAE 更低 -> drift_corr 应用符号确为 bug（应为 pred - beta·feat）。")
    else:
        print("结论: 当前(+) MAE 更低 -> 符号非 bug（与 exp49 -13MW 一致），勿改。")


if __name__ == "__main__":
    main()
