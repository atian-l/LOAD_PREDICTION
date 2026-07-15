# -*- coding: utf-8 -*-
"""v8 验证入口：对比 v7（根 model_bundle.pkl = v6 模型）与 v8，同一 val 窗口。

验证：① 全天 MAE  ② 午间(08-18) MAE  ③ 跨年泛化(trigger 命中率 OOF vs val)
      ④ 大偏差(>3000MW)点数  ⑤ 工程一致性(输出格式)。
报告写 v8/output/v8_evaluation.md。运行：python -m v8.evaluate
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

from load_pred import config as LC
from load_pred import train as T
from load_pred import features as F
from load_pred.model import EnsembleModel

from . import config as VC
from . import model as VM
from . import segments as SEG


def _mae(p, a):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(a))))


def _metrics(p, a, hours):
    err = p - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    mid = (hours >= 8) & (hours < 18)
    mae_mid = float(np.mean(np.abs(err[mid]))) if mid.any() else float("nan")
    night = hours < 8
    eve = hours >= 18
    big = np.abs(err) > 3000
    return {
        "MAE": mae, "RMSE": rmse, "Bias": bias,
        "MAE_night(00-08)": float(np.mean(np.abs(err[night]))) if night.any() else float("nan"),
        "MAE_day(08-18)": mae_mid,
        "MAE_evening(18-24)": float(np.mean(np.abs(err[eve]))) if eve.any() else float("nan"),
        "大偏差点数(>3000)": int(big.sum()),
        "大偏差均值": float(np.mean(np.abs(err[big]))) if big.any() else 0.0,
        "N": int(len(a)),
    }


def evaluate(verbose: bool = True) -> dict:
    times, X, pred_load, actual = T.build_dataset()
    val_start = pd.Timestamp(LC.VAL_START)
    val_end = pd.Timestamp(LC.VAL_END)
    buf_start = val_start - pd.Timedelta(days=7)
    em = (times >= buf_start) & (times <= val_end)
    X_eval = X[em]
    times_eval = pd.DatetimeIndex(times[em])
    actual_eval = actual.reindex(times_eval)
    vm = (times_eval >= val_start) & (times_eval <= val_end) & actual_eval.notna()
    hours_val = times_eval[vm].hour.values.astype(int)

    # ---- v7 基线（根 model_bundle.pkl）----
    if verbose:
        print("[1/3] v7 基线预测 ...")
    v7 = EnsembleModel.load(LC.MODEL_BUNDLE)
    X_v7 = v7.mismatch_model.transform(X_eval)
    v7_all = pd.Series(v7.predict_load(X_v7, pred_load), index=times_eval)
    v7_pred = v7_all[vm].values
    a = actual_eval[vm].values

    # ---- v8 ----
    if verbose:
        print("[2/3] v8 预测 ...")
    v8 = VM.V8Model.load(VC.V8_BUNDLE)
    v8_all = pd.Series(v8.predict(X_eval, pred_load, times_eval), index=times_eval)
    v8_pred = v8_all[vm].values

    m7 = _metrics(v7_pred, a, hours_val)
    m8 = _metrics(v8_pred, a, hours_val)

    # ---- 跨年泛化：trigger 命中率 OOF vs val ----
    if verbose:
        print("[3/3] 跨年泛化分析（trigger 命中率 OOF vs val）...")
    # OOF trigger 命中率（动态修正激活的点比例）
    oof_final = v8.dynamic._oof_final
    oof_base = v8.oof_pool["base_A_oof"]
    oof_trig_rate = float(np.mean(oof_final != oof_base))
    # val trigger 命中率
    val_dates = pd.DatetimeIndex(times_eval[vm]).normalize()
    val_segs = SEG.segment_array(hours_val)
    ds_cache = {}
    val_trig = np.zeros(len(val_dates), dtype=bool)
    for i, (d, s) in enumerate(zip(val_dates, val_segs)):
        key = (d, s)
        if key not in ds_cache:
            ds_cache[key] = v8.dynamic.params(d, s)
        val_trig[i] = ds_cache[key][2]
    val_trig_rate = float(np.mean(val_trig))

    delta = m8["MAE"] - m7["MAE"]
    result = {
        "v7_MAE": m7["MAE"], "v8_MAE": m8["MAE"], "ΔMAE": delta,
        "v7_午间MAE": m7["MAE_day(08-18)"], "v8_午间MAE": m8["MAE_day(08-18)"],
        "v7_大偏差点数": m7["大偏差点数(>3000)"], "v8_大偏差点数": m8["大偏差点数(>3000)"],
        "OOF_trigger命中率": oof_trig_rate, "val_trigger命中率": val_trig_rate,
        "v7_full": m7, "v8_full": m8,
    }

    lines = ["v8 vs v7 验证报告", "=" * 60]
    lines.append(f"val 窗口: {LC.VAL_START} ~ {LC.VAL_END}  (N={m7['N']})")
    lines.append("-" * 60)
    lines.append("① 全天 MAE:")
    lines.append(f"   v7 = {m7['MAE']:.2f} MW")
    lines.append(f"   v8 = {m8['MAE']:.2f} MW   (Δ={delta:+.2f} MW, {'改善' if delta<0 else '恶化/持平'})")
    lines.append("② 分段 MAE (v7 -> v8):")
    for seg, k in [("night(00-08)", "MAE_night(00-08)"), ("day(08-18)", "MAE_day(08-18)"), ("evening(18-24)", "MAE_evening(18-24)")]:
        lines.append(f"   {seg}: {m7[k]:.2f} -> {m8[k]:.2f}  (Δ={m8[k]-m7[k]:+.2f})")
    lines.append("③ 跨年泛化 (trigger 命中率):")
    lines.append(f"   OOF = {oof_trig_rate:.3f}   val = {val_trig_rate:.3f}   (一致性好则跨年可迁移)")
    lines.append("④ 大偏差 (>3000MW) 点数:")
    lines.append(f"   v7 = {m7['大偏差点数(>3000)']}   v8 = {m8['大偏差点数(>3000)']}   "
                 f"(均值 {m7['大偏差均值']:.0f} -> {m8['大偏差均值']:.0f})")
    lines.append("⑤ Bias: v7=%.1f  v8=%.1f" % (m7["Bias"], m8["Bias"]))
    lines.append("=" * 60)
    report = "\n".join(lines)
    with open(VC.V8_OUTPUT_DIR / "v8_evaluation.md", "w", encoding="utf-8") as f:
        f.write(report)
    if verbose:
        print("\n" + report)
    return result


def main():
    evaluate(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
