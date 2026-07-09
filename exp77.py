# -*- coding: utf-8 -*-
"""exp77 — 量化 9AM 部署对 val MAE 的影响（诊断，不写产物）。

事实(exp76)：所有气象 20:00 起报，D 日 20:00 issue 覆盖 D+1（lead 4-27.8h）。
9AM on D 预测 D+1 时，覆盖 D+1 的 issue(20:00 D) 尚未起报 → D+1 气象 0% 可得；
仅 20:00 D-1 issue（覆盖 D）可得。当前 val(run_time=None) 用 D+1 气象 = 21:00 部署条件。

部署语义：9AM on day D 预测 day D+1。此时可得气象 = 20:00 D-1 issue（覆盖 D）。
对 val 时刻 T（在 day X 上），等价部署运行 = 9AM on day X-1 预测 day X；可得气象
覆盖 day X-1（即 T-96 同时刻）。故：
  (a) 实际 D+1 气象(run_time=None) = 基线（21:00 部署，D+1 气象可得）
  (b) D+1 气象 = NaN（9AM 严格：D+1 气象全不可得）
  (c) D+1 气象 = D 日气象(平移 -96，9AM 持续性代理：用可得的前一日预报作 D+1 代理）

用已存 v5 模型（mismatch_model/hour_bias/drift/threshold 均在训练期拟合），
仅替换 val 气象条件评估，隔离"气象可得性"对 MAE 的影响。训练用全量历史起报
（每个历史 T 的气象 = T 前一日 20:00 起报、T 时刻前可得，无泄露）；9AM 部署
缺 D+1 气象是部署时约束，非训练泄露 —— 此实验度量该 train/serve skew 的影响。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from load_pred import config as C, data_loader as dl, features as F
from load_pred.model import EnsembleModel


def build_X(weather_dedup, model):
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    X = F.build_features(times, pred_load, weather_dedup)
    X = model.mismatch_model.transform(X)  # 用已存模型在训练期拟合的系数
    return times, X, pred_load, actual


def eval_val(model, times, X, pred_load, actual):
    vmask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    pred_v = model.predict_load(X[vmask], pred_load[vmask])
    a = actual[vmask].values
    mae = float(np.abs(pred_v - a).mean())
    return mae, int(vmask.sum())


def main():
    model = EnsembleModel.load(C.MODEL_BUNDLE)
    times0 = dl.full_time_index()
    # reindex 到完整 15min 网格（缺失预报时刻 -> NaN，weather_features 内部本就如此处理）
    w_full = dl.load_weather_dedup(run_time=None).reindex(times0)

    # 确定 val 时刻集合（用于 b/c 仅替换 val 段气象；训练段保持原样）
    val_times = times0[(times0 >= pd.Timestamp(C.VAL_START)) & (times0 <= pd.Timestamp(C.VAL_END))]

    # (a) 基线：实际 D+1 气象
    times, X, pred_load, actual = build_X(w_full, model)
    mae_a, n = eval_val(model, times, X, pred_load, actual)
    print(f"(a) 实际 D+1 气象 (21:00 部署, run_time=None): MAE={mae_a:.2f}  N={n}", flush=True)

    # (b) D+1 气象 = NaN（9AM 严格）：对 val 段气象置 NaN
    w_nan = w_full.copy()
    w_nan.loc[val_times] = np.nan
    times, X, pred_load, actual = build_X(w_nan, model)
    mae_b, n = eval_val(model, times, X, pred_load, actual)
    print(f"(b) D+1 气象=NaN (9AM 严格, D+1 气象不可得): MAE={mae_b:.2f}  Δ={mae_b-mae_a:+.2f}  N={n}", flush=True)

    # (c) D+1 气象 = D 日气象（9AM 持续性代理：w[T]=w[T-96]，前一日同时刻可得预报）
    w_pers = w_full.copy()
    for col in w_pers.columns:
        shifted = w_full[col].shift(96)  # T-96 = 前一日同时刻（20:00 D-1 issue 覆盖 D，可得）
        w_pers.loc[val_times, col] = shifted.loc[val_times].values
    times, X, pred_load, actual = build_X(w_pers, model)
    mae_c, n = eval_val(model, times, X, pred_load, actual)
    print(f"(c) D+1 气象=D日气象 (9AM 持续性代理, 平移-96): MAE={mae_c:.2f}  Δ={mae_c-mae_a:+.2f}  N={n}", flush=True)

    print(f"\n结论: 9AM 部署 D+1 气象不可得。当前 val {mae_a:.2f} 假设 21:00 条件（D+1 气象可得）。", flush=True)
    print(f"      9AM 严格(NaN)={mae_b:.2f}；9AM 持续性代理={mae_c:.2f}。", flush=True)


if __name__ == "__main__":
    main()
