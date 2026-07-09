# -*- coding: utf-8 -*-
"""exp79 — ③ 聚合方式全管线测试（含校正重估）。

exp78 raw 显示 pure-LGB trimmed-mean(20%) 较 median raw -11.56 MW。本实验在全管线
（train_ensemble + compute_hour_bias 3 折 OOF 重估 hour_bias/drift/threshold + val 评估）
下验证各聚合方式，确认增益是否在加校正后保留。不保存模型。

聚合：median(基线=1459.06) / mean / trimmed(trim_frac=0.1/0.2/0.3)
合规：仅改聚合方式，无 actual 入特征。仅诊断。
"""
from __future__ import annotations
import io, sys, copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from load_pred import config as C, data_loader as dl, features as F, train as T


def run_variant(times, X, pred_load, actual, usable, mismatch_model, agg, trim=0.2):
    cfg = copy.deepcopy(C.TRAIN_CONFIG)
    cfg["aggregation"] = agg
    cfg["trim_frac"] = trim
    best_it = cfg["best_it_fixed"]
    model = T.train_ensemble(times, X, pred_load, actual, usable, cfg, best_it)
    model.mismatch_model = mismatch_model
    model.hour_bias, model.drift_corr, model.threshold_corr = T.compute_hour_bias(
        times, X, pred_load, actual, usable, cfg, best_it)
    vmask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    pred_v = model.predict_load(X[vmask], pred_load[vmask])
    a_v = actual[vmask].values
    return float(np.abs(pred_v - a_v).mean())


def main():
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)
    X = F.build_features(times, pred_load, weather)
    usable = T.usable_mask(times, pred_load, actual)
    mismatch_model = F.MismatchModel().fit(X, usable)
    X = mismatch_model.transform(X)
    print(f"feats={X.shape[1]} train={usable.sum()}", flush=True)

    variants = [
        ("median (基线)", "median", 0.2),
        ("mean", "mean", 0.2),
        ("trimmed 0.1", "trimmed", 0.1),
        ("trimmed 0.2", "trimmed", 0.2),
        ("trimmed 0.3", "trimmed", 0.3),
    ]
    results = {}
    for name, agg, trim in variants:
        mae = run_variant(times, X, pred_load, actual, usable, mismatch_model, agg, trim)
        results[name] = mae
        print(f"{name:18s} val MAE={mae:.2f}  Δ={mae-results[variants[0][0]]:+.2f}", flush=True)

    print(f"\n基线(median)=1459.06 (v5)。最优 trimmed 增益见上。", flush=True)


if __name__ == "__main__":
    main()
