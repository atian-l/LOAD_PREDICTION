# -*- coding: utf-8 -*-
"""exp81 - ④ per-model best_iter（direct vs residual 不同迭代数）测试。

在 ⑥ MOS 基线上，扫描 (direct_it, residual_it) 组合。direct 预测 actual(绝对水平~8万)，
residual 预测 actual-corrected_pred(小残差)，二者最优迭代数可能不同。best_it_fixed=80 是
全集成 U 型最优（exp44）；本实验测 per-mode 是否更优。

baseline {80,80}=1449.20（⑥）。合规：仅改迭代数，无 actual 入特征。仅诊断。
"""
from __future__ import annotations
import io, sys, copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from load_pred import config as C, data_loader as dl, features as F, train as T


def run_combo(times, X, pred_load, actual, usable, mismatch_model, mos_model, d_it, r_it):
    cfg = copy.deepcopy(C.TRAIN_CONFIG)
    best_it = {False: d_it, True: r_it}
    model = T.train_ensemble(times, X, pred_load, actual, usable, cfg, best_it, mos_model=mos_model)
    model.mismatch_model = mismatch_model
    model.hour_bias, model.drift_corr, model.threshold_corr = T.compute_hour_bias(
        times, X, pred_load, actual, usable, cfg, best_it, mos_model=mos_model)
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
    mos_model = F.MosModel().fit(X, actual, usable)
    print(f"feats={X.shape[1]} train={usable.sum()}", flush=True)

    combos = [(80, 80), (120, 40), (40, 120), (150, 60), (60, 150), (100, 100), (50, 50)]
    base = None
    for d_it, r_it in combos:
        mae = run_combo(times, X, pred_load, actual, usable, mismatch_model, mos_model, d_it, r_it)
        if base is None:
            base = mae
        print(f"(direct={d_it:3d}, residual={r_it:3d})  val MAE={mae:.2f}  Δ={mae-base:+.2f}", flush=True)
    print(f"\nbaseline (80,80)=1449.20(⑥)。per-mode 若 Δ<0 则有益。", flush=True)


if __name__ == "__main__":
    main()
