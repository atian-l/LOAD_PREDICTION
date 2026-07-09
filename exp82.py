# -*- coding: utf-8 -*-
"""exp82 - ⑤ 联合样本权重（time × load × weather-extreme）快速测试。

当前 alpha_w=5.0 仅时间近因加权。MAE 为均匀加权评估，故对高负荷/极端天气样本加权
可能改善大误差点但伤及均值。在 ⑥ 基线上测：load-weighting(γ) + weather-extreme(δ)。
monkeypatch T._time_weights 注入联合权重，复用 ⑥ 全管线。baseline=1449.20。
合规：权重仅基于 pred_load/weather（非 actual）。仅诊断。
"""
from __future__ import annotations
import io, sys, copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from load_pred import config as C, data_loader as dl, features as F, train as T

# 全局（供 monkeypatch 闭包使用）
_STATE = {"pl": None, "X": None, "gamma": 0.0, "delta": 0.0, "alpha": 5.0}


def joint_tw(times, mask, alpha):
    w = T._time_weights_orig(times, mask, alpha)
    g = _STATE["gamma"]; d = _STATE["delta"]
    if g == 0.0 and d == 0.0:
        return w
    pl = _STATE["pl"].reindex(times)[mask].values.astype(float)
    pl_norm = pl / np.nanmean(pl)
    factor = np.ones(len(w))
    if g > 0:
        factor *= 1.0 + g * np.clip(pl_norm - 1.0, -0.5, 1.0)
    if d > 0:
        X = _STATE["X"]
        temp = X["temp"].reindex(times)[mask].values.astype(float)
        precip = X["precip"].reindex(times)[mask].values.astype(float)
        extreme = ((temp < 8.0) | (temp > 30.0) | (precip > 0.0)).astype(float)
        factor *= 1.0 + d * extreme
    factor = np.clip(factor, 0.05, None)  # 防止高 gamma 下负/零权重
    return w * factor


def run_setting(times, X, pred_load, actual, usable, mismatch_model, mos_model, gamma, delta):
    _STATE["gamma"] = gamma; _STATE["delta"] = delta
    cfg = copy.deepcopy(C.TRAIN_CONFIG)
    best_it = cfg["best_it_fixed"]
    model = T.train_ensemble(times, X, pred_load, actual, usable, cfg, best_it, mos_model=mos_model)
    model.mismatch_model = mismatch_model
    model.hour_bias, model.drift_corr, model.threshold_corr = T.compute_hour_bias(
        times, X, pred_load, actual, usable, cfg, best_it, mos_model=mos_model)
    vmask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    pred_v = model.predict_load(X[vmask], pred_load[vmask])
    return float(np.abs(pred_v - actual[vmask].values).mean())


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
    _STATE["pl"] = pred_load; _STATE["X"] = X
    # monkeypatch
    T._time_weights_orig = T._time_weights
    T._time_weights = joint_tw
    print(f"feats={X.shape[1]} train={usable.sum()}", flush=True)

    settings = [(1.0, 0.0), (1.5, 0.0), (2.0, 0.0), (2.5, 0.0)]
    base = 1449.20  # ⑥ 已知基线
    for g, d in settings:
        mae = run_setting(times, X, pred_load, actual, usable, mismatch_model, mos_model, g, d)
        if base is None:
            base = mae
        print(f"(gamma={g}, delta={d})  val MAE={mae:.2f}  Δ={mae-base:+.2f}", flush=True)
    print(f"\nbaseline (0,0)=1449.20(⑥)。⑤ 若 Δ<0 则联合权重有益。", flush=True)


if __name__ == "__main__":
    main()
