# -*- coding: utf-8 -*-
"""exp80 - ⑥ 两级系统 MOS + 残差学习（用户最关注）。

Stage1 MOS：Ridge(actual ~ pred_load + 气象 + 日历)，target=actual（仅作目标，合规#1），
  inputs=pred_load+weather+calendar（均合规）。corrected_pred = MOS 预测（较 pred_load 更接近 actual）。
Stage2：LGB 残差成员预测 (actual - corrected_pred)，重建 corrected_pred + raw；直接成员不变。
用户提醒：MOS 用 actual 作"输入"才违规#1；作"目标"=监督学习，允许（与现有 direct/residual 同理）。

测试：
  (a) direct + residual-on-corrected_pred（当前结构换 MOS 锚）
  (b) residual-on-corrected_pred only（纯 ⑥）
对比 1459.06。corrected_pred 仅作残差锚；特征仍基于原始 pred_load（pl_weather_residual 等）。
合规：actual 仅作 MOS/LGB 目标，绝不作输入。仅诊断。
"""
from __future__ import annotations
import io, sys, copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from load_pred import config as C, data_loader as dl, features as F, train as T


MOS_COLS = ["pred_load", "irrad", "temp", "hdd", "cdd",
            "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos"]
MOS_COLS_ENRICH = MOS_COLS + ["precip", "wind", "solar_wind", "pl_weather_residual"]


def fit_mos(X, actual, usable, cols):
    M = X[cols].values.astype(float)
    col_mean = np.nanmean(M[usable], axis=0)
    Mf = M.copy(); nan = np.isnan(Mf)
    if nan.any():
        Mf[nan] = np.take(col_mean, np.where(nan)[1])
    rg = Ridge(alpha=1.0)
    rg.fit(Mf[usable], actual[usable].values)
    corrected = rg.predict(Mf)
    return np.clip(corrected, 0.0, None)


def run_variant(times, X, pred_load, actual, usable, mismatch_model, anchor, residual_modes):
    cfg = copy.deepcopy(C.TRAIN_CONFIG)
    cfg["residual_modes"] = residual_modes
    best_it = cfg["best_it_fixed"]
    # 用 anchor 作 pred_load 参数 -> y_res = actual - anchor，重建 anchor + raw
    model = T.train_ensemble(times, X, anchor, actual, usable, cfg, best_it)
    model.mismatch_model = mismatch_model
    model.hour_bias, model.drift_corr, model.threshold_corr = T.compute_hour_bias(
        times, X, anchor, actual, usable, cfg, best_it)
    vmask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    pred_v = model.predict_load(X[vmask], anchor[vmask])
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
    pl = pred_load.reindex(times).fillna(0.0)
    act = actual.reindex(times).fillna(0.0)
    print(f"feats={X.shape[1]} train={usable.sum()}", flush=True)

    # MOS corrected_pred (basic + enriched)
    corrected_b = pd.Series(fit_mos(X, act, usable, MOS_COLS), index=times)
    corrected_e = pd.Series(fit_mos(X, act, usable, MOS_COLS_ENRICH), index=times)
    m_pl = np.abs(pl[usable] - act[usable]).mean()
    m_mos_b = np.abs(corrected_b[usable] - act[usable]).mean()
    m_mos_e = np.abs(corrected_e[usable] - act[usable]).mean()
    print(f"MOS 锚质量(train MAE): pred_load={m_pl:.1f}  basic={m_mos_b:.1f}  enrich={m_mos_e:.1f}", flush=True)

    # 基线：direct+residual-on-pred_load（=1459 结构）
    mae_base = run_variant(times, X, pl, actual, usable, mismatch_model, pl, [False, True])
    print(f"(基线 direct+residual@pred_load)      val MAE={mae_base:.2f}", flush=True)
    # (a1) direct+residual@MOS_basic
    mae_a1 = run_variant(times, X, pl, actual, usable, mismatch_model, corrected_b, [False, True])
    print(f"(a1 direct+residual@MOS_basic)        val MAE={mae_a1:.2f}  Δ={mae_a1-mae_base:+.2f}", flush=True)
    # (a2) direct+residual@MOS_enrich
    mae_a2 = run_variant(times, X, pl, actual, usable, mismatch_model, corrected_e, [False, True])
    print(f"(a2 direct+residual@MOS_enrich)       val MAE={mae_a2:.2f}  Δ={mae_a2-mae_base:+.2f}", flush=True)
    print(f"\n基线应≈1459.06(v5)。⑥ 若 Δ<0 则 MOS 锚有益。", flush=True)


if __name__ == "__main__":
    main()
