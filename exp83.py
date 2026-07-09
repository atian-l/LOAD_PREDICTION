# -*- coding: utf-8 -*-
"""exp83 - ② 残差 GBM 替代规则校正（hour_bias/drift/threshold）测试。

当前生产用 3 类 OOF 规则校正（hour_bias 96维 + drift β@午间 + threshold 场景位移）。
② 提议：用单个 LightGBM 学习 OOF 残差(oof_pred − actual) ~ 特征，predict 时加回。
合规：GBM 仅用训练期 OOF 残差训练（actual 仅作目标），apply 到 val；无泄露。

对比（⑥+⑤ 基线 = 1445.62）：
  A) raw + 规则校正              （= 生产 1445.62）
  B) raw + GBM 校正（替代规则）  （② 核心）
  C) raw + 规则校正 + GBM        （② 叠加，GBM 在规则校正后残差上训练）
仅诊断，不写产物。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import lightgbm as lgb
from load_pred import config as C, data_loader as dl, features as F, train as T

# GBM 校正特征（聚焦，防过拟合）：日历 + pred_load + 气象 + pl_weather_residual
CORR_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
             "pred_load", "temp", "irrad", "precip", "wind", "clearness",
             "pl_weather_residual", "solar_mismatch", "hdd", "cdd"]


def get_oof_pred(times, X, pred_load, actual, usable, cfg, best_it, mos_model):
    """3 折 walk-forward OOF 原始集成预测（复用 compute_hour_bias 的折划分；无校正）。"""
    oof = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        fm = T.train_ensemble(times, X, pred_load, actual, ftr, cfg, best_it, mos_model=mos_model)
        oof[fva] = fm.predict_load(X[fva], pred_load[fva])
    return oof


def train_corr_gbm(X, resid_target, mask, cols, n_iter=120):
    """在 OOF 残差上训练保守 LightGBM 校正器。resid_target = oof_pred − actual（待校正量）。"""
    dtr = lgb.Dataset(X[mask][cols], label=resid_target[mask].values,
                      weight=T._time_weights(times_state, mask, C.TRAIN_CONFIG["alpha_w"],
                                             pred_load=X["pred_load"],
                                             load_gamma=C.TRAIN_CONFIG.get("weight_load_gamma", 0.0)))
    params = dict(objective="regression", metric="mae", learning_rate=0.05,
                  num_leaves=31, min_data_in_leaf=200, lambda_l2=8.0,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                  verbose=-1, force_col_wise=True, seed=42)
    return lgb.train(params, dtr, num_boost_round=n_iter)


times_state = None  # 供 train_corr_gbm 闭包（权重需要 times）


def main():
    global times_state
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index(); times_state = times
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)
    X = F.build_features(times, pred_load, weather)
    usable = T.usable_mask(times, pred_load, actual)
    mm = F.MismatchModel().fit(X, usable); X = mm.transform(X)
    mos = F.MosModel().fit(X, actual, usable)
    cfg = C.TRAIN_CONFIG; best_it = cfg["best_it_fixed"]
    vmask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values

    # 生产模型（含规则校正）-> A
    model = T.train_ensemble(times, X, pred_load, actual, usable, cfg, best_it, mos_model=mos)
    model.mismatch_model = mm
    model.hour_bias, model.drift_corr, model.threshold_corr = T.compute_hour_bias(
        times, X, pred_load, actual, usable, cfg, best_it, mos_model=mos)
    pred_A = model.predict_load(X[vmask], pred_load[vmask])
    mae_A = float(np.abs(pred_A - actual[vmask].values).mean())

    # OOF 原始预测（无校正）+ 规则校正后的 OOF 残差
    oof_raw = get_oof_pred(times, X, pred_load, actual, usable, cfg, best_it, mos)
    oof_mask = usable & oof_raw.notna().values
    resid_raw = (oof_raw - actual)                                   # GBM-B 目标
    # 规则校正后的残差（用于 GBM-C）：对 OOF 点应用 hour_bias/drift/threshold
    resid_after_rules = pd.Series(resid_raw.values.copy(), index=times)
    hb = model.hour_bias
    dt = pd.DatetimeIndex(times)
    if len(hb) == 24:
        sidx = dt.hour.values
    else:
        sidx = ((dt.hour.values * 60 + dt.minute.values) // (1440 // len(hb))).astype(int)
    resid_after_rules = resid_after_rules - hb[sidx]
    hrs = dt.hour.values
    for fn, beta in model.drift_corr:
        beta = np.asarray(beta, dtype=float)
        resid_after_rules = resid_after_rules - beta[hrs] * X[fn].values.astype(float)
    for tc in model.threshold_corr:
        fv = X[tc["feature"]].values.astype(float); op = tc.get("op", ">"); thr = tc["thr"]
        if op == "range":
            sel = (fv >= thr[0]) & (fv < thr[1])
        elif op == ">=": sel = fv >= thr
        elif op == "<": sel = fv < thr
        elif op == "<=": sel = fv <= thr
        else: sel = fv > thr
        if tc.get("hours") is not None:
            sel = sel & np.isin(hrs, list(tc["hours"]))
        resid_after_rules = resid_after_rules - tc.get("shift", 0.0) * sel

    cols = [c for c in CORR_COLS if c in X.columns]
    print(f"feats={X.shape[1]} cols_corr={len(cols)} oof={oof_mask.sum()} A_MAE={mae_A:.2f}", flush=True)

    # B) GBM 替代规则：在 resid_raw 上训练，val 用 raw_pred + gbm
    gbmB = train_corr_gbm(X, resid_raw, oof_mask, cols)
    raw_val = model.predict_load(X[vmask], pred_load[vmask])  # 含规则校正！需无校正版
    # 重新算无校正 raw val（临时清零校正）
    saved = (model.hour_bias, model.drift_corr, model.threshold_corr)
    model.hour_bias = None; model.drift_corr = []; model.threshold_corr = []
    raw_val_nocorr = model.predict_load(X[vmask], pred_load[vmask])
    model.hour_bias, model.drift_corr, model.threshold_corr = saved
    corrB_val = gbmB.predict(X[vmask][cols])
    pred_B = raw_val_nocorr + corrB_val
    mae_B = float(np.abs(pred_B - actual[vmask].values).mean())

    # C) GBM 叠加：在 resid_after_rules 上训练，val 用 (raw+规则) + gbm
    gbmC = train_corr_gbm(X, resid_after_rules, oof_mask, cols)
    corrC_val = gbmC.predict(X[vmask][cols])
    pred_C = pred_A + corrC_val
    mae_C = float(np.abs(pred_C - actual[vmask].values).mean())

    print(f"A) raw + 规则校正              val MAE={mae_A:.2f}  (基线)", flush=True)
    print(f"B) raw + GBM 校正(替代规则)    val MAE={mae_B:.2f}  Δ={mae_B-mae_A:+.2f}", flush=True)
    print(f"C) raw + 规则 + GBM(叠加)      val MAE={mae_C:.2f}  Δ={mae_C-mae_A:+.2f}", flush=True)


if __name__ == "__main__":
    main()
