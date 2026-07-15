# -*- coding: utf-8 -*-
"""OOF 引擎：3 折 walk-forward 产生校正后 OOF 预测池。

复用 load_pred.train.train_ensemble + 复刻 train.compute_hour_bias 的估计逻辑 +
复刻 model.predict_load 的校正应用，使 OOF base A = 生产 v6 在 fva 的无泄露模拟。

合规：3 折全在训练期（best_it_folds），fva 不重叠，fold 模型未见过 fva；val 零参与。
mos_model 全量 fit（与生产 v6 一致，继承 v6 既有 OOF 设计；actual 仅作 MOS 目标）。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from load_pred import train as T
from . import config as VC
from . import segments as SEG


# --------------------------------------------------------------------------- #
# 校正估计（复刻 train.compute_hour_bias 的估计段，用 OOF 残差；无泄露）
# --------------------------------------------------------------------------- #
def _estimate_corrections(times, X, actual, oof_raw, oof_mask, cfg):
    """用 OOF 残差 (oof_raw - actual) 估计 hour_bias / drift_corr / threshold_corr。
    逻辑与 train.compute_hour_bias 完全一致（确保 OOF base A = 生产 v6 校正配置）。"""
    resid = (oof_raw - actual).values
    dt_all = pd.DatetimeIndex(times)
    h_all = dt_all.hour.values
    n_slots = int(cfg.get("hour_bias_slots", 24))
    step = 1440 // n_slots
    mod_all = h_all * 60 + dt_all.minute.values
    slot_all = (mod_all // step).astype(int)
    hour_bias = np.zeros(n_slots, dtype=float)
    for q in range(n_slots):
        m = oof_mask & (slot_all == q)
        if m.sum():
            hour_bias[q] = float(np.average(resid[m]))

    drift_corr = []
    dc_cfg = cfg.get("drift_corr")
    if dc_cfg:
        feat_name = dc_cfg["feature"]
        hours_set = set(dc_cfg["hours"])
        feat = X[feat_name].values.astype(float)
        beta = np.zeros(24, dtype=float)
        for h in range(24):
            if h not in hours_set:
                continue
            m = oof_mask & (h_all == h)
            f = feat[m]; e = resid[m]
            good = np.isfinite(f) & np.isfinite(e)
            d = float(np.dot(f[good], f[good]))
            if d > 0:
                beta[h] = float(np.dot(f[good], e[good]) / d)
        drift_corr.append((feat_name, beta))

    threshold_corr = []
    for tc in cfg.get("threshold_corr", []):
        feat_name = tc["feature"]; op = tc.get("op", ">"); thr = tc["thr"]
        hours_list = tc["hours"]; shrink = float(tc["shrinkage"])
        feat = X[feat_name].values.astype(float)
        if op == "range":
            lo, hi = thr; m = oof_mask & (feat >= lo) & (feat < hi)
        elif op == ">=":
            m = oof_mask & (feat >= thr)
        elif op == "<":
            m = oof_mask & (feat < thr)
        elif op == "<=":
            m = oof_mask & (feat <= thr)
        else:
            m = oof_mask & (feat > thr)
        if hours_list is not None:
            m = m & np.isin(h_all, list(hours_list))
        shift = float(np.average(resid[m])) * shrink if m.sum() else 0.0
        threshold_corr.append({"feature": feat_name, "op": op, "thr": thr,
                               "hours": hours_list, "shift": shift})
    return hour_bias, drift_corr, threshold_corr


# --------------------------------------------------------------------------- #
# 校正应用（复刻 model.predict_load 的校正段）
# --------------------------------------------------------------------------- #
def apply_corrections(times, X, pred, hour_bias, drift_corr, threshold_corr) -> np.ndarray:
    """对 raw 预测应用 hour_bias/drift_corr/threshold_corr（与 EnsembleModel.predict_load 一致）。"""
    pred = np.asarray(pred, dtype=float).copy()
    dt = pd.DatetimeIndex(times)
    hours = dt.hour.values.astype(int)
    if hour_bias is not None:
        n = len(hour_bias)
        mod = hours * 60 + dt.minute.values
        idx = ((mod * n) // 1440).astype(int)
        pred = pred - hour_bias[idx]
    for feat_name, beta in drift_corr:
        beta = np.asarray(beta, dtype=float)
        pred = pred + beta[hours] * X[feat_name].values.astype(float)
    for tc in threshold_corr:
        feat_name = tc["feature"]
        fv = X[feat_name].values.astype(float)
        op = tc.get("op", ">"); thr = tc["thr"]
        if op == "range":
            lo, hi = thr; sel = (fv >= lo) & (fv < hi)
        elif op == ">=":
            sel = fv >= thr
        elif op == "<":
            sel = fv < thr
        elif op == "<=":
            sel = fv <= thr
        else:
            sel = fv > thr
        hours_list = tc.get("hours")
        if hours_list is not None:
            sel = sel & np.isin(hours, list(hours_list))
        shift = tc.get("shift", 0.0)
        if shift != 0.0:
            pred[sel] = pred[sel] - shift
    return np.clip(pred, 0.0, None)


# --------------------------------------------------------------------------- #
# 校正后 OOF（3 折 walk-forward）
# --------------------------------------------------------------------------- #
def compute_oof_corrected(times, X, pred_load, actual, usable, cfg, best_it, mos_model):
    """3 折 walk-forward OOF（复用 cfg['best_it_folds']），返回校正后 OOF 预测 + 校正参数 + oof_mask。

    每折：fold train_ensemble(ftr) -> predict fva (raw，fold_model 无校正) -> 入 OOF。
    再用 OOF 残差估 hour_bias/drift/threshold，应用得校正后 OOF。无泄露（fold 未见过 fva）。
    """
    oof_raw = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        fold_model = T.train_ensemble(times, X, pred_load, actual, ftr, cfg, best_it, mos_model=mos_model)
        oof_raw[fva] = fold_model.predict_load(X[fva], pred_load[fva])
    oof_mask = usable & oof_raw.notna().values
    hour_bias, drift_corr, threshold_corr = _estimate_corrections(
        times, X, actual, oof_raw, oof_mask, cfg)
    oof_corr = apply_corrections(times, X, oof_raw.values, hour_bias, drift_corr, threshold_corr)
    return oof_corr, hour_bias, drift_corr, threshold_corr, oof_mask


# --------------------------------------------------------------------------- #
# OOF 池（base A + base B）
# --------------------------------------------------------------------------- #
def run_oof(times, X_full, pred_load, actual, usable, cfg_A, cfg_B, best_it, full_mos, verbose=True):
    """产生 OOF 池：base A (v6 40 成员) + base B (reg_only 10 成员) 校正后 OOF。

    返回 oof_pool dict + base B 校正参数（生产 base B 用）+ base A 校正参数（参考）。
    生产 base A 加载根 model_bundle.pkl（=v6），其校正参数已在 bundle 内。
    """
    if verbose:
        print("  [OOF] base A (v6 40 成员) 3 折 walk-forward ...")
    base_A_oof, hb_A, dc_A, tc_A, mask_A = compute_oof_corrected(
        times, X_full, pred_load, actual, usable, cfg_A, best_it, full_mos)
    mae_A = float(np.abs(base_A_oof[mask_A] - actual.values[mask_A]).mean())
    if verbose:
        print(f"        base A OOF 点={int(mask_A.sum())}  OOF MAE={mae_A:.2f}")

    if verbose:
        print("  [OOF] base B (reg_only 10 成员) 3 折 walk-forward ...")
    base_B_oof, hb_B, dc_B, tc_B, mask_B = compute_oof_corrected(
        times, X_full, pred_load, actual, usable, cfg_B, best_it, full_mos)
    mae_B = float(np.abs(base_B_oof[mask_B] - actual.values[mask_B]).mean())
    if verbose:
        print(f"        base B OOF 点={int(mask_B.sum())}  OOF MAE={mae_B:.2f}")

    oof_mask = mask_A & mask_B
    idx = np.where(oof_mask)[0]
    times_idx = pd.DatetimeIndex(times)[idx]
    hours = times_idx.hour.values.astype(int)
    seg = SEG.segment_array(hours)
    dates = times_idx.normalize()
    oof_pool = {
        "idx": idx,
        "base_A_oof": base_A_oof[idx],
        "base_B_oof": base_B_oof[idx],
        "actual": actual.values[idx].astype(float),
        "times": np.asarray(times)[idx],
        "hours": hours,
        "seg": seg,
        "dates": dates,
    }
    return oof_pool, (hb_A, dc_A, tc_A), (hb_B, dc_B, tc_B), mae_A, mae_B
