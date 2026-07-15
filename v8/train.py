# -*- coding: utf-8 -*-
"""v8 训练入口：OOF 估计 + 全量 correction + 组装 V8Model + 保存 + val 评估。

运行：python -m v8.train
合规：所有动态参数来自训练期 OOF；val 仅最终评估，不参与任何参数学习。
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
from . import oof as OOF
from . import base as BASE
from . import correction as CORR
from . import weather_sim as WS
from . import model as VM


def _fmt(dt) -> str:
    return pd.Timestamp(dt).strftime("%Y/%m/%d %H:%M:%S")


def _mae(pred, actual):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(actual))))


def run_train(verbose: bool = True) -> dict:
    VC.ensure_dirs()
    cfg_A = LC.TRAIN_CONFIG
    cfg_B = {**LC.TRAIN_CONFIG, **VC.BASE_B_CFG_OVERRIDE}
    best_it = int(cfg_A["best_it_fixed"])

    if verbose:
        print("[1/8] 构建数据集（复用 load_pred.build_dataset）...")
    times, X, pred_load, actual = T.build_dataset()
    usable = T.usable_mask(times, pred_load, actual)
    mismatch_model = F.MismatchModel().fit(X, usable)
    X_full = mismatch_model.transform(X)
    mc = cfg_A["mos"]
    full_mos = F.MosModel(cols=mc.get("cols"), alpha=mc.get("alpha", 1.0)).fit(X_full, actual, usable)
    feat_cols = list(X_full.columns)
    if verbose:
        print(f"      特征数={len(feat_cols)}  可训练点={int(usable.sum())}")

    # ---- OOF 池 ----
    if verbose:
        print("[2/8] OOF 3 折 walk-forward（base A + base B）...")
    oof_pool, corr_A, corr_B, mae_A_oof, mae_B_oof = OOF.run_oof(
        times, X_full, pred_load, actual, usable, cfg_A, cfg_B, best_it, full_mos, verbose)
    if verbose:
        print(f"      OOF 池点={len(oof_pool['idx'])}  base A OOF MAE={mae_A_oof:.2f}  base B OOF MAE={mae_B_oof:.2f}")

    # ---- correction OOF（嵌套 2 折，无泄露）----
    if verbose:
        print("[3/8] correction 嵌套 2 折 OOF（trigger/α/w 估计用）...")
    corr_oof = CORR.correction_oof_nested(oof_pool, X_full, feat_cols)

    # ---- 天气相似度 + τ grid + DynamicEstimator ----
    if verbose:
        print("[4/8] 天气相似度池 + τ grid + trigger/α/w 动态估计 ...")
    day_vec_all = WS.day_weather_vectors(X_full, times)
    oof_dates = pd.DatetimeIndex(oof_pool["dates"]).normalize().unique()
    day_vec_pool = day_vec_all.loc[oof_dates].sort_index()
    fold_windows = cfg_A.get("best_it_folds")  # 3 折评估窗，minimax 跨季稳定性选 trigger/τ
    best_tau, best_worst, best_ws, best_dyn = None, None, None, None
    for tau in VC.WEATHER_SIM_TAU_GRID:
        ws = WS.WeatherSim(k=VC.WEATHER_SIM_K, tau=tau)
        ws.fit(day_vec_pool)
        dyn = CORR.DynamicEstimator(ws, oof_pool, corr_oof, day_vec_pool,
                                    fold_windows=fold_windows).fit(verbose=False)
        # τ 亦按 minimax 最差折稳定性选（跨年泛化代理），平局取 OOF 均值更小者
        if best_worst is None or dyn._oof_worst < best_worst - 1e-6 or (
                abs(dyn._oof_worst - best_worst) <= 1e-6 and dyn._oof_mae < best_dyn._oof_mae):
            best_worst, best_tau, best_ws, best_dyn = dyn._oof_worst, tau, ws, dyn
    best_mae_dyn = best_dyn._oof_mae
    if verbose:
        print(f"      τ grid 选 tau={best_tau}  动态修正 OOF MAE={best_mae_dyn:.2f}  最差折MAE={best_worst:.2f}")
        oof_base_mae = float(np.mean(np.abs(best_dyn.oof["base_A_oof"] - best_dyn.oof["actual"])))
        oof_trig_rate = float(np.mean(best_dyn._oof_final != best_dyn.oof["base_A_oof"]))
        print(f"      [dyn] trig_frac={best_dyn.trig_frac} min_gain={best_dyn.min_gain} "
              f"OOF base MAE={oof_base_mae:.2f} -> 动态={best_mae_dyn:.2f} "
              f"(Δ{best_mae_dyn-oof_base_mae:+.2f}, OOF trigger命中率={oof_trig_rate:.3f})")

    # ---- 全量 correction model + base B ----
    if verbose:
        print("[5/8] 全量训练 correction model（3 段）+ base B（reg_only）...")
    correction_models = CORR.train_correction_models(oof_pool, X_full, feat_cols)
    base_B = BASE.train_base_B_full(times, X_full, pred_load, actual, usable, cfg_B, best_it,
                                    full_mos, mismatch_model, corr_B)

    # ---- adaptive preference ----
    if verbose:
        print("[6/8] adaptive model selection 偏好表（天气型分桶）...")
    adaptive_pref = BASE.adaptive_preference(oof_pool, day_vec_pool)
    n_B = sum(1 for v in adaptive_pref.values() if v == "B")
    if verbose:
        print(f"      天气型桶={len(adaptive_pref)}  偏好 B 的桶={n_B}  其余=A")

    # ---- base A 加载 + 组装 ----
    if verbose:
        print("[7/8] 加载 base A（v6 根 bundle）+ 组装 V8Model + 保存 ...")
    base_A = BASE.load_base_A()
    model = VM.V8Model(
        feat_cols=feat_cols, cfg=cfg_A, mismatch_model=mismatch_model, full_mos=full_mos,
        base_A=base_A, base_B=base_B, correction_models=correction_models,
        weather_sim=best_ws, dynamic=best_dyn, adaptive_pref=adaptive_pref,
        corr_B=corr_B, oof_pool=oof_pool, corr_oof=corr_oof, day_vec_pool=day_vec_pool,
    )
    model.save(VC.V8_BUNDLE)
    if verbose:
        print(f"      已保存: {VC.V8_BUNDLE}")

    # ---- val 评估（仅报告，不参与参数）----
    if verbose:
        print("[8/8] val 评估 ...")
    metrics = _eval_val(model, times, X, pred_load, actual)
    if verbose:
        print("\n================ v8 val 评估 ================")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print("============================================")
    return metrics


def _eval_val(model, times, X, pred_load, actual) -> dict:
    """val 窗口评估（含 7 天 buffer 供 mismatch rolling）。"""
    val_start = pd.Timestamp(LC.VAL_START)
    val_end = pd.Timestamp(LC.VAL_END)
    buf_start = val_start - pd.Timedelta(days=7)
    em = (times >= buf_start) & (times <= val_end)
    X_eval = X[em]
    pred_eval = model.predict(X_eval, pred_load, times[em])
    pred_s = pd.Series(pred_eval, index=times[em])
    actual_s = actual.reindex(times[em])
    vm = (times[em] >= val_start) & (times[em] <= val_end) & actual_s.notna()
    p = pred_s[vm]; a = actual_s[vm]
    err = p - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    # 午间 08-18
    hours = pd.DatetimeIndex(p.index).hour.values
    mid = (hours >= 8) & (hours < 18)
    mae_mid = float(np.mean(np.abs(err[mid]))) if mid.any() else float("nan")
    # 大偏差
    big = np.abs(err) > 3000
    return {
        "MAE": mae, "RMSE": rmse, "Bias": bias,
        "午间MAE(08-18)": mae_mid,
        "大偏差点数(>3000)": int(big.sum()),
        "大偏差均值": float(np.mean(np.abs(err[big]))) if big.any() else 0.0,
        "N_points": int(len(a)),
        "v7_MAE参考": 1445.62,
    }


def main():
    metrics = run_train(verbose=True)
    return 0 if metrics["MAE"] < 1500 else 1


if __name__ == "__main__":
    sys.exit(main())
