# -*- coding: utf-8 -*-
"""诊断：trigger 阈值扫描（OOF 触发率/最差折 MAE vs val 触发率/MAE）。

目的：理解 trigger 阈值与跨年(val)表现的关系，验证跨年稳定性（用户验证法 ③）。
合规：本脚本只读已存 v8_bundle，扫描阈值用于"理解/验证"，不把 val MAE 写回任何生产参数；
生产 trigger 仍由 train 期 OOF(minimax) 选定。val 在此仅作跨年泛化诊断。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from load_pred import config as LC
from load_pred import train as T
from . import config as VC
from . import model as VM
from . import segments as SEG
from . import weather_sim as WS


def _metrics(p, a, hours):
    err = np.asarray(p) - np.asarray(a)
    mid = (hours >= 8) & (hours < 18)
    return float(np.mean(np.abs(err))), float(np.mean(np.abs(err[mid]))) if mid.any() else float("nan")


def run():
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
    a = actual_eval[vm].values

    v8 = VM.V8Model.load(VC.V8_BUNDLE)

    # OOF 触发率 / 最差折 MAE / 均值 MAE（逐阈值）
    fid = v8.dynamic._fold_ids()
    actual_oof = v8.dynamic.oof["actual"]
    base_oof = v8.dynamic.oof["base_A_oof"]
    oof_base_mae = float(np.mean(np.abs(base_oof - actual_oof)))
    use_minimax = (fid >= 0).any()
    folds = [k for k in range(len(v8.dynamic.fold_windows or [])) if (fid == k).any()] if use_minimax else []

    # val 逐点 base/corr 预算一次（与阈值无关），再按阈值组合
    X_t = v8.mismatch_model.transform(X_eval)
    base_A_pred = v8.base_A.predict_load(X_t, pred_load)
    base_B_pred = v8.base_B.predict_load(X_t, pred_load)
    day_vec = WS.day_weather_vectors(X_t, times_eval)
    dates = pd.DatetimeIndex(times_eval).normalize()
    hours = times_eval.hour.values.astype(int)
    seg_arr = SEG.segment_array(hours)
    date_sel = {}
    for d in np.unique(dates):
        d = pd.Timestamp(d).normalize()
        wt = VM.BASE.weather_type(day_vec.loc[d]) if d in day_vec.index else None
        date_sel[d] = VM.BASE.select_base(wt, v8.adaptive_pref) if wt is not None else "A"
    # 各段 corr 预测（与阈值无关）
    corr_full = np.empty(len(times_eval), dtype=float)
    for seg in VC.SEGMENTS:
        m = seg_arr == seg
        if not m.any():
            continue
        idx_seg = np.where(m)[0]
        corr_full[idx_seg] = VM.CORR.correction_predict(v8.correction_models[seg], X_t.iloc[idx_seg][v8.feat_cols])

    print(f"OOF base MAE={oof_base_mae:.2f}  val N={int(vm.sum())}  v7_MAE=1445.62")
    print(f"{'tf':>5} {'mg':>6} | {'OOF_trig':>8} {'OOF_MAE':>8} {'worst':>8} | {'val_trig':>8} {'val_MAE':>8} {'val_mid':>8}")
    print("-" * 78)
    for tf in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for mg in VC.MIN_GAIN_GRID:
            # OOF
            oof_final = v8.dynamic._simulate(mg, tf=tf)
            oof_trig = float(np.mean(oof_final != base_oof))
            oof_mae = float(np.mean(np.abs(oof_final - actual_oof)))
            worst = max(float(np.mean(np.abs(oof_final[fid == k] - actual_oof[fid == k]))) for k in folds) if folds else oof_mae
            # val
            v8.dynamic.min_gain = mg
            final = np.empty(len(times_eval), dtype=float)
            trig_flag = np.zeros(len(times_eval), dtype=bool)
            for seg in VC.SEGMENTS:
                m = seg_arr == seg
                if not m.any():
                    continue
                idx_seg = np.where(m)[0]
                ds_cache = {}
                for d in pd.DatetimeIndex(np.unique(dates[idx_seg])).normalize():
                    q_vec = day_vec.loc[d].values if d in day_vec.index else None
                    ds_cache[d] = v8.dynamic.params(d, seg, q_vec=q_vec, tf=tf)
                for j in idx_seg:
                    d = pd.Timestamp(dates[j]).normalize()
                    sel = date_sel.get(d, "A")
                    base_val = base_A_pred[j] if sel == "A" else base_B_pred[j]
                    al, w, trig = ds_cache[d]
                    if trig and al > 0 and w > 0:
                        final[j] = base_val + w * al * corr_full[j]
                        trig_flag[j] = True
                    else:
                        final[j] = base_val
            final = np.clip(final, 0.0, None)
            val_trig = float(np.mean(trig_flag[vm]))
            v_mae, v_mid = _metrics(final[vm], a, hours_val)
            print(f"{tf:>5} {mg:>6.0f} | {oof_trig:>8.3f} {oof_mae:>8.2f} {worst:>8.2f} | {val_trig:>8.3f} {v_mae:>8.2f} {v_mid:>8.2f}")
    # baseline: 永不触发（pure base A，adaptive 也强制 A）= v7
    v_mae, v_mid = _metrics(base_A_pred[vm], a, hours_val)
    print(f"{'never':>5} {'-':>6} | {0.0:>8.3f} {oof_base_mae:>8.2f} {oof_base_mae:>8.2f} | {0.0:>8.3f} {v_mae:>8.2f} {v_mid:>8.2f}  (pure base A = v7)")


if __name__ == "__main__":
    run()
