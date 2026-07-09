# -*- coding: utf-8 -*-
"""exp55: 现实 MOS 楼层（仅用 ≤ D-1 的过去实际负荷，无未来泄露；但违反约束#1）。

为用户决策提供具体数字：若放宽 #1 允许“过去实际负荷”做 MOS 偏置校正，能达到多少。
对每个 val 时刻 T(h)，bias_h = mean(model_pred - actual) over 过去 K 天同时刻 h (T 之前)。
校正 = model_pred(T) - bias_h。
合规警示：仅诊断。使用实际负荷做输入违反 #1，除非用户明确批准绝不进入生产。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd

from load_pred import config as C, data_loader as dl, features as F, train as T


def main():
    times, X, pred_load, actual = T.build_dataset()
    usable = T.usable_mask(times, pred_load, actual)
    mm = F.MismatchModel().fit(X, usable)
    X = mm.transform(X)
    val = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
           & actual.notna()).values
    cfg = dict(C.TRAIN_CONFIG)
    cfg["best_it_fixed"] = 80

    # 生产模型预测（含 per-hour + drift_corr）
    print("training full ensemble ...", flush=True)
    model = T.train_ensemble(times, X, pred_load, actual, usable, cfg, 80)
    model.hour_bias, model.drift_corr = T.compute_hour_bias(times, X, pred_load, actual, usable, cfg, 80)
    pred_prod = pd.Series(model.predict_load(X, pred_load), index=times)  # 生产预测（全量）

    actual_v = actual[val]
    mae_prod = (pred_prod[val] - actual_v).abs().mean()
    mae_ext = (pred_load[val] - actual_v).abs().mean()
    print(f"external pred_load val MAE = {mae_ext:.2f}", flush=True)
    print(f"production model val MAE = {mae_prod:.2f}  (基准)", flush=True)
    print(flush=True)

    hours = pd.DatetimeIndex(times).hour.values.astype(int)
    # 过去实际负荷 MOS：逐时刻用过去 K 天同时刻的 (pred-actual) 均值校正
    pred_arr = pred_prod.values.astype(float)
    act_arr = actual.values.astype(float)
    for K in (3, 7, 14):
        corrected = pred_arr.copy()
        # 仅对 val 校正；用整个序列的过去（含训练期实际）
        for i in np.where(val)[0]:
            h = hours[i]
            # 过去 K 天同时刻：索引 i-96*k ... i-96*1，同 hour
            idxs = [i - 96 * k for k in range(1, K + 1)]
            idxs = [j for j in idxs if j >= 0 and np.isfinite(pred_arr[j]) and np.isfinite(act_arr[j]) and hours[j] == h]
            if idxs:
                b = np.mean(pred_arr[idxs] - act_arr[idxs])
                corrected[i] = pred_arr[i] - b
        mae = np.abs(corrected[val] - act_arr[val]).mean()
        print(f"prod + past-actuals MOS(K={K:2d}d, per-hour) val MAE = {mae:.2f}  (Δprod {mae-mae_prod:+.2f})", flush=True)

    # 也测：仅外部预测 + MOS（看 MOS 单独威力）
    print(flush=True)
    for K in (7,):
        corrected = pred_load.values.astype(float).copy()
        act_full = actual.values.astype(float)
        for i in np.where(val)[0]:
            h = hours[i]
            idxs = [i - 96 * k for k in range(1, K + 1)]
            idxs = [j for j in idxs if j >= 0 and np.isfinite(pred_load.values[j]) and np.isfinite(act_full[j]) and hours[j] == h]
            if idxs:
                b = np.mean(pred_load.values[idxs] - act_full[idxs])
                corrected[i] = pred_load.values[i] - b
        mae = np.abs(corrected[val] - act_full[val]).mean()
        print(f"external + past-actuals MOS(K={K}d) val MAE = {mae:.2f}  (Δext {mae-mae_ext:+.2f})", flush=True)

    # 日级 MOS（过去 K 天整体均值偏置，更粗但更稳）
    print(flush=True)
    dates = pd.DatetimeIndex(times).normalize()
    # 每日 (pred-actual) 均值
    df_d = pd.DataFrame({"d": dates, "r": pred_arr - act_arr, "ok": np.isfinite(pred_arr) & np.isfinite(act_arr)})
    daily = df_d[df_d.ok].groupby("d")["r"].mean().sort_index()
    for K in (3, 7, 14, 30):
        # 过去 K 天滚动均值（不含当天）
        roll = daily.shift(1).rolling(f"{K}D").mean()
        corr = pred_arr.copy()
        for i in np.where(val)[0]:
            d = dates[i]
            if d in roll.index and np.isfinite(roll.loc[d]):
                corr[i] = pred_arr[i] - roll.loc[d]
        mae = np.abs(corr[val] - act_arr[val]).mean()
        print(f"prod + DAY-level MOS(K={K:2d}d) val MAE = {mae:.2f}  (Δprod {mae-mae_prod:+.2f})", flush=True)


if __name__ == "__main__":
    main()
