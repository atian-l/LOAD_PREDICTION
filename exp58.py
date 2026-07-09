# -*- coding: utf-8 -*-
"""exp58: 阴雨/云天场景误差分析（规范重点场景）。

用生产预测(full_predictions.csv)分析 val 上 cloudy/clear/rainy 子集的：
  - MAE、样本数、贡献
  - 偏置方向 mean(pred-actual)
  - pl_wr 与误差方向的相关（是否已被 pl_wr 捕获）
判断阴雨天误差是否有“可纠正的系统性方向”，还是不可约噪声。
仅诊断，不写产物。
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
    mm = F.MismatchModel().fit(X, usable); X = mm.transform(X)

    # 生产预测
    fp = pd.read_csv(C.FULL_PRED_CSV, encoding="utf-8-sig")
    fp[C.COL_TIME] = pd.to_datetime(fp[C.COL_TIME])
    fp = fp.set_index(C.COL_TIME)
    pred_prod = fp[C.COL_PRED_LOAD].reindex(times)

    val = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END)) & actual.notna()).values
    act_v = actual[val].values
    pred_v = pred_prod.values[val]
    err = pred_v - act_v
    plwr = X["pl_weather_residual"].values[val]
    clear = X["clearness"].values[val]
    precip = X["precip"].values[val]
    hours = pd.DatetimeIndex(times[val]).hour.values
    midday = (hours >= 11) & (hours <= 13)

    print(f"val 总体: MAE={np.abs(err).mean():.1f}  bias={err.mean():.1f}  n={val.sum()}", flush=True)
    print(f"  corr(pl_wr, err) = {np.corrcoef(plwr, err)[0,1]:.3f}  (全时段)", flush=True)
    print(flush=True)

    def sub(name, mask):
        if mask.sum() == 0:
            print(f"  {name}: n=0"); return
        e = err[mask]
        print(f"  {name:22s}: n={mask.sum():5d} ({100*mask.mean():4.1f}%)  MAE={np.abs(e).mean():6.1f}  "
              f"bias={e.mean():+7.1f}  corr(pl_wr,err)={np.corrcoef(plwr[mask], e)[0,1]:+.3f}", flush=True)

    print("=== 按云量 clearness ===", flush=True)
    sub("clear (>0.7)", clear > 0.7)
    sub("mid (0.3-0.7)", (clear >= 0.3) & (clear <= 0.7))
    sub("cloudy (<0.3)", clear < 0.3)
    sub("very cloudy (<0.15)", clear < 0.15)
    print(flush=True)
    print("=== 按降水 precip ===", flush=True)
    sub("rainy (precip>0)", precip > 0)
    sub("dry (precip=0)", precip <= 0)
    print(flush=True)
    print("=== 午间 × 云量 ===", flush=True)
    sub("midday clear", midday & (clear > 0.7))
    sub("midday cloudy", midday & (clear < 0.3))
    sub("midday very cloudy", midday & (clear < 0.15))
    print(flush=True)

    # 阴天是否有可纠正的恒定偏置？（对比 pl_wr 已捕获的部分）
    cloudy = clear < 0.3
    if cloudy.sum() > 50:
        e = err[cloudy]
        # 减去 pl_wr 线性解释后的残余偏置
        p = plwr[cloudy]
        good = np.isfinite(p) & np.isfinite(e)
        if good.sum() > 10 and np.dot(p[good], p[good]) > 0:
            b = np.dot(p[good], e[good]) / np.dot(p[good], p[good])
            resid = e[good] - b * p[good]
            print(f"cloudy: pl_wr β={b:+.4f}, 去除 pl_wr 后残余 bias={resid.mean():+.1f} MAE={np.abs(resid).mean():.1f}", flush=True)
            print(f"  (若残余 bias 显著非零，可能有未被 pl_wr 捕获的阴天系统偏置)", flush=True)


if __name__ == "__main__":
    main()
