# -*- coding: utf-8 -*-
"""verify_skew_fix.py - 验证 bug#1 修复后 predict 与 val 特征/预测 parity。

方法（只读，不修改生产代码）：
  1) 用修复后的 predict.predict_d1(run_date=2026-06-14, run_hour=21) 预测 D+1=2026-06-15。
     run-hour=21 复现 val 的"20:00 day-D issue 覆盖 D+1"条件（CSV 仅含 20:00 issue）。
  2) 与 diag_val.csv 中 2026-06-15 的 pred 比对（diag_val 用 run_time=None，同为最晚=20:00 issue）。
     修复前：predict 仅在 96 点上构造特征 -> pl_wr_diff_*/pl_norm 等全 NaN -> 与 val 预测显著不同。
     修复后：predict 在 14 天回看窗口构造特征 -> 与 val bit-identical -> 预测应一致。
  3) 直接检查修复后 D+1 时刻"曾 skew 的特征"是否非 NaN。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd

from load_pred import config as C, data_loader as dl, features as F
from load_pred.predict import predict_d1
from load_pred.model import EnsembleModel

DIAG = Path(__file__).parent / "FDS" / "output" / "diag_val.csv"
RUN_DATE = "2026-06-14"   # D；预测 D+1 = 2026-06-15（val 最后一天）

# bug#1 中曾因 96 点构造而全 NaN 的特征列
SKEW_COLS = ["plnorm_x_temp", "plnorm_x_hour", "irrad_anom_672", "pl_x_irrad_anom",
             "pl_wr_diff_96", "pl_wr_diff_672", "pl_wr_roll_mean_672", "pl_wr_roll_std_672",
             "pl_dip_96", "solar_mismatch"]


def main():
    # ---- 1) 修复后 predict D+1=2026-06-15 ----
    out = predict_d1(RUN_DATE, run_hour=21, run_minute=0, verbose=False)
    out["时间"] = pd.to_datetime(out["时间"])
    out = out.set_index("时间")
    pred_fix = out["预测负荷"]

    # ---- 2) diag_val 中 2026-06-15 ----
    dval = pd.read_csv(DIAG, encoding="utf-8-sig", index_col=0, parse_dates=True)
    dval = dval[dval.index.normalize() == pd.Timestamp("2026-06-15")]
    pred_val = dval["pred"]

    common = pred_fix.index.intersection(pred_val.index)
    pf = pred_fix.reindex(common)
    pv = pred_val.reindex(common)
    diff = (pf - pv).abs()
    print("=" * 64)
    print("bug#1 修复验证：predict(固定) vs diag_val(全历史) @ 2026-06-15")
    print("=" * 64)
    print(f"对齐点数 = {len(common)}")
    print(f"预测 max|Δ|  = {diff.max():.4f} MW")
    print(f"预测 mean|Δ| = {diff.mean():.4f} MW")
    print(f"predict 均值 = {pf.mean():.2f}   val 均值 = {pv.mean():.2f}")
    print("-" * 64)
    print("（修复前因 skew，predict 与 val 应显著不同；修复后应趋近一致，仅余四舍五入差异）")

    # ---- 3) 直接检查 D+1 时刻 skew 特征非 NaN ----
    print("=" * 64)
    print("D+1 时刻「曾 skew 特征」NaN 检查（修复后应全部 0 NaN）")
    print("=" * 64)
    run_date = pd.Timestamp(RUN_DATE).normalize()
    d1_start = run_date + pd.Timedelta(days=1)
    d1_end = d1_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    pred_load = dl.pred_load_series()
    weather = dl.load_weather_dedup(run_time=run_date + pd.Timedelta(hours=21))
    max_hist = max(max(C.PRED_LAGS), max(C.PRED_ROLLING_WINDOWS))
    build_times = pd.date_range(d1_start - pd.Timedelta(minutes=15) * (max_hist * 2), d1_end, freq=C.FREQ)
    X = F.build_features(build_times, pred_load, weather)
    model = EnsembleModel.load(C.MODEL_BUNDLE)
    if getattr(model, "mismatch_model", None) is not None:
        X = model.mismatch_model.transform(X)
    xd1 = X[X.index >= d1_start]   # D+1 96 行
    n_nan_total = 0
    for c in SKEW_COLS:
        if c in xd1.columns:
            n = int(xd1[c].isna().sum())
            n_nan_total += n
            flag = "OK" if n == 0 else f"!!! {n} NaN"
            print(f"  {c:24s} D+1 NaN = {n:3d}  {flag}")
        else:
            print(f"  {c:24s} (列不存在)")
    print("-" * 64)
    print(f"曾 skew 特征 D+1 总 NaN 数 = {n_nan_total}   {'PASS - skew 已消除' if n_nan_total == 0 else 'FAIL'}")
    print("=" * 64)


if __name__ == "__main__":
    main()
