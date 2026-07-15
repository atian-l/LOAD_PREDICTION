# -*- coding: utf-8 -*-
"""
预测模式入口（生产部署，TFT 集成）。

每日运行：
  - 加载 models/ 中已训练模型（不重训，Constraint #6）；
  - 仅利用运行当天能够获得的数据：
      * 外部预测负荷：运行日 D 最远获 D+1（Constraint #2）；
      * 气象：仅使用 起报时间 <= 运行时刻 的最新起报版本（Constraint #4/#5）；
  - 输出 D+1 全天 96 个时刻预测。

TFT 按预测日序列前向，需 encoder 历史在 X 内。build_features 在 14 天回看窗口上构造
（> encoder_len=288 + 96），使 D+1 各点特征与训练 bit-identical 且 encoder 历史充足（skew fix，
与 v6/TCN 同），再取 D+1 96 点。

用法:
    python -m load_pred_tft.predict --run-date 2026-05-18
    python -m load_pred_tft.predict                 # 默认=今天
"""
from __future__ import annotations
import sys
import argparse
import pandas as pd

from . import config as C
from . import data_loader as dl
from . import features as F
from .model import LoadModel


def _fmt_time(dt) -> str:
    return pd.Timestamp(dt).strftime("%Y/%m/%d %H:%M:%S")


def predict_d1(run_date, run_hour=C.DEFAULT_RUN_HOUR, run_minute=C.DEFAULT_RUN_MINUTE,
               verbose: bool = True) -> pd.DataFrame:
    """
    对 run_date(D) 的次日 D+1 全天 96 点进行预测。

    返回 DataFrame: 时间(str), 预测负荷(float)
    """
    run_date = pd.Timestamp(run_date).normalize()
    run_dt = run_date + pd.Timedelta(hours=run_hour, minutes=run_minute)
    d1_start = run_date + pd.Timedelta(days=1)
    d1_end = d1_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    times = pd.date_range(d1_start, d1_end, freq=C.FREQ)

    if verbose:
        print(f"运行日期 D   : {run_date.strftime('%Y/%m/%d')}")
        print(f"运行时刻     : {run_dt.strftime('%Y/%m/%d %H:%M')}")
        print(f"预测目标 D+1 : {d1_start.strftime('%Y/%m/%d')}  ({len(times)} 点)")

    # ---- 仅使用运行时可获得的数据 ----
    pred_load = dl.pred_load_series()
    weather = dl.load_weather_dedup(run_time=run_dt)

    # 特征（与训练完全一致的构造逻辑）；在足够长历史回看窗口上构造再取 D+1 96 点
    # （skew fix：rolling/shift 回看 ≤672，14 天窗口已完整覆盖；且 TFT encoder_len=288 < 14 天）。
    max_hist_steps = max(max(C.PRED_LAGS), max(C.PRED_ROLLING_WINDOWS))  # 672
    lookback = pd.Timedelta(minutes=15) * (max_hist_steps * 2)           # 14 天
    build_times = pd.date_range(d1_start - lookback, d1_end, freq=C.FREQ)
    X = F.build_features(build_times, pred_load, weather)

    # ---- 加载模型并推理（不训练）----
    model = LoadModel.load(C.MODEL_BUNDLE)
    if getattr(model, "mismatch_model", None) is not None:
        X = model.mismatch_model.transform(X)
    if verbose:
        n_mem = len(getattr(model, "members", []))
        print(f"已加载模型   : {C.MODEL_BUNDLE}  (members={n_mem}, shrinkage={getattr(model,'shrinkage',1.0)})")

    pred_all = model.predict_load(X, pred_load)
    pred = pd.Series(pred_all, index=build_times).reindex(times).clip(lower=0.0)
    dec = int(model.train_meta.get("config", {}).get("round_decimals", 2))
    pred = pred.round(dec)

    out = pd.DataFrame({
        "时间": [_fmt_time(t) for t in times],
        "预测负荷": pred.values,
    })
    return out


def _save_outputs(out: pd.DataFrame, run_date):
    dec = int(C.TRAIN_CONFIG.get("round_decimals", 2))
    run_date = pd.Timestamp(run_date).strftime("%Y%m%d")
    named = C.OUTPUT_DIR / f"prediction_{run_date}.csv"
    out.to_csv(named, index=False, encoding="utf-8-sig", float_format=f"%.{dec}f")
    out.to_csv(C.LATEST_PRED_CSV, index=False, encoding="utf-8-sig", float_format=f"%.{dec}f")
    return named, C.LATEST_PRED_CSV


def main():
    ap = argparse.ArgumentParser(description="日前(D+1)负荷预测 - 预测模式（TFT）")
    ap.add_argument("--run-date", default=None, help="运行日期 D，如 2026-05-18；默认=今天")
    ap.add_argument("--run-hour", type=int, default=C.DEFAULT_RUN_HOUR, help="运行时刻-时")
    ap.add_argument("--run-minute", type=int, default=C.DEFAULT_RUN_MINUTE, help="运行时刻-分")
    args = ap.parse_args()

    run_date = args.run_date if args.run_date else pd.Timestamp.now().normalize()
    C.ensure_dirs()

    out = predict_d1(run_date, args.run_hour, args.run_minute, verbose=True)
    named, latest = _save_outputs(out, run_date)
    print(f"\n已输出 D+1 预测:")
    print(f"  {named}")
    print(f"  {latest}")
    print(f"  预测均值={out['预测负荷'].mean():.2f}  最小={out['预测负荷'].min():.2f}  最大={out['预测负荷'].max():.2f}")


if __name__ == "__main__":
    main()
