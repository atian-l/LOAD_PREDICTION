# -*- coding: utf-8 -*-
"""v8 预测入口（生产部署）。

每日 09:00 运行，加载 v8/models/v8_bundle.pkl（不重训），输出 D+1 全天 96 点。
五层串联：分段 -> adaptive base -> 天气相似度 KNN -> trigger/α/w -> correction -> fusion。
复用 load_pred 的 data_loader/features（skew fix：14 天回看窗口构建特征）。

用法:
    python -m v8.predict --run-date 2026-05-18
    python -m v8.predict                 # 默认=今天
"""
from __future__ import annotations
import sys
import argparse
import pandas as pd

from load_pred import config as LC
from load_pred import data_loader as dl
from load_pred import features as F

from . import config as VC
from . import model as VM


def _fmt_time(dt) -> str:
    return pd.Timestamp(dt).strftime("%Y/%m/%d %H:%M:%S")


def predict_d1(run_date, run_hour=LC.DEFAULT_RUN_HOUR, run_minute=LC.DEFAULT_RUN_MINUTE,
               verbose: bool = True) -> pd.DataFrame:
    """对 run_date(D) 的次日 D+1 全天 96 点预测。返回 DataFrame: 时间, 预测负荷。"""
    run_date = pd.Timestamp(run_date).normalize()
    run_dt = run_date + pd.Timedelta(hours=run_hour, minutes=run_minute)
    d1_start = run_date + pd.Timedelta(days=1)
    d1_end = d1_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=15)
    times = pd.date_range(d1_start, d1_end, freq=LC.FREQ)

    if verbose:
        print(f"运行日期 D   : {run_date.strftime('%Y/%m/%d')}")
        print(f"运行时刻     : {run_dt.strftime('%Y/%m/%d %H:%M')}")
        print(f"预测目标 D+1 : {d1_start.strftime('%Y/%m/%d')}  ({len(times)} 点)")

    # 仅使用运行时可获得的数据
    pred_load = dl.pred_load_series()
    weather = dl.load_weather_dedup(run_time=run_dt)

    # 14 天回看窗口构建特征（skew fix，与训练一致；窗口>672+96 使 D+1 特征 bit-identical）
    max_hist = max(max(LC.PRED_LAGS), max(LC.PRED_ROLLING_WINDOWS))
    lookback = pd.Timedelta(minutes=15) * (max_hist * 2)
    build_times = pd.date_range(d1_start - lookback, d1_end, freq=LC.FREQ)
    X = F.build_features(build_times, pred_load, weather)

    # 加载 v8 模型并推理（不训练）
    model = VM.V8Model.load(VC.V8_BUNDLE)
    if verbose:
        n_corr = sum(len(v) for v in model.correction_models.values())
        print(f"已加载 v8 模型: {VC.V8_BUNDLE}  (correction 成员={n_corr}, "
              f"trig_frac={model.dynamic.trig_frac}, min_gain={model.dynamic.min_gain})")

    pred_all = model.predict(X, pred_load, build_times)
    pred = pd.Series(pred_all, index=build_times).reindex(times).clip(lower=0.0)
    dec = int(LC.TRAIN_CONFIG.get("round_decimals", 2))
    pred = pred.round(dec)

    out = pd.DataFrame({
        "时间": [_fmt_time(t) for t in times],
        "预测负荷": pred.values,
    })
    return out


def _save_outputs(out: pd.DataFrame, run_date):
    dec = int(LC.TRAIN_CONFIG.get("round_decimals", 2))
    run_date = pd.Timestamp(run_date).strftime("%Y%m%d")
    named = VC.V8_OUTPUT_DIR / f"prediction_{run_date}.csv"
    out.to_csv(named, index=False, encoding="utf-8-sig", float_format=f"%.{dec}f")
    latest = VC.V8_OUTPUT_DIR / "latest_prediction.csv"
    out.to_csv(latest, index=False, encoding="utf-8-sig", float_format=f"%.{dec}f")
    return named, latest


def main():
    ap = argparse.ArgumentParser(description="v8 日前(D+1)负荷预测 - 预测模式")
    ap.add_argument("--run-date", default=None, help="运行日期 D，如 2026-05-18；默认=今天")
    ap.add_argument("--run-hour", type=int, default=LC.DEFAULT_RUN_HOUR)
    ap.add_argument("--run-minute", type=int, default=LC.DEFAULT_RUN_MINUTE)
    args = ap.parse_args()

    run_date = args.run_date if args.run_date else pd.Timestamp.now().normalize()
    VC.ensure_dirs()
    out = predict_d1(run_date, args.run_hour, args.run_minute, verbose=True)
    named, latest = _save_outputs(out, run_date)
    print(f"\n已输出 D+1 预测:")
    print(f"  {named}")
    print(f"  {latest}")
    print(f"  预测均值={out['预测负荷'].mean():.2f}  最小={out['预测负荷'].min():.2f}  最大={out['预测负荷'].max():.2f}")


if __name__ == "__main__":
    main()
