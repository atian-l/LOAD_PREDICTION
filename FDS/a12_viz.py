# -*- coding: utf-8 -*-
"""FDS/a12_viz.py - (十二) 模型拟合能力可视化。"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from diag_lib import load_val, save_fig


def plot_day(ax, g, title):
    ax.plot(g.index, g["actual"], "k-", lw=1.5, label="实际")
    ax.plot(g.index, g["pred"], "r-", lw=1.2, label="预测")
    ax.plot(g.index, g["pred_load"], "b--", lw=1, alpha=0.6, label="外部预测")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.tick_params(axis="x", labelsize=7, rotation=30)


def main():
    d = load_val()
    print("== (十二) 拟合能力可视化 ==", flush=True)

    # 1) pred vs actual 散点
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.scatter(d["actual"], d["pred"], s=2, alpha=0.15)
    mn, mx = d["actual"].min(), d["actual"].max()
    ax.plot([mn, mx], [mn, mx], "r-", lw=1.5, label="y=x")
    ax.set_xlabel("actual"); ax.set_ylabel("pred"); ax.set_title("预测 vs 实际")
    ax.legend()
    ax = axes[1]
    ax.scatter(d["actual"], d["error"], s=2, alpha=0.15, color="tab:purple")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("actual"); ax.set_ylabel("error (pred-actual)")
    ax.set_title("残差图 (正=高估)")
    fig.tight_layout()
    save_fig(fig, "12_pred_vs_actual")

    # 2) 误差时间序列（按周采样画全期）
    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.plot(d.index, d["error"], lw=0.4, color="tab:blue", alpha=0.7)
    ax.axhline(0, color="k", lw=1)
    roll = d["error"].rolling(96 * 7, center=True).mean()
    ax.plot(roll.index, roll, color="tab:red", lw=1.5, label="7日滚动均值")
    ax.set_title("验证期误差时间序列 (蓝=逐点, 红=7日滚动Bias)")
    ax.set_ylabel("error (MW)"); ax.legend()
    fig.tight_layout()
    save_fig(fig, "12_error_timeseries")

    # 3) Worst / Best / Peak 日案例
    daily_mae = d["abs_error"].resample("D").mean()
    worst_day = daily_mae.idxmax()
    best_day = daily_mae.idxmin()
    # 峰值误差日：当日 max|error|
    daily_max = d["abs_error"].resample("D").max()
    peak_err_day = daily_max.idxmax()
    # 高峰低估日：实际峰值日
    daily_peak = d["actual"].resample("D").max()
    peak_load_day = daily_peak.idxmax()

    days = [("Worst Day (日均MAE最高)", worst_day),
            ("Best Day (日均MAE最低)", best_day),
            ("Peak-Error Day (单点最大误差)", peak_err_day),
            ("Peak-Load Day (实际最高负荷)", peak_load_day)]
    print("案例日: worst=%s MAE=%.0f, best=%s MAE=%.0f, peak_err=%s, peak_load=%s (%.0f MW)" % (
        worst_day.date(), daily_mae[worst_day], best_day.date(), daily_mae[best_day],
        peak_err_day.date(), peak_load_day.date(), daily_peak[peak_load_day]), flush=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (lbl, day) in zip(axes.flat, days):
        g = d.loc[day.strftime("%Y-%m-%d")]
        mae = g["abs_error"].mean()
        plot_day(ax, g, f"{lbl} {day.date()} (日MAE={mae:.0f})")
    fig.tight_layout()
    save_fig(fig, "12_case_days")

    # 4) 连续 2 周详图
    mid = d.index[len(d) // 2]
    g = d.loc[mid: mid + pd.Timedelta(days=14)]
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(g.index, g["actual"], "k-", lw=1, label="实际")
    ax.plot(g.index, g["pred"], "r-", lw=0.8, alpha=0.8, label="预测")
    ax.fill_between(g.index, g["actual"], g["pred"], alpha=0.15, color="tab:red", label="误差")
    ax.set_title("连续2周 预测 vs 实际")
    ax.legend(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.tight_layout()
    save_fig(fig, "12_two_week_detail")


if __name__ == "__main__":
    main()
