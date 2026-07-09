# -*- coding: utf-8 -*-
"""FDS/a02_temporal.py - (二) 误差时间分析。"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diag_lib import load_val, metrics_by, save_fig, save_table

DOW = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]


def main():
    d = load_val()
    print("== (二) 误差时间分析 ==", flush=True)

    by_hour = metrics_by(d, "hour")
    by_slot = metrics_by(d, "slot")
    by_dow = metrics_by(d, "dow")
    by_dow.index = [DOW[i] for i in by_dow.index]
    by_month = metrics_by(d, "month")
    by_season = metrics_by(d, "season")
    by_we = metrics_by(d, "is_weekend")
    by_hol = metrics_by(d, "is_holiday")
    for nm, t in [("hour", by_hour), ("slot", by_slot), ("dow", by_dow), ("month", by_month),
                  ("season", by_season), ("weekend", by_we), ("holiday", by_hol)]:
        save_table(t, f"02_by_{nm}")
    print("按小时 MAE 范围: %.0f ~ %.0f (差 %.0f)" %
          (by_hour["MAE"].min(), by_hour["MAE"].max(), by_hour["MAE"].max() - by_hour["MAE"].min()), flush=True)
    print("最差3小时:", by_hour.sort_values("MAE", ascending=False).head(3)[["MAE", "Bias"]].to_string(), flush=True)
    print("最佳3小时:", by_hour.sort_values("MAE").head(3)[["MAE", "Bias"]].to_string(), flush=True)
    print("按月:\n", by_month[["MAE", "Bias"]].to_string(), flush=True)
    print("按季节:\n", by_season[["MAE", "Bias"]].to_string(), flush=True)
    print("工作日 vs 周末:\n", by_we[["MAE", "Bias"]].to_string(), flush=True)
    print("节假日 vs 非:\n", by_hol[["MAE", "Bias"]].to_string(), flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    ax.plot(by_hour.index, by_hour["MAE"], "o-", color="tab:blue", label="MAE")
    ax.axhline(by_hour["MAE"].mean(), color="tab:blue", ls="--", alpha=0.5)
    axb = ax.twinx()
    axb.plot(by_hour.index, by_hour["Bias"], "s-", color="tab:red", label="Bias")
    axb.axhline(0, color="tab:red", ls=":", alpha=0.5)
    ax.set_title("按小时 MAE / Bias"); ax.set_xlabel("hour"); ax.set_ylabel("MAE", color="tab:blue")
    axb.set_ylabel("Bias", color="tab:red")
    ax = axes[0, 1]
    ax.plot(by_slot.index, by_slot["MAE"], "-", color="tab:blue")
    ax.set_title("按 15min slot MAE (96维)"); ax.set_xlabel("slot 0-95"); ax.set_ylabel("MAE")
    ax = axes[1, 0]
    ax.bar(by_dow.index, by_dow["MAE"], color="tab:green")
    ax.set_title("按星期 MAE"); ax.set_ylabel("MAE")
    ax = axes[1, 1]
    ax.bar(by_month.index.astype(str), by_month["MAE"], color="tab:purple")
    axb = ax.twinx()
    axb.plot(by_month.index.astype(str), by_month["Bias"], "o-", color="tab:red")
    ax.set_title("按月份 MAE / Bias"); ax.set_ylabel("MAE", color="tab:purple")
    axb.set_ylabel("Bias", color="tab:red")
    fig.tight_layout()
    save_fig(fig, "02_temporal")


if __name__ == "__main__":
    main()
