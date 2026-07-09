# -*- coding: utf-8 -*-
"""FDS/a04_weather.py - (四) 天气条件分析。"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diag_lib import load_val, metrics_by, save_fig, save_table


def bin_series(s, edges, labels):
    return pd.cut(s, bins=edges, labels=labels, include_lowest=True)


def main():
    d = load_val().copy()
    print("== (四) 天气条件分析 ==", flush=True)

    d["temp_bin"] = bin_series(d["temp"], [-99, 0, 8, 15, 22, 28, 32, 99],
                               ["<0", "0-8", "8-15", "15-22", "22-28", "28-32", ">32"])
    d["irrad_bin"] = bin_series(d["irrad"], [-1, 0, 200, 400, 600, 800, 1500],
                                ["0", "0-200", "200-400", "400-600", "600-800", ">800"])
    d["precip_bin"] = bin_series(d["precip"], [-0.01, 0.01, 0.5, 2, 5, 100],
                                 ["0", "0-0.5", "0.5-2", "2-5", ">5"])
    d["wind_bin"] = bin_series(d["wind"], [-1, 2, 4, 6, 8, 100],
                               ["<2", "2-4", "4-6", "6-8", ">8"])
    d["clear_bin"] = bin_series(d["clearness"], [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
                                ["<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", ">0.8"])
    d["cloud_bin"] = bin_series(d["cloud_deficit"], [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01],
                                ["<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", ">0.8"])
    res = {}
    for nm, col in [("temp", "temp_bin"), ("irrad", "irrad_bin"), ("precip", "precip_bin"),
                    ("wind", "wind_bin"), ("clearness", "clear_bin"), ("cloud", "cloud_bin")]:
        t = metrics_by(d, col)
        res[nm] = t
        save_table(t, f"04_by_{nm}")
    for nm, t in res.items():
        print(f"按{nm}:\n", t[["N", "MAE", "Bias"]].to_string(), flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (nm, t) in zip(axes.flat, res.items()):
        x = range(len(t))
        ax.bar(x, t["MAE"], color="tab:blue", alpha=0.7)
        axb = ax.twinx()
        axb.plot(x, t["Bias"], "o-", color="tab:red")
        axb.axhline(0, color="tab:red", ls=":", alpha=0.5)
        ax.set_xticks(x); ax.set_xticklabels(t.index, rotation=30, fontsize=8)
        ax.set_title(f"按 {nm}"); ax.set_ylabel("MAE", color="tab:blue")
        axb.set_ylabel("Bias", color="tab:red")
    fig.tight_layout()
    save_fig(fig, "04_weather")


if __name__ == "__main__":
    main()
