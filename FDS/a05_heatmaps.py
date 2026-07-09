# -*- coding: utf-8 -*-
"""FDS/a05_heatmaps.py - (五) 二维 Heatmap 分析。"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diag_lib import load_val, save_fig


def heat(d, row, col, row_bins, col_bins, row_labels, col_labels, val="error", agg="mean"):
    d = d.copy()
    d["rb"] = pd.cut(d[row], bins=row_bins, labels=row_labels, include_lowest=True) \
        if row_bins is not None else d[row]
    d["cb"] = pd.cut(d[col], bins=col_bins, labels=col_labels, include_lowest=True) \
        if col_bins is not None else d[col]
    return d.pivot_table(index="rb", columns="cb", values=val, aggfunc=agg)


def plot_heat(ax, piv, title, cmap="RdBu_r", center=0, fmt=".0f"):
    arr = piv.values.astype(float)
    if center is not None:
        vmax = np.nanmax(np.abs(arr))
        im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    else:
        im = ax.imshow(arr, aspect="auto", cmap=cmap)
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, fontsize=7, rotation=45)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=8)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main():
    d = load_val()
    print("== (五) 二维 Heatmap ==", flush=True)
    temp_e = [-99, 0, 8, 15, 22, 28, 99]; temp_l = ["<0", "0-8", "8-15", "15-22", "22-28", ">28"]
    clr_e = [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01]; clr_l = ["<.2", ".2-.4", ".4-.6", ".6-.8", ">.8"]
    irr_e = [-1, 1, 200, 400, 600, 800, 1500]; irr_l = ["0", "0-200", "200-400", "400-600", "600-800", ">800"]
    pl_e = d["pred_load"].quantile([0, .2, .4, .6, .8, 1]).values; pl_l = ["P0-20", "P20-40", "P40-60", "P60-80", "P80-100"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plot_heat(axes[0, 0], heat(d, "temp", "hour", temp_e, None, temp_l, list(range(24))), "Bias: 温度 × 小时")
    plot_heat(axes[0, 1], heat(d, "clearness", "hour", clr_e, None, clr_l, list(range(24))), "Bias: Clearness × 小时")
    plot_heat(axes[0, 2], heat(d, "irrad", "hour", irr_e, None, irr_l, list(range(24))), "Bias: 辐照 × 小时")
    # Month x Hour
    d2 = d.copy(); d2["m"] = d2.index.month
    piv = d2.pivot_table(index="m", columns="hour", values="error", aggfunc="mean")
    plot_heat(axes[1, 0], piv, "Bias: 月份 × 小时")
    plot_heat(axes[1, 1], heat(d, "temp", "pred_load", temp_e, pl_e, temp_l, pl_l), "Bias: 温度 × 预测负荷")
    # MAE 版本：Hour x Temperature
    plot_heat(axes[1, 2], heat(d, "temp", "hour", temp_e, None, temp_l, list(range(24)), val="abs_error"),
              "MAE: 温度 × 小时", cmap="viridis", center=None)
    fig.tight_layout()
    save_fig(fig, "05_heatmaps")
    # 找系统性失配最大的格子
    piv_bh = heat(d, "temp", "hour", temp_e, None, temp_l, list(range(24)))
    flat = piv_bh.stack().sort_values()
    print("Bias 最大失配格(温度×小时) Top5 高估:\n", flat.tail(5).to_string(), flush=True)
    print("Bias 最大失配格(温度×小时) Top5 低估:\n", flat.head(5).to_string(), flush=True)


if __name__ == "__main__":
    main()
