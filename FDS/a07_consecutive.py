# -*- coding: utf-8 -*-
"""FDS/a07_consecutive.py - (七) 连续误差分析（连续高估/低估/漂移）。"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diag_lib import load_val, save_fig, save_table


def runs_of_sign(e):
    s = np.sign(e).astype(int)
    runs = []
    i = 0
    while i < len(s):
        j = i
        while j < len(s) and s[j] == s[i] and s[i] != 0:
            j += 1
        if s[i] != 0:
            runs.append((int(s[i]), j - i))
        i = j
    return runs


def main():
    d = load_val()
    e = d["error"].values
    print("== (七) 连续误差 ==", flush=True)
    runs = runs_of_sign(e)
    over = [r[1] for r in runs if r[0] > 0]
    under = [r[1] for r in runs if r[0] < 0]
    over = np.array(over); under = np.array(under)
    allr = np.array([r[1] for r in runs])
    stats = pd.DataFrame({
        "连续高估run": [over.mean(), np.median(over), over.max(), (over >= 96).sum(), (over >= 192).sum(), len(over)],
        "连续低估run": [under.mean(), np.median(under), under.max(), (under >= 96).sum(), (under >= 192).sum(), len(under)],
    }, index=["平均长度", "中位长度", "最长(15min)", "≥1天(96)个数", "≥2天(192)个数", "run总数"])
    save_table(stats, "07_consecutive_runs")
    print(stats.to_string(), flush=True)

    # 连续大误差（|e|>MAE）run
    big = (np.abs(e) > np.abs(e).mean()).astype(int)
    bigruns = []
    i = 0
    while i < len(big):
        if big[i] == 1:
            j = i
            while j < len(big) and big[j] == 1:
                j += 1
            bigruns.append(j - i)
            i = j
        else:
            i += 1
    bigruns = np.array(bigruns) if bigruns else np.array([0])
    print(f"连续大误差(|e|>MAE) run: 平均={bigruns.mean():.1f} 最长={bigruns.max()} (={bigruns.max()*15/60:.1f}h)", flush=True)

    # 每日 bias 漂移：按日聚合 bias，看日间波动
    daily = d["error"].resample("D").mean()
    print(f"逐日 Bias: 均值={daily.mean():.0f} std={daily.std():.0f} 范围=[{daily.min():.0f},{daily.max():.0f}]  "
          f"日间方差占比={daily.var()/(e.var())*100:.1f}% (相对总误差方差)", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(over, bins=np.arange(0, max(over.max(), 1) + 2), alpha=0.5, label=f"高估(N={len(over)})", color="tab:red")
    ax.hist(under, bins=np.arange(0, max(under.max(), 1) + 2), alpha=0.5, label=f"低估(N={len(under)})", color="tab:blue")
    ax.axvline(96, color="g", ls="--", label="1天(96)")
    ax.set_xlabel("连续 run 长度 (15min步)"); ax.set_ylabel("频次")
    ax.set_title("连续高估/低估 run 长度分布"); ax.legend()
    ax = axes[1]
    daily.plot(ax=ax, color="k")
    ax.axhline(0, color="r", lw=1)
    ax.fill_between(daily.index, daily, 0, where=daily > 0, alpha=0.3, color="tab:red")
    ax.fill_between(daily.index, daily, 0, where=daily < 0, alpha=0.3, color="tab:blue")
    ax.set_title("逐日 Bias 时间序列 (日间漂移)")
    ax.set_ylabel("日均 Bias (MW)")
    fig.tight_layout()
    save_fig(fig, "07_consecutive")


if __name__ == "__main__":
    main()
