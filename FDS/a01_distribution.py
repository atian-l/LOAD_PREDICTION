# -*- coding: utf-8 -*-
"""FDS/a01_distribution.py - (一) 误差分布分析。"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from diag_lib import load_val, metrics, save_fig, save_table, OUT


def main():
    d = load_val()
    e = d["error"].values
    ext = d["ext_error"].values
    print("== (一) 误差分布 ==", flush=True)

    # 基本统计
    m = metrics(e, d["actual"].values)
    m_ext = metrics(ext, d["actual"].values)
    skew = float(stats.skew(e))
    kurt = float(stats.kurtosis(e))
    t_stat, p_val = stats.ttest_1samp(e, 0)
    jb_stat, jb_p = stats.jarque_bera(e)
    summary = pd.DataFrame({
        "模型误差": [m["MAE"], m["RMSE"], m["Bias"], m["std"], m["MAPE"], skew, kurt,
                    float(np.percentile(np.abs(e), 50)), float(np.percentile(np.abs(e), 90)),
                    float(np.percentile(np.abs(e), 99)), float(t_stat), float(p_val), float(jb_stat), float(jb_p)],
        "外部预测误差": [m_ext["MAE"], m_ext["RMSE"], m_ext["Bias"], m_ext["std"], m_ext["MAPE"],
                    float(stats.skew(ext)), float(stats.kurtosis(ext)),
                    float(np.percentile(np.abs(ext), 50)), float(np.percentile(np.abs(ext), 90)),
                    float(np.percentile(np.abs(ext), 99)), 0, 0, 0, 0],
    }, index=["MAE", "RMSE", "Bias", "std", "MAPE", "偏度", "峰度", "P50|e|", "P90|e|",
              "P99|e|", "t(Bias=0)", "p", "JB统计量", "JB_p"])
    save_table(summary, "01_dist_summary")
    print(summary.to_string(), flush=True)

    # 误差区间占比
    ae = np.abs(e)
    bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 1e9)]
    names = ["<500", "500-1k", "1k-2k", "2k-4k", ">4k"]
    props = [float(np.mean((ae >= lo) & (ae < hi)) * 100) for lo, hi in bands]
    contr = [float(np.mean(np.where((ae >= lo) & (ae < hi), ae, 0))) for lo, hi in bands]
    bt = pd.DataFrame({"占比%": props, "对MAE贡献MW": contr}, index=names)
    save_table(bt, "01_error_bands")
    print("误差区间占比:\n", bt.to_string(), flush=True)

    # 图1: 直方图 + KDE
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (x, lbl, c) in zip(axes, [(e, "模型误差", "tab:blue"), (ext, "外部预测误差", "tab:orange")]):
        ax.hist(x, bins=80, density=True, alpha=0.5, color=c, label=lbl)
        kde = stats.gaussian_kde(x)
        xs = np.linspace(x.min(), x.max(), 400)
        ax.plot(xs, kde(xs), color=c, lw=1.5)
        ax.axvline(0, color="k", ls="--", lw=1)
        ax.axvline(x.mean(), color=c, ls=":", lw=1.5, label=f"Bias={x.mean():.0f}")
        ax.set_title(f"{lbl} 分布 (MAE={np.mean(np.abs(x)):.0f})")
        ax.legend(fontsize=8)
    save_fig(fig, "01_hist_kde")

    # 图2: QQ plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (x, lbl) in zip(axes, [(e, "模型误差"), (ext, "外部预测误差")]):
        stats.probplot(x, dist="norm", plot=ax)
        ax.set_title(f"QQ Plot - {lbl}")
        ax.get_lines()[0].set_markerfacecolor("tab:blue")
        ax.get_lines()[0].set_markersize(2)
    save_fig(fig, "01_qqplot")

    # 图3: 误差箱线 by 符号 + 长尾
    fig, ax = plt.subplots(figsize=(8, 4.5))
    parts = ax.boxplot([e[e >= 0], e[e < 0]], tick_labels=["高估(e>0)", "低估(e<0)"], showfliers=True,
                       patch_artist=True)
    for p, c in zip(parts["boxes"], ["tab:red", "tab:blue"]):
        p.set_facecolor(c); p.set_alpha(0.4)
    ax.axhline(0, color="k", lw=1)
    ax.set_title(f"高估/低估箱线 (高估占比={np.mean(e>0)*100:.1f}%,  长尾>3σ={np.mean(np.abs(e)>3*e.std())*100:.1f}%)")
    save_fig(fig, "01_box_overunder")
    print("高估占比: %.1f%%  低估占比: %.1f%%" % (np.mean(e > 0) * 100, np.mean(e < 0) * 100), flush=True)
    print(f"系统偏移 Bias={m['Bias']:.1f} MW (t={t_stat:.1f}, p={p_val:.1e})  "
          f"长尾: 偏度={skew:.2f} 峰度={kurt:.2f}", flush=True)


if __name__ == "__main__":
    main()
