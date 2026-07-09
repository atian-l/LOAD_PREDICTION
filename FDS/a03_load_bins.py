# -*- coding: utf-8 -*-
"""FDS/a03_load_bins.py - (三) 负荷区间分析。"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diag_lib import load_val, metrics, metrics_by, save_fig, save_table


def main():
    d = load_val()
    print("== (三) 负荷区间分析 ==", flush=True)

    # 按预测负荷十分位
    d = d.copy()
    d["pl_decile"] = pd.qcut(d["pred_load"], 10, labels=[f"P{i*10}-{i*10+10}" for i in range(10)])
    d["act_decile"] = pd.qcut(d["actual"], 10, labels=[f"P{i*10}-{i*10+10}" for i in range(10)])
    by_pl = metrics_by(d, "pl_decile")
    by_act = metrics_by(d, "act_decile")
    save_table(by_pl, "03_by_pred_load_decile")
    save_table(by_act, "03_by_actual_decile")
    print("按预测负荷十分位:\n", by_pl[["N", "MAE", "Bias", "RMSE"]].to_string(), flush=True)

    # 峰谷：实际负荷最高/最低 5% 点
    hi = d["actual"].quantile(0.95); lo = d["actual"].quantile(0.05)
    peak = metrics(d[d["actual"] >= hi]["error"].values, d[d["actual"] >= hi]["actual"].values)
    valley = metrics(d[d["actual"] <= lo]["error"].values, d[d["actual"] <= lo]["actual"].values)
    print(f"峰值(实际≥P95={hi:.0f}): MAE={peak['MAE']:.0f} Bias={peak['Bias']:.0f} N={peak['N']}", flush=True)
    print(f"谷值(实际≤P5={lo:.0f}): MAE={valley['MAE']:.0f} Bias={valley['Bias']:.0f} N={valley['N']}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.bar(range(10), by_pl["MAE"], color="tab:blue")
    ax.set_xticks(range(10)); ax.set_xticklabels(by_pl.index, rotation=45)
    ax.set_title("按预测负荷十分位 MAE"); ax.set_ylabel("MAE")
    axb = ax.twinx(); axb.plot(range(10), by_pl["Bias"], "o-", color="tab:red")
    axb.axhline(0, color="tab:red", ls=":", alpha=0.5); axb.set_ylabel("Bias", color="tab:red")
    ax = axes[1]
    ax.scatter(d["actual"], d["error"], s=2, alpha=0.15, color="tab:blue")
    ax.axhline(0, color="k", lw=1)
    # 拟合趋势
    z = np.polyfit(d["actual"], d["error"], 2)
    xs = np.linspace(d["actual"].min(), d["actual"].max(), 200)
    ax.plot(xs, np.polyval(z, xs), "r-", lw=2, label=f"二次拟合")
    ax.set_xlabel("actual load (MW)"); ax.set_ylabel("error (pred-actual)")
    ax.set_title("误差 vs 实际负荷 (正=高估)"); ax.legend()
    fig.tight_layout()
    save_fig(fig, "03_load_bins")


if __name__ == "__main__":
    main()
