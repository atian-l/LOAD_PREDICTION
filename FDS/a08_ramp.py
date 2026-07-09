# -*- coding: utf-8 -*-
"""FDS/a08_ramp.py - (八) Ramp（爬坡）分析。"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diag_lib import load_val, metrics, save_fig, save_table


def main():
    d = load_val().dropna(subset=["actual_ramp", "pred_ramp"]).copy()
    print("== (八) Ramp 分析 ==", flush=True)
    ar = d["actual_ramp"].values
    bins = [-1e9, -1500, -800, -300, -100, 100, 300, 800, 1500, 1e9]
    lbl = ["<-1500", "-1500~-800", "-800~-300", "-300~-100", "-100~100",
           "100~300", "300~800", "800~1500", ">1500"]
    d["ramp_bin"] = pd.cut(d["actual_ramp"], bins=bins, labels=lbl, include_lowest=True)
    rows = []
    for key, g in d.groupby("ramp_bin", sort=True):
        m = metrics(g["error"].values, g["actual"].values)
        ramp_err = (g["pred_ramp"] - g["actual_ramp"]).abs().mean()
        # 爬坡方向预测成功率
        success = float(np.mean(np.sign(g["pred_ramp"]) == np.sign(g["actual_ramp"])) * 100)
        rows.append({"N": m["N"], "MAE": m["MAE"], "Bias": m["Bias"], "RampMAE": ramp_err,
                     "方向成功率%": success})
    t = pd.DataFrame(rows, index=lbl)
    save_table(t, "08_ramp")
    print(t.to_string(), flush=True)
    # 大爬坡(>800 或 <-800) 的占比与 MAE 贡献
    big = d["actual_ramp"].abs() > 800
    print(f"大爬坡(|Δ|>800MW/15min) 占比={big.mean()*100:.1f}%  其MAE={d[big]['error'].abs().mean():.0f} "
          f"vs 全体MAE={d['error'].abs().mean():.0f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.bar(range(len(t)), t["MAE"], color="tab:blue")
    ax.set_xticks(range(len(t))); ax.set_xticklabels(t.index, rotation=45, fontsize=8)
    ax.set_title("按实际爬坡区间 MAE"); ax.set_ylabel("MAE")
    axb = ax.twinx(); axb.plot(range(len(t)), t["方向成功率%"], "o-", color="tab:red")
    axb.set_ylabel("爬坡方向成功率%", color="tab:red")
    ax = axes[1]
    ax.scatter(d["actual_ramp"], d["ramp_error"], s=2, alpha=0.15)
    ax.axhline(0, color="k", lw=1); ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("actual ramp (MW/15min)"); ax.set_ylabel("ramp error (pred-actual)")
    ax.set_title("爬坡误差 vs 实际爬坡")
    fig.tight_layout()
    save_fig(fig, "08_ramp")


if __name__ == "__main__":
    main()
