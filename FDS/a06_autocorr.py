# -*- coding: utf-8 -*-
"""FDS/a06_autocorr.py - (六) 残差自相关分析 (ACF/PACF)。"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diag_lib import load_val, load_oof, acf, pacf_yw, save_fig, save_table


def main():
    d = load_val()
    e = d["error"].values
    N = len(e)
    print("== (六) 残差自相关 ==", flush=True)
    nlags = 200
    ac = acf(e, nlags)
    pc = pacf_yw(e, min(nlags, 96))
    bound = 1.96 / np.sqrt(N)
    key_lags = [1, 4, 96, 192, 672] if N > 672 else [1, 4, 96, 192]
    rows = []
    for lg in key_lags:
        if lg < len(ac):
            rows.append({"lag": lg, "ACF": float(ac[lg]), "PACF": float(pc[lg]) if lg < len(pc) else float("nan"),
                         "显著(±%.3f)" % bound: "是" if abs(ac[lg]) > bound else "否"})
    t = pd.DataFrame(rows).set_index("lag")
    save_table(t, "06_acf_key_lags")
    print(t.to_string(), flush=True)
    # 找 ACF 显著峰
    sig = np.where(np.abs(ac[1:nlags + 1]) > bound)[0] + 1
    print(f"显著自相关 lag 数: {len(sig)}/{nlags}  (bound=±{bound:.3f})", flush=True)
    print(f"ACF@lag1={ac[1]:.3f} lag4={ac[4]:.3f} lag96={ac[96]:.3f} "
          f"lag192={ac[192]:.3f}" + (f" lag672={ac[672]:.3f}" if N > 672 and len(ac) > 672 else ""), flush=True)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7))
    ax = axes[0]
    ax.bar(range(nlags + 1), ac, width=1.0, color="tab:blue")
    ax.axhline(bound, color="r", ls="--", lw=1); ax.axhline(-bound, color="r", ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.5)
    for lg in [96, 192]:
        if lg <= nlags:
            ax.axvline(lg, color="g", ls=":", lw=1)
    ax.set_title(f"残差 ACF (val, N={N})  红虚线=95%置信界  绿线=96/192(日/双日)")
    ax.set_xlabel("lag (15min)"); ax.set_ylabel("ACF")
    ax = axes[1]
    npc = len(pc)
    ax.bar(range(npc), pc, width=1.0, color="tab:orange")
    ax.axhline(bound, color="r", ls="--", lw=1); ax.axhline(-bound, color="r", ls="--", lw=1)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("残差 PACF (前96阶)")
    ax.set_xlabel("lag"); ax.set_ylabel("PACF")
    fig.tight_layout()
    save_fig(fig, "06_acf_pacf")


if __name__ == "__main__":
    main()
