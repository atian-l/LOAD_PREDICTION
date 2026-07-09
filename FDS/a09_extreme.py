# -*- coding: utf-8 -*-
"""FDS/a09_extreme.py - (九) 极端天气分析。"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diag_lib import load_val, metrics, save_fig, save_table


def main():
    d = load_val().copy()
    print("== (九) 极端天气 ==", flush=True)
    scen = {
        "高温(temp>30)": d["temp"] > 30,
        "低温(temp<0)": d["temp"] < 0,
        "寒冷(temp<8)": d["temp"] < 8,
        "暴雨(precip>2)": d["precip"] > 2,
        "有雨(precip>0)": d["precip"] > 0,
        "大风(wind>8)": d["wind"] > 8,
        "晴午(clear>0.8@11-14)": (d["clearness"] > 0.8) & d["hour"].between(11, 14),
        "多云午(clr0.2-0.5@11-14)": (d["clearness"] >= 0.2) & (d["clearness"] < 0.5) & d["hour"].between(11, 14),
        "阴天日间(clear<0.2,day)": (d["clearness"] < 0.2) & (d["is_daytime"] == 1),
    }
    rows = []
    total_mae = d["abs_error"].sum()
    for nm, mask in scen.items():
        g = d[mask]
        if len(g) == 0:
            continue
        m = metrics(g["error"].values, g["actual"].values)
        rows.append({"场景": nm, "N": m["N"], "占比%": len(g) / len(d) * 100,
                     "MAE": m["MAE"], "Bias": m["Bias"], "MAPE": m["MAPE"],
                     "对总MAE贡献%": g["abs_error"].sum() / total_mae * 100})
    t = pd.DataFrame(rows).set_index("场景")
    t = t.sort_values("MAE", ascending=False)
    save_table(t, "09_extreme_weather")
    print(t.to_string(), flush=True)


if __name__ == "__main__":
    main()
