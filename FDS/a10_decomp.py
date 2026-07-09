# -*- coding: utf-8 -*-
"""FDS/a10_decomp.py - (十) 误差来源拆解（方差分解）。

将最终误差 e = pred - actual 的方差，按因子组顺序回归（sequential OLS R²增量）拆分为：
外部预测误差 / 时间规律 / 天气 / 节假日与周末 / 负荷水平 / 随机残差。
顺序：先 ext_error（外部预测漏过未校正的部分），再时间、天气、节假日、负荷水平。
另报告模型对外部误差的校正有效率。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diag_lib import load_val, save_fig, save_table


def ols_r2(y, X):
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    return 1.0 - np.var(resid) / np.var(y)


def dummies(s):
    return pd.get_dummies(s, prefix=s.name, drop_first=True).astype(float).values


def main():
    d = load_val().dropna().copy()
    e = d["error"].values
    y = e - e.mean()
    TSS = np.var(y)
    print("== (十) 误差来源拆解 ==", flush=True)

    # 因子组
    ext = d["ext_error"].values.reshape(-1, 1)
    temporal = np.column_stack([dummies(d["hour"].rename("h")), dummies(d["dow"].rename("d")),
                                dummies(d["month"].rename("m"))])
    d["temp_bin"] = pd.cut(d["temp"], [-99, 0, 8, 15, 22, 28, 99], labels=False, include_lowest=True)
    d["clr_bin"] = pd.cut(d["clearness"], [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01], labels=False, include_lowest=True)
    d["irr_bin"] = pd.cut(d["irrad"], [-1, 1, 200, 400, 600, 800, 1500], labels=False, include_lowest=True)
    weather = np.column_stack([dummies(d["temp_bin"].rename("t")), dummies(d["clr_bin"].rename("c")),
                               dummies(d["irr_bin"].rename("i"))])
    hol = np.column_stack([d["is_holiday"].values.reshape(-1, 1), d["is_weekend"].values.reshape(-1, 1)]).reshape(len(d), -1)
    d["pl_dec"] = pd.qcut(d["pred_load"], 10, labels=False)
    loadlv = dummies(d["pl_dec"].rename("p"))

    groups = [("外部预测误差(ext_error)", ext), ("时间规律(时/周/月)", temporal),
              ("天气(温/晴/辐)", weather), ("节假日与周末", hol), ("负荷水平(pl十分位)", loadlv)]
    seq_r2 = []
    cumX = np.empty((len(d), 0))
    cum = 0.0
    for nm, X in groups:
        cumX = np.column_stack([cumX, X])
        r2 = ols_r2(y, cumX)
        seq_r2.append({"因子组": nm, "累计R²": r2, "边际R²增量": r2 - cum, "方差解释%": (r2 - cum) * 100})
        cum = r2
    total_r2 = cum
    seq_r2.append({"因子组": "随机残差", "累计R²": 1.0, "边际R²增量": 1.0 - total_r2, "方差解释%": (1.0 - total_r2) * 100})
    t = pd.DataFrame(seq_r2).set_index("因子组")
    save_table(t, "10_decomposition")
    print(t.to_string(), flush=True)
    print(f"总可解释 R²={total_r2:.3f}  随机残差占={(1-total_r2)*100:.1f}%", flush=True)

    # 模型对外部误差的校正有效率
    ext_abs = d["ext_error"].abs()
    e_abs = d["abs_error"]
    removed = (ext_abs - e_abs)
    print(f"外部预测 MAE={ext_abs.mean():.0f} -> 模型后 MAE={e_abs.mean():.0f}  "
          f"校正有效率={removed.mean()/ext_abs.mean()*100:.1f}%", flush=True)
    print(f"corr(e, ext_error)={np.corrcoef(e, d['ext_error'])[0,1]:.3f}  "
          f"corr(e, model_corr)={np.corrcoef(e, d['model_corr'])[0,1]:.3f}", flush=True)

    # 占比饼图
    fig, ax = plt.subplots(figsize=(8, 8))
    labels = [r["因子组"] for r in seq_r2]
    sizes = [max(r["边际R²增量"], 0) for r in seq_r2]
    colors = ["tab:orange", "tab:blue", "tab:green", "tab:purple", "tab:brown", "tab:gray"]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
    ax.set_title(f"误差方差来源拆解 (总可解释R²={total_r2:.3f})")
    save_fig(fig, "10_decomposition")


if __name__ == "__main__":
    main()
