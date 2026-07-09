# -*- coding: utf-8 -*-
"""FDS/a13_learnable.py - (十三) 剩余可学习信息分析。

判断 v6 残差是否仍有可学习规律。核心区分：
- 可从特征学习 (calendar/weather/pl_load/pl_wr 等 predict-time 可得特征) 的结构；
- 不可学习 = 外部预测的特异性误差 (ext_error，predict-time 不可得) + 纯随机。
方法：结构特征 OLS R²、逐日 oracle 上界、val vs train-OOF 结构对比、ACF 解读。
不训练残差模型（避免"为降MAE而实验"），仅做结构化 R² 诊断。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from diag_lib import load_val, load_oof, acf, save_fig, save_table


def ols_r2(y, X):
    if X.shape[1] == 0:
        return 0.0
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    return 1.0 - np.var(resid) / np.var(y)


def dummies(s):
    return pd.get_dummies(s, prefix=str(s.name), drop_first=True).astype(float).values


def main():
    d = load_val().dropna().copy()
    e = d["error"].values
    y = e - e.mean()
    print("== (十三) 剩余可学习信息 ==", flush=True)

    # 结构特征组（predict-time 可得，不含 ext_error）
    d["temp_bin"] = pd.cut(d["temp"], [-99, 0, 8, 15, 22, 28, 99], labels=False, include_lowest=True)
    d["clr_bin"] = pd.cut(d["clearness"], [-0.01, 0.2, 0.4, 0.6, 0.8, 1.01], labels=False, include_lowest=True)
    d["irr_bin"] = pd.cut(d["irrad"], [-1, 1, 200, 400, 600, 800, 1500], labels=False, include_lowest=True)
    d["pl_dec"] = pd.qcut(d["pred_load"], 10, labels=False)
    groups = {
        "时/周/月": np.column_stack([dummies(d["hour"].rename("h")), dummies(d["dow"].rename("d")), dummies(d["month"].rename("m"))]),
        "天气(温/晴/辐)": np.column_stack([dummies(d["temp_bin"].rename("t")), dummies(d["clr_bin"].rename("c")), dummies(d["irr_bin"].rename("i"))]),
        "节假日/周末": np.column_stack([d["is_holiday"].values, d["is_weekend"].values]).astype(float),
        "负荷水平": dummies(d["pl_dec"].rename("p")),
        "pl_weather_residual": d["pl_weather_residual"].values.reshape(-1, 1),
        "solar_mismatch": d["solar_mismatch"].values.reshape(-1, 1),
    }
    # 各组单独 R²（v6 已用这些，残差上应小）+ 全部合起来 R²
    rows = []
    cumX = np.empty((len(d), 0))
    cum = 0.0
    for nm, X in groups.items():
        cumX = np.column_stack([cumX, X])
        r2_alone = ols_r2(y, X)
        r2_cum = ols_r2(y, cumX)
        rows.append({"特征组": nm, "单独R²": r2_alone, "累计R²": r2_cum})
        cum = r2_cum
    total_struct_r2 = cum
    ext_r2 = ols_r2(y, d["ext_error"].values.reshape(-1, 1))
    rows.append({"特征组": "ext_error(不可作特征,参考)", "单独R²": ext_r2, "累计R²": float("nan")})
    rows.append({"特征组": "随机/不可学习(1-结构R²)", "单独R²": 1 - total_struct_r2, "累计R²": float("nan")})
    t = pd.DataFrame(rows).set_index("特征组")
    save_table(t, "13_learnable_r2")
    print(t.round(4).to_string(), flush=True)
    print(f"结构特征总可解释 R²={total_struct_r2:.3f}  (v6 已捕获大部分; 残余为未捕获/不可迁移)", flush=True)
    print(f"ext_error 单独 R²={ext_r2:.3f}  (外部预测误差, predict-time 不可得 -> 不可学习)", flush=True)
    print(f"随机+不可学习占比={(1-total_struct_r2)*100:.1f}%", flush=True)

    # 逐日 oracle 上界（per-day 均值偏置已知时 MAE）
    daily_bias = d["error"].resample("D").mean()
    d["day_bias"] = d.index.normalize().map(daily_bias)
    e_oracle = d["error"] - d["day_bias"]
    oracle_mae = float(e_oracle.abs().mean())
    print(f"逐日 oracle (已知当日均值偏置): MAE {d['abs_error'].mean():.0f} -> {oracle_mae:.0f} "
          f"(日级上限, 但 exp70/73 证明不可迁移)", flush=True)

    # val vs train-OOF 结构对比：小时 R²
    oof = load_oof()
    r2_hour_val = ols_r2(y, dummies(d["hour"].rename("h")))
    eoof = oof["error"].values
    r2_hour_oof = ols_r2(eoof - eoof.mean(), dummies(oof["hour"].rename("h")))
    print(f"小时结构 R²: train-OOF(未校正)={r2_hour_oof:.3f}  vs val(v6校正后)={r2_hour_val:.3f}  "
          f"(校正应使 val 更小)", flush=True)

    # ACF 解读
    ac = acf(e, 200)
    print(f"ACF lag1={ac[1]:.3f}(目标平滑性,非可操作) lag96={ac[96]:.3f}(日级) "
          f"lag192={ac[192]:.3f}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    bars = [r["单独R²"] for r in rows[:-2]]
    labels = [r["特征组"] for r in rows[:-2]]
    ax.barh(range(len(bars)), bars, color="tab:blue")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(total_struct_r2, color="r", ls="--", label=f"结构总R²={total_struct_r2:.3f}")
    ax.set_title("残差在各结构特征组上的单独 R² (v6 残余)"); ax.legend()
    ax = axes[1]
    daily_bias.plot(ax=ax, kind="bar", color="tab:gray")
    ax.set_title("逐日均值偏置 (日级特异性误差, 不可迁移)")
    ax.set_ylabel("日均 Bias (MW)"); ax.tick_params(axis="x", labelsize=6, rotation=45)
    fig.tight_layout()
    save_fig(fig, "13_learnable")


if __name__ == "__main__":
    main()
