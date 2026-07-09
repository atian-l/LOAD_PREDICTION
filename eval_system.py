# -*- coding: utf-8 -*-
"""eval_system.py - 完善的评价体系（只读诊断，不修改生产代码/模型/特征）。

在 v6 官方验证窗预测（FDS/output/diag_val.csv）基础上输出：
  1. 分时 MAE（00-06/06-11/11-14/14-18/18-24）
  2. 分天气 MAE（晴/多云/阴雨 + 高温/低温/正常）及 分时×天气 矩阵
  3. 新能源敏感：当前无新能源出力数据 -> 仅标注，不可分析
  4. 可视化：hour bias 曲线 / actual vs pred / residual 时序 / residual ACF / TOP 误差日期
产物：eval_system/（图+表）。合规：actual 仅评估目标（#1）；不新增特征、不改训练/模型。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from FDS import diag_lib as DL

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "eval_system"; FIG = OUT / "figures"; TBL = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True); TBL.mkdir(parents=True, exist_ok=True)
VAL_CSV = ROOT / "FDS" / "output" / "diag_val.csv"


def _load(path):
    head = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns
    return pd.read_csv(path, encoding="utf-8-sig", parse_dates=[head[0]]).set_index(head[0])


def save_fig(fig, name):
    p = FIG / f"{name}.png"; fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  [fig] {p.name}", flush=True)


def save_tbl(df, name):
    p = TBL / f"{name}.csv"; df.to_csv(p, encoding="utf-8-sig", float_format="%.4f")
    print(f"  [tbl] {p.name}", flush=True)


d = _load(VAL_CSV)
d["hour"] = d["hour"].astype(int)
d["slot"] = d["slot"].astype(int)
print(f"VAL N={len(d)} MAE={d['abs_error'].mean():.2f} Bias={d['error'].mean():.2f}", flush=True)

# ---- 分时 ----
BANDS = [("00-06", d["hour"].between(0, 5)),
         ("06-11", d["hour"].between(6, 10)),
         ("11-14", d["hour"].between(11, 13)),
         ("14-18", d["hour"].between(14, 17)),
         ("18-24", d["hour"].between(18, 23))]

# ---- 天气类型（互斥）----
has_precip = "precip" in d.columns
precip = d["precip"].fillna(0).values if has_precip else np.zeros(len(d))
clr = d["clearness"].fillna(0.5).values
wtype = np.where(precip > 0, "阴雨", np.where(clr >= 0.8, "晴", "多云"))
d["wtype"] = wtype
# 温度极端（与天气类型交叉，独立标注）
temp = d["temp"].fillna(15).values
d["ttype"] = np.where(temp >= 30, "高温(>=30)", np.where(temp < 8, "低温(<8)", "正常"))


def stat(mask):
    g = d[mask]
    return {"N": len(g), "MAE": g["abs_error"].mean(), "Bias": g["error"].mean(),
            "RMSE": np.sqrt((g["error"] ** 2).mean())}


# 1. 分时
print("\n=== 分时 MAE ===", flush=True)
rows = [dict({"时段": n}, **stat(m)) for n, m in BANDS]
p_band = pd.DataFrame(rows).set_index("时段")
print(p_band.round(1).to_string(), flush=True)
save_tbl(p_band, "01_by_band")

# 2. 分天气
print("\n=== 分天气 MAE ===", flush=True)
rows = [{"天气": "全天", **stat(np.ones(len(d), bool))}]
for w in ["晴", "多云", "阴雨"]:
    rows.append({"天气": w, **stat(d["wtype"] == w)})
for t in ["高温(>=30)", "低温(<8)", "正常"]:
    rows.append({"温度": t, **stat(d["ttype"] == t)})
p_wx = pd.DataFrame(rows).fillna("")
print(p_wx.round(1).to_string(index=False), flush=True)
save_tbl(p_wx, "02_by_weather")

# 分时 × 天气类型 矩阵
print("\n=== 分时×天气 MAE 矩阵 ===", flush=True)
mat = pd.DataFrame(index=[n for n, _ in BANDS], columns=["晴", "多云", "阴雨"], dtype=float)
mat_n = pd.DataFrame(index=[n for n, _ in BANDS], columns=["晴", "多云", "阴雨"], dtype=int)
for n, m in BANDS:
    for w in ["晴", "多云", "阴雨"]:
        mm = m & (d["wtype"] == w)
        mat.loc[n, w] = d.loc[mm, "abs_error"].mean()
        mat_n.loc[n, w] = int(mm.sum())
print("MAE:"); print(mat.round(0).to_string(), flush=True)
print("N:"); print(mat_n.to_string(), flush=True)
save_tbl(mat, "03_band_x_weather_mae")
save_tbl(mat_n, "03_band_x_weather_n")

# 4. 可视化
# hour bias 曲线
by_h = d.groupby("hour")["error"].agg(["mean", "std", lambda x: x.abs().mean()]).rename(
    columns={"mean": "Bias", "<lambda_0>": "MAE"})
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(by_h.index, by_h["Bias"], color="seagreen")
ax.axhline(0, color="k", lw=0.8); ax.axvspan(11, 14, color="orange", alpha=0.12)
ax.set_title("Hour-level Bias (pred-actual)；橙=午间"); ax.set_xlabel("hour"); ax.set_ylabel("Bias")
save_fig(fig, "04_hour_bias")

# actual vs pred（全天 + 午间）
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, mask, title in [(axes[0], np.ones(len(d), bool), "全天"),
                        (axes[1], d["hour"].between(11, 13), "午间 11-14")]:
    g = d[mask]
    ax.scatter(g["actual"], g["pred"], c=g["clearness"], cmap="viridis", s=5, alpha=0.6)
    lims = [min(g["actual"].min(), g["pred"].min()), max(g["actual"].max(), g["pred"].max())]
    ax.plot(lims, lims, "k--", lw=0.8); ax.set_xlabel("actual"); ax.set_ylabel("pred"); ax.set_title(title)
save_fig(fig, "05_actual_vs_pred")

# residual 时序（全程 + 午间滚动）
fig, axes = plt.subplots(2, 1, figsize=(13, 7))
axes[0].plot(d.index, d["error"].values, lw=0.3, alpha=0.5, color="gray")
axes[0].plot(d.index, d["error"].rolling(96 * 7, min_periods=96).mean(), lw=1.2, color="red")
axes[0].axhline(0, color="k", lw=0.8); axes[0].set_title("残差时序 pred-actual（红=7日滚动均值）"); axes[0].set_ylabel("MW")
mid = d[d["hour"].between(11, 13)]
axes[1].plot(mid.index, mid["error"].values, lw=0.4, alpha=0.5, color="gray")
axes[1].plot(mid.index, mid["error"].rolling(96, min_periods=20).mean(), lw=1.3, color="red")
axes[1].axhline(0, color="k", lw=0.8); axes[1].set_title("午间残差时序（红=滚动均值）"); axes[1].set_ylabel("MW")
save_fig(fig, "06_residual_ts")

# residual ACF
ac = DL.acf(d["error"].values, nlags=200)
band = 1.96 / np.sqrt(len(d))
fig, ax = plt.subplots(figsize=(11, 4))
ax.stem(range(97), ac[:97], basefmt=" ")
ax.axhline(band, color="r", ls="--", lw=0.7); ax.axhline(-band, color="r", ls="--", lw=0.7)
ax.axhline(0, color="k", lw=0.8); ax.set_title("全程残差 ACF（lag0-96，红虚线=95%带）")
ax.set_xlabel("lag (15min)"); ax.set_ylabel("ACF")
save_fig(fig, "07_residual_acf")

# TOP 误差日期（按日绝对误差排序）
day_err = d.groupby(d.index.normalize())["abs_error"].mean().sort_values(ascending=False)
top_days = day_err.head(10)
print("\n=== TOP10 误差日（日均 abs_error）===", flush=True)
print(top_days.round(1).to_string(), flush=True)
save_tbl(top_days.reset_index().rename(columns={"index": "date", "abs_error": "daily_mean_abs_error"}),
         "08_top10_error_days")
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(range(len(top_days)), top_days.values, color="firebrick")
ax.set_xticks(range(len(top_days))); ax.set_xticklabels([t.strftime("%m-%d") for t in top_days.index], rotation=45)
ax.set_ylabel("日均 abs_error"); ax.set_title("TOP10 误差日")
save_fig(fig, "08_top_error_days")

print("\n=== 新能源敏感分析 ===", flush=True)
print("当前 data/ 仅有 direct_load_latest.csv 与 shandong_weather_15min.csv，无新能源(光伏/风电)出力数据。", flush=True)
print("-> 高新能源占比/高新能源爬坡拆分不可执行；须等待新增新能源出力数据。", flush=True)
print("\neval_system done. 产物目录:", OUT, flush=True)
