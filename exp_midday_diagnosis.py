# -*- coding: utf-8 -*-
"""exp_midday_diagnosis.py - 午间(11:00-14:00)误差专项诊断体系（只读诊断）。

强制约束（与 /goal 一致）：
  - 不修改任何生产代码 / v6 模型结构 / 训练流程 / 数据泄露约束。
  - 不增加新特征；不调整 threshold_corr / drift_corr / MOS 任何参数。
  - 不使用官方验证窗口重新调参。
  - 所有分析仅基于现有预测结果、真实值、已有特征。
  - 任何候选改进方向仅为诊断后候选，不代表立即实施；须经理论分析+泄露检查+
    walk-forward 跨折验证+独立实验确认稳定收益后方可考虑入生产。

数据源（只读）：
  FDS/output/diag_val.csv  v6 模型在官方验证窗 2026/03/01-06/15 的无泄露预测 + 全诊断列
  FDS/output/diag_oof.csv  训练期 3 折 walk-forward OOF（2025 春/秋 + 2026 冬），用于跨年对比
误差符号约定：error = pred - actual（正=高估，负=低估）；ext_error = pred_load - actual。
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
# 复用 FDS 诊断库：导入即配置 CJK 字体；借用其纯函数 acf/pacf_yw/metrics
from FDS import diag_lib as DL

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "FDS_midday"
FIG = OUT / "figures"; TBL = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True); TBL.mkdir(parents=True, exist_ok=True)
VAL_CSV = ROOT / "FDS" / "output" / "diag_val.csv"
OOF_CSV = ROOT / "FDS" / "output" / "diag_oof.csv"

MID_HOURS = [11, 12, 13]  # 11:00-13:59 = "11-14 时段"
MID_SLOTS = list(range(44, 56))  # slot 44=11:00 ... 55=13:45


# --------------------------------------------------------------------------- #
def _load(path: Path) -> pd.DataFrame:
    head = pd.read_csv(path, encoding="utf-8-sig", nrows=0).columns
    return pd.read_csv(path, encoding="utf-8-sig", parse_dates=[head[0]]).set_index(head[0])


def save_fig(fig, name):
    p = FIG / f"{name}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  [fig] {p.name}", flush=True)


def save_tbl(df: pd.DataFrame, name: str):
    p = TBL / f"{name}.csv"
    df.to_csv(p, encoding="utf-8-sig", float_format="%.4f")
    print(f"  [tbl] {p.name}", flush=True)


def met(e, a=None):
    m = DL.metrics(np.asarray(e, float), None if a is None else np.asarray(a, float))
    return m  # {MAE,RMSE,Bias,std,N,MAPE?}


def season_of(m: int) -> str:
    return {12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春", 6: "夏",
            7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋"}[m]


def shade_mid(ax, ymax=None):
    ax.axvspan(11, 14, color="orange", alpha=0.12)


# --------------------------------------------------------------------------- #
val = _load(VAL_CSV)
oof = _load(OOF_CSV)
for df in (val, oof):
    df["hour"] = df["hour"].astype(int)
    df["slot"] = df["slot"].astype(int)
val["month"] = val["month"].astype(int)
oof["month"] = oof["month"].astype(int)
val["season"] = [season_of(m) for m in val["month"].values]
oof["season"] = [season_of(m) for m in oof["month"].values]
val["model_corr"] = val["pred"] - val["pred_load"]
oof["model_corr"] = oof["pred"] - oof["pred_load"]  # = error - ext_error
mid = val["hour"].isin(MID_HOURS)
print(f"VAL N={len(val)}  MAE={val['abs_error'].mean():.2f}  "
      f"Bias={val['error'].mean():.2f}  RMSE={np.sqrt((val['error']**2).mean()):.2f}", flush=True)
print(f"MID N={mid.sum()}  MAE={val.loc[mid,'abs_error'].mean():.2f}  "
      f"Bias={val.loc[mid,'error'].mean():.2f}\n", flush=True)


# ============================= 第一部分：午间问题是否存在 ============================= #
print("=" * 70 + "\n第一部分：午间(11-14) vs 全天及其它时段\n" + "=" * 70, flush=True)
bands = [("全天", np.ones(len(val), dtype=bool)),
         ("夜间 23-06", val["hour"].isin(list(range(23, 24)) + list(range(0, 6)))),
         ("早高峰 06-11", val["hour"].between(6, 10)),
         ("午间 11-14", mid),
         ("下午 14-18", val["hour"].between(14, 17)),
         ("晚高峰 18-23", val["hour"].between(18, 22))]
rows = []
for name, mask in bands:
    g = val[mask]
    m = met(g["error"].values, g["actual"].values)
    m["时段"] = name
    rows.append(m)
p1 = pd.DataFrame(rows).set_index("时段")[["N", "MAE", "RMSE", "Bias", "MAPE", "std"]]
print(p1.round(2).to_string(), flush=True)
save_tbl(p1, "01_midday_vs_bands")
all_mae = val["abs_error"].mean()
mid_mae = val.loc[mid, "abs_error"].mean()
print(f"\n=> 午间 MAE {mid_mae:.1f} vs 全天 {all_mae:.1f} = 全天的 {mid_mae/all_mae*100:.0f}%；"
      f"午间 Bias {val.loc[mid,'error'].mean():.1f} vs 全天 {val['error'].mean():.1f}", flush=True)

# ============================= 第二部分：小时级 / 96-slot 误差剖面 ============================= #
print("\n" + "=" * 70 + "\n第二部分：小时级 & 96-slot 误差剖面\n" + "=" * 70, flush=True)
by_h = []
for h in range(24):
    g = val[val["hour"] == h]
    m = met(g["error"].values, g["actual"].values)
    m["hour"] = h
    by_h.append(m)
p2h = pd.DataFrame(by_h).set_index("hour")[["N", "MAE", "RMSE", "Bias", "MAPE", "std"]]
save_tbl(p2h, "02_by_hour")

by_s = []
for s in range(96):
    g = val[val["slot"] == s]
    m = met(g["error"].values, g["actual"].values)
    m["slot"] = s
    by_s.append(m)
p2s = pd.DataFrame(by_s).set_index("slot")[["N", "MAE", "Bias", "std"]]
save_tbl(p2s, "02_by_slot")
worst10 = p2s.sort_values("MAE", ascending=False).head(10)
print("最差 10 个 slot（按 MAE）：", flush=True)
print(worst10.round(2).to_string(), flush=True)
save_tbl(worst10, "02_worst10_slots")

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
axes[0].bar(p2h.index, p2h["MAE"], color="steelblue"); shade_mid(axes[0])
axes[0].set_ylabel("MAE"); axes[0].set_title("Hour-level Error Profile (val 2026/03-06)")
axes[1].bar(p2h.index, p2h["RMSE"], color="darkorange"); shade_mid(axes[1]); axes[1].set_ylabel("RMSE")
axes[2].bar(p2h.index, p2h["Bias"], color="seagreen"); shade_mid(axes[2])
axes[2].axhline(0, color="k", lw=0.8); axes[2].set_ylabel("Bias (pred-actual)"); axes[2].set_xlabel("hour")
save_fig(fig, "02_hour_profile")
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
for ax, c, col in zip(axes, ["steelblue", "seagreen", "darkorange"], ["MAE", "Bias", "std"]):
    ax.bar(p2s.index, p2s[col], color=c)
    for s in MID_SLOTS:
        ax.axvspan(s - 0.5, s + 0.5, color="orange", alpha=0.12)
    ax.set_ylabel(col)
axes[2].set_xlabel("slot (0-95)"); axes[0].set_title("96-slot Error Profile (橙色=午间 11-14)")
save_fig(fig, "02_slot_profile")


# ============================= 第三部分：午间预测曲线拟合分析 ============================= #
print("\n" + "=" * 70 + "\n第三部分：午间(11-14)曲线拟合\n" + "=" * 70, flush=True)
# 选午间误差最大的一周做折线
vmid = val[mid].copy()
daily_mid_err = vmid["abs_error"].resample("D").mean()
peak_day = daily_mid_err.idxmax()
w0 = peak_day - pd.Timedelta(days=3)
w1 = peak_day + pd.Timedelta(days=3)
week = val.loc[w0:w1]
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes[0, 0].plot(week.index, week["actual"], label="actual", lw=1.3)
axes[0, 0].plot(week.index, week["pred"], label="pred(v6)", lw=1.1, alpha=0.85)
axes[0, 0].plot(week.index, week["pred_load"], label="pred_load(外部)", lw=0.9, alpha=0.6, ls="--")
for d in pd.date_range(w0.normalize(), w1.normalize()):
    for h in (11, 14):
        axes[0, 0].axvline(d + pd.Timedelta(hours=h), color="orange", alpha=0.25, lw=0.7)
axes[0, 0].set_title(f"预测 vs 实际（午间误差最大周 {w0.date()}~{w1.date()}，橙线=11/14时）")
axes[0, 0].legend(fontsize=8); axes[0, 0].set_ylabel("MW")
# 残差曲线（午间，全程）
res_mid = vmid["error"]
axes[0, 1].plot(res_mid.index, res_mid.values, lw=0.4, alpha=0.5, color="gray")
axes[0, 1].plot(res_mid.index, res_mid.rolling(96, min_periods=20).mean(), lw=1.4, color="red", label="滚动均值(96点)")
axes[0, 1].axhline(0, color="k", lw=0.8); axes[0, 1].legend(fontsize=8)
axes[0, 1].set_title("午间残差 pred-actual（红=滚动均值，负=低估）"); axes[0, 1].set_ylabel("MW")
# 散点 pred vs actual
sc = axes[1, 0].scatter(vmid["actual"], vmid["pred"], c=vmid["clearness"], cmap="viridis", s=6, alpha=0.7)
lims = [min(vmid["actual"].min(), vmid["pred"].min()), max(vmid["actual"].max(), vmid["pred"].max())]
axes[1, 0].plot(lims, lims, "k--", lw=0.8); axes[1, 0].set_xlabel("actual"); axes[1, 0].set_ylabel("pred")
axes[1, 0].set_title("午间 pred vs actual（色=clearness）")
plt.colorbar(sc, ax=axes[1, 0], label="clearness")
# 散点 actual vs error（看高负荷压缩）
axes[1, 1].scatter(vmid["actual"], vmid["error"], c=vmid["pred_load"], cmap="coolwarm", s=6, alpha=0.7)
axes[1, 1].axhline(0, color="k", lw=0.8); axes[1, 1].set_xlabel("actual"); axes[1, 1].set_ylabel("error (pred-actual)")
axes[1, 1].set_title("午间 actual vs error（色=pred_load，负=低估）")
save_fig(fig, "03_midday_fit")
# 拟合形态判断
g_hi = vmid[vmid["actual"] >= vmid["actual"].quantile(0.9)]
g_lo = vmid[vmid["actual"] <= vmid["actual"].quantile(0.1)]
print(f"午间整体 Bias={vmid['error'].mean():.1f}（{'低估' if vmid['error'].mean()<0 else '高估'}）", flush=True)
print(f"  高负荷(P90+)点 Bias={g_hi['error'].mean():.1f}  低负荷(P10-)点 Bias={g_lo['error'].mean():.1f}", flush=True)
print(f"  pred-actual 相关={vmid['pred'].corr(vmid['actual']):.3f}  "
      f"pred std/actual std={vmid['pred'].std()/vmid['actual'].std():.3f}（<1=波动不足）", flush=True)


# ============================= 第四部分：Bias 问题诊断 ============================= #
print("\n" + "=" * 70 + "\n第四部分：Bias 诊断（residual=pred-actual）\n" + "=" * 70, flush=True)
def bias_by(col, bins=None, labels=None):
    g = val[mid].copy()
    if bins is not None:
        g["_g"] = pd.cut(g[col], bins=bins, labels=labels, include_lowest=True)
    else:
        g["_g"] = g[col]
    rows = []
    for k, gg in g.groupby("_g", observed=True):
        rows.append({"组": str(k), "N": len(gg), "mean_res": gg["error"].mean(),
                     "MAE": gg["abs_error"].mean(), "std": gg["error"].std()})
    return pd.DataFrame(rows)
p4h = bias_by("hour"); save_tbl(p4h, "04_bias_by_hour")
p4m = bias_by("month"); save_tbl(p4m, "04_bias_by_month")
p4clr = bias_by("clearness", bins=[-0.01, 0.2, 0.5, 0.8, 1.01],
                labels=["<0.2", "0.2-0.5", "0.5-0.8", ">=0.8"]); save_tbl(p4clr, "04_bias_by_clearness")
p4t = bias_by("temp", bins=[-99, 0, 8, 15, 25, 35, 999],
              labels=["<0", "0-8", "8-15", "15-25", "25-35", ">=35"]); save_tbl(p4t, "04_bias_by_temp")
p4p = bias_by("precip", bins=[-0.01, 0.001, 1e9], labels=["无降水", "有降水"]); save_tbl(p4p, "04_bias_by_precip")
print("午间 Bias by hour:\n", p4h.round(1).to_string(index=False), flush=True)
print("午间 Bias by clearness:\n", p4clr.round(1).to_string(index=False), flush=True)
print("午间 Bias by month:\n", p4m.round(1).to_string(index=False), flush=True)


# ============================= 第五部分：clearness / 天气专项（11-14） ============================= #
print("\n" + "=" * 70 + "\n第五部分：午间 clearness 分档表现（含外部 vs 模型）\n" + "=" * 70, flush=True)
g = val[mid].copy()
g["clr_bin"] = pd.cut(g["clearness"], bins=[-0.01, 0.2, 0.5, 0.8, 1.01],
                      labels=["<0.2", "0.2-0.5", "0.5-0.8", ">=0.8"])
rows = []
for k, gg in g.groupby("clr_bin", observed=True):
    rows.append({"clearness": str(k), "N": len(gg),
                 "ext_Bias": gg["ext_error"].mean(), "ext_MAE": gg["ext_error"].abs().mean(),
                 "model_Bias": gg["error"].mean(), "model_MAE": gg["abs_error"].mean(),
                 "RMSE": np.sqrt((gg["error"] ** 2).mean()),
                 "corr": gg["model_corr"].mean()})
p5 = pd.DataFrame(rows)
print(p5.round(1).to_string(index=False), flush=True)
save_tbl(p5, "05_midday_clearness")


# ============================= 第六部分：pred_load 误差拆解（午间） ============================= #
print("\n" + "=" * 70 + "\n第六部分：午间 pred_load 误差拆解（外部 vs 模型校正）\n" + "=" * 70, flush=True)
g = val[mid].copy()
p50 = g["pred_load"].quantile(0.5); p90 = g["pred_load"].quantile(0.9)
g["pl_bin"] = pd.cut(g["pred_load"], bins=[-1e9, p50, p90, 1e9], labels=["P0-P50", "P50-P90", "P90+"])
rows = []
for k, gg in g.groupby("pl_bin", observed=True):
    rows.append({"pred_load档": str(k), "N": len(gg),
                 "ext_Bias(p_l-a)": gg["ext_error"].mean(), "ext_MAE": gg["ext_error"].abs().mean(),
                 "model_Bias(p-a)": gg["error"].mean(), "model_MAE": gg["abs_error"].mean(),
                 "校正量": gg["model_corr"].mean()})
p6 = pd.DataFrame(rows)
print(p6.round(1).to_string(index=False), flush=True)
save_tbl(p6, "06_pred_load_decomp")
print(f"\n午间整体: ext_Bias={g['ext_error'].mean():.1f}  model_Bias={g['error'].mean():.1f}  "
      f"校正量={g['model_corr'].mean():.1f}", flush=True)
print(f"  ext_MAE={g['ext_error'].abs().mean():.1f}  model_MAE={g['abs_error'].mean():.1f}  "
      f"模型相对外部改善={(g['ext_error'].abs()-g['abs_error']).mean():.1f}", flush=True)


# ============================= 第七部分：峰值误差分析（午间） ============================= #
print("\n" + "=" * 70 + "\n第七部分：午间峰值误差（按 pred_load 分位）\n" + "=" * 70, flush=True)
g = val[mid].copy()
qs = [0, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
qlabels = ["<=P50", "P50-75", "P75-90", "P90-95", "P95-99", ">P99"]
g["pq"] = pd.cut(g["pred_load"], bins=np.quantile(g["pred_load"], qs), labels=qlabels, include_lowest=True)
rows = []
for k, gg in g.groupby("pq", observed=True):
    rows.append({"pred_load档": str(k), "N": len(gg), "MAE": gg["abs_error"].mean(),
                 "Bias": gg["error"].mean(), "RMSE": np.sqrt((gg["error"] ** 2).mean()),
                 "ext_Bias": gg["ext_error"].mean()})
p7 = pd.DataFrame(rows)
print(p7.round(1).to_string(index=False), flush=True)
save_tbl(p7, "07_peak_error")


# ============================= 第八部分：残差可学习性（ACF/PACF + 信息耗尽检验） ============================= #
print("\n" + "=" * 70 + "\n第八部分：残差可学习性\n" + "=" * 70, flush=True)
# 全程残差 ACF/PACF（15min 网格，lag 96=1天 192=2天）
full_res = val["error"].values
ac = DL.acf(full_res, nlags=200)
pac = DL.pacf_yw(full_res, nlags=50)
for lag in [1, 4, 96, 192]:
    print(f"  全程残差 ACF(lag={lag})={ac[lag]:.3f}", flush=True)
# 白噪声检验：前 50 阶 ACF 超过 ±1.96/sqrt(N) 的比例
N = len(full_res); band = 1.96 / np.sqrt(N)
n_over = int(np.sum(np.abs(ac[1:51]) > band))
print(f"  前50阶 ACF 超 ±{band:.3f} 的个数: {n_over}/50（白噪声期望~5%）", flush=True)
# 午间日均值残差的自相关（跨日稳定偏置？）
day_mid = val[mid]["error"].resample("D").mean().dropna()
dac = DL.acf(day_mid.values, nlags=14)
print(f"  午间日均残差 ACF(lag1天)={dac[1]:.3f}  ACF(lag7天)={dac[7]:.3f}", flush=True)
fig, axes = plt.subplots(2, 1, figsize=(11, 7))
axes[0].stem(range(len(ac[:97])), ac[:97], basefmt=" ")
axes[0].axhline(band, color="r", ls="--", lw=0.7); axes[0].axhline(-band, color="r", ls="--", lw=0.7)
axes[0].axhline(0, color="k", lw=0.8); axes[0].set_title("全程残差 ACF（lag 0-96，红虚线=95%带）")
axes[0].set_xlabel("lag (15min)"); axes[0].set_ylabel("ACF")
axes[1].stem(range(len(pac[:49])), pac[:49], basefmt=" ")
axes[1].axhline(band, color="r", ls="--", lw=0.7); axes[1].axhline(-band, color="r", ls="--", lw=0.7)
axes[1].axhline(0, color="k", lw=0.8); axes[1].set_title("全程残差 PACF（lag 0-48）")
axes[1].set_xlabel("lag (15min)"); axes[1].set_ylabel("PACF")
save_fig(fig, "08_acf_pacf")
# 信息耗尽检验：用现有特征预测午间残差，时序切分 R²（仅诊断，不入生产）
from sklearn.linear_model import Ridge
feat_cols = [c for c in ["clearness", "temp", "irrad", "precip", "wind", "clear_sky",
                         "pl_weather_residual", "solar_mismatch", "cloud_deficit",
                         "pred_load", "hdd", "cdd", "irrad_anom_672", "pl_dip_96"] if c in val.columns]
vm = val[mid].sort_index().copy()
Xm = vm[feat_cols].fillna(0.0).values
ym = vm["error"].values
n_tr = int(len(vm) * 0.7)
Xtr, Xte = Xm[:n_tr], Xm[n_tr:]; ytr, yte = ym[:n_tr], ym[n_tr:]
base_mae = np.abs(yte - yte.mean()).mean()  # 恒定预测=训练均值(近似)
ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
pr = ridge.predict(Xte)
r2 = 1 - np.sum((yte - pr) ** 2) / np.sum((yte - yte.mean()) ** 2)
print(f"  信息耗尽检验(午间残差~现有特征, 时序7:3): test R²={r2:.3f}  "
      f"恒定MAE={base_mae:.1f} -> Ridge MAE={np.abs(yte-pr).mean():.1f}", flush=True)
coef = pd.Series(ridge.coef_, index=feat_cols).sort_values(key=np.abs, ascending=False)
print("  Ridge top 系数:\n" + coef.round(3).head(8).to_string(), flush=True)


# ============================= 第九部分：午间失败案例 Top20 ============================= #
print("\n" + "=" * 70 + "\n第九部分：午间最大误差 Top20\n" + "=" * 70, flush=True)
g = val[mid].copy()
g["sunny"] = np.where(g["clearness"] >= 0.5, "晴", "阴/多云")
top = g.sort_values("abs_error", ascending=False).head(20)
cols = ["actual", "pred", "pred_load", "temp", "irrad", "clearness", "sunny",
        "ext_error", "error", "abs_error"]
p9 = top[cols].round(1)
print(p9.to_string(), flush=True)
save_tbl(p9, "09_top20_midday")


# ============================= 第十部分：跨年稳定性（OOF 2025 vs val 2026） ============================= #
print("\n" + "=" * 70 + "\n第十部分：跨年稳定性（OOF 2025 春/秋 + 2026 冬  vs  val 2026 春/夏）\n" + "=" * 70, flush=True)
def midstat(df, label):
    g = df[df["hour"].isin(MID_HOURS)]
    return {"来源": label, "N": len(g),
            "model_Bias": g["error"].mean(), "model_MAE": g["abs_error"].mean(),
            "ext_Bias": g["ext_error"].mean(), "ext_MAE": g["ext_error"].abs().mean(),
            "校正量": g["model_corr"].mean()}
rows = []
for se in ["春", "秋", "冬"]:
    rows.append(midstat(oof[oof["season"] == se], f"OOF 2025{se}"))
for se in ["春", "夏"]:
    rows.append(midstat(val[val["season"] == se], f"VAL 2026{se}"))
p10 = pd.DataFrame(rows)
print(p10.round(1).to_string(index=False), flush=True)
save_tbl(p10, "10_cross_year_midday")
# 同季节跨年：OOF 2025春 vs VAL 2026春
o_s = oof[(oof["season"] == "春") & (oof["hour"].isin(MID_HOURS))]
v_s = val[(val["season"] == "春") & (val["hour"].isin(MID_HOURS))]
print(f"\n同季节跨年对比（午间）：", flush=True)
print(f"  OOF 2025春: model_Bias={o_s['error'].mean():.1f}  ext_Bias={o_s['ext_error'].mean():.1f}", flush=True)
print(f"  VAL 2026春: model_Bias={v_s['error'].mean():.1f}  ext_Bias={v_s['ext_error'].mean():.1f}", flush=True)
# 各小时跨年 Bias 稳定性
print("\n午间逐小时跨年 model_Bias（OOF2025春 vs VAL2026春）：", flush=True)
for h in MID_HOURS:
    ob = oof[(oof["season"] == "春") & (oof["hour"] == h)]["error"].mean()
    vb = val[(val["season"] == "春") & (val["hour"] == h)]["error"].mean()
    print(f"  hour {h}: OOF={ob:+.1f}  VAL={vb:+.1f}  Δ={vb-ob:+.1f}", flush=True)

print("\nexp_midday_diagnosis done. 输出目录:", OUT, flush=True)
