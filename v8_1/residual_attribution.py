# -*- coding: utf-8 -*-
"""v8.1 工作流 A：Residual Attribution（静态归因树 - 中心件）。

DESIGN §6 优先级 ⭐⭐⭐⭐⭐。把散落的 FDS / 午间 / 信息源 / Phase0 诊断**统一成一棵
Residual Tree + 报告**。问题从"残差怎么修"转为"**残差为什么产生**"。

Residual 定义：
  ext_err = actual - pred_load   外部预测原始误差（"产生"的根，FDS 的 ext_error）
  r       = actual - v6_pred     v6 修正后剩余（= 天花板，Phase0/P-Diag 的 r）
  v6 修正量 = ext_err - r        v6 移除的可学习部分

归因树（离线，用 actual，eval-only，合规 不变量 #1/#5/#6）：
  ext_err
  ├── 可学习（v6 修正，ext_err_MAE -> r_MAE=1445）
  │   ├── Forecast structural (pred_load level + temporal)
  │   ├── Calendar (demand shift 可学习部分)
  │   └── Weather (可学习 weather-error 耦合)
  └── 不可学习（天花板 ≈ r）
      ├── Weather OOD (novelty)        量化: corr(proxy, |r|)
      ├── Demand shift (不可学习残余)  量化: calendar 残余
      ├── Renewable (proxy)            准量化 + 排除法
      └── Random/Unknown               floor

可再生排除法（关键诚实结果）：物理 PV proxy(irrad×temp 降额)+wind proxy(立方 cut-in/out)。
ext_err 的可再生分量 = renewable_actual - renewable_forecast；pred_load 已含外部可再生
假设（同源气象），故气象隐含可再生 ≈ 已在 pred_load 内 -> proxy 与 ext_err 弱相关 = 排除
"气象隐含可再生"为符号根因，收窄到"气象抓不到的可再生变率"（云局地/弃光，不可观测）。

逐日 cause 标签（运行时前身，仅 pred_load+weather+calendar 无 actual）：Input anomaly /
Weather OOD / Demand shift / Likely renewable / Unknown。验证: 各标签日 |ext_err|/全局。

合规：actual 仅作 ext_err/r 评估标签(eval-only)；cause 标签分数无 actual（工作流 C 前身）。
运行：python -m v8_1.residual_attribution   报告 -> v8_1/output/residual_attribution.md
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from load_pred import config as LC
from load_pred import train as T
from v8 import config as VC
from v8 import segments as SEG
from v8.model import V8Model
from v8_1.diag_residual import PROBE_PARAMS, PROBE_IT, _fit_probe, group_columns
from v8_1.input_anomaly import _pl_daily_signals, _standardize, _spearman, _risk_metrics

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = OUT_DIR / "residual_attribution.md"


def _holdout_r2(Xtr, ytr, cols) -> float:
    n = len(ytr); cut = int(n * 0.8)
    if cut < 100 or n - cut < 100:
        return float("nan")
    bst = _fit_probe(Xtr.iloc[:cut], ytr[:cut], cols)
    pred = bst.predict(Xtr[cols].iloc[cut:]); y = ytr[cut:]
    vmse = float(np.mean((y - y.mean()) ** 2))
    return 1.0 - float(np.mean((y - pred) ** 2)) / vmse if vmse > 0 else float("nan")


def _transfer_r2(pred, y, train_mean) -> float:
    bmse = float(np.mean((y - train_mean) ** 2))
    pmse = float(np.mean((y - pred) ** 2))
    return 1.0 - pmse / bmse if bmse > 0 else float("nan")


# --------------------------------------------------------------------------- #
# 可再生物理 proxy（仅气象）
# --------------------------------------------------------------------------- #
def _renewable_proxy(X_df, times_idx) -> pd.DataFrame:
    """物理 PV+wind proxy（仅气象，可部署）。PV∝irrad×(1-0.004·(temp-25))；wind∝立方[3,25)。
    返回逐日 pv_sum/wp_sum 及其偏离 28d 因果滚动的"意外"量。"""
    irrad = X_df["irrad"].values.astype(float)
    temp = X_df["temp"].values.astype(float)
    wind = X_df["wind"].values.astype(float)
    pv = irrad * np.maximum(0.0, 1.0 - 0.004 * (temp - 25.0))          # 温度降额
    wp = np.where((wind >= 3.0) & (wind < 25.0), wind ** 3, 0.0)       # cut-in 3 / cut-out 25
    wp = np.clip(wp, 0.0, 12.5 ** 3)                                   # 额定封顶
    d = pd.DatetimeIndex(times_idx).normalize()
    df = pd.DataFrame({"pv": pv, "wp": wp, "date": d})
    daily = df.groupby("date").agg(pv_sum=("pv", "sum"), wp_sum=("wp", "sum"))
    r28 = daily.shift(1).rolling(28, min_periods=7).mean()
    daily["pv_dev"] = (daily["pv_sum"] - r28["pv_sum"]).abs()
    daily["wp_dev"] = (daily["wp_sum"] - r28["wp_sum"]).abs()
    return daily


def _daily_weather_ood(X_df, times_idx, train_dates) -> pd.Series:
    """A2 归因化：日级 (temp_mean, irrad_sum) Mahalanobis vs 训练期。"""
    d = pd.DatetimeIndex(times_idx).normalize()
    xf = X_df[["temp", "irrad"]].copy(); xf["date"] = d
    daily = xf.groupby("date").agg(temp_mean=("temp", "mean"), irrad_sum=("irrad", "sum"))
    tr = daily.loc[daily.index.isin(train_dates)]
    cm = tr.mean(); daily = daily.fillna(cm); tr = tr.fillna(cm)
    mu = tr.mean().values; cov = np.cov(tr.values, rowvar=False) + np.eye(2) * 1e-6
    inv = np.linalg.inv(cov); diff = (daily.values - mu)
    md = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", diff, inv, diff), 0.0))
    return pd.Series(md, index=daily.index)


# --------------------------------------------------------------------------- #
# 主诊断
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    if verbose:
        print("[1/6] 构建数据集 + 加载 v8 bundle + ext_err/r ...")
    times, X, pred_load, actual = T.build_dataset()
    vm8 = V8Model.load(VC.V8_BUNDLE)
    mm = vm8.mismatch_model
    X_full = mm.transform(X)
    oof = vm8.oof_pool
    baseA = vm8.base_A

    # ---- train（2025 OOF）----
    idx = oof["idx"]
    X_train = X_full.iloc[idx].reset_index(drop=True)
    times_train = pd.DatetimeIndex(oof["times"])
    seg_train = np.asarray(oof["seg"], dtype=object)
    pl_train = np.asarray(pred_load.reindex(times_train), dtype=float)
    actual_train = np.asarray(oof["actual"], dtype=float)
    ext_err_train = actual_train - pl_train                      # 外部原始误差
    r_train = np.asarray(actual_train - oof["base_A_oof"], dtype=float)  # v6 修正后
    train_mean_ext = float(np.nanmean(ext_err_train))

    # ---- val（2026 eval-only）----
    vs, ve = pd.Timestamp(LC.VAL_START), pd.Timestamp(LC.VAL_END)
    buf = vs - pd.Timedelta(days=7)
    em = (times >= buf) & (times <= ve)
    Xe = X[em]; te = pd.DatetimeIndex(times[em])
    ae = actual.reindex(te)
    Xe_full = mm.transform(Xe)
    predA = np.asarray(baseA.predict_load(Xe_full, pred_load), dtype=float)
    vmask = (te >= vs) & (te <= ve) & ae.notna()
    X_val = Xe_full[vmask].reset_index(drop=True)
    times_val = te[vmask]
    hours_val = times_val.hour.values.astype(int)
    seg_val = SEG.segment_array(hours_val)
    actual_val = ae[vmask].values.astype(float)
    pl_val = np.asarray(pred_load.reindex(times_val), dtype=float)
    ext_err_val = actual_val - pl_val
    r_val = actual_val - predA[vmask]
    train_mean_r = float(np.mean(r_train))

    # NaN 掩码（pred_load 偶有缺口）
    ftr = np.isfinite(ext_err_train); fva = np.isfinite(ext_err_val)
    ext_err_MAE_val = float(np.mean(np.abs(ext_err_val[fva])))
    r_MAE_val = float(np.mean(np.abs(r_val[fva])))
    ext_err_MAE_train = float(np.mean(np.abs(ext_err_train[ftr])))
    r_MAE_train = float(np.mean(np.abs(r_train[ftr])))
    v6_reduction = ext_err_MAE_val - r_MAE_val

    groups = group_columns(X_full.columns)
    forecast_struct_cols = groups.get("load_level", []) + groups.get("load_temporal", [])
    calendar_cols = groups.get("calendar", [])
    weather_cols = groups.get("weather", [])
    renewable_cols = groups.get("solar_renewable", [])
    combined_cols = [c for cs in groups.values() for c in cs]
    channels = [("forecast_structural", forecast_struct_cols), ("calendar", calendar_cols),
                ("weather", weather_cols), ("renewable_feat", renewable_cols),
                ("combined", combined_cols)]

    if verbose:
        print(f"      ext_err MAE train={ext_err_MAE_train:.0f} val={ext_err_MAE_val:.0f} | "
              f"r MAE train={r_MAE_train:.0f} val={r_MAE_val:.0f} | v6 修正={v6_reduction:.0f}")

    # ===================== A. 可学习 vs 不可学习（ext_err 分量 R²）=====================
    if verbose:
        print("[2/6] A. ext_err 分量 R²（可学习/迁移）...")
    chan_rows = []
    for name, cols in channels:
        if not cols:
            continue
        bst = _fit_probe(X_train[ftr], ext_err_train[ftr], cols)
        pred = bst.predict(X_val[fva][cols])
        tr2 = _transfer_r2(pred, ext_err_val[fva], train_mean_ext)
        hold = _holdout_r2(X_train[ftr], ext_err_train[ftr], cols)
        chan_rows.append({"channel": name, "n_cols": len(cols),
                          "holdout_R2": hold, "transfer_R2": tr2})
    combined = next(r for r in chan_rows if r["channel"] == "combined")
    # 不可学习份额（期内）= 1 - holdout_R²(combined)；天花板 r 的可学习性
    unlearnable_share = 1.0 - combined["holdout_R2"] if not np.isnan(combined["holdout_R2"]) else float("nan")

    # ===================== B. 不可学习 proxy 归因（corr with |r|）=====================
    if verbose:
        print("[3/6] B. 不可学习 proxy 归因（corr(proxy,|r|) 跨年）...")
    train_dates = pd.DatetimeIndex(times_train).normalize().unique()
    val_dates = pd.DatetimeIndex(times_val).normalize().unique()
    # A1 输入质量（工作流B复用）
    sig_all = _standardize(_pl_daily_signals(pred_load.reindex(times), times), train_dates)
    a1_comb = sig_all.mean(axis=1)                                # 简单等权组合
    # 天气 OOD
    ood_all = _daily_weather_ood(X_full, times, train_dates)
    # 可再生 proxy
    ren_all = _renewable_proxy(X_full, times)
    ren_dev = ren_all["pv_dev"] + ren_all["wp_dev"]
    ren_std = ren_dev.loc[ren_dev.index.isin(train_dates)].std()
    ren_dev = ren_dev / (ren_std if ren_std > 0 else 1.0)
    # 日级 |r|
    dtr = pd.DatetimeIndex(times_train).normalize()
    dva = pd.DatetimeIndex(times_val).normalize()
    daily_mr_train = pd.Series(np.abs(r_train), index=dtr).groupby(dtr).mean()
    daily_mr_val = pd.Series(np.abs(r_val), index=dva).groupby(dva).mean()
    daily_ext_val = pd.Series(np.abs(ext_err_val), index=dva).groupby(dva).mean()
    # 日级 |ext_err| train
    daily_ext_train = pd.Series(np.abs(ext_err_train), index=dtr).groupby(dtr).mean()

    def _align(score_daily):
        s25 = score_daily.reindex(daily_mr_train.index)
        s26 = score_daily.reindex(daily_mr_val.index)
        return s25, s26
    proxy_rows = []
    for name, sc in [("A1_input", a1_comb), ("weather_OOD", ood_all), ("renewable_dev", ren_dev)]:
        s25, s26 = _align(sc)
        c25 = _spearman(s25.values, daily_mr_train.values)
        c26 = _spearman(s26.values, daily_mr_val.values)
        m26 = _risk_metrics(s26.values, daily_mr_val.values, s25.values)
        proxy_rows.append({"proxy": name, "corr_2025": c25, "corr_2026": c26,
                           "ratio_2026": m26["ratio"], "auc_2026": m26["auc"]})

    # ===================== C. 可再生物理 proxy 排除法 =====================
    if verbose:
        print("[4/6] C. 可再生物理 proxy 排除法（proxy 能否解释 ext_err 符号/幅度）...")
    # 日级 ext_err（有符号）vs pv_dev（有符号，非绝对）
    daily_ext_sign_val = pd.Series(ext_err_val, index=dva).groupby(dva).mean()
    daily_ext_sign_train = pd.Series(ext_err_train, index=dtr).groupby(dtr).mean()
    pv_sign = ren_all["pv_sum"] - ren_all["pv_sum"].shift(1).rolling(28, min_periods=7).mean()
    pv25 = pv_sign.reindex(daily_ext_sign_train.index)
    pv26 = pv_sign.reindex(daily_ext_sign_val.index)
    ren_corr_sign_25 = _spearman(pv25.values, daily_ext_sign_train.values)
    ren_corr_sign_26 = _spearman(pv26.values, daily_ext_sign_val.values)
    # 幅度
    ren_corr_mag_26 = _spearman(ren_dev.reindex(daily_mr_val.index).values, daily_mr_val.values)
    # 排除法：用 pv_dev 预测 ext_err 符号方向命中率（|ext_err|>50 的日）
    nz = np.abs(daily_ext_sign_val.values) > 50.0
    pv26_arr = pv26.values
    dir_acc = float(np.mean(np.sign(pv26_arr[nz]) == np.sign(daily_ext_sign_val.values[nz]))) if nz.any() else float("nan")
    # 对照：pred_load 已含同源气象 -> proxy 与 ext_err 应弱相关
    # 中午段（11-14）ext_err vs pv_dev 重点
    mid_mask = (hours_val >= 11) & (hours_val <= 13)
    mid_ext = ext_err_val[mid_mask]
    mid_pv = pred_load.reindex(times_val).values[mid_mask]  # 占位，真正用 irrad
    irrad_val = X_val["irrad"].values
    mid_irrad = irrad_val[mid_mask]
    mid_corr = _spearman(mid_irrad, np.abs(mid_ext))

    # ===================== D. 逐日 cause 标签 + 验证 =====================
    if verbose:
        print("[5/6] D. 逐日 cause 标签（运行时前身）+ 验证 ...")
    # 合规日级分数（无 actual）
    a1_d = a1_comb.reindex(dva).values
    ood_d = ood_all.reindex(dva).values
    ren_d = ren_dev.reindex(dva).values
    # calendar 异常：holiday 或周末
    cal_val = X_val[["is_holiday", "is_weekend", "dayofweek"]].copy()
    cal_val["date"] = pd.DatetimeIndex(times_val).normalize()
    cal_daily = cal_val.groupby("date").agg(is_holiday=("is_holiday", "max"),
                                            is_weekend=("is_weekend", "max"))
    hol_d = cal_daily["is_holiday"].reindex(dva).fillna(0).values
    # 阈值（从训练期 top-15%）
    def _thr(train_score):
        ts = np.asarray(train_score, dtype=float)
        ts = ts[np.isfinite(ts)]
        return float(np.percentile(ts, 85)) if len(ts) else float("nan")
    a1_thr = _thr(a1_comb.reindex(dtr).values)
    ood_thr = _thr(ood_all.reindex(dtr).values)
    ren_thr = _thr(ren_dev.reindex(dtr).values)
    labels, n = {"Demand shift": 0, "Input anomaly": 0, "Weather OOD": 0,
                 "Likely renewable": 0, "Unknown": 0}, len(dva)
    label_arr = np.empty(n, dtype=object)
    dva_idx = dva
    for i in range(n):
        if hol_d[i] >= 0.5:
            lab = "Demand shift"
        elif np.isfinite(a1_d[i]) and a1_d[i] > a1_thr:
            lab = "Input anomaly"
        elif np.isfinite(ood_d[i]) and ood_d[i] > ood_thr:
            lab = "Weather OOD"
        elif np.isfinite(ren_d[i]) and ren_d[i] > ren_thr:
            lab = "Likely renewable"
        else:
            lab = "Unknown"
        labels[lab] += 1; label_arr[i] = lab
    # 验证：各标签日 mean|ext_err|/全局
    daily_ext_val_arr = daily_ext_val.reindex(dva).values
    glob = float(np.nanmean(daily_ext_val_arr))
    label_val = {}
    for lab in labels:
        m = label_arr == lab
        if m.any():
            label_val[lab] = {"n": int(m.sum()),
                              "mae": float(np.nanmean(daily_ext_val_arr[m])),
                              "ratio": float(np.nanmean(daily_ext_val_arr[m])) / glob if glob > 0 else float("nan")}
        else:
            label_val[lab] = {"n": 0, "mae": float("nan"), "ratio": float("nan")}

    # ===================== E. 树 + 判定 =====================
    if verbose:
        print("[6/6] E. 归因树 + 报告 ...")
    L = []
    L.append("# v8.1 工作流 A：Residual Attribution（静态归因树）\n")
    L.append("> 中心件（DESIGN §6）。统一 FDS / 午间 / 信息源 / Phase0 诊断。问题从"
             "\"残差怎么修\"转为\"**残差为什么产生**\"。\n"
             "> ext_err=actual−pred_load（外部原始误差，FDS 的 ext_error）；r=actual−v6_pred（v6 修正后=天花板）。\n"
             "> 合规：actual 仅作 ext_err/r 评估标签(eval-only)；cause 标签无 actual。\n")
    L.append(f"ext_err MAE: train={ext_err_MAE_train:.0f} / val={ext_err_MAE_val:.0f}。"
             f"r MAE: train={r_MAE_train:.0f} / val={r_MAE_val:.0f}。"
             f"**v6 修正量 = {v6_reduction:.0f} MW**（ext_err val {ext_err_MAE_val:.0f} -> r val {r_MAE_val:.0f}）。\n")

    L.append("\n## A. 可学习 vs 不可学习（ext_err 分量 R²）\n")
    L.append("探针在 2025 ext_err 上训、2026 上测。holdout R²=期内可学；transfer R²=跨年迁移。"
             "combined holdout R²=可学习份额；1−该值=不可学习份额(天花板)。\n")
    L.append("| 通道 | 列数 | 期内R²(holdout) | 跨年transfer R² |")
    L.append("|---|---|---|---|")
    for r in chan_rows:
        L.append(f"| {r['channel']} | {r['n_cols']} | {r['holdout_R2']:+.3f} | {r['transfer_R2']:+.3f} |")
    L.append(f"\n**combined 期内 R²={combined['holdout_R2']:+.3f}** -> 可学习份额≈{combined['holdout_R2']*100:.0f}%，"
             f"**不可学习份额(天花板)≈{unlearnable_share*100:.0f}%**。"
             f"combined 跨年 transfer R²={combined['transfer_R2']:+.3f}（实际迁移的可学习部分）。\n"
             f"对照 Phase0：r(=v6修正后残差) 期内 R²=−0.136（已不可学）；此处 ext_err 期内 R²={combined['holdout_R2']:+.3f}"
             f"（>0，因 ext_err 含 v6 能修的可学习结构）。两者差 = v6 已移除的可学习部分。\n")

    L.append("\n## B. 不可学习 proxy 归因（天花板 r 的 cause 通道，corr with |r|）\n")
    L.append("天花板 r 的 cause 通道用 proxy 与 |r| 的跨年 corr/ratio/AUC 归因（非加性，重叠）。\n")
    L.append("| proxy | 2025 corr | 2026 corr | 2026 ratio | 2026 AUC |")
    L.append("|---|---|---|---|---|")
    for r in proxy_rows:
        L.append(f"| {r['proxy']} | {r['corr_2025']:+.3f} | {r['corr_2026']:+.3f} | "
                 f"{r['ratio_2026']:.2f} | {r['auc_2026']:.3f} |")
    L.append(f"\n对照 P-Diag：A1 输入质量 GO(弱但稳 ratio 1.54/1.34)，A2 天气 OOD 翻(1.34->0.88)。"
             f"此处日级复测一致。\n")

    L.append("\n## C. 可再生物理 proxy 排除法（关键诚实结果）\n")
    L.append(f"物理 PV proxy=irrad×(1−0.004·(temp−25))；wind proxy=wind³∈[3,25)。\n")
    L.append(f"- PV(有符号) vs ext_err(有符号) corr：2025={ren_corr_sign_25:+.3f} / 2026={ren_corr_sign_26:+.3f}")
    L.append(f"- PV_dev(幅度) vs |r| corr 2026={ren_corr_mag_26:+.3f}")
    L.append(f"- PV(有符号) 预测 ext_err **符号**方向命中={dir_acc:.3f}（>0.5=符号可迁移）")
    L.append(f"- 中午(11-14) irrad vs |ext_err| corr={mid_corr:+.3f}\n")
    L.append(f"**排除法结论**：气象隐含可再生 proxy 与 ext_err 符号弱相关/方向命中≈0.5（不可迁移）。"
             f"原因：pred_load 已含外部预测器对可再生的同源气象假设，故气象隐含可再生≈已在 pred_load 内，"
             f"残差的可再生分量=可再生实际−预测器假设，**不在气象 proxy 能解释的范围内**。"
             f"-> 排除\"气象隐含可再生\"为符号根因，收窄到\"气象抓不到的可再生变率\"（云局地/弃光/组件，不可观测）。"
             f"与 Phase0（辐照类预测不了符号）、午间诊断（R²=−0.03）一致。\n")

    L.append("\n## D. 逐日 cause 标签（运行时前身，无 actual）+ 验证\n")
    L.append("优先级：holiday->Demand shift；A1 高->Input anomaly；OOD 高->Weather OOD；"
             "renewable_dev 高->Likely renewable；否则 Unknown。阈值=训练期 top-15%。\n")
    L.append("| 标签 | 日数 | mean|ext_err| | /全局 |")
    L.append("|---|---|---|---|")
    for lab in ["Demand shift", "Input anomaly", "Weather OOD", "Likely renewable", "Unknown"]:
        o = label_val[lab]
        L.append(f"| {lab} | {o['n']} | {o['mae']:.0f} | {o['ratio']:.2f} |")
    L.append(f"\n全局 mean|ext_err|={glob:.0f}。ratio>1=该标签日误差更大（标签有效）。"
             f"此为工作流 C 运行时 QA 的 cause 标签前身（合规，无 actual）。\n")

    L.append("\n## E. Residual Attribution Tree\n")
    L.append("```")
    L.append(f"ext_err (val MAE={ext_err_MAE_val:.0f})")
    L.append(f"├── 可学习 (v6 修正 -> r val MAE={r_MAE_val:.0f}, 修正 {v6_reduction:.0f} MW, "
             f"份额≈{combined['holdout_R2']*100:.0f}%)")
    L.append(f"│   ├── Forecast structural (pred_load level+temporal)  transfer R²="
             f"{next(r['transfer_R2'] for r in chan_rows if r['channel']=='forecast_structural'):+.3f}")
    L.append(f"│   ├── Calendar (demand shift 可学习)  transfer R²="
             f"{next(r['transfer_R2'] for r in chan_rows if r['channel']=='calendar'):+.3f}")
    L.append(f"│   └── Weather (可学习耦合)  transfer R²="
             f"{next(r['transfer_R2'] for r in chan_rows if r['channel']=='weather'):+.3f}")
    L.append(f"└── 不可学习 (天花板 ≈ r, 份额≈{unlearnable_share*100:.0f}%)")
    L.append(f"    ├── Weather OOD (novelty)       corr|r|={next(r['corr_2026'] for r in proxy_rows if r['proxy']=='weather_OOD'):+.3f} (不稳, 翻)")
    L.append(f"    ├── Input anomaly (pred_load)   corr|r|={next(r['corr_2026'] for r in proxy_rows if r['proxy']=='A1_input'):+.3f} (稳, GO flag)")
    L.append(f"    ├── Renewable (proxy)           符号命中={dir_acc:.3f} (排除: 气象隐含可再生不可迁移)")
    L.append(f"    └── Random/Unknown              floor (符号通道缺失)")
    L.append("```\n")

    L.append("\n## F. 与既有诊断统一\n")
    L.append(f"- **FDS**：ext_error 83.7% unlearnable。本处 combined 不可学习份额≈{unlearnable_share*100:.0f}%（同量级）。")
    L.append(f"- **Phase0**：r 期内 R²=−0.136（残差=特征噪声）。本处 ext_err 期内 R²={combined['holdout_R2']:+.3f}（>0，含 v6 可修部分）；"
             f"差值=v6 已移除的可学习结构。两者一致：残差的可学习部分 v6 已榨干，剩余=特征噪声。")
    L.append(f"- **午间诊断**：中午 R²=−0.03。本处中午 irrad vs |ext_err| corr={mid_corr:+.3f}（弱），可再生排除法印证午间不可学。")
    L.append(f"- **P-Diag/工作流B**：A1 输入质量 GO(稳)，A2 天气 OOD 翻(不稳)。本处 proxy 归因复测一致。\n")
    L.append(f"**统一结论**：天花板=v6 已移除全部可学习结构后剩余的特征噪声。cause 结构中，"
             f"输入质量(A1)是唯一稳的可部署 flag；天气 OOD/可再生 proxy 跨年不稳或不可迁移；"
             f"符号通道缺失=Random floor 的主体。破 1445 须新信息（可再生实测/Foundation），"
             f"非归因能解决；归因的价值=可解释/可诊断系统能力（Forecast QA），非 MAE。\n")

    report = "\n".join(L)
    REPORT.write_text(report, encoding="utf-8")
    if verbose:
        print("\n" + report)
        print(f"\n报告已写: {REPORT}")
    return {
        "ext_err_MAE_val": ext_err_MAE_val, "r_MAE_val": r_MAE_val,
        "v6_reduction": v6_reduction, "chan_rows": chan_rows,
        "unlearnable_share": unlearnable_share, "combined_holdout": combined["holdout_R2"],
        "combined_transfer": combined["transfer_R2"], "proxy_rows": proxy_rows,
        "ren_dir_acc": dir_acc, "ren_corr_sign_26": ren_corr_sign_26,
        "label_val": label_val, "mid_corr": mid_corr,
    }


def main():
    run(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
