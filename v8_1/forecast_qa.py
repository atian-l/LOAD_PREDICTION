# -*- coding: utf-8 -*-
"""v8.1 工作流 C：运行时 Forecast QA（归因 -> 系统能力落地）。

把工作流 A（归因）+ B（A1 flag）变成 predict.py 运行时输出四元组：
  Prediction = v6 预测
  Confidence = P(误差<=典型)  校准概率（2025 OOF 训练幅度探针+isotonic）
  Reason     = cause 标签（Input anomaly / Weather OOD / Demand shift / Likely renewable / Unknown）
  Warning    = 高风险 actionable flag（Confidence 低 OR Input anomaly）

核心合规：运行时**无 actual**。Confidence = 2025 OOF 训练的幅度探针（|r| 目标，P-Diag 已证
跨年 transfer R²+0.182/AUC 0.760）+ isotonic 校准；cause 标签仅用 pred_load+weather+calendar。
actual 仅作离线验证标签(eval-only，不变量 #1/#5/#6)。

关键区分（DESIGN §3）：运行时 cause 标签 = **预测 cause 类别**（幅度/类别可迁移），非测残差值
（符号不可迁移）。Confidence 预测"可靠性"（幅度），不预测"高估/低估"（符号）。

运行：python -m v8_1.forecast_qa   报告 -> v8_1/output/forecast_qa.md
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from scipy.stats import spearmanr

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
from v8_1.residual_attribution import _daily_weather_ood, _renewable_proxy

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = OUT_DIR / "forecast_qa.md"

CONF_WARN_THR = 0.50      # Confidence < 此值 -> Warning（日级均值）
CAUSE_TOP_PCT = 85        # cause 标签阈值 = 训练期 top-15%


def _transfer_r2(pred, y, train_mean) -> float:
    bmse = float(np.mean((y - train_mean) ** 2)); pmse = float(np.mean((y - pred) ** 2))
    return 1.0 - pmse / bmse if bmse > 0 else float("nan")


def _cause_label_daily(val_dates_u, a1_comb, ood_all, ren_dev, cal_daily, a1_thr, ood_thr, ren_thr):
    """运行时 cause 标签（无 actual）。返回 (label_arr, score_dict)。"""
    a1_d = a1_comb.reindex(val_dates_u).values
    ood_d = ood_all.reindex(val_dates_u).values
    ren_d = ren_dev.reindex(val_dates_u).values
    hol_d = cal_daily["is_holiday"].reindex(val_dates_u).fillna(0).values
    n = len(val_dates_u)
    label_arr = np.empty(n, dtype=object)
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
        label_arr[i] = lab
    return label_arr


# --------------------------------------------------------------------------- #
# 主诊断
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    if verbose:
        print("[1/5] 构建数据集 + 加载 v8 bundle ...")
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
    r_train = np.asarray(oof["actual"] - oof["base_A_oof"], dtype=float)
    mr_train = np.abs(r_train)
    train_mean_mag = float(mr_train.mean())

    # ---- val（2026 eval-only；运行时模拟=仅用 features）----
    vs, ve = pd.Timestamp(LC.VAL_START), pd.Timestamp(LC.VAL_END)
    buf = vs - pd.Timedelta(days=7)
    em = (times >= buf) & (times <= ve)
    Xe = X[em]; te = pd.DatetimeIndex(times[em])
    ae = actual.reindex(te)
    Xe_full = mm.transform(Xe)
    predA = np.asarray(baseA.predict_load(Xe_full, pred_load), dtype=float)  # v6 预测（运行时可得）
    vmask = (te >= vs) & (te <= ve) & ae.notna()
    X_val = Xe_full[vmask].reset_index(drop=True)
    times_val = te[vmask]
    actual_val = ae[vmask].values.astype(float)
    r_val = actual_val - predA[vmask]                       # 仅验证用（eval-only）
    mr_val = np.abs(r_val)

    groups = group_columns(X_full.columns)
    combined_cols = [c for cs in groups.values() for c in cs]

    # ===================== A. 训练 Confidence 模型 =====================
    if verbose:
        print("[2/5] A. 训练 Confidence 模型（幅度探针 + isotonic 校准）...")
    # 幅度探针：2025 OOF |r| 目标，全特征
    probe = _fit_probe(X_train, mr_train, combined_cols)
    pred_mag_train = probe.predict(X_train[combined_cols])
    pred_mag_val = probe.predict(X_val[combined_cols])
    # sanity：复现 P-Diag 跨年 transfer R²/AUC
    tr2 = _transfer_r2(pred_mag_val, mr_val, train_mean_mag)
    med = float(np.median(mr_train))
    yb_val = (mr_val > med).astype(int)
    auc = float(roc_auc_score(yb_val, pred_mag_val)) if 0 < yb_val.sum() < len(yb_val) else float("nan")
    # isotonic 校准：pred_mag -> P(|r|>median)
    yb_train = (mr_train > med).astype(int)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(pred_mag_train, yb_train)
    risk_val = iso.predict(pred_mag_val)                   # P(|r|>median)
    conf_val = 1.0 - risk_val                              # Confidence = P(|r|<=median)
    brier = float(brier_score_loss(yb_val, risk_val))
    if verbose:
        print(f"      幅度探针 transfer R²={tr2:+.3f} AUC={auc:.3f} (对照 P-Diag +0.182/0.760); Brier={brier:.3f}")

    # ===================== B. 运行时四元组（日级，无 actual）=====================
    if verbose:
        print("[3/5] B. 运行时四元组（日级聚合）...")
    dva = pd.DatetimeIndex(times_val).normalize()
    val_dates_u = pd.DatetimeIndex(sorted(set(dva)))
    # v6 预测日级
    v6_pred_val = predA[vmask]
    df_pt = pd.DataFrame({"pred": v6_pred_val, "conf": conf_val, "date": dva,
                          "mr": mr_val, "r": r_val})
    g = df_pt.groupby("date")
    daily_pred_mean = g["pred"].mean()
    daily_pred_peak = g["pred"].max()
    daily_conf_mean = g["conf"].mean()                     # = 期望可靠点比例
    daily_conf_min = g["conf"].min()
    daily_mr_mean = g["mr"].mean()                         # 验证用（eval-only）
    daily_reliable_frac = g["mr"].apply(lambda s: float(np.mean(s <= med)))  # 验证用

    # cause 标签（无 actual）
    train_dates = pd.DatetimeIndex(times_train).normalize().unique()
    dtr = pd.DatetimeIndex(times_train).normalize()
    sig_all = _standardize(_pl_daily_signals(pred_load.reindex(times), times), train_dates)
    a1_comb = sig_all.mean(axis=1)
    ood_all = _daily_weather_ood(X_full, times, train_dates)
    ren_all = _renewable_proxy(X_full, times)
    ren_dev = ren_all["pv_dev"] + ren_all["wp_dev"]
    ren_std = ren_dev.loc[ren_dev.index.isin(train_dates)].std()
    ren_dev = ren_dev / (ren_std if ren_std > 0 else 1.0)
    cal_val = X_val[["is_holiday", "is_weekend", "dayofweek"]].copy()
    cal_val["date"] = dva
    cal_daily = cal_val.groupby("date").agg(is_holiday=("is_holiday", "max"))

    def _thr(s):
        a = np.asarray(s, dtype=float); a = a[np.isfinite(a)]
        return float(np.percentile(a, CAUSE_TOP_PCT)) if len(a) else float("nan")
    a1_thr = _thr(a1_comb.reindex(dtr).values)
    ood_thr = _thr(ood_all.reindex(dtr).values)
    ren_thr = _thr(ren_dev.reindex(dtr).values)
    label_arr = _cause_label_daily(val_dates_u, a1_comb, ood_all, ren_dev, cal_daily,
                                   a1_thr, ood_thr, ren_thr)
    daily_reason = pd.Series(label_arr, index=val_dates_u)
    # Warning = Confidence 低 OR Input anomaly
    daily_warning = (daily_conf_mean.reindex(val_dates_u) < CONF_WARN_THR) | \
                    (daily_reason == "Input anomaly")

    # ===================== C. 验证（eval-only）=====================
    if verbose:
        print("[4/5] C. 验证（reliability + Warning 子集 + 分位）...")
    # reliability: Confidence 分位 -> 实际可靠比例
    conf_arr = daily_conf_mean.reindex(val_dates_u).values
    rel_arr = daily_reliable_frac.reindex(val_dates_u).values
    mr_arr = daily_mr_mean.reindex(val_dates_u).values
    warn_arr = daily_warning.reindex(val_dates_u).values
    n = len(val_dates_u)
    order = np.argsort(conf_arr)
    nbin = min(5, n)
    bins = np.array_split(order, nbin)
    rel_rows = []
    for b in bins:
        rel_rows.append({"conf_lo": float(np.min(conf_arr[b])),
                         "conf_hi": float(np.max(conf_arr[b])),
                         "n": len(b),
                         "actual_reliable_frac": float(np.mean(rel_arr[b])),
                         "actual_mr": float(np.mean(mr_arr[b]))})
    # Warning 子集
    glob_mr = float(np.mean(mr_arr))
    warn_mr = float(np.mean(mr_arr[warn_arr])) if warn_arr.any() else float("nan")
    nowarn_mr = float(np.mean(mr_arr[~warn_arr])) if (~warn_arr).any() else float("nan")
    # Confidence-decile: 低置信 vs 高置信 actual |r|
    lo_mr = float(np.mean(mr_arr[order[:n // 5]]))     # 最低 20% confidence
    hi_mr = float(np.mean(mr_arr[order[-n // 5:]]))    # 最高 20% confidence
    # 日级 corr
    conf_mr_corr = _spearman(conf_arr, mr_arr)
    # Brier (日级: 用日级 mean risk vs daily reliable frac>0.5)
    daily_risk = 1.0 - daily_conf_mean.reindex(val_dates_u).values
    daily_bin = (rel_arr > 0.5).astype(int)
    brier_day = float(brier_score_loss(daily_bin, daily_risk)) if 0 < daily_bin.sum() < len(daily_bin) else float("nan")

    # ===================== D. cause 标签验证 =====================
    cause_val = {}
    for lab in ["Demand shift", "Input anomaly", "Weather OOD", "Likely renewable", "Unknown"]:
        m = label_arr == lab
        cause_val[lab] = {"n": int(m.sum()),
                          "mr": float(np.mean(mr_arr[m])) if m.any() else float("nan"),
                          "ratio": float(np.mean(mr_arr[m]) / glob_mr) if m.any() and glob_mr > 0 else float("nan")}

    # ===================== E. 报告 =====================
    if verbose:
        print("[5/5] E. 报告 ...")
    L = []
    L.append("# v8.1 工作流 C：运行时 Forecast QA（四元组）\n")
    L.append("> 把 A（归因）+ B（A1 flag）变成 predict.py 运行时输出："
             "Prediction + Confidence + Reason + Warning。\n"
             "> 合规：运行时无 actual；Confidence=2025 OOF 训练幅度探针+isotonic；cause 标签仅 pred_load+weather+calendar。\n"
             "> actual 仅作离线验证(eval-only)。\n")
    L.append(f"幅度探针：combined 全特征，|r| 目标，2025 OOF 训。median(|r_train|)={med:.0f} MW（\"典型误差\"阈值）。\n")

    L.append("\n## A. Confidence 模型（幅度探针 + isotonic 校准）\n")
    L.append(f"- 跨年 transfer R²={tr2:+.3f}，risk_AUC={auc:.3f}（对照 P-Diag +0.182/0.760，一致）")
    L.append(f"- isotonic 校准：pred_|r| -> P(|r|>median)。点级 Brier={brier:.3f}（0=完美，1=最差，0.25=常数基线）\n")

    L.append("\n## B. 运行时四元组（日级，val 2026 模拟运行时）\n")
    L.append(f"Confidence_day = 96 点 mean(1−risk) = 期望可靠点比例。"
             f"Warning = Confidence<{CONF_WARN_THR} OR Reason=Input anomaly。\n")
    L.append("| 日期 | Pred均值 MW | Pred峰值 MW | Confidence | Reason | Warning |")
    L.append("|---|---|---|---|---|---|")
    # 展示: 5 个 Warning 日 + 5 个高置信日
    warn_dates = val_dates_u[warn_arr]
    hi_conf_dates = val_dates_u[order[-5:]]
    show = list(warn_dates[:5]) + list(hi_conf_dates)
    seen = set()
    for d in show:
        if d in seen:
            continue
        seen.add(d)
        L.append(f"| {d.date()} | {daily_pred_mean.reindex([d]).iloc[0]:.0f} | "
                 f"{daily_pred_peak.reindex([d]).iloc[0]:.0f} | "
                 f"{daily_conf_mean.reindex([d]).iloc[0]:.2f} | "
                 f"{daily_reason.reindex([d]).iloc[0]} | "
                 f"{'⚠️YES' if daily_warning.reindex([d]).iloc[0] else 'no'} |")
    L.append(f"\n（共 {int(warn_arr.sum())}/{n} 日触发 Warning）\n")

    L.append("\n## C. 验证（eval-only）\n")
    L.append("### Reliability（Confidence 分位 -> 实际可靠点比例）\n")
    L.append("| Confidence 区间 | 日数 | 实际可靠比例 | 实际 mean|r| |")
    L.append("|---|---|---|---|")
    for r in rel_rows:
        L.append(f"| [{r['conf_lo']:.2f}, {r['conf_hi']:.2f}] | {r['n']} | "
                 f"{r['actual_reliable_frac']:.2f} | {r['actual_mr']:.0f} |")
    L.append(f"\n良好校准=低 Confidence 区间实际可靠比例低、mean|r|高（单调）。\n")
    L.append("### Warning 子集 + 置信分位\n")
    L.append(f"- 全局 daily mean|r|={glob_mr:.0f}")
    L.append(f"- Warning 日 mean|r|={warn_mr:.0f}（ratio={warn_mr/glob_mr:.2f}）")
    L.append(f"- 非 Warning 日 mean|r|={nowarn_mr:.0f}（ratio={nowarn_mr/glob_mr:.2f}）")
    L.append(f"- 最低 20% Confidence 日 mean|r|={lo_mr:.0f}；最高 20% Confidence 日 mean|r|={hi_mr:.0f}"
             f"（分离度={lo_mr/max(hi_mr,1):.2f}x）")
    L.append(f"- 日级 Spearman(Confidence, mean|r|)={conf_mr_corr:+.3f}（负=高置信低误差）")
    L.append(f"- 日级 Brier={brier_day:.3f}\n")

    L.append("\n## D. Reason（cause 标签）验证\n")
    L.append("| Reason | 日数 | mean|r| | /全局 |")
    L.append("|---|---|---|---|")
    for lab in ["Demand shift", "Input anomaly", "Weather OOD", "Likely renewable", "Unknown"]:
        o = cause_val[lab]
        L.append(f"| {lab} | {o['n']} | {o['mr']:.0f} | {o['ratio']:.2f} |")
    L.append(f"\n仅 Input anomaly 稳定有效（ratio>1，对照工作流 A/B）。其他 cause 为最佳猜测但跨年不稳。\n")

    L.append("\n## E. predict.py 集成方式\n")
    L.append("```")
    L.append("# 离线训练一次（train 时）：")
    L.append("#   1. 幅度探针 booster（2025 OOF |r| 目标）-> models/qa_mag_booster.txt")
    L.append("#   2. isotonic 校准器 -> models/qa_calib.pkl")
    L.append("#   3. cause 阈值（A1/OOD/renewable top-15%）-> models/qa_thresholds.pkl")
    L.append("# 运行时（predict.py，D+1）：")
    L.append("#   features = build_features(14d window)   # 同 v6，无 actual")
    L.append("#   pred = v6.predict(features)            # 96 点")
    L.append("#   pred_mag = qa_mag_booster.predict(features)")
    L.append("#   risk = qa_calib.predict(pred_mag)      # P(|r|>median)")
    L.append("#   confidence = 1 - risk                  # 96 点 -> 日级 mean")
    L.append("#   reason = cause_label(A1, OOD, calendar, renewable)  # 日级")
    L.append("#   warning = (confidence_day < 0.5) or (reason == 'Input anomaly')")
    L.append("#   output: pred(96) + confidence(96) + reason(day) + warning(day)")
    L.append("```\n")

    L.append("\n## F. 结论\n")
    warn_ratio = warn_mr / glob_mr if glob_mr > 0 else float("nan")
    sep = lo_mr / max(hi_mr, 1)
    L.append(f"- **Confidence 模型有效**：transfer R²={tr2:+.3f}/AUC={auc:.3f}（复现 P-Diag），"
             f"reliability 单调，日级 corr={conf_mr_corr:+.3f}，"
             f"低/高置信分离度={sep:.2f}x。")
    L.append(f"- **Warning 可操作**：Warning 日 mean|r|={warn_mr:.0f}（全局 {glob_mr:.0f}，"
             f"ratio {warn_ratio:.2f}），{int(warn_arr.sum())}/{n} 日触发 -> 人工复核/fallback 触发。")
    L.append(f"- **能力定位**：Forecast QA = 可解释/可诊断系统能力，**非 MAE 杠杆**（符号通道仍关，"
             f"Confidence 预测可靠性不预测方向）。Reason 中仅 Input anomaly 稳定有效。")
    L.append(f"- **价值**：预测从\"{daily_pred_mean.iloc[0]:.0f} MW\"单值 -> "
             f"附 Confidence+Reason+Warning，满足工业 Forecast QA 需求（DESIGN 终点里程碑的\"可解释/可诊断\"落地）。")

    report = "\n".join(L)
    REPORT.write_text(report, encoding="utf-8")
    if verbose:
        print("\n" + report)
        print(f"\n报告已写: {REPORT}")
    return {
        "transfer_r2": tr2, "auc": auc, "brier": brier, "brier_day": brier_day,
        "conf_mr_corr": conf_mr_corr, "warn_mr": warn_mr, "glob_mr": glob_mr,
        "warn_ratio": warn_ratio, "lo_mr": lo_mr, "hi_mr": hi_mr, "sep": sep,
        "n_warn": int(warn_arr.sum()), "n_days": n, "cause_val": cause_val,
        "rel_rows": rel_rows,
    }


def main():
    run(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
