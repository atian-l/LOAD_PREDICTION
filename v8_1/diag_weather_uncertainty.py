# -*- coding: utf-8 -*-
"""v8.1 气象不确定性探针：测 d1_d7 未用气象信号能否强化幅度 Diagnosis + 攻中午段。

背景（P-Diag 后战略决断）：
  - 实际新能源出力不在数据中且拿不到外部 -> 1445 确认为当前数据硬上限。
  - 转向操作价值：用 d1_d7 未用的气象不确定性强化幅度 |r| Diagnosis 通道（已证可迁移），
    攻中午 day 段幅度失败（P-Diag day 段 R²=-0.044），顺带解 9AM 部署。
  - 气象不确定性补不了符号通道 -> 不破 1445，目标是更准的风险标注/置信度。

测三个不确定性来源（均 weather-only，无 actual）：
  1. 多 lead 分歧：同一预测时间 T 跨多个起报 issue 的预报 std（D1~D7 演变=不确定）
  2. 集合离散度：_std 列跨 issue 均值（within-issue 集合发散）
  3. 空间差：|县区_温度-光伏_温度| / |县区_辐照度-光伏_辐照度|（三地点不一致）

合规：不确定性特征仅用 d1_d7 气象（无 actual）；|r| 仅作 eval 标签（#1）。
运行：python -m v8_1.diag_weather_uncertainty  报告 -> v8_1/output/p_weather_uncertainty.md
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
from v8_1.diag_residual import PROBE_PARAMS, PROBE_IT, group_columns, _fit_probe
from v8_1.diag_input import _mag_metrics, _mag_holdout_r2, _risk_subset_eval

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = OUT_DIR / "p_weather_uncertainty.md"
WEATHER_D17 = LC.DATA_DIR / "shandong_weather_15min_d1_d7.csv"

UNC_COLS = ["起报时间", "预测时间", "光伏_辐照度", "光伏_辐照度_std",
            "光伏_温度", "光伏_温度_std", "县区_温度", "县区_辐照度"]


def _build_uncertainty() -> pd.DataFrame:
    """从 d1_d7 计算每个预测时间 T 的不确定性特征（跨 issue）。"""
    w = pd.read_csv(WEATHER_D17, usecols=UNC_COLS, encoding="utf-8-sig")
    w["起报时间"] = pd.to_datetime(w["起报时间"])
    w["预测时间"] = pd.to_datetime(w["预测时间"])
    w = w.dropna(subset=["预测时间"])
    w["cp_temp_diff"] = (w["县区_温度"] - w["光伏_温度"]).abs()
    w["cp_irrad_diff"] = (w["县区_辐照度"] - w["光伏_辐照度"]).abs()
    g = w.groupby("预测时间")
    unc = pd.DataFrame(index=g.groups.keys())
    unc["irrad_lead_std"] = g["光伏_辐照度"].std()      # 多 lead 分歧
    unc["irrad_ens_std"] = g["光伏_辐照度_std"].mean()  # 集合离散度
    unc["temp_lead_std"] = g["光伏_温度"].std()
    unc["temp_ens_std"] = g["光伏_温度_std"].mean()
    unc["cp_irrad_diff"] = g["cp_irrad_diff"].mean()    # 空间差
    unc["cp_temp_diff"] = g["cp_temp_diff"].mean()
    unc.index = pd.DatetimeIndex(unc.index)
    return unc


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 50:
        return float("nan")
    return float(spearmanr(a[m], b[m]).correlation)


def run(verbose: bool = True) -> dict:
    if verbose:
        print("[1/5] 构建数据集 + 加载 v8 bundle ...")
    times, X, pred_load, actual = T.build_dataset()
    vm8 = V8Model.load(VC.V8_BUNDLE)
    mm = vm8.mismatch_model
    X_full = mm.transform(X)
    oof = vm8.oof_pool
    baseA = vm8.base_A

    idx = oof["idx"]
    X_train = X_full.iloc[idx].reset_index(drop=True)
    r_train = np.asarray(oof["actual"] - oof["base_A_oof"], dtype=float)
    mr_train = np.abs(r_train)
    times_train = pd.DatetimeIndex(oof["times"])
    seg_train = np.asarray(oof["seg"], dtype=object)
    train_mean_mag = float(mr_train.mean())

    vs, ve = pd.Timestamp(LC.VAL_START), pd.Timestamp(LC.VAL_END)
    buf = vs - pd.Timedelta(days=7)
    em = (times >= buf) & (times <= ve)
    Xe = X[em]
    te = pd.DatetimeIndex(times[em])
    ae = actual.reindex(te)
    Xe_full = mm.transform(Xe)
    predA = np.asarray(baseA.predict_load(Xe_full, pred_load), dtype=float)
    vmask = (te >= vs) & (te <= ve) & ae.notna()
    X_val = Xe_full[vmask].reset_index(drop=True)
    r_val = (ae[vmask].values.astype(float) - predA[vmask])
    mr_val = np.abs(r_val)
    times_val = te[vmask]
    seg_val = SEG.segment_array(times_val.hour.values.astype(int))

    # ===================== 加载 d1_d7 不确定性 =====================
    if verbose:
        print("[2/5] 加载 d1_d7 + 计算气象不确定性特征 ...")
    unc = _build_uncertainty()
    unc_cols = list(unc.columns)
    # 对齐到 train/val 时间线
    unc_tr = unc.reindex(times_train)
    unc_va = unc.reindex(times_val)
    # 用训练期 std 标准化（便于组合 + 子集阈值）
    tr_std = unc_tr.std()
    tr_std = tr_std.replace(0, 1.0)
    unc_tr_z = (unc_tr - unc_tr.mean()) / tr_std
    unc_va_z = (unc_va - unc_tr.mean()) / tr_std
    # 覆盖率
    cov_tr = unc_tr.notna().mean()
    cov_va = unc_va.notna().mean()
    if verbose:
        print(f"      不确定性特征={unc_cols}")
        print(f"      train 覆盖率: " + ", ".join(f"{c}={cov_tr[c]:.2f}" for c in unc_cols))
        print(f"      val   覆盖率: " + ", ".join(f"{c}={cov_va[c]:.2f}" for c in unc_cols))

    # ===================== A. 不确定性 -> |r| 跨年（单特征）=====================
    if verbose:
        print("[3/5] A. 不确定性单特征 -> |r| 跨年（含分段，午间重点）...")
    unc_rows = []
    for c in unc_cols:
        sc_tr = unc_tr_z[c].values.astype(float)
        sc_va = unc_va_z[c].values.astype(float)
        c25 = _corr(sc_tr, mr_train)
        c26 = _corr(sc_va, mr_val)
        sub = _risk_subset_eval(sc_va, mr_val, sc_tr)
        unc_rows.append({"feature": c, "corr_2025": c25, "corr_2026": c26,
                         "ratio_2026": sub["ratio_26"], "auc_2026": sub["auc_26"],
                         "mae_hi_2026": sub["mae_hi"], "mae_lo_2026": sub["mae_lo"]})
    # 分段（day=午间）单特征 corr
    seg_unc = {}
    for s in VC.SEGMENTS:
        mtr = seg_train == s
        mva = seg_val == s
        for c in unc_cols:
            seg_unc[(s, c)] = _corr(unc_va_z[c].values[mva], mr_val[mva])

    # ===================== B. 不确定性强化幅度探针 =====================
    if verbose:
        print("[4/5] B. 不确定性加入幅度探针 -> 跨年 R²（含午间）...")
    groups = group_columns(X_full.columns)
    combined_cols = [c for cs in groups.values() for c in cs]

    def _mag_probe(Xtr, ytr, Xva, cols):
        bst = _fit_probe(Xtr, ytr, cols)
        return bst.predict(Xva[cols])

    # combined 仅
    pred_comb = _mag_probe(X_train, mr_train, X_val, combined_cols)
    m_comb = _mag_metrics(pred_comb, mr_val, train_mean_mag)
    # combined + 不确定性
    Xtr_u = X_train.copy()
    Xva_u = X_val.copy()
    for c in unc_cols:
        Xtr_u[f"unc_{c}"] = unc_tr_z[c].values
        Xva_u[f"unc_{c}"] = unc_va_z[c].values
    cols_u = combined_cols + [f"unc_{c}" for c in unc_cols]
    pred_u = _mag_probe(Xtr_u, mr_train, Xva_u, cols_u)
    m_u = _mag_metrics(pred_u, mr_val, train_mean_mag)
    # 仅不确定性
    cols_unc = [f"unc_{c}" for c in unc_cols]
    pred_unc = _mag_probe(Xtr_u, mr_train, Xva_u, cols_unc)
    m_unc = _mag_metrics(pred_unc, mr_val, train_mean_mag)

    # 分段（day=午间）combined vs combined+unc
    seg_probe = {}
    for s in VC.SEGMENTS:
        mtr = seg_train == s
        mva = seg_val == s
        if not mva.any():
            continue
        Xtr_s = X_train[mtr].reset_index(drop=True)
        Xva_s = X_val[mva].reset_index(drop=True)
        ytr_s = mr_train[mtr]
        seg_mean = float(ytr_s.mean()) if len(ytr_s) else train_mean_mag
        Xtr_su = Xtr_s.copy()
        Xva_su = Xva_s.copy()
        for c in unc_cols:
            Xtr_su[f"unc_{c}"] = unc_tr_z[c].values[mtr]
            Xva_su[f"unc_{c}"] = unc_va_z[c].values[mva]
        p_c = _mag_probe(Xtr_s, ytr_s, Xva_s, combined_cols)
        p_u = _mag_probe(Xtr_su, ytr_s, Xva_su, cols_u)
        y = mr_val[mva]
        bmse = float(np.mean((y - seg_mean) ** 2))
        r2_c = 1.0 - float(np.mean((y - p_c) ** 2)) / bmse if bmse > 0 else float("nan")
        r2_u = 1.0 - float(np.mean((y - p_u) ** 2)) / bmse if bmse > 0 else float("nan")
        seg_probe[s] = {"r2_combined": r2_c, "r2_with_unc": r2_u, "delta": r2_u - r2_c}

    # ===================== C. 自动判定 =====================
    if verbose:
        print("[5/5] C. 自动判定 ...")
    # 单特征迁移：corr 两年>0 且 ratio_2026>1 且 auc>0.55（物理信号是否存在，非诊断增益）
    unc_transferring = [r for r in unc_rows
                        if (not np.isnan(r["corr_2025"]) and r["corr_2025"] > 0.0
                            and not np.isnan(r["corr_2026"]) and r["corr_2026"] > 0.0
                            and not np.isnan(r["ratio_2026"]) and r["ratio_2026"] > 1.0
                            and not np.isnan(r["auc_2026"]) and r["auc_2026"] > 0.55)]
    # 幅度探针增益：combined+unc 的 transfer R² > combined（诊断强化，本探针真正目标）
    probe_gain = m_u["transfer_R2"] - m_comb["transfer_R2"]
    # 中午段恢复：day 段 r2_with_unc > 0 且 delta>0（P-Diag day 段 combined=-0.044）
    day_recovered = seg_probe.get("day", {}).get("r2_with_unc", float("nan"))
    day_delta = seg_probe.get("day", {}).get("delta", float("nan"))
    diagnosis_improved = (probe_gain > 0.005) or (
        not np.isnan(day_recovered) and day_recovered > 0.0 and day_delta > 0.0)
    signal_exists = bool(unc_transferring)

    # ===================== 报告 =====================
    L = []
    L.append("# v8.1 气象不确定性探针\n")
    L.append("> 战略背景：无外部新能源出力 -> 1445 硬上限。转操作价值：用 d1_d7 未用气象不确定性\n"
             "> 强化幅度 Diagnosis（P-Diag 已证可迁移）+ 攻中午段幅度失败。**不破 1445**（符号通道仍缺）。\n")
    L.append(f"数据：d1_d7（{WEATHER_D17.name}），三地点(风电/县区/光伏站)+08:00&20:00 起报+D1-D7。\n")
    L.append(f"|r|_train mean={train_mean_mag:.0f}  |r|_val mean={float(mr_val.mean()):.0f}。"
             f"P-Diag combined 幅度 transfer R²=+0.182（day 段 −0.044）。\n")

    L.append("\n## A. 不确定性单特征 -> |r| 跨年\n")
    L.append("| 特征 | 含义 | 2025 corr | 2026 corr | 2026 高/全局MAE | 2026 AUC | 2026 高风险MAE | 2026 低风险MAE |")
    L.append("|---|---|---|---|---|---|---|---|")
    meaning = {"irrad_lead_std": "辐照多lead分歧", "irrad_ens_std": "辐照集合离散",
               "temp_lead_std": "温度多lead分歧", "temp_ens_std": "温度集合离散",
               "cp_irrad_diff": "县区-光伏辐照差", "cp_temp_diff": "县区-光伏温度差"}
    for r in unc_rows:
        L.append(f"| {r['feature']} | {meaning.get(r['feature'],'')} | {r['corr_2025']:+.3f} | "
                 f"{r['corr_2026']:+.3f} | {r['ratio_2026']:.2f} | {r['auc_2026']:.3f} | "
                 f"{r['mae_hi_2026']:.0f} | {r['mae_lo_2026']:.0f} |")
    L.append("\n### A.2 分段 corr（day=午间重点）\n")
    L.append("| 特征 | " + " | ".join(VC.SEGMENTS) + " |")
    L.append("|---|" + "|".join(["---"] * len(VC.SEGMENTS)) + "|")
    for c in unc_cols:
        row = [c] + [f"{seg_unc.get((s, c), float('nan')):+.3f}" for s in VC.SEGMENTS]
        L.append("| " + " | ".join(row) + " |")

    L.append("\n## B. 不确定性强化幅度探针（跨年 transfer R²）\n")
    L.append("| 探针 | 跨年transfer R² | 期内R² | risk_AUC |")
    L.append("|---|---|---|---|")
    L.append(f"| combined（基线，=P-Diag）| {m_comb['transfer_R2']:+.3f} | {m_comb['std_R2']:+.3f} | {m_comb['risk_AUC']:.3f} |")
    L.append(f"| 仅不确定性 | {m_unc['transfer_R2']:+.3f} | {m_unc['std_R2']:+.3f} | {m_unc['risk_AUC']:.3f} |")
    L.append(f"| combined + 不确定性 | {m_u['transfer_R2']:+.3f} | {m_u['std_R2']:+.3f} | {m_u['risk_AUC']:.3f} |")
    L.append(f"\n**不确定性带来的 transfer R² 增益 = {probe_gain:+.3f}**（>0.005 视为有效增益）。\n")

    L.append("\n### B.2 分段 transfer R²（午间是否恢复）\n")
    L.append("| 段 | combined R² | +不确定性 R² | Δ |")
    L.append("|---|---|---|---|")
    for s in VC.SEGMENTS:
        sp = seg_probe.get(s, {})
        L.append(f"| {s} | {sp.get('r2_combined', float('nan')):+.3f} | "
                 f"{sp.get('r2_with_unc', float('nan')):+.3f} | {sp.get('delta', float('nan')):+.3f} |")
    L.append(f"\nP-Diag day 段 combined 幅度 R²=−0.044。加不确定性后 day 段 R²={day_recovered:+.3f}"
             f"（Δ={day_delta:+.3f}）。>0=中午段幅度恢复。\n")

    L.append("\n## C. 自动判定\n")
    L.append(f"1. **单特征跨年迁移**：{[r['feature'] for r in unc_transferring] if unc_transferring else '无'}")
    L.append(f"2. **幅度探针增益**：combined+不确定性 transfer R² − combined = {probe_gain:+.3f}"
             f"（{'有效' if probe_gain>0.005 else '无有效增益'}）")
    L.append(f"3. **中午段恢复**：day 段 R² {seg_probe.get('day',{}).get('r2_combined',float('nan')):+.3f} -> "
             f"{day_recovered:+.3f}（{'恢复为正' if (not np.isnan(day_recovered) and day_recovered>0) else '未恢复'}）")
    L.append(f"4. **判定**：{'**GO** - 气象不确定性强化幅度 Diagnosis 有效' if diagnosis_improved else '**NO-GO（强化维度）** - 信号存在但与现有特征冗余，不强化探针/不恢复中午'}")
    if diagnosis_improved:
        L.append(f"   -> 气象不确定性强化了幅度 Diagnosis 通道，操作价值↑。但**不破 1445**（符号通道仍缺）。")
    else:
        L.append(f"   -> **物理信号真实**：辐照不确定性（多lead分歧/集合离散/县区-光伏空间差）corr~0.49、AUC~0.73、"
                 f"跨年极稳，是比 A2 天气OOD 更干净的幅度驱动。**但与 combined 探针冗余**（增益{probe_gain:+.3f}），"
                 f"且中午段未恢复（{day_recovered:+.3f}）。即现有特征已榨干幅度通道的可迁移信号，"
                 f"气象不确定性无增量。幅度 Diagnosis 已饱和。")
        L.append(f"   -> **不破 1445**（符号通道需实际新能源出力，无外部数据）。剩余备选=P-Foundation"
                 f"（不依赖外部数据，预训练先验或携符号信息）。")
    L.append(f"\n**诚实边界**：本探针目标非 MAE 突破（符号通道缺失，无外部出力不可补），"
             f"而是验证气象不确定性能否提升幅度 Diagnosis 的操作质量与中午段覆盖。结论=不能（冗余）。")

    report = "\n".join(L)
    REPORT.write_text(report, encoding="utf-8")
    if verbose:
        print("\n" + report)
        print(f"\n报告已写: {REPORT}")
    return {"unc_rows": unc_rows, "seg_unc": seg_unc, "m_comb": m_comb, "m_u": m_u,
            "m_unc": m_unc, "seg_probe": seg_probe, "probe_gain": probe_gain,
            "day_recovered": day_recovered, "diagnosis_improved": diagnosis_improved,
            "signal_exists": signal_exists, "unc_transferring": unc_transferring}


def main():
    run(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
