# -*- coding: utf-8 -*-
"""v8.1 工作流 B：A1 输入异常独立验证（Diagnosis 层 - 输入质量任务）。

DESIGN §5/§6 优先级 ⭐⭐⭐⭐⭐。P-Diag 把 A1(pred_load 突变)+A2(天气 OOD) 打包测过，
报告了 A2 比率跨年翻转(1.34->0.88 不稳)，但**A1 自身的比率稳定性从未单独报告**。
本探针隔离 A1，回答：作为**输入质量**信号(新任务，非残差方向)，A1 是否跨年**稳定**
且可迁移 -> 能否作可部署置信度 flag。

核心区分（DESIGN §3）：
  - A1 信号 = 输入质量，仅用 pred_load（shift(1) 因果滚动，可部署，无 weather/actual）。
  - |r| = 验证标签（eval-only，actual 仅作可靠性度量，不变量 #1）。
  - 任务 = "输入异常 -> 预测是否更不可靠"，非"预测残差符号/值"（符号通道已关闭）。

稳定性判据（关键，对照 A2 翻转）：
  - ratio(高A1子集 mean|r| / 全局) 跨年是否同向(都>1 或都<1) -> 不翻转=可部署。
  - corr(signal, |r|) 跨年同号。
  - risk_AUC 跨年都>0.5。
  弱但稳 = 可部署置信度 flag（强但翻如 A2 = 不可部署）。

合规：A1 仅用 pred_load；actual 仅作 |r| 评估标签(eval-only，不变量 #1/#5/#6)。
运行：python -m v8_1.input_anomaly   报告 -> v8_1/output/a1_input_anomaly.md
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
from v8_1.diag_residual import PROBE_PARAMS, PROBE_IT, _fit_probe

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = OUT_DIR / "a1_input_anomaly.md"

# 日级探针：样本少(~200-270 日)，用更浅配置防过拟合（min_data_in_leaf 降、叶子少、L2 强）
DAILY_PARAMS = dict(PROBE_PARAMS, min_data_in_leaf=20, num_leaves=15, lambda_l2=12.0)
DAILY_IT = 60


def _fit_daily_probe(Xtr, ytr, cols):
    d = lgb.Dataset(Xtr[cols], label=ytr)
    return lgb.train(DAILY_PARAMS, d, num_boost_round=DAILY_IT)


# --------------------------------------------------------------------------- #
# A1 日级输入质量信号（仅 pred_load，因果滚动，可部署）
# --------------------------------------------------------------------------- #
def _pl_daily_signals(pred_load_s: pd.Series, times_idx) -> pd.DataFrame:
    """日级 pred_load 输入质量信号。全部 shift(1) 因果滚动（无未来泄露，可部署）。
    返回逐日 DataFrame（未标准化，调用方用训练期 std 标准化）。"""
    d = pd.DatetimeIndex(times_idx).normalize()
    daily = pred_load_s.groupby(d).mean()
    dmax = pred_load_s.groupby(d).max()
    dstd = pred_load_s.groupby(d).std()
    r7med = daily.shift(1).rolling(7, min_periods=3).median()
    r7mean = daily.shift(1).rolling(7, min_periods=3).mean()
    r7std = daily.shift(1).rolling(7, min_periods=3).std()
    r28med = daily.shift(1).rolling(28, min_periods=7).median()
    r7max = dmax.shift(1).rolling(7, min_periods=3).median()
    r7dstd = dstd.shift(1).rolling(7, min_periods=3).median()
    sig = pd.DataFrame(index=daily.index)
    sig["jump_7d"] = (daily - r7med).abs()                       # 水平突变 vs 7日中位（原 A1）
    sig["jump_28d"] = (daily - r28med).abs()                     # 月度漂移
    sig["z_7d"] = ((daily - r7mean) / r7std).abs().replace([np.inf, -np.inf], np.nan)
    sig["peak_jump"] = (dmax - r7max).abs()                      # 峰值水平跳变
    sig["vol_anom"] = (dstd - r7dstd).abs()                      # 日内波动率异常
    return sig


def _standardize(sig: pd.DataFrame, train_dates) -> pd.DataFrame:
    tr = sig.index.isin(train_dates)
    std = sig.loc[tr].std().replace(0, 1.0).fillna(1.0)
    return sig / std


def _spearman(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return float("nan")
    return float(spearmanr(a[m], b[m]).correlation)


def _risk_metrics(score, y_mag, train_score) -> dict:
    """ratio = 高分(top-20%，阈值取自 train_score)子集 mean|r| / 全局；auc = |r|>median 二分类。
    score/y_mag 在同一期（2025 或 2026）上评估；train_score 提供阈值。"""
    score = np.asarray(score, dtype=float); y_mag = np.asarray(y_mag, dtype=float)
    train_score = np.asarray(train_score, dtype=float)
    tr_fin = np.isfinite(train_score)
    thr = float(np.nanpercentile(train_score[tr_fin], 80)) if tr_fin.any() else float("nan")
    fin = np.isfinite(score) & np.isfinite(y_mag) & np.isfinite(thr)
    if fin.sum() < 20:
        return {"ratio": float("nan"), "auc": float("nan"), "n_hi": 0,
                "mae_hi": float("nan"), "mae_global": float("nan")}
    s = score[fin]; y = y_mag[fin]
    hi = s > thr
    glob = float(np.mean(y))
    mae_hi = float(np.mean(y[hi])) if hi.any() else float("nan")
    ratio = mae_hi / glob if glob > 0 else float("nan")
    med = float(np.median(y))
    yb = (y > med).astype(int)
    auc = float(roc_auc_score(yb, s)) if 0 < yb.sum() < len(yb) else float("nan")
    return {"ratio": ratio, "auc": auc, "n_hi": int(hi.sum()),
            "mae_hi": mae_hi, "mae_global": glob}


# --------------------------------------------------------------------------- #
# 主诊断
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    if verbose:
        print("[1/4] 构建数据集 + 加载 v8 bundle + 残差 ...")
    times, X, pred_load, actual = T.build_dataset()
    vm8 = V8Model.load(VC.V8_BUNDLE)
    mm = vm8.mismatch_model
    X_full = mm.transform(X)
    oof = vm8.oof_pool
    baseA = vm8.base_A

    # r_train（2025 OOF，无泄露）
    r_train = np.asarray(oof["actual"] - oof["base_A_oof"], dtype=float)
    mr_train = np.abs(r_train)
    times_train = pd.DatetimeIndex(oof["times"])
    seg_train = np.asarray(oof["seg"], dtype=object)
    train_mean_mag = float(mr_train.mean())

    # r_val（2026 eval-only）
    vs, ve = pd.Timestamp(LC.VAL_START), pd.Timestamp(LC.VAL_END)
    buf = vs - pd.Timedelta(days=7)
    em = (times >= buf) & (times <= ve)
    Xe = X[em]; te = pd.DatetimeIndex(times[em])
    ae = actual.reindex(te)
    Xe_full = mm.transform(Xe)
    predA = np.asarray(baseA.predict_load(Xe_full, pred_load), dtype=float)
    vmask = (te >= vs) & (te <= ve) & ae.notna()
    r_val = (ae[vmask].values.astype(float) - predA[vmask])
    mr_val = np.abs(r_val)
    times_val = te[vmask]
    hours_val = times_val.hour.values.astype(int)
    seg_val = SEG.segment_array(hours_val)

    train_dates = pd.DatetimeIndex(times_train).normalize().unique()
    val_dates = pd.DatetimeIndex(times_val).normalize().unique()

    # ---- A1 日级信号（全连续 pred_load 因果滚动，再按训练/验证日期切）----
    if verbose:
        print("[2/4] A1 日级输入质量信号（仅 pred_load，因果滚动）...")
    pl_all = pred_load.reindex(times)
    sig_all = _pl_daily_signals(pl_all, times)
    sig_all = _standardize(sig_all, train_dates)
    sig_cols = list(sig_all.columns)

    # 日级 |r| 目标
    dtr = pd.DatetimeIndex(times_train).normalize()
    dva = pd.DatetimeIndex(times_val).normalize()
    daily_mr_train = pd.Series(mr_train, index=dtr).groupby(dtr).mean()
    daily_mr_val = pd.Series(mr_val, index=dva).groupby(dva).mean()
    s_tr = sig_all.loc[sig_all.index.isin(train_dates)].join(daily_mr_train.rename("mr")).dropna(subset=["mr"])
    s_va = sig_all.loc[sig_all.index.isin(val_dates)].join(daily_mr_val.rename("mr")).dropna(subset=["mr"])
    daily_mean_mag = float(s_tr["mr"].mean())
    if verbose:
        print(f"      日级 训练日 N={len(s_tr)}  验证日 N={len(s_va)}  "
              f"daily mean|r|_train={daily_mean_mag:.0f}")

    # ===================== A. 单信号跨年稳定性（核心）=====================
    if verbose:
        print("[3/4] A. 单信号跨年稳定性（corr/ratio/auc 2025 vs 2026）...")
    rows = []
    for col in sig_cols:
        c25 = _spearman(s_tr[col].values, s_tr["mr"].values)
        c26 = _spearman(s_va[col].values, s_va["mr"].values)
        m25 = _risk_metrics(s_tr[col].values, s_tr["mr"].values, s_tr[col].values)
        m26 = _risk_metrics(s_va[col].values, s_va["mr"].values, s_tr[col].values)
        rows.append({
            "signal": col, "corr_2025": c25, "corr_2026": c26,
            "ratio_2025": m25["ratio"], "ratio_2026": m26["ratio"],
            "auc_2025": m25["auc"], "auc_2026": m26["auc"],
            "mae_hi_2026": m26["mae_hi"], "mae_global_2026": m26["mae_global"],
            "n_hi_2026": m26["n_hi"],
        })

    # ===================== B. 组合 A1 日级转移 R² + 稳定性 =====================
    bst = _fit_daily_probe(s_tr, s_tr["mr"].values, sig_cols)
    pred_tr = bst.predict(s_tr[sig_cols])
    pred_va = bst.predict(s_va[sig_cols])
    base_mse = float(np.mean((s_va["mr"].values - daily_mean_mag) ** 2))
    probe_mse = float(np.mean((s_va["mr"].values - pred_va) ** 2))
    transfer_r2 = 1.0 - probe_mse / base_mse if base_mse > 0 else float("nan")
    val_mse = float(np.mean((s_va["mr"].values - s_va["mr"].mean()) ** 2))
    std_r2 = 1.0 - probe_mse / val_mse if val_mse > 0 else float("nan")
    # 期内 holdout（2025 内 80/20）
    n = len(s_tr); cut = int(n * 0.8)
    if cut >= 20 and n - cut >= 10:
        bst_h = _fit_daily_probe(s_tr.iloc[:cut], s_tr["mr"].values[:cut], sig_cols)
        ph = bst_h.predict(s_tr[sig_cols].iloc[cut:]); yh = s_tr["mr"].values[cut:]
        vh = float(np.mean((yh - yh.mean()) ** 2))
        holdout_r2 = 1.0 - float(np.mean((yh - ph) ** 2)) / vh if vh > 0 else float("nan")
    else:
        holdout_r2 = float("nan")
    comb_c25 = _spearman(pred_tr, s_tr["mr"].values)
    comb_c26 = _spearman(pred_va, s_va["mr"].values)
    comb_m25 = _risk_metrics(pred_tr, s_tr["mr"].values, pred_tr)
    comb_m26 = _risk_metrics(pred_va, s_va["mr"].values, pred_tr)

    # ===================== C. 点级广播（对照 P-Diag 单 A1 AUC≈0.540）=====================
    dtr_pt = pd.DatetimeIndex(times_train).normalize()
    dva_pt = pd.DatetimeIndex(times_val).normalize()
    sc_tr_pt = pd.Series(pred_tr, index=s_tr.index).reindex(dtr_pt).values
    sc_va_pt = pd.Series(pred_va, index=s_va.index).reindex(dva_pt).values
    pt_m26 = _risk_metrics(sc_va_pt, mr_val, sc_tr_pt)
    # 原 A1(jump_7d) 点级复算，对照 bundled 0.540
    a1_tr_pt = s_tr["jump_7d"].reindex(dtr_pt).values
    a1_va_pt = s_va["jump_7d"].reindex(dva_pt).values
    a1_pt_m26 = _risk_metrics(a1_va_pt, mr_val, a1_tr_pt)

    # ===================== D. 分段稳定性（午间重点）=====================
    seg_rows = {}
    for s in VC.SEGMENTS:
        mtr = seg_train == s; mva = seg_val == s
        if not mva.any() or not mtr.any():
            continue
        m25s = _risk_metrics(sc_tr_pt[mtr], mr_train[mtr], sc_tr_pt[mtr])
        m26s = _risk_metrics(sc_va_pt[mva], mr_val[mva], sc_tr_pt[mtr])
        seg_rows[s] = {"ratio_2025": m25s["ratio"], "ratio_2026": m26s["ratio"],
                       "auc_2025": m25s["auc"], "auc_2026": m26s["auc"]}

    # ===================== E. 判定 =====================
    def _stable(r):
        r25, r26 = r["ratio_2025"], r["ratio_2026"]
        c25, c26 = r["corr_2025"], r["corr_2026"]
        a25, a26 = r["auc_2025"], r["auc_2026"]
        if any(np.isnan(x) for x in [r25, r26, c25, c26, a25, a26]):
            return False
        ratio_same = (r25 > 1 and r26 > 1) or (r25 < 1 and r26 < 1)
        corr_same = (c25 > 0) == (c26 > 0)
        auc_ok = a25 > 0.5 and a26 > 0.5
        return ratio_same and corr_same and auc_ok
    stable_signals = [r["signal"] for r in rows if _stable(r)]
    flipped = [r["signal"] for r in rows
               if (not np.isnan(r["ratio_2025"]) and not np.isnan(r["ratio_2026"])
                   and ((r["ratio_2025"] > 1) != (r["ratio_2026"] > 1)))]
    comb_stable = (not np.isnan(comb_m25["ratio"]) and not np.isnan(comb_m26["ratio"])
                   and ((comb_m25["ratio"] > 1) == (comb_m26["ratio"] > 1))
                   and (not np.isnan(comb_c25) and not np.isnan(comb_c26)
                        and (comb_c25 > 0) == (comb_c26 > 0)))
    # 部署（可作置信度 flag）：稳定 且 (auc_2026>0.55 或 transfer R²>0 或 ratio 两年都>1)
    deployable = bool(comb_stable and (
        (not np.isnan(comb_m26["auc"]) and comb_m26["auc"] > 0.55)
        or transfer_r2 > 0.0
        or (comb_m25["ratio"] > 1 and comb_m26["ratio"] > 1)))

    # ===================== 报告 =====================
    L = []
    L.append("# v8.1 工作流 B：A1 输入异常独立验证（输入质量任务）\n")
    L.append("> Diagnosis 层 - 输入质量（DESIGN §3/§5/§6）。隔离 A1，测其作为**输入质量**信号"
             "（仅 pred_load，非残差方向）的跨年**稳定性**与可迁移性。\n"
             "> A1 信号=输入质量（因果滚动，可部署）；|r|=验证标签（eval-only）。\n"
             "> 对照：P-Diag 报告 A2 比率跨年翻转 1.34->0.88（不稳）；A1 比率稳定性此前未单测。\n")
    L.append(f"r=actual−base_A(v6)。mr_train=|r_train|(2025 OOF, N={len(mr_train)}, "
             f"mean={train_mean_mag:.0f})；mr_val=|r_val|(2026 eval-only, N={len(mr_val)}, "
             f"mean={float(mr_val.mean()):.0f})。日级 训练日 N={len(s_tr)}，验证日 N={len(s_va)}。\n")

    L.append("\n## A. 单信号跨年稳定性（核心：A1 是否翻如 A2）\n")
    L.append("ratio=高分(top20%)子集 mean|r|/全局（>1=高异常子集误差更大=信号有效）。"
             "稳定=ratio 跨年同向 且 corr 同号 且 AUC 两年都>0.5。\n")
    L.append("| 信号 | 2025 corr | 2026 corr | 2025 ratio | 2026 ratio | 2025 AUC | 2026 AUC | "
             "2026 高异常MAE | 2026 全局MAE |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['signal']} | {r['corr_2025']:+.3f} | {r['corr_2026']:+.3f} | "
                 f"{r['ratio_2025']:.2f} | {r['ratio_2026']:.2f} | {r['auc_2025']:.3f} | "
                 f"{r['auc_2026']:.3f} | {r['mae_hi_2026']:.0f} | {r['mae_global_2026']:.0f} |")
    L.append(f"\n- **稳定信号**：{stable_signals if stable_signals else '无'}")
    L.append(f"- **翻转信号(ratio 跨年跨过 1，类 A2)**：{flipped if flipped else '无'}")
    L.append(f"- 对照 A2：ratio 1.34->0.88（翻）。A1 若 ratio 两年同向 -> 不翻 -> 可部署。\n")

    L.append("\n## B. 组合 A1（5 信号）日级跨年迁移\n")
    L.append("| 期内R²(2025holdout) | 跨年transfer R² | 标准R² | 2025 corr | 2026 corr | "
             "2025 ratio | 2026 ratio | 2026 AUC |")
    L.append("|---|---|---|---|---|---|---|---|")
    L.append(f"| {holdout_r2:+.3f} | {transfer_r2:+.3f} | {std_r2:+.3f} | {comb_c25:+.3f} | "
             f"{comb_c26:+.3f} | {comb_m25['ratio']:.2f} | {comb_m26['ratio']:.2f} | "
             f"{comb_m26['auc']:.3f} |")
    L.append(f"\n基线=mean(daily|r|_train)={daily_mean_mag:.0f}（do-nothing）。transfer R²>0=组合 A1 "
             f"跨年可迁移。对照 P-Diag combined 幅度探针 transfer R²=+0.182（用全特征；本处仅 pred_load 输入质量）。\n")

    L.append("\n## C. 点级广播（对照 P-Diag 单 A1 AUC≈0.540）\n")
    L.append(f"- 组合 A1 点级 risk_AUC(2026) = {pt_m26['auc']:.3f}（ratio={pt_m26['ratio']:.2f}）")
    L.append(f"- 原 A1(jump_7d) 点级 risk_AUC(2026) = {a1_pt_m26['auc']:.3f}（对照 bundled ≈0.540）\n")

    L.append("\n## D. 分段稳定性（午间重点）\n")
    L.append("| 段 | 2025 ratio | 2026 ratio | 2025 AUC | 2026 AUC |")
    L.append("|---|---|---|---|---|")
    for s in VC.SEGMENTS:
        if s in seg_rows:
            o = seg_rows[s]
            L.append(f"| {s} | {o['ratio_2025']:.2f} | {o['ratio_2026']:.2f} | "
                     f"{o['auc_2025']:.3f} | {o['auc_2026']:.3f} |")

    L.append("\n## E. 判定（A1 输入质量 flag 可部署性）\n")
    L.append(f"1. **组合稳定**：{'是' if comb_stable else '否'}（ratio 跨年同向 + corr 同号）")
    L.append(f"2. **稳定单信号**：{stable_signals if stable_signals else '无'}")
    L.append(f"3. **翻转单信号**：{flipped if flipped else '无'}")
    L.append(f"4. **组合 transfer R²**：{transfer_r2:+.3f}；组合 risk_AUC(2026)={comb_m26['auc']:.3f}")
    L.append(f"5. **A1 判定**：{'**GO（可部署 flag）** - 输入质量信号跨年稳定' if deployable else '**NO-GO（当前特征下）** - 输入质量信号跨年不稳或无信号'}")
    if deployable:
        L.append(f"   -> A1 弱但**稳**（ratio 两年同向={comb_m25['ratio']:.2f}/{comb_m26['ratio']:.2f}，"
                 f"AUC={comb_m26['auc']:.3f}），可作运行时置信度 flag：输入异常->预测不可靠标注。"
                 f"与 A2 天气 OOD 翻转(1.34->0.88)对比，输入质量比天气 OOD 更可部署。")
        L.append(f"   -> 价值=Forecast QA 的一个 cause 标签（Input anomaly），非 MAE 杠杆（符号通道仍关）。"
                 f"接入工作流 C 运行时 QA。")
    else:
        L.append(f"   -> A1 输入质量信号跨年翻转或无信号（ratio {comb_m25['ratio']:.2f}->"
                 f"{comb_m26['ratio']:.2f}，corr {comb_c25:+.3f}->{comb_c26:+.3f}），同 A2 不稳。"
                 f"输入质量 flag 当前特征下不可部署。")

    report = "\n".join(L)
    REPORT.write_text(report, encoding="utf-8")
    if verbose:
        print("[4/4] 报告生成。")
        print("\n" + report)
        print(f"\n报告已写: {REPORT}")
    return {
        "rows": rows, "transfer_r2": transfer_r2, "holdout_r2": holdout_r2,
        "comb_c25": comb_c25, "comb_c26": comb_c26,
        "comb_ratio_25": comb_m25["ratio"], "comb_ratio_26": comb_m26["ratio"],
        "comb_auc_26": comb_m26["auc"], "pt_auc_26": pt_m26["auc"],
        "a1_pt_auc_26": a1_pt_m26["auc"], "seg_rows": seg_rows,
        "stable_signals": stable_signals, "flipped": flipped,
        "comb_stable": comb_stable, "deployable": deployable,
    }


def main():
    run(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
