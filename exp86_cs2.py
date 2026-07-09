# -*- coding: utf-8 -*-
"""exp86_cs2 - CS2 峰值收缩缓解的 walk-forward 跨年稳定性验证。

FDS/report.md 候选 CS2：峰值（actual≥P95）低估 −2 GW 系模型收缩（val 峰值 < 训练 max，非外推），
理论上可缓解；报告提两种机制：① 上调 residual 成员 quantile alpha（全局减收缩）② 峰值感知后验校正。
风险：exp74 证明负荷"特征"过拟合。开放问题：能否用非特征机制可迁移地缓解峰值收缩。

【关键诊断前提】(a03 表)：pred_load 条件偏差与 actual 条件偏差**方向相反**--
  高 pred_load(P90-100) 模型偏差 +382.8（过估）；高 actual(P90-100) 模型偏差 −1067.8（低估）。
  即峰值低估点（高 actual）的 pred_load 并不高（外部预测漏峰），而高 pred_load 点反而过估。
  => 基于 pred_load 的峰值校正会定位到**错误**的点；峰值低估本质是外部预测漏峰 = ext_error(83.7% 不可学)。

本实验用 walk-forward 实证两种机制是否能可迁移地【降低峰值收缩】，而非仅降单窗 MAE。
主指标 = peak-Δ（actual≥P95(val) 子集 MAE 的 treat−base；CS2 的真实目标），次指标 = overall-Δ。

设计（与 exp85 同构）：
  Part 0 诊断前提复核：官方 val 上 pred_load/actual 条件的外部预测偏差对立（无模型，纯数据）。
  Part A  机制①（quantile alpha 上调，固定 +0.10 不重搜）：4 折【原始集成】walk-forward
          baseline alpha=[0.45,0.5,0.55] vs treat=[0.55,0.6,0.65]，每折就地 mismatch/MOS 重拟合。
  Part B  机制②（pred_load≥P95(train) 峰值感知后验校正，shrinkage=1.0 固定）：4 折【全管线】
          复用 compute_hour_bias（cfg.threshold_corr 追加 pred_load 项，thr=train P95，fold 子折 OOF 估
          shift，无泄露）。treat=含 pred_load 校正 / base=不含；同一终集成，仅后验项不同。

合规：阈值/alpha 固定不重搜；shift 全部训练期 OOF 估计；不用官方 val 选参；不修改生产代码/模型；不写产物。
判定：CS2 达成目标 = peak-Δ 跨折一致<0（可迁移地降低峰值收缩）。
"""
from __future__ import annotations
import io, sys, copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from load_pred import config as C, data_loader as dl, features as F, train as T

BASE_ALPHAS = [0.45, 0.5, 0.55]      # v6 生产
TREAT_ALPHAS = [0.55, 0.6, 0.65]     # 固定 +0.10（诊断峰值低估 -2016 MW 为据，非 val 搜索）
DEC = int(C.TRAIN_CONFIG.get("round_decimals", 2))

FOLDS = [
    ("spring25", "2025-02-28 23:45:00", "2025-03-01 00:00:00", "2025-05-31 23:45:00"),
    ("autumn25", "2025-08-31 23:45:00", "2025-09-01 00:00:00", "2025-11-30 23:45:00"),
    ("winter26", "2025-12-31 23:45:00", "2026-01-01 00:00:00", "2026-02-28 23:45:00"),
    ("official", C.TRAIN_END,            C.VAL_START,            C.VAL_END),
]


def mae_on(pred, act, mask=None):
    p = np.round(np.asarray(pred, dtype=float), DEC)
    if mask is not None:
        return float(np.abs(p[mask] - act[mask]).mean()) if mask.any() else float("nan")
    return float(np.abs(p - act).mean())


def make_subfolds(train_end, n=3):
    te = pd.Timestamp(train_end)
    ts0 = pd.Timestamp(C.TRAIN_CONFIG["train_start"])
    min_train = pd.DateOffset(months=6)
    folds, end = [], te
    for _ in range(n):
        val_start = end - pd.DateOffset(months=3) + pd.Timedelta(minutes=15)
        train_cutoff = val_start - pd.Timedelta(minutes=15)
        if train_cutoff < ts0 + min_train:
            break
        folds.append((train_cutoff, val_start, end))
        end = train_cutoff - pd.Timedelta(minutes=15)
    return list(reversed(folds))


def get_masks(name, te, vs, ve, times, usable, actual, pred_load):
    te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
    train_mask = usable if name == "official" else (usable & (times <= te))
    vmask = ((times >= vs) & (times <= ve) & actual.notna() & pred_load.notna()).values
    return train_mask, vmask


# ---------------- Part A: 机制① quantile alpha（原始集成）----------------
def raw_fold_alpha(times, X_base, pred_load, actual, cfg, best_it, train_mask, vmask, alphas):
    mm = F.MismatchModel().fit(X_base, train_mask); X = mm.transform(X_base)
    mos = F.MosModel().fit(X, actual, train_mask)
    cfg_f = dict(cfg); cfg_f["quantile_alphas"] = list(alphas)
    model = T.train_ensemble(times, X, pred_load, actual, train_mask, cfg_f, best_it, mos_model=mos)
    model.mismatch_model = mm
    return model.predict_load(X[vmask], pred_load[vmask])


def part_a(times, X_base, pred_load, actual, usable, cfg, best_it):
    print("\n==== Part A: 机制① quantile alpha 上调（原始集成，固定 +0.10 不重搜）====", flush=True)
    print(f"baseline alpha={BASE_ALPHAS}  treat alpha={TREAT_ALPHAS}", flush=True)
    print(f"{'折':<10}{'n_peak':>7}{'base_MAE':>10}{'treat_MAE':>11}{'overallΔ':>9}"
          f"{'base_peak':>10}{'treat_peak':>12}{'peakΔ':>8}", flush=True)
    rows = []
    for name, te, vs, ve in FOLDS:
        train_mask, vmask = get_masks(name, te, vs, ve, times, usable, actual, pred_load)
        act_v = actual[vmask].values
        peak = act_v >= np.quantile(act_v, 0.95)
        pb = raw_fold_alpha(times, X_base, pred_load, actual, cfg, best_it, train_mask, vmask, BASE_ALPHAS)
        pt = raw_fold_alpha(times, X_base, pred_load, actual, cfg, best_it, train_mask, vmask, TREAT_ALPHAS)
        b, t = mae_on(pb, act_v), mae_on(pt, act_v)
        bp, tp = mae_on(pb, act_v, peak), mae_on(pt, act_v, peak)
        rows.append(dict(name=name, n_peak=int(peak.sum()), b=b, t=t, d=t - b, bp=bp, tp=tp, dp=tp - bp))
        print(f"{name:<10}{int(peak.sum()):>7}{b:>10.1f}{t:>11.1f}{t-b:>+9.1f}{bp:>10.1f}{tp:>12.1f}{tp-bp:>+8.1f}", flush=True)
    return rows


# ---------------- Part B: 机制② pred_load 峰值感知后验校正（全管线）----------------
def part_b(times, X_base, pred_load, actual, usable, cfg, best_it):
    print("\n==== Part B: 机制② pred_load≥P95(train) 峰值感知后验校正（全管线）====", flush=True)
    print(f"{'折':<10}{'n_peak':>7}{'shift_est':>11}{'base_MAE':>10}{'treat_MAE':>11}{'overallΔ':>9}"
          f"{'base_peak':>10}{'treat_peak':>12}{'peakΔ':>8}", flush=True)
    rows = []
    for name, te, vs, ve in FOLDS:
        train_mask, vmask = get_masks(name, te, vs, ve, times, usable, actual, pred_load)
        mm = F.MismatchModel().fit(X_base, train_mask); X = mm.transform(X_base)
        mos = F.MosModel().fit(X, actual, train_mask)

        subfolds = ([(pd.Timestamp(a), pd.Timestamp(b), pd.Timestamp(c)) for a, b, c in cfg["best_it_folds"]]
                    if name == "official" else make_subfolds(te, n=3))
        train_p95 = float(np.quantile(pred_load[train_mask].values, 0.95))
        cfg_f = dict(cfg); cfg_f["best_it_folds"] = subfolds
        cfg_f["threshold_corr"] = list(cfg["threshold_corr"]) + [
            {"feature": "pred_load", "op": ">", "thr": train_p95, "hours": None, "shrinkage": 1.0}]
        print(f"  --- fold={name} subfolds={len(subfolds)} train_P95(pred_load)={train_p95:.0f}", flush=True)
        hour_bias, drift_corr, threshold_corr = T.compute_hour_bias(
            times, X, pred_load, actual, train_mask, cfg_f, best_it, mos_model=mos)
        pl_item = next((tc for tc in threshold_corr if tc["feature"] == "pred_load"), None)
        shift_est = float(pl_item["shift"]) if pl_item else 0.0

        model = T.train_ensemble(times, X, pred_load, actual, train_mask, cfg, best_it, mos_model=mos)
        model.mismatch_model = mm; model.hour_bias = hour_bias; model.drift_corr = drift_corr
        act_v = actual[vmask].values
        peak = act_v >= np.quantile(act_v, 0.95)

        model.threshold_corr = threshold_corr  # treat：含 pred_load 校正
        pt = model.predict_load(X[vmask], pred_load[vmask])
        model.threshold_corr = [tc for tc in threshold_corr if tc["feature"] != "pred_load"]  # base：不含
        pb = model.predict_load(X[vmask], pred_load[vmask])
        b, t = mae_on(pb, act_v), mae_on(pt, act_v)
        bp, tp = mae_on(pb, act_v, peak), mae_on(pt, act_v, peak)
        rows.append(dict(name=name, n_peak=int(peak.sum()), shift=shift_est, b=b, t=t, d=t - b, bp=bp, tp=tp, dp=tp - bp))
        print(f"{name:<10}{int(peak.sum()):>7}{shift_est:>+11.1f}{b:>10.1f}{t:>11.1f}{t-b:>+9.1f}"
              f"{bp:>10.1f}{tp:>12.1f}{tp-bp:>+8.1f}", flush=True)
    return rows


def main():
    cfg = C.TRAIN_CONFIG; best_it = cfg["best_it_fixed"]
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)
    X_base = F.build_features(times, pred_load, weather)
    usable = T.usable_mask(times, pred_load, actual)
    print(f"CS2 峰值收缩缓解跨年稳定性验证  特征数={X_base.shape[1]}  best_it={best_it}", flush=True)

    # ---- Part 0: 诊断前提复核（纯数据，无模型）----
    print("\n==== Part 0: 诊断前提复核（官方 val 外部预测偏差结构）====", flush=True)
    vmask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna() & pred_load.notna()).values
    pl_v = pred_load[vmask].values; ac_v = actual[vmask].values
    ext = pl_v - ac_v  # 外部预测误差
    hp = pl_v >= np.quantile(pl_v, 0.95)   # 高 pred_load
    ha = ac_v >= np.quantile(ac_v, 0.95)   # 高 actual
    print(f"  高 pred_load(P95+): 外部偏差={np.mean(ext[hp]):+.1f} MW (n={hp.sum()})", flush=True)
    print(f"  高 actual  (P95+): 外部偏差={np.mean(ext[ha]):+.1f} MW (n={ha.sum()})", flush=True)
    print(f"  => 方向{'一致' if np.sign(np.mean(ext[hp]))==np.sign(np.mean(ext[ha])) else '相反'}："
          f"高 pred_load {'过估' if np.mean(ext[hp])>0 else '低估'} vs 高 actual {'过估' if np.mean(ext[ha])>0 else '低估'}", flush=True)
    print(f"  解读：峰值低估（高 actual）点 pred_load 不高（外部漏峰），pred_load 校正无法定位->峰值收缩本质=ext_error", flush=True)

    rows_a = part_a(times, X_base, pred_load, actual, usable, cfg, best_it)
    rows_b = part_b(times, X_base, pred_load, actual, usable, cfg, best_it)

    # ---- 判定 ----
    print("\n================ CS2 判定 ================", flush=True)

    def mech_status(rows):
        inner = [r for r in rows if r["name"] != "official"]
        peak_ok = all(r["dp"] < 0 for r in inner)          # 跨内折一致降低峰值收缩
        overall_ok = all(r["d"] < 10 for r in inner)        # 无显著整体代价（整体 Δ<+10）
        return peak_ok, overall_ok

    for label, rows in [("机制①alpha", rows_a), ("机制②load校正", rows_b)]:
        peak_deltas = [r["dp"] for r in rows]
        overall_deltas = [r["d"] for r in rows]
        pk, ov = mech_status(rows)
        print(f"\n[{label}] peak-Δ(各折)={[round(d,1) for d in peak_deltas]}  "
              f"overall-Δ(各折)={[round(d,1) for d in overall_deltas]}", flush=True)
        print(f"  peak-Δ跨内折一致<0: {pk}   无显著整体代价(各折Δ<+10): {ov}", flush=True)
    a_peak, a_ovl = mech_status(rows_a)
    b_peak, b_ovl = mech_status(rows_b)
    a_valid = a_peak and a_ovl
    b_valid = b_peak and b_ovl
    if a_valid or b_valid:
        verdict = "VALIDATED 某机制可迁移降低峰值收缩且无显著整体代价"
    elif a_peak and not a_ovl:
        verdict = "REJECTED 机制①虽降峰值但整体代价过大（全局上移过估非峰值点）-> 非可行方案"
    elif not a_peak and not b_peak:
        verdict = "REJECTED 两机制均无法可迁移降低峰值收缩--峰值收缩不可从 pred_load 定位（=ext_error 不可学）"
    else:
        verdict = "REJECTED 机制不稳定--峰值收缩不可迁移缓解"
    print(f"\n综合结论: {verdict}", flush=True)


if __name__ == "__main__":
    main()
