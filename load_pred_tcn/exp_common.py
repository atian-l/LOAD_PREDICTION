# -*- coding: utf-8 -*-
"""
TCN 调优实验共享 harness（exp_common）。

提供所有 exp_*.py 复用的原语：数据缓存、参数化训练、3 折 walk-forward OOF、
OOF 校正估计、原始/全校正集成预测、val 评估、统一对比表打印。

设计原则（见 .claude/plans/plan.md）：
  - 选择信号用 WF-CV（训练期 3 折 best_it_folds），官方 val 仅读数、不参与选择。
  - 复用生产 train.build_dataset / train_ensemble / compute_hour_bias / _evaluate，
    通过 cfg 覆盖传超参（标准化/样本权重/OOF 校正逻辑全部复用，实验与生产同构）。
  - 不写生产 artifact（不存模型到 models/、不覆盖 model_bundle.pkl）。
  - 缩成员快速模式：cfg 覆盖 seeds/objectives/residual_modes（如 5 成员广搜）。

运行：从项目根目录 `python -m load_pred_tcn.exp_fit_diag` 等。
"""
from __future__ import annotations
import contextlib
import io
import sys
import time

import numpy as np
import pandas as pd

from . import config as C
from . import features as F
from .train import build_dataset, usable_mask, train_ensemble, _evaluate

# 参考基线
V6_VAL_MAE = 1445.62        # LightGBM v6 官方 val MAE（生产基线）
TCN_BASE_MAE = 2027.47      # TCN 60ep/dropout0.1/wd1e-5 标准化修复后基线

# 缩成员快速模式（广搜用；top 配置由用户用全量 40 成员 train.py 复跑确认）
# SINGLE_MEM=1 成员（direct-regression）：最便宜的 WF-CV 探针，用于大网格 Stage A
# QUICK_MEM=2 成员（direct+residual regression）：保留残差多样性，Stage B val 读数用
SINGLE_MEM = {"seeds": [42], "objectives": ["regression"], "residual_modes": [False]}
QUICK_MEM = {"seeds": [42], "objectives": ["regression"], "residual_modes": [False, True]}

_CACHE: dict = {}


# --------------------------------------------------------------------------- #
# 指标工具
# --------------------------------------------------------------------------- #
def _mae(pred, actual):
    a = np.asarray(actual, dtype=float); p = np.asarray(pred, dtype=float)
    m = np.isfinite(a) & np.isfinite(p)
    return float(np.mean(np.abs(p[m] - a[m]))) if m.sum() else float("nan")


def _r2(pred, actual):
    a = np.asarray(actual, dtype=float); p = np.asarray(pred, dtype=float)
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    ss_res = float(np.sum((p - a) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _debiased(err):
    """去均值绝对误差（剔除 bias 后的纯波动地板）。"""
    e = np.asarray(err, dtype=float)
    m = np.isfinite(e)
    e = e[m]
    return float(np.mean(np.abs(e - e.mean()))) if len(e) else float("nan")


def _midday_mae(pred, actual, times):
    """午间 11-14 点 MAE（该项目已知最难时段）。"""
    h = pd.DatetimeIndex(times).hour.values
    sel = (h >= 11) & (h < 15)
    if not sel.any():
        return float("nan")
    return _mae(np.asarray(pred)[sel], np.asarray(actual)[sel])


def _metrics(pred_vals, actual_vals, times):
    err = np.asarray(pred_vals, dtype=float) - np.asarray(actual_vals, dtype=float)
    return {
        "MAE": _mae(pred_vals, actual_vals),
        "R2": _r2(pred_vals, actual_vals),
        "Bias": float(np.nanmean(err)),
        "debiased": _debiased(err),
        "midday": _midday_mae(pred_vals, actual_vals, times),
    }


# --------------------------------------------------------------------------- #
# 数据缓存（一次性构造）
# --------------------------------------------------------------------------- #
def build_cached():
    """build_dataset + MismatchModel + MosModel 一次性构造，缓存复用。返回 dict。"""
    if _CACHE:
        return _CACHE
    times, X, pred_load, actual = build_dataset()
    usable = usable_mask(times, pred_load, actual)
    mm = F.MismatchModel().fit(X, usable)
    X = mm.transform(X)
    mos_model = None
    if C.TRAIN_CONFIG.get("mos"):
        mc = C.TRAIN_CONFIG["mos"]
        mos_model = F.MosModel(cols=mc.get("cols"), alpha=mc.get("alpha", 1.0)).fit(X, actual, usable)
    anchor = (pd.Series(mos_model.transform(X), index=X.index)
              if mos_model is not None else pred_load)
    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    _CACHE.update(dict(times=times, X=X, pred_load=pred_load, actual=actual, usable=usable,
                       mos_model=mos_model, anchor=anchor, feat_cols=list(X.columns),
                       val_m=val_m))
    return _CACHE


def _cfg(override: dict | None = None) -> dict:
    cfg = dict(C.TRAIN_CONFIG)
    if override:
        cfg.update(override)
    return cfg


# --------------------------------------------------------------------------- #
# 参数化训练（复用生产 train_ensemble）
# --------------------------------------------------------------------------- #
def train_ens(override: dict | None = None, usable=None, best_it=None,
              mos_model=None, verbose=False):
    """参数化训练 TCN 集成。override 合并入 TRAIN_CONFIG。返回 EnsembleModel（无 OOF 校正：
    hour_bias=None / drift_corr=[] / threshold_corr=[]，故 predict_load 返回原始集成预测）。"""
    d = build_cached()
    if usable is None:
        usable = d["usable"]
    if mos_model is None:
        mos_model = d["mos_model"]
    cfg = _cfg(override)
    if best_it is None:
        best_it = cfg["best_it_fixed"]
    if verbose:
        model = train_ensemble(d["times"], d["X"], d["pred_load"], d["actual"],
                               usable, cfg, best_it, mos_model=mos_model)
    else:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model = train_ensemble(d["times"], d["X"], d["pred_load"], d["actual"],
                                   usable, cfg, best_it, mos_model=mos_model)
    return model


def ensemble_raw(model, X_sub, pred_load_sub):
    """无校正的原始集成预测（anchor+λ(ens-anchor)，含 clip；无 hour_bias/drift/threshold）。
    model 须为 train_ens 返回（未设校正）。"""
    return np.asarray(model.predict_load(X_sub, pred_load_sub), dtype=float)


def apply_corrections(model, hour_bias, drift_corr, threshold_corr):
    """把 OOF 校正装到 model 上（后续 predict_load 即含校正）。"""
    model.hour_bias = None if hour_bias is None else np.asarray(hour_bias, dtype=float)
    model.drift_corr = list(drift_corr or [])
    model.threshold_corr = list(threshold_corr or [])
    return model


# --------------------------------------------------------------------------- #
# 3 折 walk-forward OOF（训练期内，无泄露）
# --------------------------------------------------------------------------- #
def _oof_loop(override: dict | None = None, best_it=None, mos_model=None, verbose=False):
    """3 折 walk-forward OOF：每折在 ftr(usable&<=te) 上训练、预测 fva(vs<=t<=ve)。
    返回 (oof_pred: Series, oof_mask: bool array)。复用生产 train_ensemble（含 cfg 覆盖）。"""
    d = build_cached()
    times, X, pred_load, actual, usable = (d["times"], d["X"], d["pred_load"],
                                           d["actual"], d["usable"])
    cfg = _cfg(override)
    if best_it is None:
        best_it = cfg["best_it_fixed"]
    oof_pred = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        m = train_ens(override, usable=ftr, best_it=best_it, mos_model=mos_model, verbose=verbose)
        oof_pred[fva] = ensemble_raw(m, X[fva], pred_load[fva])
    oof_mask = usable & oof_pred.notna().values
    return oof_pred, oof_mask


def estimate_corrections(oof_pred, oof_mask, cfg, times, X, actual):
    """由 OOF 残差估计 hour_bias / drift_corr / threshold_corr（逻辑与 train.compute_hour_bias
    逐行一致，仅改为接收已有 oof_pred，避免重复 OOF 训练）。无泄露（仅训练期 OOF 残差）。"""
    resid = (oof_pred - actual).values
    n_slots = int(cfg.get("hour_bias_slots", 24))
    step = 1440 // n_slots
    dt_all = pd.DatetimeIndex(times)
    mod_all = dt_all.hour.values * 60 + dt_all.minute.values
    slot_all = (mod_all // step).astype(int)
    h_all = dt_all.hour.values
    hour_bias = np.zeros(n_slots, dtype=float)
    for q in range(n_slots):
        m = oof_mask & (slot_all == q)
        if m.sum():
            hour_bias[q] = float(np.average(resid[m]))

    drift_corr = []
    dc = cfg.get("drift_corr")
    if dc:
        fn = dc["feature"]; hs = set(dc["hours"])
        feat = X[fn].values.astype(float)
        beta = np.zeros(24, dtype=float)
        for h in range(24):
            if h not in hs:
                continue
            m = oof_mask & (h_all == h)
            f = feat[m]; e = resid[m]
            good = np.isfinite(f) & np.isfinite(e)
            d = float(np.dot(f[good], f[good]))
            if d > 0:
                beta[h] = float(np.dot(f[good], e[good]) / d)
        drift_corr.append((fn, beta))

    threshold_corr = []
    for tc in cfg.get("threshold_corr", []):
        fn = tc["feature"]; op = tc.get("op", ">"); thr = tc["thr"]
        hl = tc["hours"]; shrink = float(tc["shrinkage"])
        feat = X[fn].values.astype(float)
        if op == "range":
            m = oof_mask & (feat >= thr[0]) & (feat < thr[1])
        elif op == ">=":
            m = oof_mask & (feat >= thr)
        elif op == "<":
            m = oof_mask & (feat < thr)
        elif op == "<=":
            m = oof_mask & (feat <= thr)
        else:
            m = oof_mask & (feat > thr)
        if hl is not None:
            m = m & np.isin(h_all, list(hl))
        shift = float(np.average(resid[m])) * shrink if m.sum() else 0.0
        threshold_corr.append({"feature": fn, "op": op, "thr": thr, "hours": hl, "shift": shift})
    return hour_bias, drift_corr, threshold_corr


def compute_oof(override: dict | None = None, best_it=None, mos_model=None, verbose=False):
    """3 折 OOF -> (hour_bias, drift_corr, threshold_corr, oof_pred, oof_mask, fold_mae[])。
    一次 OOF 训练循环同时给出校正与折内 MAE（WF-CV 选择信号）。"""
    d = build_cached()
    times, X, actual = d["times"], d["X"], d["actual"]
    cfg = _cfg(override)
    oof_pred, oof_mask = _oof_loop(override, best_it=best_it, mos_model=mos_model, verbose=verbose)
    hour_bias, drift_corr, threshold_corr = estimate_corrections(
        oof_pred, oof_mask, cfg, times, X, actual)
    fold_mae = []
    for te, vs, ve in cfg["best_it_folds"]:
        vs, ve = pd.Timestamp(vs), pd.Timestamp(ve)
        fm = oof_mask & (times >= vs) & (times <= ve)
        if fm.sum():
            fold_mae.append(float(np.mean(np.abs(oof_pred[fm].values - actual[fm].values))))
    arr = np.array(fold_mae)
    return {
        "hour_bias": hour_bias, "drift_corr": drift_corr, "threshold_corr": threshold_corr,
        "oof_pred": oof_pred, "oof_mask": oof_mask, "fold_mae": fold_mae,
        "wfcv_mean": float(arr.mean()) if len(arr) else float("nan"),
        "wfcv_std": float(arr.std()) if len(arr) else float("nan"),
        "wfcv_cv": float(arr.std() / arr.mean()) if len(arr) >= 2 and arr.mean() > 0 else float("nan"),
    }


def wf_cv(override: dict | None = None, best_it=None, mos_model=None, verbose=False):
    """仅 WF-CV（折内 MAE 均值/std/cv），不含校正估计。用于纯选择信号。"""
    r = compute_oof(override, best_it=best_it, mos_model=mos_model, verbose=verbose)
    return {"fold_mae": r["fold_mae"], "mean": r["wfcv_mean"],
            "std": r["wfcv_std"], "cv": r["wfcv_cv"]}


# --------------------------------------------------------------------------- #
# 单配置全流程（HP 搜索脚本用：训练+OOF+val raw/full 指标）
# --------------------------------------------------------------------------- #
def run_config(tag: str, override: dict | None = None, best_it=None,
               mos_model=None, verbose=False) -> dict:
    """跑一个配置：OOF(校正+折MAE) + 全量训练 -> val_raw/val_full 指标。返回指标 dict。
    选择信号 = wfcv_mean（不用 val）；val 仅读数。"""
    d = build_cached()
    times, X, pred_load, actual, usable, val_m = (d["times"], d["X"], d["pred_load"],
                                                  d["actual"], d["usable"], d["val_m"])
    cfg = _cfg(override)
    if best_it is None:
        best_it = cfg["best_it_fixed"]
    ts = time.perf_counter()
    # OOF（校正 + 折 MAE）
    oof = compute_oof(override, best_it=best_it, mos_model=mos_model, verbose=verbose)
    # 全量训练（usable）-> val_raw
    model = train_ens(override, usable=usable, best_it=best_it, mos_model=mos_model, verbose=verbose)
    val_raw = ensemble_raw(model, X[val_m], pred_load[val_m])
    # val_full（装校正后）
    apply_corrections(model, oof["hour_bias"], oof["drift_corr"], oof["threshold_corr"])
    val_full = ensemble_raw(model, X[val_m], pred_load[val_m])
    va = actual[val_m].values
    vt = times[val_m]
    m_raw = _metrics(val_raw, va, vt)
    m_full = _metrics(val_full, va, vt)
    oof_mae = _mae(oof["oof_pred"].values[oof["oof_mask"]], actual.values[oof["oof_mask"]])
    return {
        "tag": tag, "MAE": m_full["MAE"], "MAE_raw": m_raw["MAE"],
        "Bias": m_full["Bias"], "R2": m_full["R2"], "debiased": m_full["debiased"],
        "midday": m_full["midday"], "oof_mae": oof_mae,
        "wfcv_mean": oof["wfcv_mean"], "wfcv_std": oof["wfcv_std"], "wfcv_cv": oof["wfcv_cv"],
        "fold_mae": oof["fold_mae"], "dt": time.perf_counter() - ts,
    }


# --------------------------------------------------------------------------- #
# 对比表打印
# --------------------------------------------------------------------------- #
def print_table(rows: list[dict], title: str, ref_mae: float = V6_VAL_MAE,
                extra_cols: tuple = ()):
    """打印配置对比表。rows 为 run_config 返回的 dict 列表。"""
    print("\n" + "=" * 78)
    print(title + f"   (val vs v6={ref_mae})")
    print("=" * 78)
    cols = ["tag", "MAE", "MAE_raw", "oof_mae", "wfcv_mean", "wfcv_cv", "Bias",
            "debiased", "midday"] + list(extra_cols)
    head = f"{'tag':16} {'MAE':>8} {'raw':>8} {'oof':>8} {'wfcv':>8} {'cv':>6} {'Bias':>8} {'debia':>7} {'午间':>6}"
    print(head)
    for r in rows:
        print(f"{r['tag']:16} {r['MAE']:>8.1f} {r['MAE_raw']:>8.1f} {r['oof_mae']:>8.1f} "
              f"{r['wfcv_mean']:>8.1f} {r['wfcv_cv']:>6.3f} {r['Bias']:>+8.1f} "
              f"{r['debiased']:>7.1f} {r['midday']:>6.0f}")
    if rows:
        best = min(rows, key=lambda r: r["wfcv_mean"])  # 按 WF-CV 选（不用 val）
        print(f"\nWF-CV 最优: {best['tag']}  wfcv={best['wfcv_mean']:.1f}  "
              f"val={best['MAE']:.1f} (Δv6 {best['MAE']-ref_mae:+.1f})  "
              f"debiased={best['debiased']:.1f}  折CV={best['wfcv_cv']:.3f}  ({best['dt']:.0f}s)")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# 两阶段 HP 搜索（HP-sweep 脚本通用）
#   Stage A：用 SINGLE_MEM（1 成员）跑 wf_cv 做大网格廉价筛选 -> 选 wfcv_mean 最低的 top-K
#   Stage B：top-K 用 QUICK_MEM（2 成员）跑 run_config 出 val 读数（仍快速；最终全量 40 成员由
#            用户 train.py 复跑确认）。
# 选择信号始终为 wfcv_mean（WF-CV），val 仅读数、不参与选择（防 val 过拟合/泄露）。
# --------------------------------------------------------------------------- #
def print_stage_a(stage_a: list[dict], title: str) -> None:
    """打印 Stage A 全配置 WF-CV 表（按 wfcv_mean 升序）。"""
    print("\n" + "-" * 78)
    print(f"[Stage A] {title}  (1 成员 WF-CV 筛选；按 wfcv 升序)")
    print("-" * 78)
    print(f"{'tag':28} {'wfcv':>8} {'std':>7} {'cv':>6}   折MAE")
    for r in sorted(stage_a, key=lambda x: x["wfcv"]):
        folds = "[" + ",".join(f"{x:.0f}" for x in r["folds"]) + "]"
        mark = " *" if r.get("top") else ""
        print(f"{r['tag']:28} {r['wfcv']:>8.1f} {r['std']:>7.1f} {r['cv']:>6.3f}   {folds}{mark}")
    print("-" * 78)


def hp_sweep(configs: list[tuple[str, dict]], title: str,
             single: dict | None = None, quick: dict | None = None,
             topk: int = 3, best_it=None, ref_mae: float = V6_VAL_MAE,
             verbose=False):
    """两阶段 HP 搜索。

    configs : [(tag, override), ...]  override 合并入 cfg（与 SINGLE/QUICK 成员覆盖合并）
    single  : Stage A 成员覆盖（默认 SINGLE_MEM）
    quick   : Stage B 成员覆盖（默认 QUICK_MEM）
    topk    : Stage B 确认的配置数
    返回 (rows_B, stage_a)：rows_B 为 run_config 结果列表，stage_a 为全配置 WF-CV 记录。
    """
    single = single or SINGLE_MEM
    quick = quick or QUICK_MEM
    build_cached()
    # ---- Stage A：廉价 WF-CV 筛选 ----
    stage_a = []
    for i, (tag, ov) in enumerate(configs):
        ov_a = {**single, **ov}
        r = wf_cv(ov_a, best_it=best_it, verbose=verbose)
        stage_a.append({"tag": tag, "override": ov, "wfcv": r["mean"],
                        "std": r["std"], "cv": r["cv"], "folds": r["fold_mae"]})
        print(f"  [A {i+1}/{len(configs)}] {tag:28} wfcv={r['mean']:.1f} "
              f"±{r['std']:.1f} cv={r['cv']:.3f}")
    order = sorted(stage_a, key=lambda x: x["wfcv"])
    for r in order[:topk]:
        r["top"] = True
    print_stage_a(stage_a, title)
    # ---- Stage B：top-K val 读数 ----
    rows = []
    for r in [x for x in order if x.get("top")]:
        ov_b = {**quick, **r["override"]}
        rc = run_config(r["tag"], ov_b, best_it=best_it, verbose=verbose)
        rows.append(rc)
        print(f"  [B] {r['tag']:28} wfcv={r['wfcv']:.1f} -> val={rc['MAE']:.1f} "
              f"raw={rc['MAE_raw']:.1f} oof={rc['oof_mae']:.1f} ({rc['dt']:.0f}s)")
    print_table(rows, f"{title} [Stage B: top{topk} 2成员 val 读数]", ref_mae=ref_mae)
    return rows, stage_a
