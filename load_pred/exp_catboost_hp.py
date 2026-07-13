# -*- coding: utf-8 -*-
"""
Phase 3-B1：CatBoost 超参搜索（聚焦缩小 vs v6 的结构性 38MW 差距）

3-A 结论：CatBoost 纯误差地板 debiased~1483 > v6 1445，非 bias/过拟合，是结构性。
本轮重点方向：
  - grow_policy=Lossguide（leaf-wise 树）对齐 v6 LightGBM leaf-wise+num_leaves=255 的核心优势
  - min_data_in_leaf=200 对齐 v6（3-A 默认 1，叶子过细疑致 debiased 偏高）
  - depth/l2/lr 常规扫

每配置 best_it 固定 80（3-A 最优点），完整管线（40 成员 + OOF 校正重估）。
Lossguide 在 GPU 支持未验证（rsm 已知不支持），逐配置 try/except。

运行：python -m load_pred.exp_catboost_hp   （4090 上约 10~18 min）
"""
from __future__ import annotations
import sys
import time
import io
import contextlib
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from . import config as C
from .train import build_dataset, usable_mask, _time_weights
from .features import MismatchModel, MosModel
from .exp_catboost_ab import _predict_load, _metrics, _arr, V6_VAL_MAE
from . import exp_catboost_ab as ab

# (tag, depth, lr, l2, bagging_temp, grow_policy, max_leaves, best_it)
CONFIGS = [
    ("base_d8",      8, 0.03, 4.0, 1.0, "SymmetricTree", None, 80),
    ("d6",           6, 0.03, 4.0, 1.0, "SymmetricTree", None, 80),
    ("d10",         10, 0.03, 4.0, 1.0, "SymmetricTree", None, 80),
    ("l2_2",         8, 0.03, 2.0, 1.0, "SymmetricTree", None, 80),
    ("l2_8",         8, 0.03, 8.0, 1.0, "SymmetricTree", None, 80),
    ("lr001_bi160",  8, 0.01, 4.0, 1.0, "SymmetricTree", None, 160),
    ("lossguide255", 8, 0.03, 4.0, 1.0, "Lossguide", 255, 80),
    ("lossguide128", 8, 0.03, 4.0, 1.0, "Lossguide", 128, 80),
]

MIN_DATA_IN_LEAF = 200  # 对齐 v6


# --------------------------------------------------------------------------- #
# 参数化训练原语（复用 ab._predict_load / _metrics / _arr，不依赖训练参数）
# --------------------------------------------------------------------------- #
def _params(loss, seed, iters, hp, eval_set=False):
    p = dict(
        task_type="GPU", devices="0", loss_function=loss,
        learning_rate=hp["lr"], depth=hp["depth"], l2_leaf_reg=hp["l2"],
        bootstrap_type="Bayesian", bagging_temperature=hp["bagging_temp"],
        boosting_type="Plain", random_seed=seed, verbose=0,
        allow_writing_files=False, iterations=iters,
        grow_policy=hp["grow_policy"],
        min_data_in_leaf=MIN_DATA_IN_LEAF,
    )
    if hp["grow_policy"] == "Lossguide":
        p["max_leaves"] = hp["max_leaves"]
    if eval_set:
        p["eval_metric"] = "MAE"
        p["od_type"] = "Iter"
        p["od_wait"] = 50
    return p


def _fit(Xtr, ytr, wtr, loss, seed, iters, hp, eval_set=None):
    pool = Pool(Xtr, label=ytr, weight=wtr)
    m = CatBoostRegressor(**_params(loss, seed, iters, hp, eval_set=bool(eval_set)))
    if eval_set is not None:
        evX, evy = eval_set
        m.fit(pool, eval_set=Pool(evX, label=evy))
    else:
        m.fit(pool)
    return m


def _train_ensemble(X, actual, anchor, mask, cfg, best_it, feat_cols, hp):
    y_res = actual - anchor
    Xtr = _arr(X[mask], feat_cols)
    wtr = _time_weights(ab.times_global, mask, cfg["alpha_w"],
                        pred_load=ab.pred_load_global,
                        load_gamma=cfg.get("weight_load_gamma", 0.0))
    ytr_dir = actual[mask].to_numpy(np.float64)
    ytr_res = y_res[mask].to_numpy(np.float64)
    members = []
    for residual in cfg["residual_modes"]:
        y = ytr_res if residual else ytr_dir
        for obj in cfg["objectives"]:
            alphas = cfg["quantile_alphas"] if obj == "quantile" else [None]
            for qa in alphas:
                loss = f"Quantile:alpha={qa}" if obj == "quantile" else "RMSE"
                for s in cfg["seeds"]:
                    m = _fit(Xtr, y, wtr, loss, s, best_it, hp)
                    members.append((m, bool(residual)))
    return members


def _ensemble_raw(members, X, anchor_vals, feat_cols, shrinkage):
    Xarr = _arr(X, feat_cols)
    mp = np.empty((len(members), len(X)), dtype=float)
    for i, (m, is_res) in enumerate(members):
        raw = m.predict(Xarr)
        mp[i] = anchor_vals + raw if is_res else raw
    ens = np.median(mp, axis=0)
    return anchor_vals + shrinkage * (ens - anchor_vals)


def _compute_oof(times, X, pred_load, actual, usable, anchor, cfg, best_it, feat_cols, hp):
    """3 折 OOF -> hour_bias + drift_corr + threshold_corr（逻辑同 ab._cb_compute_hour_bias）。"""
    oof_pred = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        members = _train_ensemble(X, actual, anchor, ftr, cfg, best_it, feat_cols, hp)
        oof_pred[fva] = _ensemble_raw(members, X[fva], anchor[fva].values, feat_cols, cfg["shrinkage"])
    oof_mask = usable & oof_pred.notna().values
    resid = (oof_pred - actual).values

    n_slots = int(cfg.get("hour_bias_slots", 96))
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
    return hour_bias, drift_corr, threshold_corr, oof_pred, oof_mask


# --------------------------------------------------------------------------- #
def _run_config(tag, hp, best_it, times, X, pred_load, actual, usable, anchor, cfg, feat_cols, val_m):
    ts = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        members = _train_ensemble(X, actual, anchor, usable, cfg, best_it, feat_cols, hp)
        hour_bias, drift_corr, threshold_corr, oof_pred, oof_mask = _compute_oof(
            times, X, pred_load, actual, usable, anchor, cfg, best_it, feat_cols, hp)
    pred_val = _predict_load(members, X[val_m], anchor[val_m].values, feat_cols,
                             cfg["shrinkage"], hour_bias, drift_corr, threshold_corr)
    actual_val = actual[val_m]
    mt = _metrics(pd.Series(pred_val, index=times[val_m]), actual_val, times[val_m])
    err = pred_val - actual_val.values
    debiased = float(np.mean(np.abs(err - err.mean())))
    fmaes = []
    for te, vs, ve in cfg["best_it_folds"]:
        vs, ve = pd.Timestamp(vs), pd.Timestamp(ve)
        fm = usable & (times >= vs) & (times <= ve) & oof_pred.notna().values
        if fm.sum():
            fmaes.append(float(np.mean(np.abs(oof_pred[fm].values - actual[fm].values))))
    farr = np.array(fmaes)
    fcv = float(farr.std() / farr.mean()) if len(farr) >= 2 and farr.mean() > 0 else float("nan")
    dt = time.perf_counter() - ts
    return {"tag": tag, "MAE": mt["MAE"], "Bias": mt["Bias"], "R2": mt["R2"],
            "midday": mt["midday_MAE"], "debiased": debiased, "fcv": fcv,
            "fmaes": fmaes, "dt": dt, "hp": hp, "best_it": best_it}


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print(f"Phase 3-B1: CatBoost 超参搜索   (val vs v6={V6_VAL_MAE}, min_data_in_leaf={MIN_DATA_IN_LEAF})")
    print("=" * 74)

    print("[1] 构建数据集...")
    times, X, pred_load, actual = build_dataset()
    ab.times_global, ab.pred_load_global = times, pred_load
    usable = usable_mask(times, pred_load, actual)
    mm = MismatchModel().fit(X, usable)
    X = mm.transform(X)
    mos_model = MosModel(cols=C.TRAIN_CONFIG["mos"]["cols"],
                         alpha=C.TRAIN_CONFIG["mos"]["alpha"]).fit(X, actual, usable)
    anchor = pd.Series(mos_model.transform(X), index=X.index)
    feat_cols = list(X.columns)
    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    cfg = C.TRAIN_CONFIG
    print(f"    特征数={len(feat_cols)}  val点={int(val_m.sum())}  配置数={len(CONFIGS)}")

    print(f"\n[2] 逐配置训练 + 评估 ...")
    rows = []
    for tag, depth, lr, l2, bt, gp, ml, best_it in CONFIGS:
        hp = {"depth": depth, "lr": lr, "l2": l2, "bagging_temp": bt,
              "grow_policy": gp, "max_leaves": ml}
        try:
            r = _run_config(tag, hp, best_it, times, X, pred_load, actual, usable,
                            anchor, cfg, feat_cols, val_m)
            rows.append(r)
            print(f"  {tag:14s} MAE={r['MAE']:.2f} Bias={r['Bias']:+7.1f} debiased={r['debiased']:.2f} "
                  f"午间={r['midday']:.0f} 折CV={r['fcv']:.3f} ({r['dt']:.0f}s)")
        except Exception as e:
            ename = type(e).__name__
            msg = str(e).splitlines()[0][:90]
            print(f"  {tag:14s} FAIL ({ename}: {msg})")

    if not rows:
        print("\n所有配置失败。"); return 1

    # ---- 对比表 ----
    print("\n" + "=" * 74)
    print("超参搜索对比（vs v6 1445.62）")
    print("=" * 74)
    print(f"{'tag':14} {'MAE':>8} {'Δv6':>8} {'Bias':>8} {'debiased':>9} {'午间':>6} {'折CV':>6} {'policy':>13}")
    for r in rows:
        print(f"{r['tag']:14} {r['MAE']:>8.2f} {r['MAE']-V6_VAL_MAE:>+8.2f} {r['Bias']:>+8.1f} "
              f"{r['debiased']:>9.2f} {r['midday']:>6.0f} {r['fcv']:>6.3f} {r['hp']['grow_policy']:>13}")

    best = min(rows, key=lambda r: r["MAE"])
    print(f"\n最优: {best['tag']}  MAE={best['MAE']:.2f} (Δv6 {best['MAE']-V6_VAL_MAE:+.2f})  "
          f"debiased={best['debiased']:.2f}  折CV={best['fcv']:.3f}")
    print(f"     hp={best['hp']} best_it={best['best_it']}")
    print("-" * 74)

    # ---- 诊断 ----
    print("\n诊断：")
    lossguide_rows = [r for r in rows if r["hp"]["grow_policy"] == "Lossguide"]
    sym_rows = [r for r in rows if r["hp"]["grow_policy"] == "SymmetricTree"]
    if lossguide_rows:
        lg_best = min(lossguide_rows, key=lambda r: r["MAE"])
        sym_best = min(sym_rows, key=lambda r: r["MAE"]) if sym_rows else None
        print(f"  Lossguide 最优: {lg_best['tag']} MAE={lg_best['MAE']:.2f} debiased={lg_best['debiased']:.2f}")
        if sym_best:
            print(f"  Symmetric 最优: {sym_best['tag']} MAE={sym_best['MAE']:.2f} debiased={sym_best['debiased']:.2f}")
            d = lg_best["MAE"] - sym_best["MAE"]
            print(f"  Lossguide vs Symmetric: {d:+.2f} MW ({'leaf-wise 有效' if d < -5 else 'leaf-wise 无明显优势'})")
    if best["debiased"] < V6_VAL_MAE:
        print(f"  最优 debiased={best['debiased']:.0f} < v6 {V6_VAL_MAE} -> 纯波动误差已低于 v6，")
        print("  存在无泄露校正手段可追平/超越 v6。建议 Phase 3-C: 在最优配置上做 recency-weighted hour_bias。")
    elif best["MAE"] < V6_VAL_MAE + 20:
        print(f"  最优 MAE={best['MAE']:.0f} 接近 v6({V6_VAL_MAE})，debiased={best['debiased']:.0f} 仍偏高。")
        print("  可作异质集成成员(Phase 4)或继续微调。")
    else:
        print(f"  最优 MAE={best['MAE']:.0f} 仍落后 v6 >20MW，debiased={best['debiased']:.0f}。")
        print("  CatBoost 在本特征集上结构性劣势难逆 -> 建议作异质集成成员或停止单独优化。")
    print("=" * 74)

    try:
        with open("exp_catboost_hp_result.txt", "w", encoding="utf-8") as f:
            f.write(f"v6={V6_VAL_MAE} min_data_in_leaf={MIN_DATA_IN_LEAF}\n")
            f.write("tag\tMAE\tDelta_v6\tBias\tdebiased\tmidday\tfold_cv\tpolicy\tbest_it\n")
            for r in rows:
                f.write(f"{r['tag']}\t{r['MAE']:.4f}\t{r['MAE']-V6_VAL_MAE:+.4f}\t{r['Bias']:.4f}\t"
                        f"{r['debiased']:.4f}\t{r['midday']:.4f}\t{r['fcv']:.4f}\t"
                        f"{r['hp']['grow_policy']}\t{r['best_it']}\n")
            f.write(f"best={best['tag']}\n")
        print("(已写 exp_catboost_hp_result.txt)")
    except Exception as e:
        print(f"(写结果失败: {e})")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
