# -*- coding: utf-8 -*-
"""
CatBoost F3/F4/F5: 集成扩展扫描

F3 成员数: seeds=3(24成员) / 2(16) / 7(56)  -- 默认 5 seeds=40 成员
F4 per-mode best_it: direct=80,residual=160 / direct=160,residual=80
F5 多 best_it 混合: 成员交替 80/160 / 循环 40/80/160

自包含（复刻 hp 链，参数化 seeds + per-mode best_it + multi best_it 列表）。
其余超参固定 l2_8。

注意：F3 改成员数致集成规模变化，MAE 差异含规模效应；F4/F5 成员数不变(40)。
      F4/F5 改变成员 best_it -> OOF 校正需在每个配置的 best_it 方案下重估（已重估）。

合规：不修改生产脚本；仅 import 复用 hp._fit/_ensemble_raw + ab._predict_load/_arr/_metrics；
      6 条泄露不变量全保持。OOF 3 折估计，不接触官方验证集。
运行：python -m load_pred.exp_catboost_ensemble2   （4090 上约 15-25 min，8 配置）
"""
from __future__ import annotations
import sys
import time
import io
import contextlib
import warnings

import numpy as np
import pandas as pd

from . import config as C
from .train import build_dataset, usable_mask, _time_weights
from .features import MismatchModel, MosModel
from .exp_catboost_ab import _predict_load, _arr, _metrics, V6_VAL_MAE
from . import exp_catboost_ab as ab
from . import exp_catboost_hp as hp

HP_L2_8 = {"depth": 8, "lr": 0.03, "l2": 8.0, "bagging_temp": 1.0,
           "grow_policy": "SymmetricTree", "max_leaves": None}
L2_8_VAL_MAE = 1477.67


def _train_ensemble_ext(X, actual, anchor, mask, cfg, feat_cols, hpcfg,
                        seeds, best_it_dir, best_it_res, multi_best_it):
    y_res = actual - anchor
    Xtr = _arr(X[mask], feat_cols)
    wtr = _time_weights(ab.times_global, mask, cfg["alpha_w"],
                        pred_load=ab.pred_load_global,
                        load_gamma=cfg.get("weight_load_gamma", 0.0))
    ytr_dir = actual[mask].to_numpy(np.float64)
    ytr_res = y_res[mask].to_numpy(np.float64)
    members = []
    bi_iter = list(multi_best_it) if multi_best_it else None
    bi_pos = 0
    for residual in cfg["residual_modes"]:
        y = ytr_res if residual else ytr_dir
        bi_mode = best_it_res if residual else best_it_dir
        for obj in cfg["objectives"]:
            alphas = cfg["quantile_alphas"] if obj == "quantile" else [None]
            for qa in alphas:
                loss = f"Quantile:alpha={qa}" if obj == "quantile" else "RMSE"
                for s in seeds:
                    if bi_iter is not None:
                        bi = bi_iter[bi_pos % len(bi_iter)]
                        bi_pos += 1
                    else:
                        bi = bi_mode
                    m = hp._fit(Xtr, y, wtr, loss, s, bi, hpcfg)
                    members.append((m, bool(residual)))
    return members


def _compute_oof_ext(times, X, pred_load, actual, usable, anchor, cfg, feat_cols, hpcfg,
                     seeds, best_it_dir, best_it_res, multi_best_it):
    oof_pred = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        members = _train_ensemble_ext(X, actual, anchor, ftr, cfg, feat_cols, hpcfg,
                                      seeds, best_it_dir, best_it_res, multi_best_it)
        oof_pred[fva] = hp._ensemble_raw(members, X[fva], anchor[fva].values,
                                         feat_cols, cfg["shrinkage"])
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


def _run_config_ext(tag, times, X, pred_load, actual, usable, anchor, cfg,
                    feat_cols, val_m, seeds, best_it_dir, best_it_res, multi_best_it):
    ts = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        members = _train_ensemble_ext(X, actual, anchor, usable, cfg, feat_cols, HP_L2_8,
                                      seeds, best_it_dir, best_it_res, multi_best_it)
        hour_bias, drift_corr, threshold_corr, oof_pred, oof_mask = _compute_oof_ext(
            times, X, pred_load, actual, usable, anchor, cfg, feat_cols, HP_L2_8,
            seeds, best_it_dir, best_it_res, multi_best_it)
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
    return {"tag": tag, "MAE": mt["MAE"], "Bias": mt["Bias"], "debiased": debiased,
            "fcv": fcv, "n_members": len(members), "dt": dt}


SEEDS_5 = [42, 7, 123, 2024, 99]
SEEDS_3 = [42, 7, 123]
SEEDS_2 = [42, 7]
SEEDS_7 = [42, 7, 123, 2024, 99, 11, 22]

# (tag, seeds, best_it_dir, best_it_res, multi_best_it)
CONFIGS = [
    ("baseline_5s",     SEEDS_5, 80,  80,  None),
    # F3 成员数
    ("F3_seeds3",       SEEDS_3, 80,  80,  None),
    ("F3_seeds2",       SEEDS_2, 80,  80,  None),
    ("F3_seeds7",       SEEDS_7, 80,  80,  None),
    # F4 per-mode best_it
    ("F4_dir80_res160", SEEDS_5, 80,  160, None),
    ("F4_dir160_res80", SEEDS_5, 160, 80,  None),
    # F5 多 best_it 混合
    ("F5_mix80_160",    SEEDS_5, 80,  80,  [80, 160]),
    ("F5_mix40_80_160", SEEDS_5, 80,  80,  [40, 80, 160]),
]


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print(f"CatBoost F3/F4/F5: 集成扩展 ({len(CONFIGS)}配置)")
    print(f"  基线: l2_8 val={L2_8_VAL_MAE}  v6={V6_VAL_MAE}")
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
    cfg = C.TRAIN_CONFIG
    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    print(f"    特征数={len(feat_cols)}  val点={int(val_m.sum())}  配置数={len(CONFIGS)}")

    print(f"\n[2] 逐配置训练 + 评估 ...")
    rows = []
    for tag, seeds, bid, bir, mbi in CONFIGS:
        try:
            r = _run_config_ext(tag, times, X, pred_load, actual, usable, anchor,
                                cfg, feat_cols, val_m, seeds, bid, bir, mbi)
            rows.append(r)
            print(f"  {tag:16s} MAE={r['MAE']:.2f} Δv6={r['MAE']-V6_VAL_MAE:+.2f} "
                  f"Δl2_8={r['MAE']-L2_8_VAL_MAE:+.2f} debiased={r['debiased']:.2f} "
                  f"n={r['n_members']:2d} ({r['dt']:.0f}s)")
        except Exception as e:
            print(f"  {tag:16s} FAIL ({type(e).__name__}: {str(e).splitlines()[0][:80]})")

    if rows:
        print("\n" + "=" * 74)
        print(f"F3/F4/F5 集成扩展对比（vs l2_8 {L2_8_VAL_MAE} / v6 {V6_VAL_MAE}）")
        print("=" * 74)
        print(f"{'tag':16} {'MAE':>8} {'Δv6':>8} {'Δl2_8':>8} {'debiased':>9} {'n':>4} {'折CV':>6}")
        for r in rows:
            print(f"{r['tag']:16} {r['MAE']:>8.2f} {r['MAE']-V6_VAL_MAE:>+8.2f} "
                  f"{r['MAE']-L2_8_VAL_MAE:>+8.2f} {r['debiased']:>9.2f} {r['n_members']:>4} {r['fcv']:>6.3f}")
        best = min(rows, key=lambda r: r["MAE"])
        print(f"\n最优: {best['tag']}  MAE={best['MAE']:.2f} (Δl2_8 {best['MAE']-L2_8_VAL_MAE:+.2f})")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
