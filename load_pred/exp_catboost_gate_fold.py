# -*- coding: utf-8 -*-
"""
CatBoost P2-8: 分折 OOF 场景分析 + 冬折/recency 选 w 验证

背景：P2-7 显示场景门控 val 过拟合（OOF/val 仅 2/7 一致），clear_noon 跨年翻转
（OOF CatBoost 差 / val CatBoost 好）。但翻转可能是近期趋势（2026 CatBoost 变好）。
冬折 OOF（2026-01~02，最接近 val 2026-03~06）的 clear_noon 表现是关键：
  - 若冬折 clear_noon CatBoost 也优 -> 近期趋势，场景门控可迁移，1438 可信
  - 若冬折也差 -> val 噪声，场景门控不可信，回退全局融合

流程：
  1. build_dataset + MismatchModel + MOS
  2. v6 LightGBM official + OOF 3 折（分折存储含校正预测）
  3. CatBoost d10 official + OOF 3 折（分折存储含校正预测）
  4. 分折场景分析：春(2025-03~05)/秋(2025-09~11)/冬(2026-01~02) 各折 clear_noon 等 lgb/cat MAE
  5. 选 w 策略对比 val：
     a. 全局 OOF w（3 折合并）
     b. 冬折 w（仅冬折 OOF 选）
     c. recency 加权 OOF w（冬折权重高）
     d. val 自选 w（上界参考）
  6. 判断 clear_noon 是否近期迁移

合规: 不修改生产脚本; 6 条泄露不变量全保持; val eval-only; OOF 3 折全在训练期内。
运行: python -m load_pred.exp_catboost_gate_fold  (本地 3060 约 8 min)
"""
from __future__ import annotations
import sys
import time
import copy
import io
import contextlib
import warnings

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from . import config as C
from .train import build_dataset, usable_mask, train_ensemble, compute_hour_bias
from .features import MismatchModel, MosModel
from .exp_catboost_ab import V6_VAL_MAE, _predict_load
from . import exp_catboost_ab as ab
from . import exp_catboost_hp as hp

HP_DW = {"depth": 10, "lr": 0.03, "l2": 8.0, "bagging_temp": 1.0,
         "grow_policy": "Depthwise", "max_leaves": None}
BEST_IT = 80

SCENES = ["clear_noon_high", "clear_noon_mid", "halfclear_noon",
          "cloudy_noon", "rainy", "cold", "baseline"]


def _scene_masks(clearness, precip, temp, hours):
    mid = (hours >= 11) & (hours <= 14)
    raw = {
        "clear_noon_high": (clearness >= 0.9) & mid,
        "clear_noon_mid": ((clearness >= 0.8) & (clearness < 0.9)) & mid,
        "halfclear_noon": ((clearness >= 0.5) & (clearness < 0.8)) & mid,
        "cloudy_noon": ((clearness >= 0.2) & (clearness < 0.5)) & mid,
        "rainy": (precip > 0),
        "cold": (temp < 8),
    }
    assigned = np.zeros(len(hours), dtype=bool)
    excl = {}
    for name in ["clear_noon_high", "clear_noon_mid", "halfclear_noon", "cloudy_noon"]:
        excl[name] = raw[name] & ~assigned
        assigned |= excl[name]
    for name in ["rainy", "cold"]:
        excl[name] = raw[name] & ~assigned
        assigned |= excl[name]
    excl["baseline"] = ~assigned
    return excl


def _mae(p, a):
    return float(np.mean(np.abs(p - a))) if len(p) else float('nan')


def _best_w(pl, pc, av, m, ws):
    if m.sum() == 0:
        return 0.0, float('nan')
    bw, bm = 0.0, _mae(pl[m], av[m])
    for w in ws:
        mae = _mae(w * pc[m] + (1 - w) * pl[m], av[m])
        if mae < bm:
            bw, bm = float(w), mae
    return bw, bm


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print(f"CatBoost P2-8: 分折 OOF 场景分析 + 冬折/recency 选 w (v6={V6_VAL_MAE})")
    print("=" * 74)

    print("[1] 构建数据集 ...")
    times, X, pred_load, actual = build_dataset()
    ab.times_global, ab.pred_load_global = times, pred_load
    usable = usable_mask(times, pred_load, actual)
    mismatch_model = MismatchModel().fit(X, usable)
    X = mismatch_model.transform(X)
    mc = C.TRAIN_CONFIG["mos"]
    mos_model = MosModel(cols=mc["cols"], alpha=mc["alpha"]).fit(X, actual, usable)
    anchor = pd.Series(mos_model.transform(X), index=X.index)
    feat_cols = list(X.columns)
    cfg = C.TRAIN_CONFIG
    cfg_ro = copy.deepcopy(cfg)
    cfg_ro["objectives"] = ["regression"]
    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    av = actual[val_m].to_numpy(np.float64)
    print(f"    特征数={len(feat_cols)}  val点={int(val_m.sum())}")

    WS = np.arange(0.0, 1.01, 0.1)
    fold_names = ["春2025", "秋2025", "冬2026"]

    # ---- v6 LightGBM: official + 分折 OOF 含校正 ----
    print("\n[2] 训练 v6 LightGBM official + OOF 3 折（含校正，分折存储）...")
    ts = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        lgb_model = train_ensemble(times, X, pred_load, actual, usable, cfg,
                                   BEST_IT, mos_model=mos_model)
        lgb_model.mismatch_model = mismatch_model
        hb, dc, tc = compute_hour_bias(times, X, pred_load, actual, usable, cfg,
                                       BEST_IT, mos_model=mos_model)
        lgb_model.hour_bias, lgb_model.drift_corr, lgb_model.threshold_corr = hb, dc, tc
        oof_lgb_by_fold = {}
        for fname, (te, vs, ve) in zip(fold_names, cfg["best_it_folds"]):
            te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
            ftr = usable & np.asarray(times <= te)
            fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
            if fva.sum() == 0:
                continue
            fm = train_ensemble(times, X, pred_load, actual, ftr, cfg, BEST_IT, mos_model=mos_model)
            fm.mismatch_model = mismatch_model
            fm.hour_bias, fm.drift_corr, fm.threshold_corr = hb, dc, tc
            oof_lgb_by_fold[fname] = (fva, fm.predict_load(X[fva], pred_load[fva]))
    pred_lgb_val = lgb_model.predict_load(X[val_m], pred_load[val_m])
    print(f"    v6 LightGBM  val MAE={_mae(pred_lgb_val, av):.2f}  ({time.perf_counter()-ts:.0f}s)")

    # ---- CatBoost d10: official + 分折 OOF 含校正 ----
    print("\n[3] 训练 CatBoost d10 official + OOF 3 折（含校正，分折存储）...")
    ts = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        cat_members = hp._train_ensemble(X, actual, anchor, usable, cfg_ro,
                                         BEST_IT, feat_cols, HP_DW)
        hb_c, dc_c, tc_c, _, _ = hp._compute_oof(times, X, pred_load, actual, usable,
                                                 anchor, cfg_ro, BEST_IT, feat_cols, HP_DW)
        oof_cat_by_fold = {}
        for fname, (te, vs, ve) in zip(fold_names, cfg["best_it_folds"]):
            te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
            ftr = usable & np.asarray(times <= te)
            fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
            if fva.sum() == 0:
                continue
            fm = hp._train_ensemble(X, actual, anchor, ftr, cfg_ro, BEST_IT, feat_cols, HP_DW)
            oof_cat_by_fold[fname] = (fva, _predict_load(fm, X[fva], anchor[fva].to_numpy(),
                                                          feat_cols, cfg["shrinkage"], hb_c, dc_c, tc_c))
    pred_cat_val = _predict_load(cat_members, X[val_m], anchor[val_m].to_numpy(),
                                 feat_cols, cfg["shrinkage"], hb_c, dc_c, tc_c)
    print(f"    CatBoost d10  val MAE={_mae(pred_cat_val, av):.2f}  ({time.perf_counter()-ts:.0f}s)")

    # ---- [4] 分折场景分析（重点 clear_noon）----
    print(f"\n[4] 分折场景分析（clear_noon 跨年趋势）:")
    print(f"  {'场景':>18} {'折':>6} {'n':>5} {'lgb MAE':>9} {'cat MAE':>9} {'cat-lgb':>8}")
    fold_scene_stats = {}  # (scene, fold) -> (n, lgb_mae, cat_mae)
    for fname in fold_names:
        if fname not in oof_lgb_by_fold:
            continue
        fva, pl = oof_lgb_by_fold[fname]
        _, pc = oof_cat_by_fold[fname]
        act = actual[fva].to_numpy(np.float64)
        clr = X[fva]["clearness"].values.astype(float)
        pre = X[fva]["precip"].values.astype(float)
        tm = X[fva]["temp"].values.astype(float)
        hr = pd.DatetimeIndex(times[fva]).hour.values
        scenes = _scene_masks(clr, pre, tm, hr)
        for name in SCENES:
            m = scenes[name]
            n = int(m.sum())
            ml = _mae(pl[m], act[m]) if n else float('nan')
            mc_ = _mae(pc[m], act[m]) if n else float('nan')
            fold_scene_stats[(name, fname)] = (n, ml, mc_)
            if n:
                print(f"  {name:>18} {fname:>6} {n:>5} {ml:>9.1f} {mc_:>9.1f} {mc_-ml:>+8.1f}")

    # ---- [5] 各选 w 策略 ----
    # 合并 OOF
    oof_mask_all = usable & np.zeros(len(times), dtype=bool)
    oof_lgb_all = np.array([])
    oof_cat_all = np.array([])
    oof_act_all = np.array([])
    oof_clr_all = np.array([]); oof_pre_all = np.array([]); oof_tm_all = np.array([]); oof_hr_all = np.array([])
    oof_fold_all = np.array([], dtype=int)  # 折索引 0春1秋2冬
    for fi, fname in enumerate(fold_names):
        if fname not in oof_lgb_by_fold:
            continue
        fva, pl = oof_lgb_by_fold[fname]
        _, pc = oof_cat_by_fold[fname]
        act = actual[fva].to_numpy(np.float64)
        oof_lgb_all = np.concatenate([oof_lgb_all, pl])
        oof_cat_all = np.concatenate([oof_cat_all, pc])
        oof_act_all = np.concatenate([oof_act_all, act])
        oof_clr_all = np.concatenate([oof_clr_all, X[fva]["clearness"].values.astype(float)])
        oof_pre_all = np.concatenate([oof_pre_all, X[fva]["precip"].values.astype(float)])
        oof_tm_all = np.concatenate([oof_tm_all, X[fva]["temp"].values.astype(float)])
        oof_hr_all = np.concatenate([oof_hr_all, pd.DatetimeIndex(times[fva]).hour.values])
        oof_fold_all = np.concatenate([oof_fold_all, np.full(len(pl), fi)])

    val_clr = X[val_m]["clearness"].values.astype(float)
    val_pre = X[val_m]["precip"].values.astype(float)
    val_tm = X[val_m]["temp"].values.astype(float)
    val_hr = pd.DatetimeIndex(times[val_m]).hour.values
    val_scenes = _scene_masks(val_clr, val_pre, val_tm, val_hr)
    oof_scenes = _scene_masks(oof_clr_all, oof_pre_all, oof_tm_all, oof_hr_all)

    print(f"\n[5] 各选 w 策略对比 val（OOF点={len(oof_act_all)}）:")
    # (a) 全局 OOF w
    gbw, _ = _best_w(oof_lgb_all, oof_cat_all, oof_act_all, np.ones(len(oof_act_all), dtype=bool), WS)
    # (b) 冬折 w（仅冬折 OOF 选）
    winter_mask = oof_fold_all == 2
    ww, _ = _best_w(oof_lgb_all, oof_cat_all, oof_act_all, winter_mask, WS) if winter_mask.sum() else (0.0, float('nan'))
    # (c) val 自选 w（上界）
    vw, _ = _best_w(pred_lgb_val, pred_cat_val, av, np.ones(len(av), dtype=bool), WS)

    print(f"  全局 OOF w={gbw:.1f}  冬折 OOF w={ww:.1f}  val 自选 w={vw:.1f}")
    for label, w in [("全局OOF", gbw), ("冬折OOF", ww), ("val自选", vw)]:
        mae = _mae(w * pred_cat_val + (1 - w) * pred_lgb_val, av)
        print(f"  {label:>8} w={w:.1f}  val MAE={mae:.2f} (Δv6 {mae-V6_VAL_MAE:+.2f})")

    # (d) 场景门控：冬折选场景 w vs 全局 OOF 场景 w vs val 场景 w
    print(f"\n  场景门控（各策略选场景 w，val 验证）:")
    # 冬折场景 w
    winter_scenes = _scene_masks(oof_clr_all[winter_mask], oof_pre_all[winter_mask],
                                  oof_tm_all[winter_mask], oof_hr_all[winter_mask]) if winter_mask.sum() else {}
    oof_scenes_w = {}  # 全局 OOF 场景 w
    winter_scenes_w = {}  # 冬折场景 w
    val_scenes_w = {}
    for name in SCENES:
        mo = oof_scenes[name]
        oof_scenes_w[name], _ = _best_w(oof_lgb_all, oof_cat_all, oof_act_all, mo, WS) if mo.sum() else (0.0, float('nan'))
        if winter_mask.sum():
            mw = winter_scenes.get(name, np.array([], dtype=bool))
            # 冬折场景在 oof 全局中的索引
            mw_full = winter_mask & oof_scenes[name]
            winter_scenes_w[name], _ = _best_w(oof_lgb_all, oof_cat_all, oof_act_all, mw_full, WS) if mw_full.sum() else (0.0, float('nan'))
        else:
            winter_scenes_w[name] = 0.0
        mv = val_scenes[name]
        val_scenes_w[name], _ = _best_w(pred_lgb_val, pred_cat_val, av, mv, WS) if mv.sum() else (0.0, float('nan'))

    print(f"  {'场景':>18} {'全局OOFw':>8} {'冬折OOFw':>8} {'valw':>5}")
    for name in SCENES:
        print(f"  {name:>18} {oof_scenes_w[name]:>8.1f} {winter_scenes_w[name]:>8.1f} {val_scenes_w[name]:>5.1f}")

    for label, wdict in [("全局OOF场景", oof_scenes_w), ("冬折OOF场景", winter_scenes_w), ("val场景", val_scenes_w)]:
        pred = pred_lgb_val.copy()
        for name in SCENES:
            mv = val_scenes[name]
            if mv.sum():
                w = wdict[name]
                pred[mv] = w * pred_cat_val[mv] + (1 - w) * pred_lgb_val[mv]
        mae = _mae(pred, av)
        print(f"  {label:>12} -> val MAE={mae:.2f} (Δv6 {mae-V6_VAL_MAE:+.2f})")

    # ---- 汇总 ----
    print("\n" + "=" * 74)
    print("P2-8 汇总：clear_noon 是否近期迁移？")
    print("=" * 74)
    # 冬折 clear_noon cat vs lgb
    for name in ["clear_noon_high", "clear_noon_mid"]:
        for fname in fold_names:
            if (name, fname) in fold_scene_stats:
                n, ml, mc_ = fold_scene_stats[(name, fname)]
                if n:
                    print(f"  {name} {fname}: lgb={ml:.0f} cat={mc_:.0f} cat-lgb={mc_-ml:+.0f} (n={n})")
    print(f"\n  冬折 clear_noon CatBoost 是否优于 lgb -> 决定场景门控可迁移性")
    print(f"  全局OOF场景w融合 -> val MAE 见上")
    print(f"  冬折OOF场景w融合 -> val MAE 见上（若≈val场景w，则近期迁移可信）")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
