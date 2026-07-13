# -*- coding: utf-8 -*-
"""
CatBoost B/C/D: 容量 + 正则 + 学习率扩展扫

Phase0 已证明 B/C/D 大类(拟合改善不迁移至 val)大概率无效，此处齐全验证。
复用 hp._run_config 框架；monkey-patch hp._params 以支持 boosting_type /
random_strength / border_count（hp 原版硬编码 Plain、无 random_strength/border_count）。

B2 depth=12 / B3 Depthwise / B5 Ordered(慢,GPU 可能不支持,try/except)
C1 l2=16,32 / C3 bagging_temp=0.0,0.5,2.0 / C4 random_strength=0.5,2.0,5.0
D1 lr=0.02,0.05,0.1(配 best_it ∝ 1/lr) / D3 border_count=32,128,254 / D4 lr0.06_bi40
D2 lr schedule 跳过（CatBoost 无原生 schedule，用 D4 iter×lr 替代）

合规：不修改生产脚本；仅 import 复用 hp/ab/train/features；6 条泄露不变量全保持。
运行：python -m load_pred.exp_catboost_capacity   （4090 上约 25-40 min，Ordered 慢 5x）
"""
from __future__ import annotations
import sys
import time
import warnings

import numpy as np
import pandas as pd

from . import config as C
from .train import build_dataset, usable_mask
from .features import MismatchModel, MosModel
from .exp_catboost_ab import V6_VAL_MAE
from . import exp_catboost_ab as ab
from . import exp_catboost_hp as hp

L2_8_VAL_MAE = 1477.67

# ---- monkey-patch hp._params 支持 boosting_type / random_strength / border_count ----
_orig_params = hp._params


def _params_ext(loss, seed, iters, hpcfg, eval_set=False):
    p = _orig_params(loss, seed, iters, hpcfg, eval_set)
    p["boosting_type"] = hpcfg.get("boosting_type", "Plain")
    if "random_strength" in hpcfg:
        p["random_strength"] = hpcfg["random_strength"]
    if "border_count" in hpcfg:
        p["border_count"] = hpcfg["border_count"]
    return p


hp._params = _params_ext

BASE = {"depth": 8, "lr": 0.03, "l2": 8.0, "bagging_temp": 1.0,
        "grow_policy": "SymmetricTree", "max_leaves": None}


def _cfg(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


# (tag, hp_dict, best_it)
CONFIGS = [
    # B 容量/树结构
    ("B2_depth12",     _cfg(depth=12),                 80),
    ("B3_depthwise",   _cfg(grow_policy="Depthwise"),  80),
    ("B5_ordered",     _cfg(boosting_type="Ordered"),  80),
    # C 正则
    ("C1_l2_16",       _cfg(l2=16.0),                  80),
    ("C1_l2_32",       _cfg(l2=32.0),                  80),
    ("C3_bt0.0",       _cfg(bagging_temp=0.0),         80),
    ("C3_bt0.5",       _cfg(bagging_temp=0.5),         80),
    ("C3_bt2.0",       _cfg(bagging_temp=2.0),         80),
    ("C4_rs0.5",       _cfg(random_strength=0.5),      80),
    ("C4_rs2.0",       _cfg(random_strength=2.0),      80),
    ("C4_rs5.0",       _cfg(random_strength=5.0),      80),
    # D 优化/学习率（best_it ∝ 1/lr 保持等效迭代）
    ("D1_lr0.02",      _cfg(lr=0.02),                 120),
    ("D1_lr0.05",      _cfg(lr=0.05),                  48),
    ("D1_lr0.1",       _cfg(lr=0.10),                  24),
    ("D3_bc32",        _cfg(border_count=32),          80),
    ("D3_bc128",       _cfg(border_count=128),         80),
    ("D3_bc254",       _cfg(border_count=254),         80),
    ("D4_lr0.06_bi40", _cfg(lr=0.06),                  40),
]


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print(f"CatBoost B/C/D: 容量+正则+学习率扩展 ({len(CONFIGS)}配置)")
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
    for tag, hpcfg, best_it in CONFIGS:
        try:
            r = hp._run_config(tag, hpcfg, best_it, times, X, pred_load, actual,
                               usable, anchor, cfg, feat_cols, val_m)
            rows.append(r)
            print(f"  {tag:16s} MAE={r['MAE']:.2f} Δv6={r['MAE']-V6_VAL_MAE:+.2f} "
                  f"Δl2_8={r['MAE']-L2_8_VAL_MAE:+.2f} debiased={r['debiased']:.2f} "
                  f"折CV={r['fcv']:.3f} ({r['dt']:.0f}s)")
        except Exception as e:
            print(f"  {tag:16s} FAIL ({type(e).__name__}: {str(e).splitlines()[0][:80]})")

    if rows:
        print("\n" + "=" * 74)
        print(f"B/C/D 对比（vs l2_8 {L2_8_VAL_MAE} / v6 {V6_VAL_MAE}）")
        print("=" * 74)
        print(f"{'tag':16} {'MAE':>8} {'Δv6':>8} {'Δl2_8':>8} {'debiased':>9} {'折CV':>6}")
        for r in rows:
            print(f"{r['tag']:16} {r['MAE']:>8.2f} {r['MAE']-V6_VAL_MAE:>+8.2f} "
                  f"{r['MAE']-L2_8_VAL_MAE:>+8.2f} {r['debiased']:>9.2f} {r['fcv']:>6.3f}")
        best = min(rows, key=lambda r: r["MAE"])
        print(f"\n最优: {best['tag']}  MAE={best['MAE']:.2f} "
              f"(Δl2_8 {best['MAE']-L2_8_VAL_MAE:+.2f}, Δv6 {best['MAE']-V6_VAL_MAE:+.2f})")
        if best["MAE"] >= L2_8_VAL_MAE:
            print("  -> 无配置优于 l2_8 基线，B/C/D 大类确认无效（与 Phase0 一致）")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
