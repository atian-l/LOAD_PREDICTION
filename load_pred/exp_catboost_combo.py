# -*- coding: utf-8 -*-
"""
CatBoost P1-1: Depthwise × reg_only 组合突破 + 超参细化

两个独立突破叠加：
  - B3_depthwise (1463.88, grow_policy=Depthwise)
  - E3_reg_only  (1466.02, objectives=["regression"] 去 quantile 成员)
  -> dw_regonly 叠加（核心）

配置（复用 hp._run_config）：
  sym_quant / dw_quant / sym_regonly（3 个对照）
  dw_regonly（核心叠加）
  dw_regonly × depth{6,10} × l2{8,16,32} 细化（5 个）

best_it 固定 80。reg_only 成员数=10（2 residual × 1 reg × 5 seed），quant=40。

合规：不修改生产脚本；仅 import 复用 hp._run_config + ab/train/features；6 条泄露不变量全保持。
运行：python -m load_pred.exp_catboost_combo   （4090 上约 8-15 min，9 配置）
"""
from __future__ import annotations
import sys
import time
import copy
import warnings

import pandas as pd

from . import config as C
from .train import build_dataset, usable_mask
from .features import MismatchModel, MosModel
from .exp_catboost_ab import V6_VAL_MAE
from . import exp_catboost_ab as ab
from . import exp_catboost_hp as hp

L2_8_VAL_MAE = 1477.67
DW_VAL_MAE = 1463.88      # B3_depthwise
REGONLY_VAL_MAE = 1466.02  # E3_reg_only
BEST_IT = 80

BASE = {"depth": 8, "lr": 0.03, "l2": 8.0, "bagging_temp": 1.0,
        "grow_policy": "SymmetricTree", "max_leaves": None}


def _hp(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


def _ro(cfg):
    """reg_only: objectives=['regression']（去 quantile 成员）。"""
    c = copy.deepcopy(cfg)
    c["objectives"] = ["regression"]
    return c


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print(f"CatBoost P1-1: Depthwise×reg_only 组合突破+细化 (best_it={BEST_IT})")
    print(f"  对照: l2_8={L2_8_VAL_MAE}  depthwise={DW_VAL_MAE}  regonly={REGONLY_VAL_MAE}  v6={V6_VAL_MAE}")
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
    print(f"    特征数={len(feat_cols)}  val点={int(val_m.sum())}")

    # (tag, hp, cfg, best_it)
    configs = [
        ("sym_quant",          _hp(),                             cfg,      BEST_IT),
        ("dw_quant",           _hp(grow_policy="Depthwise"),      cfg,      BEST_IT),
        ("sym_regonly",        _hp(),                             _ro(cfg), BEST_IT),
        ("dw_regonly",         _hp(grow_policy="Depthwise"),      _ro(cfg), BEST_IT),
        ("dw_regonly_d10",     _hp(grow_policy="Depthwise", depth=10),   _ro(cfg), BEST_IT),
        ("dw_regonly_d6",      _hp(grow_policy="Depthwise", depth=6),    _ro(cfg), BEST_IT),
        ("dw_regonly_l2_16",   _hp(grow_policy="Depthwise", l2=16.0),    _ro(cfg), BEST_IT),
        ("dw_regonly_l2_32",   _hp(grow_policy="Depthwise", l2=32.0),    _ro(cfg), BEST_IT),
        ("dw_regonly_d10_l216", _hp(grow_policy="Depthwise", depth=10, l2=16.0), _ro(cfg), BEST_IT),
    ]
    print(f"    配置数={len(configs)}")

    print(f"\n[2] 逐配置训练 + 评估 ...")
    rows = []
    for tag, h, c, bi in configs:
        try:
            r = hp._run_config(tag, h, bi, times, X, pred_load, actual,
                               usable, anchor, c, feat_cols, val_m)
            rows.append(r)
            print(f"  {tag:20s} MAE={r['MAE']:.2f} Δv6={r['MAE']-V6_VAL_MAE:+.2f} "
                  f"Δl2_8={r['MAE']-L2_8_VAL_MAE:+.2f} debiased={r['debiased']:.2f} "
                  f"折CV={r['fcv']:.3f} ({r['dt']:.0f}s)")
        except Exception as e:
            print(f"  {tag:20s} FAIL ({type(e).__name__}: {str(e).splitlines()[0][:80]})")

    if rows:
        print("\n" + "=" * 74)
        print(f"P1-1 组合对比（vs l2_8 {L2_8_VAL_MAE} / depthwise {DW_VAL_MAE} / v6 {V6_VAL_MAE}）")
        print("=" * 74)
        print(f"{'tag':20} {'MAE':>8} {'Δv6':>8} {'Δl2_8':>8} {'Δdw':>8} {'debiased':>9} {'折CV':>6}")
        for r in rows:
            print(f"{r['tag']:20} {r['MAE']:>8.2f} {r['MAE']-V6_VAL_MAE:>+8.2f} "
                  f"{r['MAE']-L2_8_VAL_MAE:>+8.2f} {r['MAE']-DW_VAL_MAE:>+8.2f} "
                  f"{r['debiased']:>9.2f} {r['fcv']:>6.3f}")
        best = min(rows, key=lambda r: r["MAE"])
        print(f"\n最优: {best['tag']}  MAE={best['MAE']:.2f} "
              f"(Δl2_8 {best['MAE']-L2_8_VAL_MAE:+.2f}, Δv6 {best['MAE']-V6_VAL_MAE:+.2f})")
        if best["MAE"] < DW_VAL_MAE and best["MAE"] < REGONLY_VAL_MAE:
            print("  -> 叠加有效：优于单方向 depthwise/regonly，确认组合突破")
        if best["MAE"] < V6_VAL_MAE:
            print(f"  -> 破 v6！{best['tag']} MAE={best['MAE']:.2f} < v6 {V6_VAL_MAE}")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
