# -*- coding: utf-8 -*-
"""
CatBoost G3/G4: OOF 校正扩展扫描

G3 drift_corr 扩展: hours=[6-9]/[15-18]/全天 / feature=pl_x_clearness / off(对照)
G4 threshold_corr 增场景: temp>30 / clearness<0.2@11-14 / 两场景叠加 / off(对照)

cfg 副本驱动（改 drift_corr / threshold_corr），复用 hp._run_config；OOF 校正在各 cfg 下重估。
其余超参固定 l2_8。

合规：不修改生产脚本；仅 import 复用 hp._run_config + ab/train/features；6 条泄露不变量全保持。
运行：python -m load_pred.exp_catboost_corr2   （4090 上约 15-20 min，9 配置）
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

HP_L2_8 = {"depth": 8, "lr": 0.03, "l2": 8.0, "bagging_temp": 1.0,
           "grow_policy": "SymmetricTree", "max_leaves": None}
BEST_IT = 80
L2_8_VAL_MAE = 1477.67


def _cfg_drift(cfg, hours=None, feature=None, off=False):
    c = copy.deepcopy(cfg)
    if off:
        c["drift_corr"] = None
    else:
        c["drift_corr"] = {"feature": feature or c["drift_corr"]["feature"],
                           "hours": hours if hours is not None else c["drift_corr"]["hours"]}
    return c


def _cfg_threshold(cfg, extra=None, off=False):
    c = copy.deepcopy(cfg)
    if off:
        c["threshold_corr"] = []
    elif extra:
        c["threshold_corr"] = c["threshold_corr"] + extra
    return c


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print(f"CatBoost G3/G4: OOF 校正扩展 (best_it={BEST_IT})")
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
    print(f"    特征数={len(feat_cols)}  val点={int(val_m.sum())}")

    # 构造配置列表 (tag, cfg_modified)
    extra_temp30 = {"feature": "temp", "op": ">", "thr": 30.0, "hours": None, "shrinkage": 1.0}
    extra_clr02 = {"feature": "clearness", "op": "<", "thr": 0.2, "hours": [11, 12, 13, 14], "shrinkage": 1.0}
    configs = [
        # G3 drift_corr 扩展
        ("G3_drift_off",       _cfg_drift(cfg, off=True)),
        ("G3_drift_6-9",       _cfg_drift(cfg, hours=[6, 7, 8, 9])),
        ("G3_drift_15-18",     _cfg_drift(cfg, hours=[15, 16, 17, 18])),
        ("G3_drift_allday",    _cfg_drift(cfg, hours=list(range(24)))),
        ("G3_drift_pl_x_clr",  _cfg_drift(cfg, feature="pl_x_clearness")),
        # G4 threshold_corr 增场景
        ("G4_thr_off",         _cfg_threshold(cfg, off=True)),
        ("G4_add_temp_gt30",   _cfg_threshold(cfg, extra=[extra_temp30])),
        ("G4_add_clr_lt02",    _cfg_threshold(cfg, extra=[extra_clr02])),
        ("G4_add_temp30_clr02", _cfg_threshold(cfg, extra=[extra_temp30, extra_clr02])),
    ]
    print(f"    配置数={len(configs)}")

    print(f"\n[2] 逐配置训练 + 评估 ...")
    rows = []
    for tag, cfgm in configs:
        try:
            r = hp._run_config(tag, HP_L2_8, BEST_IT, times, X, pred_load, actual,
                               usable, anchor, cfgm, feat_cols, val_m)
            rows.append(r)
            print(f"  {tag:20s} MAE={r['MAE']:.2f} Δv6={r['MAE']-V6_VAL_MAE:+.2f} "
                  f"Δl2_8={r['MAE']-L2_8_VAL_MAE:+.2f} debiased={r['debiased']:.2f} "
                  f"折CV={r['fcv']:.3f} ({r['dt']:.0f}s)")
        except Exception as e:
            print(f"  {tag:20s} FAIL ({type(e).__name__}: {str(e).splitlines()[0][:80]})")

    if rows:
        print("\n" + "=" * 74)
        print(f"G3/G4 校正扩展对比（vs l2_8 {L2_8_VAL_MAE} / v6 {V6_VAL_MAE}）")
        print("=" * 74)
        print(f"{'tag':20} {'MAE':>8} {'Δv6':>8} {'Δl2_8':>8} {'debiased':>9} {'折CV':>6}")
        for r in rows:
            print(f"{r['tag']:20} {r['MAE']:>8.2f} {r['MAE']-V6_VAL_MAE:>+8.2f} "
                  f"{r['MAE']-L2_8_VAL_MAE:>+8.2f} {r['debiased']:>9.2f} {r['fcv']:>6.3f}")
        best = min(rows, key=lambda r: r["MAE"])
        print(f"\n最优: {best['tag']}  MAE={best['MAE']:.2f} (Δl2_8 {best['MAE']-L2_8_VAL_MAE:+.2f})")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
