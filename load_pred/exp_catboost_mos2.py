# -*- coding: utf-8 -*-
"""
CatBoost I1/I2: anchor / MOS 类扫描

I1 MOS alpha(Ridge 正则): 0.1 / 0.5 / 1.0(默认=baseline) / 5.0 / 10.0
I2 MOS cols(MOS 特征列): minimal(5列) / predonly(1列=raw pred_load) / no_plwr / extended(+2列)

每配置重建 MosModel -> anchor，复用 hp._run_config；其余超参固定 l2_8。
MOS target=actual(仅作目标，合规#1)，inputs=pred_load+weather+calendar(合规)。

合规：不修改生产脚本；仅 import 复用 hp._run_config + ab/train/features；6 条泄露不变量全保持。
运行：python -m load_pred.exp_catboost_mos2   （4090 上约 15-20 min，9 配置）
"""
from __future__ import annotations
import sys
import time
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

MINIMAL_COLS = ["pred_load", "temp", "hdd", "cdd", "pl_weather_residual"]
PREDONLY_COLS = ["pred_load"]
NO_PLWR_COLS = [c for c in MosModel.DEFAULT_COLS if c != "pl_weather_residual"]
EXTENDED_COLS = list(MosModel.DEFAULT_COLS) + ["clearness", "temp_day_max"]


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print(f"CatBoost I1/I2: anchor/MOS 类扫描 (best_it={BEST_IT})")
    print(f"  基线: l2_8 val={L2_8_VAL_MAE}  v6={V6_VAL_MAE}")
    print("=" * 74)

    print("[1] 构建数据集...")
    times, X, pred_load, actual = build_dataset()
    ab.times_global, ab.pred_load_global = times, pred_load
    usable = usable_mask(times, pred_load, actual)
    mm = MismatchModel().fit(X, usable)
    X = mm.transform(X)
    feat_cols = list(X.columns)
    cfg = C.TRAIN_CONFIG
    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    print(f"    特征数={len(feat_cols)}  val点={int(val_m.sum())}")

    # 构造配置 (tag, alpha, cols)
    configs = []
    for a, tag in [(0.1, "I1_a0.1"), (0.5, "I1_a0.5"), (1.0, "baseline"),
                   (5.0, "I1_a5.0"), (10.0, "I1_a10.0")]:
        configs.append((tag, a, None))
    configs.append(("I2_cols_minimal", 1.0, MINIMAL_COLS))
    configs.append(("I2_cols_predonly", 1.0, PREDONLY_COLS))
    configs.append(("I2_cols_no_plwr", 1.0, NO_PLWR_COLS))
    configs.append(("I2_cols_extended", 1.0, EXTENDED_COLS))
    print(f"    配置数={len(configs)}")

    print(f"\n[2] 逐配置重建 MOS + 训练 + 评估 ...")
    rows = []
    for tag, alpha, cols in configs:
        try:
            mos = MosModel(cols=cols, alpha=alpha).fit(X, actual, usable)
            anchor = pd.Series(mos.transform(X), index=X.index)
            r = hp._run_config(tag, HP_L2_8, BEST_IT, times, X, pred_load, actual,
                               usable, anchor, cfg, feat_cols, val_m)
            rows.append(r)
            cols_str = "default" if cols is None else f"{len(cols)}列"
            print(f"  {tag:16s} MAE={r['MAE']:.2f} Δv6={r['MAE']-V6_VAL_MAE:+.2f} "
                  f"Δl2_8={r['MAE']-L2_8_VAL_MAE:+.2f} debiased={r['debiased']:.2f} "
                  f"MOS[{cols_str},a={alpha}] ({r['dt']:.0f}s)")
        except Exception as e:
            print(f"  {tag:16s} FAIL ({type(e).__name__}: {str(e).splitlines()[0][:80]})")

    if rows:
        print("\n" + "=" * 74)
        print(f"I1/I2 MOS 对比（vs l2_8 {L2_8_VAL_MAE} / v6 {V6_VAL_MAE}）")
        print("=" * 74)
        print(f"{'tag':16} {'MAE':>8} {'Δv6':>8} {'Δl2_8':>8} {'debiased':>9} {'折CV':>6}")
        for r in rows:
            print(f"{r['tag']:16} {r['MAE']:>8.2f} {r['MAE']-V6_VAL_MAE:>+8.2f} "
                  f"{r['MAE']-L2_8_VAL_MAE:>+8.2f} {r['debiased']:>9.2f} {r['fcv']:>6.3f}")
        best = min(rows, key=lambda r: r["MAE"])
        print(f"\n最优: {best['tag']}  MAE={best['MAE']:.2f} (Δl2_8 {best['MAE']-L2_8_VAL_MAE:+.2f})")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
