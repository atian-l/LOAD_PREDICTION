# -*- coding: utf-8 -*-
"""
CatBoost H1/H2/H3/H4: 特征 + 样本类扫描

H1 特征筛选: drop_q(去_p25/_p75/_std气象分位数) / drop_x(去_x_交互列) / core(仅核心列)
H2 alpha_w(时间近期权重): 2.5 / 10.0 / 20.0  (默认 5.0=baseline)
H3 weight_load_gamma(负荷加权): 0.0 / 0.5 / 1.5 / 2.0  (默认 1.0=baseline)
H4 train_start(训练起点): 2023-02 / 2024-07 / 2025-01  (默认 2024-01=baseline)

H1 改 feat_cols(仅成员训练输入子集；MismatchModel/MosModel/校正特征用全 X 不变)。
H2/H3 改 cfg 副本顶层参数。H4 自建 usable(因 usable_mask 读全局 train_start)。
其余超参固定 l2_8。

Caveat：H4 仅改训练 mask，MOS anchor 未随 train_start 重建（严格应重建，此处简化）。
合规：不修改生产脚本；仅 import 复用 hp._run_config + ab/train/features；6 条泄露不变量全保持。
运行：python -m load_pred.exp_catboost_feat_sample   （4090 上约 20-30 min，11 配置）
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

CORE_KEYS = ["pred_load", "lag_", "rolling", "hour", "dow", "month", "is_",
             "temp", "clearness", "precip", "irrad", "pl_weather_residual",
             "hdd", "cdd", "wind", "solar_wind"]


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print(f"CatBoost H1/H2/H3/H4: 特征+样本类扫描 (best_it={BEST_IT})")
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

    # H1 子集
    fc_drop_q = [c for c in feat_cols if not any(s in c for s in ["_p25", "_p75", "_std"])]
    fc_drop_x = [c for c in feat_cols if "_x_" not in c]
    fc_core = [c for c in feat_cols if any(k in c for k in CORE_KEYS)]
    print(f"    H1 子集: drop_q={len(fc_drop_q)}  drop_x={len(fc_drop_x)}  core={len(fc_core)} (全={len(feat_cols)})")

    # 构造配置 (tag, cfg, feat_cols, usable)
    configs = []
    configs.append(("baseline", cfg, feat_cols, usable))
    # H1
    configs.append(("H1_drop_q", cfg, fc_drop_q, usable))
    configs.append(("H1_drop_x", cfg, fc_drop_x, usable))
    configs.append(("H1_core", cfg, fc_core, usable))
    # H2 alpha_w
    for aw, tag in [(2.5, "H2_aw2.5"), (10.0, "H2_aw10.0"), (20.0, "H2_aw20.0")]:
        c = dict(cfg); c["alpha_w"] = aw
        configs.append((tag, c, feat_cols, usable))
    # H3 weight_load_gamma
    for g, tag in [(0.0, "H3_g0.0"), (0.5, "H3_g0.5"), (1.5, "H3_g1.5"), (2.0, "H3_g2.0")]:
        c = dict(cfg); c["weight_load_gamma"] = g
        configs.append((tag, c, feat_cols, usable))
    # H4 train_start（自建 usable；TRAIN_END 不变）
    tr_end = pd.Timestamp(C.TRAIN_END)
    for ts, tag in [("2023-02-01", "H4_ts2023"), ("2024-07-01", "H4_ts2024h2"),
                    ("2025-01-01", "H4_ts2025")]:
        c = dict(cfg); c["train_start"] = ts + " 00:00:00"
        ts0 = pd.Timestamp(ts)
        usable_h4 = ((times >= ts0) & (times <= tr_end)
                     & pred_load.notna() & actual.notna()).values
        configs.append((tag, c, feat_cols, usable_h4))
    print(f"    配置数={len(configs)}")

    print(f"\n[2] 逐配置训练 + 评估 ...")
    rows = []
    for tag, cfgm, fc, us in configs:
        try:
            r = hp._run_config(tag, HP_L2_8, BEST_IT, times, X, pred_load, actual,
                               us, anchor, cfgm, fc, val_m)
            rows.append(r)
            print(f"  {tag:14s} MAE={r['MAE']:.2f} Δv6={r['MAE']-V6_VAL_MAE:+.2f} "
                  f"Δl2_8={r['MAE']-L2_8_VAL_MAE:+.2f} debiased={r['debiased']:.2f} "
                  f"折CV={r['fcv']:.3f} ({r['dt']:.0f}s)")
        except Exception as e:
            print(f"  {tag:14s} FAIL ({type(e).__name__}: {str(e).splitlines()[0][:80]})")

    if rows:
        print("\n" + "=" * 74)
        print(f"H1/H2/H3/H4 特征+样本对比（vs l2_8 {L2_8_VAL_MAE} / v6 {V6_VAL_MAE}）")
        print("=" * 74)
        print(f"{'tag':14} {'MAE':>8} {'Δv6':>8} {'Δl2_8':>8} {'debiased':>9} {'折CV':>6}")
        for r in rows:
            print(f"{r['tag']:14} {r['MAE']:>8.2f} {r['MAE']-V6_VAL_MAE:>+8.2f} "
                  f"{r['MAE']-L2_8_VAL_MAE:>+8.2f} {r['debiased']:>9.2f} {r['fcv']:>6.3f}")
        best = min(rows, key=lambda r: r["MAE"])
        print(f"\n最优: {best['tag']}  MAE={best['MAE']:.2f} (Δl2_8 {best['MAE']-L2_8_VAL_MAE:+.2f})")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
