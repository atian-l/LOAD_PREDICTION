# -*- coding: utf-8 -*-
"""
路线B Phase B1: Method A 按预测时段拆模型（7 组）+ OOF 迁移验证

思路：v6 是"全局 40 成员 ensemble + 96dim slot bias"做时段差异化。Method A 改为
"按时段分组，每组独立 ensemble"，是中间粒度（介于全局与 96 模型之间）。每组合样本
6000~19000，足够 40 成员。午间 12-14（全天最差时段）单独成组以专精。

对比 baseline = v6 全局+slot bias = 1445.62（ab.V6_VAL_MAE），v6 raw(无校正)~1512。

验证纪律（P2-8 教训）：
  - OOF 3 折估计 hour_bias（每 slot 由其所属组的 OOF 残差估计），val 验证迁移
  - 报告 raw val（无校正）/ OOF 校正后 val / OOF/val slot 一致性
  - gate: OOF 校正后 val < v6 1445.62 且 OOF/val 一致才采纳

合规: 不修改生产脚本; 复用 build_dataset/train_ensemble(不变量#5 共享 build_features);
actual 仅 target/MOS-target/eval(#1); pred_load lags 不变(#2); OOF 3 折全在训练期内;
throwaway 纯 stdout。运行: python -m load_pred.exp_methoda_group
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
from .train import build_dataset, usable_mask, train_ensemble, _time_weights
from .features import MismatchModel, MosModel
from .exp_catboost_ab import V6_VAL_MAE

# 7 组时段（hour），午间 12-14 单独（全天最差）
GROUPS = [
    ("深夜00-05", [0, 1, 2, 3, 4, 5]),
    ("晨爬坡06-09", [6, 7, 8, 9]),
    ("上午10-11", [10, 11]),
    ("午间峰12-14", [12, 13, 14]),
    ("午后15-17", [15, 16, 17]),
    ("晚高峰18-21", [18, 19, 20, 21]),
    ("夜间下行22-23", [22, 23]),
]
BEST_IT = 80


def _mae(p, a):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(a)))) if len(p) else float('nan')


def _slot_idx(times, n_slots=96):
    dt = pd.DatetimeIndex(times)
    mod = dt.hour.values * 60 + dt.minute.values
    return ((mod * n_slots) // 1440).astype(int)


def _hour_group_map():
    h2g = {}
    for gi, (_, ghours) in enumerate(GROUPS):
        for h in ghours:
            h2g[h] = gi
    return h2g


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 74)
    print(f"路线B B1: Method A 7组分组模型 + OOF迁移验证 (v6 baseline={V6_VAL_MAE})")
    print("=" * 74)

    print("[1] 构建数据集 ...")
    times, X, pred_load, actual = build_dataset()
    usable = usable_mask(times, pred_load, actual)
    mismatch_model = MismatchModel().fit(X, usable)
    X = mismatch_model.transform(X)
    mc = C.TRAIN_CONFIG["mos"]
    mos_model = MosModel(cols=mc["cols"], alpha=mc["alpha"]).fit(X, actual, usable)
    cfg = C.TRAIN_CONFIG
    feat_cols = list(X.columns)
    hours = pd.DatetimeIndex(times).hour.values
    h2g = _hour_group_map()
    gidx_all = np.array([h2g[h] for h in hours], dtype=int)
    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    av = actual[val_m].to_numpy(np.float64)
    val_gidx = gidx_all[val_m]
    print(f"    特征数={len(feat_cols)}  训练点={int(usable.sum())}  val点={int(val_m.sum())}")
    print(f"    组规模(训练): " + " ".join(f"{gn}:{int((usable & (gidx_all==gi)).sum())}"
                                          for gi, (gn, _) in enumerate(GROUPS)))

    # ---- [2] 每组训练全模型 + raw val 路由预测 ----
    print("\n[2] 每组训练全模型(40成员) + raw val 路由 ...")
    group_models = [None] * len(GROUPS)
    raw_val = np.zeros(len(av))
    for gi, (gname, ghours) in enumerate(GROUPS):
        ts = time.perf_counter()
        usable_g = usable & (gidx_all == gi)
        with contextlib.redirect_stdout(io.StringIO()):
            mg = train_ensemble(times, X, pred_load, actual, usable_g, cfg, BEST_IT, mos_model=mos_model)
        mg.mismatch_model = mismatch_model
        mg.hour_bias = None
        mg.drift_corr = []
        mg.threshold_corr = []
        group_models[gi] = mg
        # val 该组点 raw 预测
        vm = val_m & (gidx_all == gi)
        if vm.sum():
            raw_val[val_gidx == gi] = mg.predict_load(X[vm], pred_load[vm])
        print(f"    {gname:>14}: 训练{int(usable_g.sum())}点 val{int((val_m&(gidx_all==gi)).sum())}点 "
              f"raw组MAE={_mae(raw_val[val_gidx==gi], av[val_gidx==gi]):.1f}  ({time.perf_counter()-ts:.0f}s)")
    print(f"  --> Method A raw val MAE = {_mae(raw_val, av):.2f}  (v6 raw~1512, v6校正1445.62)")

    # ---- [3] OOF 3折每组训练子模型 + 收集 OOF 预测（按 slot）----
    print("\n[3] OOF 3折每组子模型 + 收集 OOF 残差(估 hour_bias) ...")
    oof_pred = pd.Series(np.nan, index=times)
    for fold_name, (te, vs, ve) in zip(["春", "秋", "冬"], cfg["best_it_folds"]):
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        fva_base = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        for gi, (gname, ghours) in enumerate(GROUPS):
            ftr = usable & np.asarray(times <= te) & (gidx_all == gi)
            fva = fva_base & (gidx_all == gi)
            if fva.sum() == 0:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                fm = train_ensemble(times, X, pred_load, actual, ftr, cfg, BEST_IT, mos_model=mos_model)
            fm.mismatch_model = mismatch_model
            fm.hour_bias = None
            fm.drift_corr = []
            fm.threshold_corr = []
            oof_pred[fva] = fm.predict_load(X[fva], pred_load[fva])
    oof_mask = usable & oof_pred.notna().values
    resid_oof = (oof_pred - actual).values
    print(f"    OOF 总点数={int(oof_mask.sum())}")

    # ---- [4] hour_bias: 96dim 逐 slot 估计（OOF）+ val 校正 ----
    n_slots = 96
    slot_all = _slot_idx(times, n_slots)
    hour_bias = np.zeros(n_slots, dtype=float)
    for q in range(n_slots):
        m = oof_mask & (slot_all == q)
        if m.sum():
            hour_bias[q] = float(np.average(resid_oof[m]))
    # val 校正
    val_slot = _slot_idx(times[val_m], n_slots)
    corr_val = raw_val - hour_bias[val_slot]
    print(f"  --> Method A OOF校正后 val MAE = {_mae(corr_val, av):.2f}  "
          f"(Δv6 {_mae(corr_val, av)-V6_VAL_MAE:+.2f})")

    # ---- [5] OOF/val slot 一致性（P2-8 纪律）----
    print("\n[5] OOF/val 一致性诊断（每 slot OOF残差 vs val残差方向）:")
    val_resid = corr_val - av  # 校正后残差（应接近0若bias迁移）
    raw_val_resid = raw_val - av  # raw 残差
    # 每 slot: OOF bias 是否与 val raw 残差同号（迁移性）
    consistent = 0
    total = 0
    for q in range(n_slots):
        mo = oof_mask & (slot_all == q)
        mv = val_m & (slot_all == q)
        if mo.sum() < 5 or mv.sum() < 5:
            continue
        total += 1
        oof_b = np.average(resid_oof[mo])
        val_b = np.average(raw_val_resid[mv])
        if np.sign(oof_b) == np.sign(val_b):
            consistent += 1
    consist_pct = 100.0 * consistent / total if total else 0.0
    print(f"    slot 方向一致: {consistent}/{total} = {consist_pct:.1f}%  "
          f"(<50% 判过拟合，P2-8 场景门控仅 2/7=29%)")

    # ---- [6] 分组对比 ----
    print(f"\n[6] 分组 val MAE 对比 (raw vs OOF校正):")
    print(f"  {'组':>14} {'n_val':>6} {'raw MAE':>9} {'校正MAE':>9} {'Δv6':>8}")
    for gi, (gname, _) in enumerate(GROUPS):
        mv = val_gidx == gi
        if mv.sum() == 0:
            continue
        rm = _mae(raw_val[mv], av[mv])
        cm = _mae(corr_val[mv], av[mv])
        print(f"  {gname:>14} {int(mv.sum()):>6} {rm:>9.1f} {cm:>9.1f} {cm-V6_VAL_MAE:>+8.1f}")

    # ---- 汇总 ----
    print("\n" + "=" * 74)
    print("B1 汇总: Method A 7组分组")
    print("=" * 74)
    print(f"  v6 baseline (全局+96dim bias) = {V6_VAL_MAE}")
    print(f"  Method A raw val              = {_mae(raw_val, av):.2f}  (v6 raw~1512)")
    print(f"  Method A OOF校正后 val        = {_mae(corr_val, av):.2f}  (Δv6 {_mae(corr_val, av)-V6_VAL_MAE:+.2f})")
    print(f"  OOF/val slot 方向一致率       = {consist_pct:.1f}%")
    verdict = "通过gate" if (_mae(corr_val, av) < V6_VAL_MAE and consist_pct >= 50) else "未通过gate"
    print(f"  判定: {verdict}")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
