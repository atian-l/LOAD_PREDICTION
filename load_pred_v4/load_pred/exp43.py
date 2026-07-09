# -*- coding: utf-8 -*-
"""实验43：exp41 特征 + 残差周/日变化（核心漂移信号），并快速超参扫描（no-bias 排序）。
pl_wr_diff_672 = pl_wr[T] - pl_wr[T-672]  周同比残差变化（漂移加速度）
pl_wr_diff_96  = pl_wr[T] - pl_wr[T-96]   日变化
超参扫描用 no-bias MAE 排序（per-hour 固定约 -10MW，不影响排序）。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from .exp41 import build as build41, tw, _mae, hour_bias, FOLDS, LAM, AW, QA, SEEDS, NL, LR, MDL, L2, FF, BF, BI


def build():
    times, X, pred_load, actual = build41()
    pl_wr = pd.Series(X["pl_weather_residual"].values, index=X.index)
    X["pl_wr_diff_672"] = (pl_wr - pl_wr.shift(672)).values
    X["pl_wr_diff_96"] = (pl_wr - pl_wr.shift(96)).values
    return times, X, pred_load, actual


def train_members(times, X, pred_load, actual, feat_cols, train_mask,
                  nl=NL, lr=LR, mdl=MDL, l2=L2, ff=FF, bf=BF, bi=BI):
    y_dir = actual; y_res = actual - pred_load
    Xtr = X[train_mask][feat_cols]; wtr = tw(times, train_mask, AW)
    PP = dict(learning_rate=lr, num_leaves=nl, min_data_in_leaf=mdl, lambda_l2=l2,
              feature_fraction=ff, bagging_fraction=bf, bagging_freq=1, verbose=-1, force_col_wise=True)
    member_preds = []
    objs = [("regression", None)] + [("quantile", q) for q in QA]
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[train_mask]
        d = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj, qa_ in objs:
            for s in SEEDS:
                p = dict(PP, objective=obj, seed=s)
                if obj == "quantile":
                    p["alpha"] = qa_
                bst = lgb.train(p, d, num_boost_round=int(bi))
                raw = bst.predict(X[feat_cols])
                member_preds.append((pred_load.values + raw) if residual else raw)
    return np.array(member_preds)


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    print(f"  n_feat={len(feat_cols)}")
    vm = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END)) & actual.notna()).values
    av = actual.values[vm]
    pv_full = pred_load.values
    dt = pd.DatetimeIndex(times); hour = dt.hour.values
    pl_err = pv_full - actual.values

    print("=== val 全天方向相关性 ===")
    for col in ["pl_wr_diff_672", "pl_wr_diff_96", "pl_weather_residual"]:
        v = X[col].values[vm]
        m = np.isfinite(v) & np.isfinite(pl_err[vm])
        c = float(np.corrcoef(v[m], pl_err[vm][m])[0, 1]) if m.sum() > 10 else float("nan")
        print(f"  {col:22s} corr={c:+.4f} (n={m.sum()})")

    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values

    # 基线（当前最优配置）
    print("\n== baseline (nl=127 lr=0.03 mdl=300 l2=2.0 λ=0.9) ==")
    M = train_members(times, X, pred_load, actual, feat_cols, full_mask)
    ens = np.median(M, axis=0)
    base = np.clip(pv_full + LAM * (ens - pv_full), 0, None)
    print(f"  no-bias: VAL MAE={_mae(base[vm], av):.2f}")

    # 超参扫描（no-bias 排序）
    print("\n== 超参扫描 (no-bias) ==")
    for nl in [127, 255]:
        for mdl in [200, 300]:
            for l2 in [2.0, 4.0]:
                M = train_members(times, X, pred_load, actual, feat_cols, full_mask, nl=nl, mdl=mdl, l2=l2)
                ens = np.median(M, axis=0)
                for lam in [0.9, 1.0]:
                    b = np.clip(pv_full + lam*(ens - pv_full), 0, None)
                    print(f"  nl={nl} mdl={mdl} l2={l2} λ{lam}: no-bias MAE={_mae(b[vm], av):.2f}")


if __name__ == "__main__":
    main()
