# -*- coding: utf-8 -*-
"""实验16：新验证窗(03-01~06-15)下扫描 recency(aw) × 季节加权 × best_it。
12 成员(3 seeds, q3)，固定 best_it，快速比较。median+λ0.8 聚合。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from . import data_loader as dl
from . import features as F


def build():
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)
    X = F.build_features(times, pred_load, weather)
    return times, X, pred_load, actual


def weights(times, mask, alpha, seasonal_boost, season_months=(3, 4, 5, 6)):
    """recency × 季节加权。season_months 匹配验证集月份。"""
    t = times[mask]; tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        rec = np.ones(len(t))
    else:
        rec = (1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()).values.astype(float)
    if seasonal_boost != 1.0:
        m = t.month.values
        rec = rec * np.where(np.isin(m, season_months), seasonal_boost, 1.0)
    return rec


def _mae(p, a):
    return np.mean(np.abs(p - a))


PP = dict(learning_rate=0.02, num_leaves=127, min_data_in_leaf=300, lambda_l2=1.0,
          feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1, verbose=-1, force_col_wise=True)
QA = [0.45, 0.5, 0.55]; SEEDS = [42, 7, 123]


def run(name, times, X, pred_load, actual, feat_cols, ts0_s, alpha, season_boost, best_it):
    ts0 = pd.Timestamp(ts0_s); tr_end = pd.Timestamp(C.TRAIN_END)
    usable = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    y_dir = actual; y_res = actual - pred_load
    Xtr = X[usable][feat_cols]; wtr = weights(times, usable, alpha, season_boost)
    member_preds = []
    objs = [("regression", None)] + [("quantile", q) for q in QA]
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[usable]
        d = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj, qa_ in objs:
            for s in SEEDS:
                p = dict(PP, objective=obj, seed=s)
                if obj == "quantile":
                    p["alpha"] = qa_
                bst = lgb.train(p, d, num_boost_round=int(best_it))
                raw = bst.predict(X[feat_cols])
                member_preds.append((pred_load.values + raw) if residual else raw)
    M = np.array(member_preds)
    pv_full = pred_load.values
    ens = np.median(M, axis=0)
    pred = np.clip(pv_full + 0.8 * (ens - pv_full), 0, None)
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]
    mae = _mae(pred[vm], av)
    # 按月
    hv = times[vm]; mv = hv.month.values
    bym = {mo: _mae(pred[vm][mv == mo], av[mv == mo]) for mo in sorted(set(mv))}
    bym_s = " ".join(f"{mo}:{v:.0f}" for mo, v in bym.items())
    print(f"[{name:22s}] aw={alpha} sb={season_boost} bi={best_it}  VAL MAE={mae:.2f}  [{bym_s}]")


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    configs = [
        ("base_aw2.5",         "2024-01-01", 2.5, 1.0, 221),
        ("aw3.5",              "2024-01-01", 3.5, 1.0, 221),
        ("aw5.0",              "2024-01-01", 5.0, 1.0, 221),
        ("aw2.5_season2",      "2024-01-01", 2.5, 2.0, 221),
        ("aw2.5_season3",      "2024-01-01", 2.5, 3.0, 221),
        ("aw3.5_season2",      "2024-01-01", 3.5, 2.0, 221),
        ("aw3.5_season3",      "2024-01-01", 3.5, 3.0, 221),
        ("aw2.5_bi290",        "2024-01-01", 2.5, 1.0, 290),
        ("aw3.5_season2_bi290","2024-01-01", 3.5, 2.0, 290),
        ("ts2023_aw3.5_s2",    "2023-02-01", 3.5, 2.0, 221),
    ]
    for name, ts, aw, sb, bi in configs:
        run(name, times, X, pred_load, actual, feat_cols, ts, aw, sb, bi)


if __name__ == "__main__":
    main()
