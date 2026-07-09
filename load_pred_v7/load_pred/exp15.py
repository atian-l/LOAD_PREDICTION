# -*- coding: utf-8 -*-
"""实验15：泄露安全的二阶段校正。
1) 全量训练 -> val 预测(median+λ0.8) [基线 ~1524]
2) 3-fold OOF 残差(oof_pred - actual) 估计系统性结构，校正 val：
   (a) per-hour 偏置（OOF 按小时均值，recency 加权）
   (b) per-(hour, pred_load-quintile) 偏置
   (c) 二阶段 Ridge：oof_resid ~ hour-dummies + pred_load + weather
   全部仅用训练期 OOF，无 val 泄露。"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

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


def tw(times, mask, alpha):
    t = times[mask]; tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        return np.ones(len(t))
    return (1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()).values.astype(float)


def _mae(p, a):
    return np.mean(np.abs(p - a))


PP = dict(learning_rate=0.02, num_leaves=127, min_data_in_leaf=300, lambda_l2=1.0,
          feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1, verbose=-1, force_col_wise=True)
QA = [0.45, 0.5, 0.55]; SEEDS = [42, 7, 123]; AW = 2.5


def train_members(times, X, pred_load, actual, feat_cols, train_mask, best_it):
    """在 train_mask 上训练 24 成员，返回对全时间索引的预测矩阵 M (n_members, T)。"""
    y_dir = actual; y_res = actual - pred_load
    Xtr = X[train_mask][feat_cols]; wtr = tw(times, train_mask, AW)
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
                bst = lgb.train(p, d, num_boost_round=int(best_it))
                raw = bst.predict(X[feat_cols])
                member_preds.append((pred_load.values + raw) if residual else raw)
    return np.array(member_preds)


def best_it_for(train_mask, times, X, y_dir, feat_cols):
    folds = C.TRAIN_CONFIG["best_it_folds"]; bests = []
    for te_s, vs_s, ve_s in folds:
        te, vs, ve = pd.Timestamp(te_s), pd.Timestamp(vs_s), pd.Timestamp(ve_s)
        ftr = train_mask & (times <= te); fva = train_mask & (times >= vs) & (times <= ve)
        if ftr.sum() < 1000 or fva.sum() < 500:
            continue
        dtr = lgb.Dataset(X[ftr][feat_cols], label=y_dir[ftr].values, weight=tw(times, ftr, AW))
        dva = lgb.Dataset(X[fva][feat_cols], label=y_dir[fva].values, reference=dtr)
        bst = lgb.train({**PP, "objective": "regression", "seed": 42, "metric": ["mae", "rmse"]}, dtr,
                        num_boost_round=C.TRAIN_CONFIG["best_it_num_iterations"], valid_sets=[dva], valid_names=["va"],
                        callbacks=[lgb.early_stopping(C.TRAIN_CONFIG["best_it_early_stopping"], verbose=False,
                                                      first_metric_only=True), lgb.record_evaluation({})])
        bests.append(max(bst.best_iteration, 80))
    return int(np.mean(bests)) if bests else 290


def agg(M, pv_full, lam=0.8):
    ens = np.median(M, axis=0)
    return np.clip(pv_full + lam * (ens - pv_full), 0, None)


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]; pv_full = pred_load.values

    # ---- 1) 全量训练 ----
    print("training full ensemble (24 members) ...")
    bi_full = best_it_for(full_mask, times, X, actual, feat_cols)
    print(f"best_it(full)={bi_full}")
    M_full = train_members(times, X, pred_load, actual, feat_cols, full_mask, bi_full)
    base_val = agg(M_full, pv_full)[vm]
    print(f"[baseline median+λ0.8     ] VAL MAE={_mae(base_val, av):.2f}")

    # ---- 2) 3-fold OOF：每折用 train<=fold_end 训练，预测该折（复用 bi_full 省时） ----
    print("computing 3-fold OOF ...")
    folds = C.TRAIN_CONFIG["best_it_folds"]
    oof_pred = np.full(len(times), np.nan)
    for te_s, vs_s, ve_s in folds:
        te, vs, ve = pd.Timestamp(te_s), pd.Timestamp(vs_s), pd.Timestamp(ve_s)
        ftr = full_mask & np.asarray(times <= te)
        fva = full_mask & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        M = train_members(times, X, pred_load, actual, feat_cols, ftr, bi_full)
        oof_pred[fva] = agg(M, pv_full)[fva]
    oof_mask = full_mask & (~np.isnan(oof_pred))
    oof_resid = oof_pred - actual.values  # 全时间，nan 处不用
    print(f"OOF 覆盖 {oof_mask.sum()} 点, OOF MAE={_mae(oof_pred[oof_mask], actual.values[oof_mask]):.2f}")

    # ---- 校正 (a): per-hour 偏置（OOF, recency 加权） ----
    h_all = times.hour.values
    # 全长度 recency 权重（基于 OOF 时间范围）
    t_oof = times[oof_mask]; tmin, tmax = t_oof.min(), t_oof.max()
    w_rec = (1.0 + AW * (times - tmin).total_seconds().values / (tmax - tmin).total_seconds()).astype(float)
    hour_bias = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum() > 0:
            hour_bias[h] = np.average(oof_resid[m], weights=w_rec[m])
    corr_a = np.array([hour_bias[hh] for hh in h_all])
    val_a = np.clip(base_val - corr_a[vm], 0, None)
    print(f"[+ per-hour bias (a)      ] VAL MAE={_mae(val_a, av):.2f}  (hour_bias range {hour_bias.min():.0f}..{hour_bias.max():.0f})")

    # ---- 校正 (b): per-(hour, pred_load-quintile) ----
    pq = pd.qcut(pv_full[oof_mask], 5, labels=range(5))
    oof_pq = np.full(len(times), -1)
    oof_pq[oof_mask] = pq.codes
    # 用 OOF 分位边界
    edges = np.quantile(pv_full[oof_mask], [0, .2, .4, .6, .8, 1.0])
    def pq_of(v):
        return min(4, np.searchsorted(edges[1:-1], v, side="right"))
    cell_bias = np.zeros((24, 5)); cell_cnt = np.zeros((24, 5))
    for h in range(24):
        for q in range(5):
            m = oof_mask & (h_all == h) & (oof_pq == q)
            if m.sum() > 0:
                cell_bias[h, q] = np.average(oof_resid[m], weights=w_rec[m])
                cell_cnt[h, q] = m.sum()
    corr_b = np.array([cell_bias[hh, pq_of(pv_full[i])] for i, hh in enumerate(h_all)])
    val_b = np.clip(base_val - corr_b[vm], 0, None)
    print(f"[+ per-hour×load-quint (b)] VAL MAE={_mae(val_b, av):.2f}")

    # ---- 校正 (c): 二阶段 Ridge on OOF resid ----
    feat2 = ["hour", "dayofweek", "is_holiday", "pred_load", "pred_load_vs_mean_96",
             "temp", "hdd", "cdd", "irrad", "pred_load_diff_96"]
    X2 = X[feat2].copy()
    X2["hour"] = h_all
    oof_X = X2[oof_mask]; oof_y = oof_resid[oof_mask]
    for alpha in [0.1, 1.0, 10.0, 50.0]:
        rg = Ridge(alpha=alpha).fit(oof_X, oof_y, sample_weight=w_rec[oof_mask])
        corr_c = rg.predict(X2)
        val_c = np.clip(base_val - corr_c[vm], 0, None)
        print(f"[+ Ridge2 alpha={alpha:5.1f}  (c)] VAL MAE={_mae(val_c, av):.2f}")


if __name__ == "__main__":
    main()
