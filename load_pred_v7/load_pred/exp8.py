# -*- coding: utf-8 -*-
"""
实验8：在 winning 集成基础上，测试
  (a) 近期 OOS 偏差校正（用训练期内的留出近期窗口估计偏差，非官方验证集）；
  (b) alpha_w / 结构微调。
"""
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


def time_weights(times, mask, alpha):
    t = times[mask]; tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        return np.ones(len(t))
    return (1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()).values.astype(float)


def _m(p, a):
    err = p - a
    return np.mean(np.abs(err)), (1 - np.sum(err**2) / np.sum((a - a.mean())**2)), np.mean(err)


def train_ens(X, pred_load, actual, mask, feat_cols, alpha_w, best_it, seeds, lr, nl, md):
    y_dir = actual; y_res = actual - pred_load
    Xtr = X[mask][feat_cols]; wtr = time_weights(X.index, mask, alpha_w)
    base = dict(metric=["mae", "rmse"], learning_rate=lr, num_leaves=nl,
                min_data_in_leaf=md, lambda_l2=1.0, feature_fraction=0.80,
                bagging_fraction=0.80, bagging_freq=1, verbose=-1, force_col_wise=True)
    raw_sum = np.zeros(len(X)); n = 0
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[mask]
        dtr = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj, qa in [("regression", None), ("quantile", 0.5)]:
            for s in seeds:
                p = dict(base, objective=obj, seed=s)
                if obj == "quantile":
                    p["alpha"] = qa
                bst = lgb.train(p, dtr, num_boost_round=int(best_it))
                raw = bst.predict(X[feat_cols])
                raw_sum += (pred_load.values + raw) if residual else raw
                n += 1
    return np.clip(raw_sum / n, 0, None)


def best_it_3fold(times, X, y_dir, usable, feat_cols, alpha_w, lr, nl, md):
    base = dict(metric=["mae", "rmse"], learning_rate=lr, num_leaves=nl,
                min_data_in_leaf=md, lambda_l2=1.0, feature_fraction=0.80,
                bagging_fraction=0.80, bagging_freq=1, verbose=-1, force_col_wise=True,
                seed=42, objective="regression", num_iterations=8000, early_stopping_rounds=300)
    pp = dict(base); pp.pop("num_iterations"); pp.pop("early_stopping_rounds")
    its = []
    for te, vs, ve in [("2025-02-28","2025-03-01","2025-05-31"),
                       ("2025-08-31","2025-09-01","2025-11-30"),
                       ("2025-11-30","2025-12-01","2026-01-31")]:
        te,vs,ve=pd.Timestamp(te),pd.Timestamp(vs),pd.Timestamp(ve)
        ftr=usable&(times<=te); fva=usable&(times>=vs)&(times<=ve)
        dtr=lgb.Dataset(X[ftr][feat_cols],label=y_dir[ftr].values,weight=time_weights(times,ftr,alpha_w))
        dva=lgb.Dataset(X[fva][feat_cols],label=y_dir[fva].values,reference=dtr)
        ev={}
        b=lgb.train(pp,dtr,num_boost_round=base["num_iterations"],valid_sets=[dva],valid_names=["va"],
                    callbacks=[lgb.early_stopping(base["early_stopping_rounds"],verbose=False,first_metric_only=True),
                               lgb.record_evaluation(ev)])
        its.append(b.best_iteration)
    return int(np.mean(its))


def main():
    print("building features ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2023-02-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    usable = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    actual_arr = actual.values

    seeds5 = [42, 7, 123, 2024, 99]
    seeds8 = [42, 7, 123, 2024, 99, 31, 256, 555]

    # 候选配置
    cands = [
        ("base",        1.5, 300, 127, seeds5),
        ("aw2.5",       2.5, 300, 127, seeds5),
        ("aw3.0",       3.0, 300, 127, seeds5),
        ("md200_l159",  1.5, 200, 159, seeds5),
        ("seeds8",      1.5, 300, 127, seeds8),
        ("aw2.5_s8",    2.5, 300, 127, seeds8),
    ]
    for name, aw, md, nl, seeds in cands:
        best_it = best_it_3fold(times, X, actual, usable, feat_cols, aw, 0.02, nl, md)
        ens = train_ens(X, pred_load, actual, usable, feat_cols, aw, best_it, seeds, 0.02, nl, md)
        mae, r2, bias = _m(ens[vm], actual_arr[vm])
        # 近期 OOS 偏差校正：训练到 2025-12-15，预测 2025-12-16~2026-01-31
        cut = pd.Timestamp("2025-12-15")
        recent_tr = usable & (times <= cut)
        recent_va = usable & (times > cut)
        best_it_r = best_it_3fold(times, X, actual, recent_tr, feat_cols, aw, 0.02, nl, md)
        ens_r = train_ens(X, pred_load, actual, recent_tr, feat_cols, aw, best_it_r, seeds, 0.02, nl, md)
        oos_bias = np.mean(ens_r[recent_va] - actual_arr[recent_va])  # pred-actual
        ens_corr = np.clip(ens - oos_bias, 0, None)
        mae_c, r2_c, bias_c = _m(ens_corr[vm], actual_arr[vm])
        print(f"[{name:11s}] best_it={best_it}  VAL MAE={mae:.2f}(b={bias:.0f})  "
              f"OOS_bias={oos_bias:.0f}  VAL@corr MAE={mae_c:.2f}(b={bias_c:.0f})")


if __name__ == "__main__":
    main()
