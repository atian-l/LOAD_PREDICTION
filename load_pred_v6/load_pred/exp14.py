# -*- coding: utf-8 -*-
"""实验14：分析模型自身残差(ens-actual)与各特征的相关性，寻找未捕获的可预测结构。
用最优配置(q3,aw2.5,ts2024-01,3fold,median,λ0.8)，12 成员(3 seeds)加速。"""
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


def tw(times, mask, alpha):
    t = times[mask]; tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        return np.ones(len(t))
    return (1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()).values.astype(float)


def best_it_3fold(times, X, y_dir, feat_cols, usable, alpha_w, pp):
    folds = C.TRAIN_CONFIG["best_it_folds"]; bests = []
    for te_s, vs_s, ve_s in folds:
        te, vs, ve = pd.Timestamp(te_s), pd.Timestamp(vs_s), pd.Timestamp(ve_s)
        ftr = usable & (times <= te); fva = usable & (times >= vs) & (times <= ve)
        if ftr.sum() < 1000 or fva.sum() < 500:
            continue
        dtr = lgb.Dataset(X[ftr][feat_cols], label=y_dir[ftr].values, weight=tw(times, ftr, alpha_w))
        dva = lgb.Dataset(X[fva][feat_cols], label=y_dir[fva].values, reference=dtr)
        bst = lgb.train({**pp, "objective": "regression", "seed": 42, "metric": ["mae", "rmse"]}, dtr,
                        num_boost_round=C.TRAIN_CONFIG["best_it_num_iterations"], valid_sets=[dva], valid_names=["va"],
                        callbacks=[lgb.early_stopping(C.TRAIN_CONFIG["best_it_early_stopping"], verbose=False,
                                                      first_metric_only=True), lgb.record_evaluation({})])
        bests.append(max(bst.best_iteration, 80))
    return int(np.mean(bests)) if bests else 290


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    usable = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    aw = 2.5; qa = [0.45, 0.5, 0.55]; seeds = [42, 7, 123]  # 24 成员
    y_dir = actual; y_res = actual - pred_load
    pp = dict(learning_rate=0.02, num_leaves=127, min_data_in_leaf=300, lambda_l2=1.0,
              feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1, verbose=-1, force_col_wise=True)
    bi = best_it_3fold(times, X, y_dir, feat_cols, usable, aw, pp)
    print(f"best_it={bi}")
    Xtr = X[usable][feat_cols]; wtr = tw(times, usable, aw)
    member_preds = []
    objs = [("regression", None)] + [("quantile", q) for q in qa]
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[usable]
        d = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj, qa_ in objs:
            for s in seeds:
                p = dict(pp, objective=obj, seed=s)
                if obj == "quantile":
                    p["alpha"] = qa_
                bst = lgb.train(p, d, num_boost_round=int(bi))
                raw = bst.predict(X[feat_cols])
                member_preds.append((pred_load.values + raw) if residual else raw)
    M = np.array(member_preds)
    pv_full = pred_load.values
    ens_med = np.median(M, axis=0)
    lam = 0.8
    pred = np.clip(pv_full + lam * (ens_med - pv_full), 0, None)
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]
    model_resid = pred[vm] - av  # 模型残差
    print(f"VAL MAE={np.mean(np.abs(model_resid)):.2f} Bias={np.mean(model_resid):.1f}")
    print(f"\n=== 模型残差与各特征 corr (val) ===")
    Xv = X[vm]
    cors = Xv.corrwith(pd.Series(model_resid, index=Xv.index)).abs().sort_values(ascending=False)
    print(cors.head(25).round(3).to_string())
    # 按小时看残差 bias（是否系统性）
    hv = times[vm].hour.values
    print("\n=== 模型残差 by hour (mean/std/mae) ===")
    for h in range(0, 24, 2):
        m = hv == h
        if m.sum() > 0:
            r = model_resid[m]
            print(f"  h={h:2d}: bias={r.mean():7.0f} std={r.std():6.0f} mae={np.mean(np.abs(r)):6.0f} n={m.sum()}")
    # 按月看
    mv = times[vm].month.values
    print("\n=== 模型残差 by month ===")
    for mo in sorted(set(mv)):
        m = mv == mo
        r = model_resid[m]
        print(f"  mo={mo}: bias={r.mean():7.0f} std={r.std():6.0f} mae={np.mean(np.abs(r)):6.0f} n={m.sum()}")
    # 按预测负荷分位看
    print("\n=== 模型残差 by pred_load quartile ===")
    pv = pv_full[vm]
    q = pd.qcut(pv, 5, labels=["Q1low", "Q2", "Q3", "Q4", "Q5high"])
    for lab in ["Q1low", "Q2", "Q3", "Q4", "Q5high"]:
        m = (q == lab).values
        r = model_resid[m]
        print(f"  {lab}: bias={r.mean():7.0f} std={r.std():6.0f} mae={np.mean(np.abs(r)):6.0f} pred_mean={pv[m].mean():.0f}")


if __name__ == "__main__":
    main()
