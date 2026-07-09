# -*- coding: utf-8 -*-
"""实验23：树模型不能外推。2026年 pred_load 超出训练范围→树返回最大叶→欠预测。
测试 per-hour 线性模型 actual~pred_load（可外推）vs 集成，及混合。"""
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


def _mae(p, a):
    return np.mean(np.abs(p - a))


PP = dict(learning_rate=0.02, num_leaves=127, min_data_in_leaf=300, lambda_l2=1.0,
          feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1, verbose=-1, force_col_wise=True)
QA = [0.45, 0.5, 0.55]; SEEDS = [42, 7, 123]; AW = 5.0; BI = 221; LAM = 0.8


def train_members(times, X, pred_load, actual, feat_cols, train_mask):
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
                bst = lgb.train(p, d, num_boost_round=int(BI))
                raw = bst.predict(X[feat_cols])
                member_preds.append((pred_load.values + raw) if residual else raw)
    return np.array(member_preds)


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]; pv_full = pred_load.values
    mo_all = times.month.values; h_all = times.hour.values

    # 训练集成
    print("training ensemble ...")
    M_full = train_members(times, X, pred_load, actual, feat_cols, full_mask)
    ens_full = np.median(M_full, axis=0)
    base_full = np.clip(pv_full + LAM * (ens_full - pv_full), 0, None)
    print(f"[ensemble λ0.8] VAL MAE={_mae(base_full[vm], av):.2f}")
    print(f"[pred_load only ] VAL MAE={_mae(pv_full[vm], av):.2f}")
    print(f"[ens only       ] VAL MAE={_mae(ens_full[vm], av):.2f}")

    # per-hour 线性模型 actual ~ pred_load
    print("\n== per-hour linear actual~pred_load ==")
    from sklearn.linear_model import Ridge, LinearRegression
    lin_pred = np.zeros(len(times))
    for h in range(24):
        m = full_mask & (h_all == h)
        if m.sum() < 50:
            continue
        x = pv_full[m].reshape(-1, 1); y = actual.values[m]
        # 加 recency 权重
        w = tw(times, m, AW)
        rg = LinearRegression().fit(x, y, sample_weight=w)
        mh = (h_all == h)
        lin_pred[mh] = rg.predict(pv_full[mh].reshape(-1, 1))
    lin_pred = np.clip(lin_pred, 0, None)
    print(f"[per-hour linear] VAL MAE={_mae(lin_pred[vm], av):.2f}")
    for mo in [3,4,5,6]:
        mm = (mo_all[vm] == mo)
        if mm.sum():
            print(f"  mo={mo}: lin MAE={_mae(lin_pred[vm][mm], av[mm]):.2f} bias={np.mean(lin_pred[vm][mm]-av[mm]):.0f}  ens MAE={_mae(base_full[vm][mm], av[mm]):.2f}")

    # 混合：linear + ensemble (λ blend)
    print("\n== blend linear & ensemble ==")
    for beta in [0.3, 0.5, 0.7]:
        blend = beta * lin_pred + (1 - beta) * base_full
        print(f"[blend lin{beta}+ens{1-beta:.1f}] VAL MAE={_mae(blend[vm], av):.2f}")
    # linear + ens: pred = lin + λ*(ens - lin)
    for lam in [0.3, 0.5, 0.7, 0.8]:
        p = np.clip(lin_pred + lam * (ens_full - lin_pred), 0, None)
        print(f"[lin+λ{lam}*(ens-lin)] VAL MAE={_mae(p[vm], av):.2f}")

    # per-(hour,month) 线性 (捕捉季节×小时外推)
    print("\n== per-(hour,month) linear actual~pred_load ==")
    lin_hm = np.zeros(len(times))
    for mo in range(1, 13):
        for h in range(24):
            m = full_mask & (h_all == h) & (mo_all == mo)
            if m.sum() < 30:
                # 回退到 per-hour
                m2 = full_mask & (h_all == h)
                if m2.sum() < 30:
                    continue
                x = pv_full[m2].reshape(-1,1); y = actual.values[m2]; w = tw(times, m2, AW)
                rg = LinearRegression().fit(x, y, sample_weight=w)
            else:
                x = pv_full[m].reshape(-1,1); y = actual.values[m]; w = tw(times, m, AW)
                rg = LinearRegression().fit(x, y, sample_weight=w)
            mh = (h_all == h) & (mo_all == mo)
            lin_hm[mh] = rg.predict(pv_full[mh].reshape(-1, 1))
    lin_hm = np.clip(lin_hm, 0, None)
    print(f"[per-(h,mo) linear] VAL MAE={_mae(lin_hm[vm], av):.2f}")
    for lam in [0.5, 0.7, 0.8]:
        p = np.clip(lin_hm + lam * (ens_full - lin_hm), 0, None)
        print(f"[lin_hm+λ{lam}*(ens-lin_hm)] VAL MAE={_mae(p[vm], av):.2f}")

    # 集成成员改为残差+线性外推：树预测 (actual - lin_pred)，最终 = lin_pred + tree
    print("\n== tree-on-residual-from-linear (extrapolation fix) ==")
    # 残差 = actual - lin_pred (lin_pred 已外推)
    y_res2 = actual - pd.Series(lin_pred, index=times)
    member_preds2 = []
    objs = [("regression", None)] + [("quantile", q) for q in QA]
    Xtr = X[full_mask][feat_cols]; wtr = tw(times, full_mask, AW)
    for residual in (False, True):
        ytr = (y_res2 if residual else actual)[full_mask]
        d = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj, qa_ in objs:
            for s in SEEDS:
                p = dict(PP, objective=obj, seed=s)
                if obj == "quantile":
                    p["alpha"] = qa_
                bst = lgb.train(p, d, num_boost_round=int(BI))
                raw = bst.predict(X[feat_cols])
                member_preds2.append((lin_pred + raw) if residual else raw)
    M2 = np.array(member_preds2)
    ens2 = np.median(M2, axis=0)
    for lam in [0.5, 0.7, 0.8, 0.9, 1.0]:
        p = np.clip(lin_pred + lam * (ens2 - lin_pred), 0, None)
        print(f"[lin-base + tree-resid λ{lam}] VAL MAE={_mae(p[vm], av):.2f}")
    # per-month breakdown for best
    p = np.clip(lin_pred + 0.8 * (ens2 - lin_pred), 0, None)
    for mo in [3,4,5,6]:
        mm = (mo_all[vm] == mo)
        if mm.sum():
            print(f"  mo={mo}: MAE={_mae(p[vm][mm], av[mm]):.2f} bias={np.mean(p[vm][mm]-av[mm]):.0f}")


if __name__ == "__main__":
    main()
