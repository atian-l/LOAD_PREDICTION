# -*- coding: utf-8 -*-
"""实验11b：快速 train_start 扫描（固定 best_it=290，8 成员）+ 日级误差分析。"""
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


def _m(p, a):
    err = p - a
    return (np.mean(np.abs(err)),
            1 - np.sum(err ** 2) / np.sum((a - a.mean()) ** 2),
            np.mean(err))


def train_ens(times, X, pred_load, actual, feat_cols, usable, alpha_w, seeds, qalphas,
              lr, nl, md, best_it=290):
    y_dir = actual; y_res = actual - pred_load
    pp = dict(learning_rate=lr, num_leaves=nl, min_data_in_leaf=md, lambda_l2=1.0,
              feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1, verbose=-1,
              force_col_wise=True)
    Xtr = X[usable][feat_cols]; wtr = tw(times, usable, alpha_w)
    raw_sum = np.zeros(len(times)); n = 0
    objs = [("regression", None)] + [("quantile", q) for q in qalphas]
    for residual in (False, True):
        ytr = (y_res if residual else y_dir)[usable]
        d = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj, qa in objs:
            for s in seeds:
                p = dict(pp, objective=obj, seed=s)
                if obj == "quantile":
                    p["alpha"] = qa
                bst = lgb.train(p, d, num_boost_round=int(best_it))
                raw = bst.predict(X[feat_cols])
                raw_sum += (pred_load.values + raw) if residual else raw
                n += 1
    ens = np.clip(raw_sum / n, 0, None)
    return ens, n


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    tr_end = pd.Timestamp(C.TRAIN_END)
    s3 = [42, 7, 123]; aw = 2.5; qa = [0.5]   # 8 成员，快速
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values

    print("\n=== (a) train_start sweep (8-member, best_it=290) ===")
    starts = [("2023-02-01", "ts2023-02"), ("2024-01-01", "ts2024-01"),
              ("2024-06-01", "ts2024-06"), ("2025-01-01", "ts2025-01"),
              ("2025-06-01", "ts2025-06"), ("2025-09-01", "ts2025-09")]
    for s, name in starts:
        ts0 = pd.Timestamp(s)
        usable = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
        ens, n = train_ens(times, X, pred_load, actual, feat_cols, usable, aw, s3, qa, 0.02, 127, 300)
        mae, r2, bias = _m(ens[vm], actual.values[vm])
        print(f"[{name:12s}] n={n}  VAL MAE={mae:.2f}  R2={r2:.4f}  Bias={bias:.0f}")

    # (b) 日级误差分析（用 2023-02 起点，8 成员）
    print("\n=== (b) day-level error analysis ===")
    ts0 = pd.Timestamp("2023-02-01")
    usable = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    ens, n = train_ens(times, X, pred_load, actual, feat_cols, usable, aw, s3, qa, 0.02, 127, 300)
    tv = times[vm]; ev = ens[vm]; av = actual.values[vm]; pv = pred_load.values[vm]
    df = pd.DataFrame({"t": tv, "ens": ev, "actual": av, "pred_load": pv, "err": ev - av})
    df["date"] = df["t"].dt.date
    df["dow"] = df["t"].dt.dayofweek
    df["hour"] = df["t"].dt.hour
    hol_dates = set()
    for r in F._HOLIDAY_RANGES:
        for d in pd.date_range(r[0], r[1]):
            hol_dates.add(d.date())
    df["is_holiday"] = df["t"].dt.date.isin(hol_dates).astype(int)
    day_mae = df.groupby("date").agg(day_mae=("err", lambda x: np.mean(np.abs(x))),
                                     day_bias=("err", "mean"),
                                     dow=("dow", "first"), is_holiday=("is_holiday", "first"),
                                     actual=("actual", "mean")).reset_index().sort_values("day_mae", ascending=False)
    print("worst 12 days:"); print(day_mae.head(12).to_string(index=False))
    print("\nbest 6 days:"); print(day_mae.tail(6).to_string(index=False))
    print("\nby dow:"); print(day_mae.groupby("dow")["day_mae"].agg(["mean", "count"]).round(0).to_string())
    print("\nby holiday:"); print(day_mae.groupby("is_holiday")["day_mae"].agg(["mean", "count"]).round(0).to_string())
    df["month"] = df["t"].dt.to_period("M").astype(str)
    print("\nby month (MAE):"); print(df.groupby("month")["err"].agg(lambda x: np.mean(np.abs(x))).round(0).to_string())


if __name__ == "__main__":
    main()
