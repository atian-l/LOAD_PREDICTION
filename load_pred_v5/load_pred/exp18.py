# -*- coding: utf-8 -*-
"""实验18：日级堆叠。点级模型(aw5.0) + 用 OOF 日残差训练日级模型预测每日偏置。
oracle per-day=1395，若日级特征可预测部分日偏置，则有望突破 1500。"""
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


def agg(M, pv_full):
    ens = np.median(M, axis=0)
    return np.clip(pv_full + LAM * (ens - pv_full), 0, None)


def day_features(times, X, pred_load):
    """每日日级特征（运行时可获得：D+1 全天 pred_load + 气象预报 + 日历）。"""
    df = pd.DataFrame({"t": times, "pred_load": pred_load.values})
    df["date"] = df["t"].dt.date
    df["mo"] = df["t"].dt.month; df["dow"] = df["t"].dt.dayofweek; df["doy"] = df["t"].dt.dayofyear
    g = df.groupby("date")
    out = pd.DataFrame({
        "date": list(g.groups.keys()),
        "pl_mean": g["pred_load"].mean().values,
        "pl_std": g["pred_load"].std().values,
        "pl_min": g["pred_load"].min().values,
        "pl_max": g["pred_load"].max().values,
        "mo": g["mo"].first().values,
        "dow": g["dow"].first().values,
        "doy": g["doy"].first().values,
    })
    # 日级气象
    for col in ["temp", "hdd", "cdd", "irrad", "wind"]:
        if col in X.columns:
            df[col] = X[col].values
            out[col + "_mean"] = df.groupby("date")[col].mean().reindex(out["date"]).values
    out["is_holiday"] = out["dow"].isin([5, 6]).astype(int)  # 粗略； holiday 细节见 features
    return out


def main():
    print("building ...")
    times, X, pred_load, actual = build()
    feat_cols = list(X.columns)
    ts0 = pd.Timestamp("2024-01-01"); tr_end = pd.Timestamp(C.TRAIN_END)
    full_mask = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    vm = ((times >= C.VAL_START) & (times <= C.VAL_END) & actual.notna()).values
    av = actual.values[vm]; pv_full = pred_load.values

    print("training full ensemble ...")
    M_full = train_members(times, X, pred_load, actual, feat_cols, full_mask)
    base_full = agg(M_full, pv_full)
    print(f"[baseline              ] VAL MAE={_mae(base_full[vm], av):.2f}")

    # 3-fold OOF
    print("computing 3-fold OOF ...")
    folds = C.TRAIN_CONFIG["best_it_folds"]
    oof_pred = np.full(len(times), np.nan)
    for te_s, vs_s, ve_s in folds:
        te, vs, ve = pd.Timestamp(te_s), pd.Timestamp(vs_s), pd.Timestamp(ve_s)
        ftr = full_mask & np.asarray(times <= te)
        fva = full_mask & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        M = train_members(times, X, pred_load, actual, feat_cols, ftr)
        oof_pred[fva] = agg(M, pv_full)[fva]
    oof_mask = full_mask & ~np.isnan(oof_pred)
    oof_resid = oof_pred - actual.values
    print(f"OOF n={oof_mask.sum()} MAE={_mae(oof_pred[oof_mask], actual.values[oof_mask]):.2f}")

    # 日级特征 + OOF 日残差
    dfeat = day_features(times, X, pred_load)
    dfeat = dfeat.set_index("date")
    dates = pd.Series(times).dt.date.values
    # OOF 日残差
    oof_df = pd.DataFrame({"date": dates, "resid": oof_resid, "ok": oof_mask})
    day_resid = oof_df[oof_df["ok"]].groupby("date")["resid"].mean()
    day_resid = day_resid.reindex(dfeat.index)
    train_day_mask = day_resid.notna()
    print(f"OOF 日数={train_day_mask.sum()}  day_resid stats: mean={day_resid[train_day_mask].mean():.0f} std={day_resid[train_day_mask].std():.0f}")
    # 日级残差与日级特征相关性
    dy = day_resid[train_day_mask]
    for c in ["pl_mean", "pl_std", "temp_mean", "hdd_mean", "cdd_mean", "irrad_mean", "mo", "dow"]:
        if c in dfeat.columns:
            print(f"  corr(day_resid, {c:10s}) = {dy.corr(dfeat.loc[train_day_mask, c]):.3f}")

    feat_cols_day = [c for c in dfeat.columns if c != "is_holiday" or True]
    Xd = dfeat[feat_cols_day].copy()
    # 训练日级模型
    for mdl in ["ridge", "lgb"]:
        if mdl == "ridge":
            from sklearn.linear_model import Ridge
            # one-hot mo, dow
            Xd2 = pd.get_dummies(Xd, columns=["mo", "dow"], drop_first=False)
            Xd_tr = Xd2[train_day_mask].fillna(0)
            for alpha in [1.0, 10.0, 50.0]:
                rg = Ridge(alpha=alpha).fit(Xd_tr, dy.values)
                day_corr = rg.predict(Xd2.fillna(0))
                # 减去日级偏置
                corr_full = pd.Series(day_corr, index=dfeat.index).reindex(dates).values
                pred_corr = np.clip(base_full - corr_full, 0, None)
                print(f"[+ day-ridge a={alpha:4.1f}    ] VAL MAE={_mae(pred_corr[vm], av):.2f}")
        else:
            # LightGBM 日级
            Xd_tr = Xd[train_day_mask]
            for nl in [8, 16]:
                dd = lgb.Dataset(Xd_tr, label=dy.values)
                bst = lgb.train(dict(learning_rate=0.05, num_leaves=nl, min_data_in_leaf=5,
                                     lambda_l2=5.0, verbose=-1, objective="regression", seed=42),
                                dd, num_boost_round=200)
                day_corr = bst.predict(Xd)
                corr_full = pd.Series(day_corr, index=dfeat.index).reindex(dates).values
                pred_corr = np.clip(base_full - corr_full, 0, None)
                print(f"[+ day-lgb nl={nl:2d}        ] VAL MAE={_mae(pred_corr[vm], av):.2f}")

    # 同时保留 per-hour 校正
    h_all = times.hour.values
    hour_bias = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum():
            hour_bias[h] = np.average(oof_resid[m])
    corr_h = np.array([hour_bias[hh] for hh in h_all])
    print(f"[+ per-hour only        ] VAL MAE={_mae(np.clip(base_full[vm]-corr_h[vm],0,None), av):.2f}")


if __name__ == "__main__":
    main()
