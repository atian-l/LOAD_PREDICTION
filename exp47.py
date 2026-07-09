# -*- coding: utf-8 -*-
"""
exp47 — 在最优配置 (nl=255/mdl=200/l2=4/λ=1.0, BI=80) 上测试两类无泄露精修：
  (A) 每小时偏置的不同加权：3 折均值(基线)/仅冬折(最近)/仅春折(季节匹配)/时间近邻加权。
      背景：mismatch 特征已吸收大部分 per-hour 收益(从 -10 降到 -4)。漂移使 2025 折偏置
      不完全适配 2026 验证集；近邻加权(更看重 2026-01/02 冬折)可能更贴合近期漂移。
  (B) 显式 β×pl_wr 方向校正：oof_err ≈ β·pl_wr。低 BI 模型可能未充分利用 +0.29 方向信号；
      在 OOF 上估 β，叠加到预测。无泄露(β 仅来自 OOF)。
只算一次 OOF，多种校正廉价复用。3 种子。不写产物。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from load_pred import config as C, data_loader as dl, features as F


def build():
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)
    X = F.build_features(times, pred_load, weather)
    ts0 = pd.Timestamp(C.TRAIN_CONFIG["train_start"]); tr_end = pd.Timestamp(C.TRAIN_END)
    usable = ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values
    mm = F.MismatchModel().fit(X, usable)
    X = mm.transform(X)
    return times, X, pred_load, actual, usable


def time_weights(times, mask, alpha):
    t = times[mask]; tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        return np.ones(len(t))
    return (1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()).values.astype(float)


def train_members(times, X, pred_load, actual, tr_mask, bi, cfg):
    feat_cols = list(X.columns)
    y_dir = actual; y_res = actual - pred_load
    Xtr = X[tr_mask][feat_cols]
    wtr = time_weights(times, tr_mask, cfg["alpha_w"])
    base = dict(metric="mae", learning_rate=cfg["learning_rate"], num_leaves=cfg["num_leaves"],
                min_data_in_leaf=cfg["min_data_in_leaf"], lambda_l2=cfg["lambda_l2"],
                feature_fraction=cfg["feature_fraction"], bagging_fraction=cfg["bagging_fraction"],
                bagging_freq=cfg["bagging_freq"], verbose=-1, force_col_wise=True)
    members, flags = [], []
    for residual in cfg["residual_modes"]:
        ytr = (y_res if residual else y_dir)[tr_mask]
        dtr = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj in cfg["objectives"]:
            alphas = cfg["quantile_alphas"] if obj == "quantile" else [None]
            for qa in alphas:
                for s in cfg["seeds"]:
                    p = dict(base, objective=obj, seed=s)
                    if obj == "quantile":
                        p["alpha"] = qa
                    members.append(lgb.train(p, dtr, num_boost_round=int(bi)))
                    flags.append(residual)
    return members, flags


def ens_raw(X, pred_load, members, flags, lam):
    pl = pred_load.reindex(X.index).values.astype(float)
    mp = np.empty((len(members), len(X)), dtype=float)
    for i, (bst, is_res) in enumerate(zip(members, flags)):
        raw = bst.predict(X)
        mp[i] = pl + raw if is_res else raw
    ens = np.median(mp, axis=0)
    return pl + lam * (ens - pl)


def main():
    print("building ...", flush=True)
    times, X, pred_load, actual, usable = build()
    print(f"  n_feat={X.shape[1]}  usable={usable.sum()}", flush=True)
    cfg = dict(C.TRAIN_CONFIG)
    cfg["num_leaves"] = 255; cfg["min_data_in_leaf"] = 200; cfg["lambda_l2"] = 4.0
    cfg["shrinkage"] = 1.0; cfg["seeds"] = [42, 7, 123]
    bi = 80
    val_mask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
                & actual.notna()).values

    # 训练全模型 + 3 折 OOF
    print(f"training full ensemble (BI={bi}) + 3-fold OOF ...", flush=True)
    members, flags = train_members(times, X, pred_load, actual, usable, bi, cfg)
    pred_val_nb = ens_raw(X[val_mask], pred_load, members, flags, cfg["shrinkage"])
    mae_nb = float(np.mean(np.abs(pred_val_nb - actual[val_mask].values)))

    oof_pred = pd.Series(np.nan, index=times)
    fold_masks = []
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        fm, ff = train_members(times, X, pred_load, actual, ftr, bi, cfg)
        oof_pred[fva] = ens_raw(X[fva], pred_load, fm, ff, cfg["shrinkage"])
        fold_masks.append((te, fva))
    oof_mask = usable & oof_pred.notna().values
    oof_resid = (oof_pred - actual).values  # 全长度数组
    h_all = pd.DatetimeIndex(times).hour.values
    t_all = times.values.astype("datetime64[s]").astype("int64")
    pl_wr = X["pl_weather_residual"].values

    def hour_bias_from(weight_fn):
        hb = np.zeros(24)
        for h in range(24):
            m = oof_mask & (h_all == h)
            if m.sum():
                w = weight_fn(t_all[m])
                hb[h] = float(np.average(oof_resid[m], weights=w))
        return hb

    def apply_hb(pred, hb):
        hours = pd.DatetimeIndex(X[val_mask].index).hour.values.astype(int)
        return np.clip(pred - hb[hours], 0.0, None)

    print(f"  no-bias VAL MAE={mae_nb:.2f}", flush=True)
    print("== (A) per-hour 加权变体 ==", flush=True)
    # 基线 3 折均值
    hb_mean = hour_bias_from(lambda t: np.ones(len(t)))
    mae_mean = float(np.mean(np.abs(apply_hb(pred_val_nb, hb_mean) - actual[val_mask].values)))
    print(f"  3-fold 均值    : +per-hour={mae_mean:.2f}  (delta={mae_mean-mae_nb:+.2f})", flush=True)
    # 仅冬折 (最近)
    win_mask = fold_masks[-1][1]
    hb_win = hour_bias_from(lambda t: (win_mask[oof_mask][np.isin(t_all[oof_mask], t)]).astype(float) if False else np.ones(len(t)))
    # 冬折加权：用冬折 OOF 点
    hb_win = np.zeros(24)
    for h in range(24):
        m = oof_mask & win_mask & (h_all == h)
        if m.sum():
            hb_win[h] = float(np.average(oof_resid[m]))
    mae_win = float(np.mean(np.abs(apply_hb(pred_val_nb, hb_win) - actual[val_mask].values)))
    print(f"  仅冬折(最近)   : +per-hour={mae_win:.2f}  (delta={mae_win-mae_nb:+.2f})", flush=True)
    # 仅春折 (季节匹配)
    sp_mask = fold_masks[0][1]
    hb_sp = np.zeros(24)
    for h in range(24):
        m = oof_mask & sp_mask & (h_all == h)
        if m.sum():
            hb_sp[h] = float(np.average(oof_resid[m]))
    mae_sp = float(np.mean(np.abs(apply_hb(pred_val_nb, hb_sp) - actual[val_mask].values)))
    print(f"  仅春折(季节)   : +per-hour={mae_sp:.2f}  (delta={mae_sp-mae_nb:+.2f})", flush=True)
    # 近邻时间加权 (指数，最近的 OOF 点权重高)
    tmax = t_all[oof_mask].max()
    def recency_w(t):
        # 离 tmax 越近权重越高；半衰期 ~60 天
        days = (tmax - t) / 86400.0
        return np.exp(-np.log(2) * days / 60.0)
    hb_rec = hour_bias_from(recency_w)
    mae_rec = float(np.mean(np.abs(apply_hb(pred_val_nb, hb_rec) - actual[val_mask].values)))
    print(f"  近邻时间加权   : +per-hour={mae_rec:.2f}  (delta={mae_rec-mae_nb:+.2f})", flush=True)

    # 选最佳 per-hour
    results = {"mean": (mae_mean, hb_mean), "winter": (mae_win, hb_win),
               "spring": (mae_sp, hb_sp), "recency": (mae_rec, hb_rec)}
    best_ph = min(results, key=lambda k: results[k][0])
    best_mae, best_hb = results[best_ph]
    print(f"  -> 最佳 per-hour: {best_ph} ({best_mae:.2f})", flush=True)

    print("== (B) β×pl_wr 方向校正 (叠加最佳 per-hour) ==", flush=True)
    # β 在 OOF 上估计: oof_err = pred_oof_no_bias - actual，但 pred_oof 已含 per-hour? 否，oof_pred 是 no-bias。
    # 用 no-bias OOF 残差估 β
    m_oof = oof_mask
    pl_wr_oof = pl_wr[m_oof]; err_oof = oof_resid[m_oof]
    # 去掉 pl_wr 的 NaN
    good = np.isfinite(pl_wr_oof) & np.isfinite(err_oof)
    beta = float(np.dot(pl_wr_oof[good], err_oof[good]) / np.dot(pl_wr_oof[good], pl_wr_oof[good]))
    corr_val = beta * pl_wr[val_mask]
    pred_b = apply_hb(pred_val_nb, best_hb) + np.nan_to_num(corr_val)
    pred_b = np.clip(pred_b, 0.0, None)
    mae_b = float(np.mean(np.abs(pred_b - actual[val_mask].values)))
    print(f"  β={beta:.4f}  +per-hour+β·pl_wr={mae_b:.2f}  (vs +per-hour {best_mae:.2f}, delta={mae_b-best_mae:+.2f})", flush=True)

    # 也试 per-hour + β 仅在午间
    is_mid = (h_all >= 11) & (h_all <= 13)
    beta_mid_arr = np.zeros(24)
    for h in range(11, 14):
        m = oof_mask & (h_all == h) & np.isfinite(pl_wr)
        if m.sum():
            d = pl_wr[m]; beta_mid_arr[h] = float(np.dot(d, oof_resid[m]) / np.dot(d, d))
    hours_val = pd.DatetimeIndex(X[val_mask].index).hour.values.astype(int)
    corr_mid = np.where((hours_val >= 11) & (hours_val <= 13), beta_mid_arr[hours_val] * pl_wr[val_mask], 0.0)
    pred_mid = np.clip(apply_hb(pred_val_nb, best_hb) + np.nan_to_num(corr_mid), 0.0, None)
    mae_mid = float(np.mean(np.abs(pred_mid - actual[val_mask].values)))
    print(f"  +per-hour+β_midday·pl_wr={mae_mid:.2f}  (delta={mae_mid-best_mae:+.2f})", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
