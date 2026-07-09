# -*- coding: utf-8 -*-
"""exp54: CatBoost 作为替代基模型（用户建议）。

与 LightGBM 同配置（lr=.03, depth=8≈256叶, mdl=200, l2=4, rsm=.8, subsample=.8,
BI=80），{regression, quantile(.5)} × {direct, residual} × 3 seeds = 12 成员，中位数聚合。
计算 raw + OOF per-hour + drift_corr 的 val MAE，对比 LightGBM 生产 1512.63。
合规：仅训练期 OOF，不写产物。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from load_pred import config as C, data_loader as dl, features as F, train as T


def make_member(Xtr, ytr, wtr, residual, obj, seed, alpha=None):
    p = dict(
        iterations=80, learning_rate=0.03, depth=8,
        min_data_in_leaf=200, l2_leaf_reg=4.0, rsm=0.80,
        subsample=0.80, bootstrap_type="Bernoulli",
        random_seed=seed, verbose=False, allow_writing_files=False,
    )
    if obj == "quantile":
        p["loss_function"] = f"Quantile:alpha={alpha}"
    else:
        p["loss_function"] = "RMSE"
    m = CatBoostRegressor(**p)
    m.fit(Pool(Xtr, label=ytr, weight=wtr))
    return m


def ensemble_predict(members, X, pred_load_T, is_residual_flags):
    preds = []
    for m, isr in zip(members, is_residual_flags):
        r = np.asarray(m.predict(X), dtype=float)
        if isr:
            preds.append(np.asarray(pred_load_T) + r)
        else:
            preds.append(r)
    return np.median(np.vstack(preds), axis=0)


def train_ensemble_cb(times, X, pred_load, actual, mask, cfg, best_it, seeds):
    feat_cols = list(X.columns)
    y_dir = actual
    y_res = actual - pred_load
    Xtr = X[mask][feat_cols]
    wtr = T._time_weights(times, mask, cfg["alpha_w"])
    members, flags = [], []
    for residual in cfg["residual_modes"]:
        ytr = (y_res if residual else y_dir)[mask]
        for obj in cfg["objectives"]:
            alphas = cfg["quantile_alphas"] if obj == "quantile" else [None]
            for qa in alphas:
                for s in seeds:
                    members.append(make_member(Xtr, ytr.values, wtr, residual, obj, s, qa))
                    flags.append(residual)
    return members, flags


def main():
    times, X, pred_load, actual = T.build_dataset()
    usable = T.usable_mask(times, pred_load, actual)
    mm = F.MismatchModel().fit(X, usable)
    X = mm.transform(X)
    val = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
           & actual.notna()).values
    cfg = dict(C.TRAIN_CONFIG)
    seeds = [42, 7, 123]

    # 3-fold OOF
    print("CatBoost 3-fold OOF ...", flush=True)
    oof_pred = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        members, flags = train_ensemble_cb(times, X, pred_load, actual, ftr, cfg, 80, seeds)
        oof_pred[fva] = ensemble_predict(members, X[fva], pred_load[fva], flags)
        print(f"  fold {te.date()} done", flush=True)
    oof_mask = usable & oof_pred.notna().values
    oof_resid = (oof_pred - actual).values
    h_all = pd.DatetimeIndex(times).hour.values
    plwr_all = X["pl_weather_residual"].values.astype(float)

    # full ensemble
    print("CatBoost full ensemble ...", flush=True)
    members, flags = train_ensemble_cb(times, X, pred_load, actual, usable, cfg, 80, seeds)
    raw_v = ensemble_predict(members, X[val], pred_load[val], flags)
    actual_v = actual[val].values
    hours_v = pd.DatetimeIndex(times[val]).hour.values.astype(int)
    plwr_v = plwr_all[val]
    mae_raw = np.abs(raw_v - actual_v).mean()
    print(f"CatBoost raw val MAE = {mae_raw:.2f}  (LightGBM raw=1528.74)", flush=True)

    # OOF hour_bias (mean) + drift β (11-14)
    hb = np.zeros(24)
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum():
            hb[h] = float(np.mean(oof_resid[m]))
    beta = np.zeros(24)
    for h in (11, 12, 13, 14):
        m = oof_mask & (h_all == h)
        f = plwr_all[m]; e = oof_resid[m]
        good = np.isfinite(f) & np.isfinite(e)
        d = float(np.dot(f[good], f[good]))
        if d > 0:
            beta[h] = float(np.dot(f[good], e[good]) / d)
    pred = raw_v - hb[hours_v] + beta[hours_v] * plwr_v
    mae = np.abs(pred - actual_v).mean()
    print(f"CatBoost + OOF per-hour + drift val MAE = {mae:.2f}  (LightGBM prod=1512.63)", flush=True)


if __name__ == "__main__":
    main()
