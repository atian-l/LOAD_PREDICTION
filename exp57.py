# -*- coding: utf-8 -*-
"""exp57: LightGBM + CatBoost 混合（测试误差正交性）。

此前否定了混合但未实测误差相关性。若 LGB 与 CB 误差低相关，加权混合可能优于单模型。
流程：
  1. LGB 3-seed raw val 预测
  2. CB  3-seed raw val 预测
  3. 误差相关性 + 各权重混合 val MAE
  4. 若最优 raw 混合 < 1528.74(LGB raw)，再做 3-fold OOF per-hour+drift 校正看能否破 1512.63
合规：仅训练期 OOF，不写产物。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from load_pred import config as C, features as F, model as M, train as T


def cb_member(Xtr, ytr, wtr, residual, obj, seed, alpha=None):
    p = dict(iterations=80, learning_rate=0.03, depth=8, min_data_in_leaf=200,
             l2_leaf_reg=4.0, rsm=0.80, subsample=0.80, bootstrap_type="Bernoulli",
             random_seed=seed, verbose=False, allow_writing_files=False)
    p["loss_function"] = f"Quantile:alpha={alpha}" if obj == "quantile" else "RMSE"
    m = CatBoostRegressor(**p); m.fit(Pool(Xtr, label=ytr, weight=wtr)); return m


def cb_ensemble(times, X, pred_load, actual, mask, cfg, seeds):
    feat = list(X.columns); y_dir = actual; y_res = actual - pred_load
    Xtr = X[mask][feat]; wtr = T._time_weights(times, mask, cfg["alpha_w"])
    members, flags = [], []
    for residual in cfg["residual_modes"]:
        ytr = (y_res if residual else y_dir)[mask]
        for obj in cfg["objectives"]:
            for qa in (cfg["quantile_alphas"] if obj == "quantile" else [None]):
                for s in seeds:
                    members.append(cb_member(Xtr, ytr.values, wtr, residual, obj, s, qa))
                    flags.append(residual)
    return members, flags


def cb_predict(members, X, pl_T, flags):
    preds = [(np.asarray(pl_T) + np.asarray(m.predict(X))) if f else np.asarray(m.predict(X))
             for m, f in zip(members, flags)]
    return np.median(np.vstack(preds), axis=0)


def main():
    times, X, pred_load, actual = T.build_dataset()
    usable = T.usable_mask(times, pred_load, actual)
    mm = F.MismatchModel().fit(X, usable); X = mm.transform(X)
    val = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END)) & actual.notna()).values
    cfg = dict(C.TRAIN_CONFIG); cfg["best_it_fixed"] = 80
    seeds = [42, 7, 123]
    act_v = actual[val].values

    print("LGB 3-seed full ensemble ...", flush=True)
    lgb_m = T.train_ensemble(times, X, pred_load, actual, usable, cfg, 80)
    lgb_v = lgb_m.predict_load(X[val], pred_load[val])
    print(f"  LGB raw val MAE = {np.abs(lgb_v - act_v).mean():.2f}", flush=True)

    print("CB 3-seed full ensemble ...", flush=True)
    cbs, cflags = cb_ensemble(times, X, pred_load, actual, usable, cfg, seeds)
    cb_v = cb_predict(cbs, X[val], pred_load[val], cflags)
    print(f"  CB  raw val MAE = {np.abs(cb_v - act_v).mean():.2f}", flush=True)

    el = lgb_v - act_v; ec = cb_v - act_v
    corr = np.corrcoef(el, ec)[0, 1]
    print(f"  误差相关性 corr(LGB,CB) = {corr:.4f}", flush=True)
    print(flush=True)

    best_w, best_mae = 1.0, np.abs(lgb_v - act_v).mean()
    for w in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        b = w * lgb_v + (1 - w) * cb_v
        mae = np.abs(b - act_v).mean()
        tag = "  <- best" if mae < best_mae else ""
        print(f"  blend w(LGB)={w:.2f}: raw val MAE = {mae:.2f}{tag}", flush=True)
        if mae < best_mae:
            best_w, best_mae = w, mae
    print(f"\n  best raw blend: w(LGB)={best_w:.2f} -> {best_mae:.2f}  (LGB raw={np.abs(lgb_v-act_v).mean():.2f})", flush=True)

    # 若 raw 混合优于 LGB raw，做 OOF per-hour+drift 校正
    if best_mae < np.abs(lgb_v - act_v).mean() - 0.5:
        print("\nraw blend 优于 LGB raw -> 做 3-fold OOF 校正 ...", flush=True)
        h_all = pd.DatetimeIndex(times).hour.values
        plwr_all = X["pl_weather_residual"].values.astype(float)
        oof = pd.Series(np.nan, index=times)
        for te, vs, ve in cfg["best_it_folds"]:
            te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
            ftr = usable & np.asarray(times <= te)
            fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
            if fva.sum() == 0: continue
            lm = T.train_ensemble(times, X, pred_load, actual, ftr, cfg, 80)
            lv = lm.predict_load(X[fva], pred_load[fva])
            cm, cf = cb_ensemble(times, X, pred_load, actual, ftr, cfg, seeds)
            cv = cb_predict(cm, X[fva], pred_load[fva], cf)
            oof[fva] = best_w * lv + (1 - best_w) * cv
        oof_mask = usable & oof.notna().values
        resid = (oof - actual).values
        hb = np.zeros(24)
        for h in range(24):
            m = oof_mask & (h_all == h)
            if m.sum(): hb[h] = resid[m].mean()
        beta = np.zeros(24)
        for h in (11, 12, 13, 14):
            m = oof_mask & (h_all == h)
            f = plwr_all[m]; e = resid[m]; g = np.isfinite(f) & np.isfinite(e)
            d = float(np.dot(f[g], f[g]))
            if d > 0: beta[h] = float(np.dot(f[g], e[g]) / d)
        hours_v = pd.DatetimeIndex(times[val]).hour.values.astype(int)
        plwr_v = plwr_all[val]
        raw_blend_v = best_w * lgb_v + (1 - best_w) * cb_v
        corr_v = raw_blend_v - hb[hours_v] + beta[hours_v] * plwr_v
        mae = np.abs(corr_v - act_v).mean()
        print(f"  blend + OOF per-hour + drift val MAE = {mae:.2f}  (LGB 生产=1512.63)", flush=True)
    else:
        print("\nraw blend 不优于 LGB raw -> 混合无收益，跳过校正。", flush=True)


if __name__ == "__main__":
    main()
