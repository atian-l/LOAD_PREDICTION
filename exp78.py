# -*- coding: utf-8 -*-
"""exp78 — ① 异构集成 raw 测试（无校正，隔离集成组成效果）。

在已存 v5 的 40 个 LGB 成员基础上，加入 CatBoost/XGBoost/ExtraTrees 成员
（direct+residual × 3 seeds），比较 raw val MAE（无 hour_bias/drift/threshold 校正）：
  - pure-LGB(40) median
  - LGB+CatBoost
  - LGB+CatBoost+XGBoost
  - LGB+CatBoost+XGBoost+ExtraTrees
  - 全异构 trimmed-mean(20%)
若异构 raw < pure-LGB raw，则异构集成固有更优，值得做全管线（含校正重估）。

合规：CB/XGB/ET 目标=actual(direct)/actual-pred_load(residual)，actual 仅作目标；
特征同 LGB（pred_load+weather+calendar），无 actual 输入。仅诊断。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from load_pred import config as C, data_loader as dl, features as F, train as T
from load_pred.model import EnsembleModel


def build_X():
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)
    X = F.build_features(times, pred_load, weather)
    return times, X, pred_load, actual


def main():
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    from sklearn.ensemble import ExtraTreesRegressor

    model = EnsembleModel.load(C.MODEL_BUNDLE)  # v5 LGB 40 成员 + mismatch_model
    times, X, pred_load, actual = build_X()
    X = model.mismatch_model.transform(X)
    feat_cols = model.feature_cols
    usable = T.usable_mask(times, pred_load, actual)
    vmask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    pl = pred_load.reindex(times).values.astype(float)
    y_dir = actual.reindex(times).values.astype(float)
    y_res = y_dir - pl
    Xtr = X[usable][feat_cols]
    Xv = X[vmask][feat_cols]
    pl_v = pred_load.reindex(times)[vmask].values.astype(float)
    a_v = actual.reindex(times)[vmask].values.astype(float)
    wtr = T._time_weights(times, usable, C.TRAIN_CONFIG["alpha_w"])
    print(f"train={usable.sum()} val={vmask.sum()} feats={len(feat_cols)}", flush=True)

    # ---- LGB 成员 val 原始预测（来自已存 v5）----
    lgb_preds = []  # list of (val_pred_array, is_residual)
    for booster, is_res in zip(model.members, model.member_residual):
        raw = booster.predict(Xv)
        lgb_preds.append((pl_v + raw if is_res else raw, is_res))
    print(f"LGB 成员: {len(lgb_preds)}", flush=True)

    # ---- 训练 CB/XGB/ET 成员（direct+residual × 3 seeds）----
    seeds = [42, 7, 123]
    extra_preds = []  # (val_pred, is_residual, tag)

    for is_res in [False, True]:
        y = (y_res if is_res else y_dir)[usable]
        for s in seeds:
            # CatBoost
            cb = CatBoostRegressor(iterations=80, depth=8, learning_rate=0.03,
                                   l2_leaf_reg=4.0, random_seed=s, verbose=False)
            cb.fit(Xtr, y, sample_weight=wtr)
            raw = cb.predict(Xv)
            extra_preds.append((pl_v + raw if is_res else raw, is_res, "CB"))
            # XGBoost
            xg = XGBRegressor(n_estimators=80, max_depth=8, learning_rate=0.03,
                              reg_lambda=4.0, subsample=0.8, colsample_bytree=0.8,
                              random_state=s, n_jobs=-1, verbosity=0)
            xg.fit(Xtr, y, sample_weight=wtr)
            raw = xg.predict(Xv)
            extra_preds.append((pl_v + raw if is_res else raw, is_res, "XGB"))
            # ExtraTrees
            et = ExtraTreesRegressor(n_estimators=200, max_depth=12, min_samples_leaf=200,
                                     random_state=s, n_jobs=-1)
            et.fit(Xtr, y, sample_weight=wtr)
            raw = et.predict(Xv)
            extra_preds.append((pl_v + raw if is_res else raw, is_res, "ET"))
    print(f"异构成员: {len(extra_preds)}", flush=True)

    def mae_arr(arrs):
        M = np.vstack(arrs)
        ens = np.median(M, axis=0)
        return float(np.abs(ens - a_v).mean())

    def trimmed_mae(arrs, trim=0.2):
        M = np.vstack(arrs)
        n = M.shape[0]
        k = int(np.floor(n * trim / 2))
        M_sorted = np.sort(M, axis=0)
        if k > 0 and (n - 2 * k) > 0:
            ens = np.mean(M_sorted[k:n - k], axis=0)
        else:
            ens = np.mean(M, axis=0)
        return float(np.abs(ens - a_v).mean())

    # pure-LGB
    mae_lgb = mae_arr([x[0] for x in lgb_preds])
    print(f"(pure-LGB 40 median)        raw val MAE={mae_lgb:.2f}", flush=True)

    cb = [x[0] for x in extra_preds if x[2] == "CB"]
    xg = [x[0] for x in extra_preds if x[2] == "XGB"]
    et = [x[0] for x in extra_preds if x[2] == "ET"]

    m_lgb_cb = mae_arr([x[0] for x in lgb_preds] + cb)
    print(f"(LGB+CatBoost median)       raw val MAE={m_lgb_cb:.2f}  Δ={m_lgb_cb-mae_lgb:+.2f}", flush=True)
    m_lgb_cb_xg = mae_arr([x[0] for x in lgb_preds] + cb + xg)
    print(f"(LGB+CB+XGB median)         raw val MAE={m_lgb_cb_xg:.2f}  Δ={m_lgb_cb_xg-mae_lgb:+.2f}", flush=True)
    m_all = mae_arr([x[0] for x in lgb_preds] + cb + xg + et)
    print(f"(LGB+CB+XGB+ET median)      raw val MAE={m_all:.2f}  Δ={m_all-mae_lgb:+.2f}", flush=True)
    m_trim = trimmed_mae([x[0] for x in lgb_preds] + cb + xg + et, trim=0.2)
    print(f"(全异构 trimmed-mean 20%)   raw val MAE={m_trim:.2f}  Δ={m_trim-mae_lgb:+.2f}", flush=True)
    m_trim_lgb = trimmed_mae([x[0] for x in lgb_preds], trim=0.2)
    print(f"(pure-LGB trimmed-mean 20%) raw val MAE={m_trim_lgb:.2f}  Δ={m_trim_lgb-mae_lgb:+.2f}", flush=True)


if __name__ == "__main__":
    main()
