# -*- coding: utf-8 -*-
"""exp71: 结构性杠杆测试 —— 午间专用模型 + 更多成员 + BI 调参。

午间(9-15)占 MAE ~55%。测试：
  1. 午间专用模型（仅午间点训练）vs 全天模型的午间 MAE。
  2. 80 成员 vs 40 成员。
  3. BI=120 vs 80。
仅诊断。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
import lightgbm as lgb
from load_pred import config as C, features as F, train as T


def train_cfg(times, X, pred_load, actual, usable, cfg, best_it, seeds, midday_only=False):
    """训练集成，可选仅午间点。返回 EnsembleModel。"""
    feat_cols = list(X.columns)
    y_dir = actual; y_res = actual - pred_load
    if midday_only:
        h = pd.DatetimeIndex(times).hour.values
        mask = usable & np.isin(h, list(range(9,16)))
    else:
        mask = usable
    Xtr = X[mask][feat_cols]
    wtr = T._time_weights(times, mask, cfg["alpha_w"])
    base = dict(metric=["mae","rmse"], learning_rate=cfg["learning_rate"], num_leaves=cfg["num_leaves"],
                min_data_in_leaf=cfg["min_data_in_leaf"], lambda_l2=cfg["lambda_l2"],
                feature_fraction=cfg["feature_fraction"], bagging_fraction=cfg["bagging_fraction"],
                bagging_freq=cfg["bagging_freq"], verbose=-1, force_col_wise=True)
    from load_pred.model import EnsembleModel
    model = EnsembleModel(feature_cols=feat_cols, shrinkage=cfg["shrinkage"], train_meta={})
    n=0
    for residual in cfg["residual_modes"]:
        ytr = (y_res if residual else y_dir)[mask]
        dtr = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj in cfg["objectives"]:
            alphas = cfg["quantile_alphas"] if obj=="quantile" else [None]
            for qa in alphas:
                for s in seeds:
                    p = dict(base, objective=obj, seed=s)
                    if obj=="quantile": p["alpha"]=qa
                    bst = lgb.train(p, dtr, num_boost_round=int(best_it))
                    model.add_member(bst, is_residual=residual); n+=1
    print(f"      (成员数={n}{' 午间专用' if midday_only else ''})", flush=True)
    return model


def main():
    times, X, pred_load, actual = T.build_dataset()
    usable = T.usable_mask(times, pred_load, actual)
    mm = F.MismatchModel().fit(X, usable); X = mm.transform(X)
    val = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END)) & actual.notna()).values
    hours_v = pd.DatetimeIndex(times[val]).hour.values.astype(int)
    midday_v = np.isin(hours_v, list(range(9,16)))
    cfg = dict(C.TRAIN_CONFIG)

    # 基线（生产）
    cfg["best_it_fixed"]=80
    m0 = T.train_ensemble(times, X, pred_load, actual, usable, cfg, 80)
    m0.hour_bias, m0.drift_corr, m0.threshold_corr = T.compute_hour_bias(times, X, pred_load, actual, usable, cfg, 80)
    p0 = m0.predict_load(X[val], pred_load[val]); a = actual[val].values
    print(f"基线 全天模型: 全MAE={np.abs(p0-a).mean():.2f} 午间MAE={np.abs(p0[midday_v]-a[midday_v]).mean():.2f} 非午间MAE={np.abs(p0[~midday_v]-a[~midday_v]).mean():.2f}", flush=True)

    # 1) 午间专用模型（仅午间训练），用其预测午间，全天模型预测非午间
    print("\n[1] 午间专用模型...", flush=True)
    mm_model = train_cfg(times, X, pred_load, actual, usable, cfg, 80, cfg["seeds"], midday_only=True)
    # 午间专用模型无 hour_bias/drift/threshold（简化），直接预测午间
    p_mid = mm_model.predict_load(X[val], pred_load[val])
    blend = p0.copy(); blend[midday_v] = p_mid[midday_v]
    print(f"   午间专用(午间)+全天(非午间): 全MAE={np.abs(blend-a).mean():.2f} 午间MAE={np.abs(blend[midday_v]-a[midday_v]).mean():.2f}  Δ全={np.abs(blend-a).mean()-np.abs(p0-a).mean():+.2f}", flush=True)
    # 午间专用 + 午间 hour_bias（用全天模型的 hour_bias 简单校正）
    hb = m0.hour_bias
    p_mid_hb = p_mid - hb[hours_v]
    blend2 = p0.copy(); blend2[midday_v] = p_mid_hb[midday_v]
    print(f"   午间专用+hour_bias: 全MAE={np.abs(blend2-a).mean():.2f} 午间MAE={np.abs(blend2[midday_v]-a[midday_v]).mean():.2f}  Δ全={np.abs(blend2-a).mean()-np.abs(p0-a).mean():+.2f}", flush=True)

    # 2) 80 成员（更多种子）
    print("\n[2] 80 成员（10 种子）...", flush=True)
    seeds80 = [42,7,123,2024,99, 11,22,33,44,55]
    cfg2 = dict(cfg); cfg2["seeds"]=seeds80
    m80 = T.train_ensemble(times, X, pred_load, actual, usable, cfg2, 80)
    m80.hour_bias, m80.drift_corr, m80.threshold_corr = T.compute_hour_bias(times, X, pred_load, actual, usable, cfg2, 80)
    p80 = m80.predict_load(X[val], pred_load[val])
    print(f"   80成员: 全MAE={np.abs(p80-a).mean():.2f}  Δ={np.abs(p80-a).mean()-np.abs(p0-a).mean():+.2f}", flush=True)

    # 3) BI=120
    print("\n[3] BI=120...", flush=True)
    m120 = T.train_ensemble(times, X, pred_load, actual, usable, cfg, 120)
    m120.hour_bias, m120.drift_corr, m120.threshold_corr = T.compute_hour_bias(times, X, pred_load, actual, usable, cfg, 120)
    p120 = m120.predict_load(X[val], pred_load[val])
    print(f"   BI=120: 全MAE={np.abs(p120-a).mean():.2f}  Δ={np.abs(p120-a).mean()-np.abs(p0-a).mean():+.2f}", flush=True)


if __name__ == "__main__":
    main()
