# -*- coding: utf-8 -*-
"""
Walk-forward 交叉验证实验（不写输出文件）。

用 3 个跨季节 fold（春/秋/冬，均在训练期内）做早停与配置选择，
避免单段 Jan 内部验证导致的早停偏置。最终在全部训练数据上以平均 best_iter
重训，报告官方验证集 MAE/R²。
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


FOLDS = [
    ("spring25", "2023-02-01", "2025-02-28", "2025-03-01", "2025-05-31"),
    ("fall25",   "2023-02-01", "2025-08-31", "2025-09-01", "2025-11-30"),
    ("winter26", "2023-02-01", "2025-11-30", "2025-12-01", "2026-01-31"),
]


def _metrics(pred, act):
    err = pred - act
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    ss_res = np.sum(err**2); ss_tot = np.sum((act - act.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return mae, rmse, r2, np.mean(err)


def run_config(times, X, pred_load, actual, cfg, residual, fixed_iters=None):
    """返回每个 fold 的 best_iter/val_mae，以及最终官方验证集指标。"""
    feat_cols = list(X.columns)
    y = (actual - pred_load) if residual else actual
    tr_end = pd.Timestamp(C.TRAIN_END)
    v_start = pd.Timestamp(C.VAL_START); v_end = pd.Timestamp(C.VAL_END)
    usable_all = (times >= pd.Timestamp(C.TRAIN_CONFIG["train_start"])) & (times <= tr_end) \
                 & pred_load.notna() & actual.notna()

    fold_results = []
    for name, ts, te, vs, ve in FOLDS:
        ts, te, vs, ve = pd.Timestamp(ts), pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        tr_mask = usable_all & (times >= ts) & (times <= te)
        va_mask = usable_all & (times >= vs) & (times <= ve)
        dtr = lgb.Dataset(X[tr_mask][feat_cols], label=y[tr_mask].values)
        dva = lgb.Dataset(X[va_mask][feat_cols], label=y[va_mask].values, reference=dtr)
        p = dict(cfg); n_iter = p.pop("num_iterations"); es = p.pop("early_stopping_rounds")
        ev = {}
        if fixed_iters is None:
            bst = lgb.train(p, dtr, num_boost_round=n_iter, valid_sets=[dva], valid_names=["va"],
                            callbacks=[lgb.early_stopping(es, verbose=False, first_metric_only=True),
                                       lgb.record_evaluation(ev)])
            best_it = bst.best_iteration
            bst = bst  # 早停模型
        else:
            bst = lgb.train(p, dtr, num_boost_round=int(fixed_iters))
            best_it = int(fixed_iters)
        raw_va = bst.predict(X[va_mask][feat_cols])
        pred_va = pred_load[va_mask].values + raw_va if residual else raw_va
        mae, rmse, r2, bias = _metrics(np.clip(pred_va, 0, None), actual[va_mask].values)
        fold_results.append((name, best_it, mae, r2, len(tr_mask), len(va_mask)))

    # 平均 best_iter -> 最终全量重训
    avg_it = int(np.mean([r[1] for r in fold_results if r[1] and r[1] > 0]))
    p2 = dict(cfg); p2.pop("num_iterations"); p2.pop("early_stopping_rounds")
    dtr_full = lgb.Dataset(X[usable_all][feat_cols], label=y[usable_all].values)
    bst_full = lgb.train(p2, dtr_full, num_boost_round=avg_it)
    raw_all = bst_full.predict(X[feat_cols])
    pred_all = pred_load.values + raw_all if residual else raw_all
    pred_all = np.clip(pred_all, 0, None)
    vmask = ((times >= v_start) & (times <= v_end) & actual.notna()).values
    mae, rmse, r2, bias = _metrics(pred_all[vmask], actual.values[vmask])
    imp = pd.Series(bst_full.feature_importance(importance_type="gain"), index=feat_cols)\
        .sort_values(ascending=False)
    return {"folds": fold_results, "avg_it": avg_it, "val_mae": mae, "val_r2": r2,
            "val_bias": bias, "val_rmse": rmse, "imp": imp}


def main():
    print("building features ...")
    times, X, pred_load, actual = build()
    print(f"feats={X.shape[1]}")

    base = dict(objective="regression", metric=["mae", "rmse"], learning_rate=0.02,
               feature_fraction=0.80, bagging_fraction=0.80, bagging_freq=1,
               verbose=-1, seed=42, force_col_wise=True,
               num_iterations=8000, early_stopping_rounds=300)

    configs = [
        ("direct_med", dict(num_leaves=127, min_data_in_leaf=300, lambda_l2=1.0), False),
        ("resid_med",  dict(num_leaves=127, min_data_in_leaf=300, lambda_l2=1.0), True),
        ("resid_loose",dict(num_leaves=255, min_data_in_leaf=120, lambda_l2=0.5), True),
        ("direct_med_lr01", dict(num_leaves=127, min_data_in_leaf=300, lambda_l2=1.0, learning_rate=0.01), False),
        ("resid_med_lr01",  dict(num_leaves=127, min_data_in_leaf=300, lambda_l2=1.0, learning_rate=0.01), True),
    ]
    best = None
    for name, ov, residual in configs:
        cfg = dict(base); cfg.update(ov)
        r = run_config(times, X, pred_load, actual, cfg, residual)
        print(f"\n[{name}] residual={residual}  avg_it={r['avg_it']}")
        for fn, bi, fm, fr, ntr, nva in r["folds"]:
            print(f"  fold {fn:9s} best_it={bi:5d}  fold_mae={fm:.1f}  fold_r2={fr:.4f}  (tr={ntr},va={nva})")
        print(f"  >>> OFFICIAL VAL  MAE={r['val_mae']:.2f}  RMSE={r['val_rmse']:.2f}  R2={r['val_r2']:.4f}  Bias={r['val_bias']:.1f}")
        print("  top10 features: " + ", ".join(f"{k}({v:.0f})" for k, v in r["imp"].head(10).items()))
        if best is None or r["val_mae"] < best["val_mae"]:
            best = {"name": name, **r}
    print(f"\n==== BEST: {best['name']}  MAE={best['val_mae']:.2f}  R2={best['val_r2']:.4f}  avg_it={best['avg_it']} ====")


if __name__ == "__main__":
    main()
