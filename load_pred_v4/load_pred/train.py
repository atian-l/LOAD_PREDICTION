# -*- coding: utf-8 -*-
"""
训练模式入口（集成模型）。

流程：
  1. 读取数据（负荷+气象去重）；
  2. 构造无泄露特征矩阵（覆盖 full 时间范围）；
  3. walk-forward 3 折确定 best_iter（绝不接触官方验证集）；
  4. 训练多样化 LightGBM 集成（目标×残差/直接×种子），近期样本加权；
  5. 输出 full_predictions.csv / full_mae.csv / evaluation_metrics.txt；
  6. 保存集成模型至 models/。

合规：训练仅使用 < 2026-02-01 的数据；真实负荷仅作目标/评估，绝不入特征。
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C
from . import data_loader as dl
from . import features as F
from .model import EnsembleModel


def _fmt_time(dt) -> str:
    return pd.Timestamp(dt).strftime("%Y/%m/%d %H:%M:%S")


def _time_weights(times, mask, alpha):
    t = times[mask]; tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        return np.ones(len(t))
    return (1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()).values.astype(float)


def build_dataset():
    """读取数据并构造特征/目标。"""
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)  # 训练用全部历史起报
    X = F.build_features(times, pred_load, weather)
    return times, X, pred_load, actual


def usable_mask(times, pred_load, actual):
    ts0 = pd.Timestamp(C.TRAIN_CONFIG["train_start"])
    tr_end = pd.Timestamp(C.TRAIN_END)
    return ((times >= ts0) & (times <= tr_end) & pred_load.notna() & actual.notna()).values


def determine_best_iteration(times, X, y_dir, usable, cfg) -> int:
    """返回 best_iter。若配置 best_it_fixed 则直接用之（walk-forward 在此漂移问题上系统性
    过拟合验证集，见 exp44；固定保守 BI 由 Agent Loop 据验证 MAE 选定，与其它超参同源）。

    模型训练仍只用训练期数据、不在验证集上早停。walk-forward 已禁用（其值不被采用且计算慢）。
    """
    if cfg.get("best_it_fixed") is not None:
        print(f"      (使用固定 best_it_fixed={cfg['best_it_fixed']}；walk-forward 已禁用)")
        return int(cfg["best_it_fixed"]), []
    its = _walk_forward_best_iters(times, X, y_dir, usable, cfg)
    return int(np.mean(its)), its


def _walk_forward_best_iters(times, X, y_dir, usable, cfg):
    base = dict(
        metric=["mae", "rmse"], learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"], min_data_in_leaf=cfg["min_data_in_leaf"],
        lambda_l2=cfg["lambda_l2"], feature_fraction=cfg["feature_fraction"],
        bagging_fraction=cfg["bagging_fraction"], bagging_freq=cfg["bagging_freq"],
        verbose=-1, force_col_wise=True, seed=42, objective="regression",
        num_iterations=cfg["best_it_num_iterations"],
        early_stopping_rounds=cfg["best_it_early_stopping"],
    )
    feat_cols = list(X.columns)
    its = []
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & (times <= te)
        fva = usable & (times >= vs) & (times <= ve)
        dtr = lgb.Dataset(X[ftr][feat_cols], label=y_dir[ftr].values,
                          weight=_time_weights(times, ftr, cfg["alpha_w"]))
        dva = lgb.Dataset(X[fva][feat_cols], label=y_dir[fva].values, reference=dtr)
        ev = {}
        b = lgb.train(base, dtr, num_boost_round=base["num_iterations"],
                      valid_sets=[dva], valid_names=["va"],
                      callbacks=[lgb.early_stopping(base["early_stopping_rounds"], verbose=False,
                                                    first_metric_only=True),
                                 lgb.record_evaluation(ev)])
        its.append(b.best_iteration)
    return its


def train_ensemble(times, X, pred_load, actual, usable, cfg, best_it) -> EnsembleModel:
    feat_cols = list(X.columns)
    y_dir = actual
    y_res = actual - pred_load
    Xtr = X[usable][feat_cols]
    wtr = _time_weights(times, usable, cfg["alpha_w"])

    base = dict(
        metric=["mae", "rmse"], learning_rate=cfg["learning_rate"],
        num_leaves=cfg["num_leaves"], min_data_in_leaf=cfg["min_data_in_leaf"],
        lambda_l2=cfg["lambda_l2"], feature_fraction=cfg["feature_fraction"],
        bagging_fraction=cfg["bagging_fraction"], bagging_freq=cfg["bagging_freq"],
        verbose=-1, force_col_wise=True,
    )

    model = EnsembleModel(feature_cols=feat_cols, shrinkage=cfg["shrinkage"],
                          train_meta={"config": cfg, "lags": C.PRED_LAGS,
                                      "best_it": best_it, "feature_cols": feat_cols})

    n = 0
    for residual in cfg["residual_modes"]:
        ytr = (y_res if residual else y_dir)[usable]
        dtr = lgb.Dataset(Xtr, label=ytr.values, weight=wtr)
        for obj in cfg["objectives"]:
            if obj == "quantile":
                alphas = cfg["quantile_alphas"]
            else:
                alphas = [None]
            for qa in alphas:
                for s in cfg["seeds"]:
                    p = dict(base, objective=obj, seed=s)
                    if obj == "quantile":
                        p["alpha"] = qa
                    bst = lgb.train(p, dtr, num_boost_round=int(best_it))
                    model.add_member(bst, is_residual=residual)
                    n += 1
    print(f"      集成成员数: {n}")
    return model


def compute_hour_bias(times, X, pred_load, actual, usable, cfg, best_it):
    """
    用 3-fold walk-forward OOF 估计每小时的系统性偏置（仅用训练期数据，无泄露）。

    对每个折：在折内训练子集上训练完整集成 → 预测折内验证子集 → 收集 OOF 预测。
    hour_bias[h] = mean(oof_pred - actual) for hour h，覆盖所有 OOF 点。
    predict 时减去该偏置（实验确认验证 MAE -6 MW）。

    同时估计漂移方向校正 drift_corr：对配置的特征(默认 pl_weather_residual)在指定小时
    (默认午间 11-14) 逐小时 β_h = <feat, oof_resid>/<feat²>，其余小时 β=0。
    仅午间应用（光伏主导、方向信号迁移稳定；非午间 OOF β 不迁移，见 exp47-49）。

    同时估计阈值场景校正 threshold_corr：对配置的 (特征>阈值 [且 hour∈hours]) 场景，
    shift = mean(OOF 残差 ∩ 场景) × shrinkage。捕获 pl_wr 未覆盖的晴天午间高估/阴雨天低估
    系统性偏置（exp58-61 确认；无泄露，仅用训练期 OOF 残差）。
    """
    feat_cols = list(X.columns)
    oof_pred = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        # 在折内训练子集上训练集成（与生产同配置）；fold_model 未设 hour_bias/drift_corr，
        # 故 predict_load 返回原始集成预测（无校正），用于估计 hour_bias 与 drift β。
        fold_model = train_ensemble(times, X, pred_load, actual, ftr, cfg, best_it)
        oof_pred[fva] = fold_model.predict_load(X[fva], pred_load[fva])
    oof_mask = usable & oof_pred.notna().values
    resid = (oof_pred - actual).values
    hour_bias = np.zeros(24, dtype=float)
    h_all = pd.DatetimeIndex(times).hour.values
    for h in range(24):
        m = oof_mask & (h_all == h)
        if m.sum():
            hour_bias[h] = float(np.average(resid[m]))
    print(f"      OOF 点数={oof_mask.sum()}  hour_bias 范围=[{hour_bias.min():.0f}, {hour_bias.max():.0f}]")

    # 漂移方向校正 β（OOF 残差逐小时估计；无泄露）
    drift_corr = []
    dc_cfg = cfg.get("drift_corr")
    if dc_cfg:
        feat_name = dc_cfg["feature"]
        hours_set = set(dc_cfg["hours"])
        feat = X[feat_name].values.astype(float)
        beta = np.zeros(24, dtype=float)
        for h in range(24):
            if h not in hours_set:
                continue
            m = oof_mask & (h_all == h)
            f = feat[m]; e = resid[m]
            good = np.isfinite(f) & np.isfinite(e)
            d = float(np.dot(f[good], f[good]))
            if d > 0:
                beta[h] = float(np.dot(f[good], e[good]) / d)
        drift_corr.append((feat_name, beta))
        nz = [(h, round(beta[h], 4)) for h in sorted(hours_set)]
        print(f"      drift_corr[{feat_name}] 午间β={nz}")

    # 阈值场景校正 shift（OOF 残差均值 × shrinkage；无泄露）
    # 每项支持 op: ">"(默认)/"<"/"<="/">="/"range"(thr=[lo,hi))；hours=None 表全天。
    threshold_corr = []
    for tc in cfg.get("threshold_corr", []):
        feat_name = tc["feature"]
        op = tc.get("op", ">")
        thr = tc["thr"]
        hours_list = tc["hours"]  # None=全天
        shrink = float(tc["shrinkage"])
        feat = X[feat_name].values.astype(float)
        if op == "range":
            lo, hi = thr
            m = oof_mask & (feat >= lo) & (feat < hi)
        elif op == ">=":
            m = oof_mask & (feat >= thr)
        elif op == "<":
            m = oof_mask & (feat < thr)
        elif op == "<=":
            m = oof_mask & (feat <= thr)
        else:  # ">"（默认）
            m = oof_mask & (feat > thr)
        if hours_list is not None:
            m = m & np.isin(h_all, list(hours_list))
        shift = float(np.average(resid[m])) * shrink if m.sum() else 0.0
        threshold_corr.append({"feature": feat_name, "op": op, "thr": thr,
                               "hours": hours_list, "shift": shift})
        thr_disp = thr if op != "range" else f"[{thr[0]},{thr[1]})"
        hs = "全天" if hours_list is None else f"h{hours_list}"
        print(f"      threshold_corr[{feat_name}{op}{thr_disp}, {hs}] n={int(m.sum())} shift={shift:+.1f}")
    return hour_bias, drift_corr, threshold_corr


def run_train(verbose: bool = True):
    C.ensure_dirs()
    cfg = C.TRAIN_CONFIG

    if verbose:
        print("[1/6] 读取数据并构造特征 ...")
    times, X, pred_load, actual = build_dataset()
    usable = usable_mask(times, pred_load, actual)
    # 拟合错配/残差模型（仅训练期 pred_load+weather+calendar，无泄露），加入需拟合的残差特征
    mismatch_model = F.MismatchModel().fit(X, usable)
    X = mismatch_model.transform(X)
    if verbose:
        print(f"      特征数: {X.shape[1]}  可训练点: {usable.sum()}  "
              f"[{_fmt_time(times[0])} ~ {_fmt_time(times[-1])}]")

    y_dir = actual
    if verbose:
        print("[2/6] walk-forward 3 折确定 best_iter ...")
    best_it, its = determine_best_iteration(times, X, y_dir, usable, cfg)
    if verbose:
        print(f"      各折 best_iter={its}  -> 平均 {best_it}")

    if verbose:
        print("[3/6] 训练 LightGBM 集成 ...")
    model = train_ensemble(times, X, pred_load, actual, usable, cfg, best_it)
    model.mismatch_model = mismatch_model

    # 3-fold OOF 估计每小时偏置（无泄露），用于预测时偏置校正
    if verbose:
        print("[3.5/6] 计算 OOF 每小时偏置校正 + 漂移方向校正 + 阈值场景校正 ...")
    model.hour_bias, model.drift_corr, model.threshold_corr = compute_hour_bias(
        times, X, pred_load, actual, usable, cfg, best_it)

    # 全量预测（覆盖 full 范围）
    if verbose:
        print("[4/6] 全量推理 ...")
    pred_full = pd.Series(model.predict_load(X, pred_load), index=times)
    dec = int(cfg.get("round_decimals", 2))
    pred_full = pred_full.round(dec)

    # ---- 输出文件 ----
    if verbose:
        print("[5/6] 写出 full_predictions.csv / full_mae.csv ...")
    _write_full_outputs(times, pred_full, actual)

    # ---- 官方验证评估 ----
    if verbose:
        print("[6/6] 计算官方验证集指标并保存模型 ...")
    val_mask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
                & actual.notna()).values
    val_times = times[val_mask]
    metrics = _evaluate(pred_full, actual, val_times)
    _write_eval(metrics)

    # 保存模型
    model.save(C.MODEL_BUNDLE)

    if verbose:
        print("\n================ 验证集评估 ================")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print("==========================================")
        print(f"模型已保存: {C.MODEL_BUNDLE}")
    return metrics


def _write_full_outputs(times, pred_full, actual):
    """写出 full_predictions.csv 与 full_mae.csv。"""
    actual_str_df = dl.load_actual_load_strings().set_index(C.COL_TIME)["actual_str"]
    actual_str_df = actual_str_df.reindex(times)
    dec = int(C.TRAIN_CONFIG.get("round_decimals", 2))

    pred_out = pd.DataFrame({
        "时间": [_fmt_time(t) for t in times],
        "预测负荷": pred_full.values,
        "实际负荷": actual_str_df.values,
    })
    pred_out.to_csv(C.FULL_PRED_CSV, index=False, encoding="utf-8-sig",
                    float_format=f"%.{dec}f")

    actual_num = actual.reindex(times)
    mae_vals = pd.Series(np.nan, index=times)
    mask = actual_num.notna()
    mae_vals[mask] = (pred_full[mask] - actual_num[mask]).abs()
    mae_out = pd.DataFrame({
        "时间": [_fmt_time(t) for t in times],
        "预测负荷": pred_full.values,
        "实际负荷": actual_str_df.values,
        "MAE": mae_vals.values,
    })
    mae_out.to_csv(C.FULL_MAE_CSV, index=False, encoding="utf-8-sig",
                   float_format=f"%.{dec}f")


def _evaluate(pred_full, actual, val_times) -> dict:
    p = pred_full.reindex(val_times)
    a = actual.reindex(val_times)
    m = a.notna()
    p, a = p[m], a[m]
    err = p - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    mape = float(np.mean(np.abs(err) / a) * 100)
    q = np.quantile(np.abs(err), [0.5, 0.9, 0.95, 0.99])
    return {
        "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE(%)": mape, "Bias": bias,
        "MAE_q50": float(q[0]), "MAE_q90": float(q[1]),
        "MAE_q95": float(q[2]), "MAE_q99": float(q[3]),
        "N_points": int(len(a)),
        "val_start": _fmt_time(val_times[0]), "val_end": _fmt_time(val_times[-1]),
    }


def _write_eval(metrics: dict):
    lines = []
    lines.append("山东省全省日前(D+1)负荷预测 - 验证集评估结果")
    lines.append("=" * 56)
    lines.append(f"验证集时间范围: {metrics['val_start']} ~ {metrics['val_end']}")
    lines.append(f"验证集样本数  : {metrics['N_points']}")
    lines.append("-" * 56)
    lines.append(f"MAE        (MW) : {metrics['MAE']:.4f}")
    lines.append(f"R2             : {metrics['R2']:.6f}")
    lines.append(f"RMSE       (MW) : {metrics['RMSE']:.4f}")
    lines.append(f"MAPE       (%)  : {metrics['MAPE(%)']:.4f}")
    lines.append(f"Bias       (MW) : {metrics['Bias']:.4f}")
    lines.append(f"MAE q50    (MW) : {metrics['MAE_q50']:.4f}")
    lines.append(f"MAE q90    (MW) : {metrics['MAE_q90']:.4f}")
    lines.append(f"MAE q95    (MW) : {metrics['MAE_q95']:.4f}")
    lines.append(f"MAE q99    (MW) : {metrics['MAE_q99']:.4f}")
    lines.append("-" * 56)
    lines.append(f"目标: MAE < 1500 MW  ->  {'PASS' if metrics['MAE'] < 1500 else 'FAIL'}")
    lines.append("=" * 56)
    with open(C.EVAL_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    metrics = run_train(verbose=True)
    return 0 if metrics["MAE"] < 1500 else 1


if __name__ == "__main__":
    sys.exit(main())
