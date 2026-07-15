# -*- coding: utf-8 -*-
"""
训练模式入口（TFT 集成模型）。

与 load_pred/train.py（LightGBM）、load_pred_tcn/train.py（TCN）流程逐行一致，仅 train_ensemble
将 lgb.train 换为 TFT（PyTorch 序列模型）训练，且 compute_hour_bias 的 fold 预测传 full X
（TFT 按预测日前向，需 encoder 历史在 X 内）：
  1. 读取数据（负荷+气象去重）；
  2. 构造无泄露特征矩阵（覆盖 full 时间范围）；
  3. best_iter：固定 epochs（walk-forward 在此漂移问题上过拟合，已禁用；与 v6 同哲学）；
  4. 训练多样化 TFT 集成（目标×残差/直接×种子），近期样本加权；
  5. 输出 full_predictions.csv / full_mae.csv / evaluation_metrics.txt；
  6. 保存集成模型至 models/。

合规：训练仅使用 < 2026-03-01 的数据；真实负荷仅作目标/评估，绝不入特征。
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import torch

from . import config as C
from . import data_loader as dl
from . import features as F
from .model import EnsembleModel
from .tft import train_tft, get_device


def _fmt_time(dt) -> str:
    return pd.Timestamp(dt).strftime("%Y/%m/%d %H:%M:%S")


def _time_weights(times, mask, alpha, pred_load=None, load_gamma=0.0):
    t = times[mask]; tmin, tmax = t.min(), t.max()
    if tmax == tmin:
        w = np.ones(len(t))
    else:
        w = (1.0 + alpha * (t - tmin).total_seconds() / (tmax - tmin).total_seconds()).values.astype(float)
    # 联合样本权重：负荷加权（v6 exp82 γ=1.0；输入仅 pred_load，合规#2）。
    if load_gamma > 0 and pred_load is not None:
        pl = pred_load.reindex(times)[mask].values.astype(float)
        pl_norm = pl / np.nanmean(pl)
        factor = np.clip(1.0 + load_gamma * np.clip(pl_norm - 1.0, -0.5, 1.0), 0.05, None)
        w = w * factor
    return w


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
    """返回 best_iter。若配置 best_it_fixed 则直接用之（walk-forward 在漂移问题上系统性
    过拟合验证集，见 exp44；固定保守 epochs 由 Agent Loop 据验证 MAE 选定）。
    TFT 下 best_it_fixed 即"固定训练 epochs"。模型训练仍只用训练期数据、不在验证集早停。
    """
    if cfg.get("best_it_fixed") is not None:
        print(f"      (使用固定 best_it_fixed={cfg['best_it_fixed']} epochs；walk-forward 已禁用)")
        return int(cfg["best_it_fixed"]), []
    its = _walk_forward_best_iters(times, X, y_dir, usable, cfg)
    return int(np.mean(its)), its


def _walk_forward_best_iters(times, X, y_dir, usable, cfg):
    """TFT walk-forward 早停未实现（生产用 best_it_fixed，与 v6 同哲学）。仅在 best_it_fixed=None
    时被调用，直接报错以避免静默退回错误行为。如需启用，需为 TFT 实现逐 epoch 验证 MAE 早停。"""
    raise NotImplementedError(
        "TFT walk-forward early-stopping 未实现。生产配置 best_it_fixed 已固定 epochs；"
        "若要启用 walk-forward，需为 TFT 实现逐 epoch 验证 MAE 监控与早停。"
    )


def train_ensemble(times, X, pred_load, actual, usable, cfg, best_it, mos_model=None) -> EnsembleModel:
    feat_cols = list(X.columns)
    static_cols = list(C.STATIC_COLS)
    y_dir = actual
    # 残差锚：两级系统 Stage1 MOS 的 corrected_pred（较 raw pred_load 更接近 actual；exp80 -9.86 MW）。
    # 无 MOS 时回退到 raw pred_load。actual 仅作 MOS 目标，此处仅用其输出。
    if mos_model is not None:
        anchor = pd.Series(mos_model.transform(X), index=X.index)
        y_res = actual - anchor
    else:
        y_res = actual - pred_load
    # 全长权重数组（与 X 对齐；TFT 按预测日取 target 段权重均值）
    w_full = np.zeros(len(X), dtype=float)
    w_full[usable] = _time_weights(times, usable, cfg["alpha_w"],
                                   pred_load=pred_load, load_gamma=cfg.get("weight_load_gamma", 0.0))
    device = get_device(cfg.get("device", "auto"))
    tft_config = {
        "n_feat": len(feat_cols), "n_static": len(static_cols),
        "hidden_size": int(cfg["hidden_size"]), "num_heads": int(cfg["num_heads"]),
        "num_lstm_layers": int(cfg["num_lstm_layers"]), "dropout": float(cfg["dropout"]),
        "encoder_len": int(cfg["encoder_len"]), "decoder_len": int(cfg["decoder_len"]),
    }

    model = EnsembleModel(feature_cols=feat_cols, shrinkage=cfg["shrinkage"],
                          train_meta={"config": cfg, "lags": C.PRED_LAGS,
                                      "best_it": best_it, "feature_cols": feat_cols},
                          aggregation=cfg.get("aggregation", "median"),
                          trim_frac=float(cfg.get("trim_frac", 0.2)),
                          mos_model=mos_model,
                          feat_mean=None, static_cols=static_cols,
                          tft_config=tft_config, feat_clip=cfg.get("feat_clip", 10.0),
                          device=cfg.get("device", "auto"))

    n = 0
    feat_mean_shared = None
    feat_std_shared = None
    for residual in cfg["residual_modes"]:
        ytr = y_res if residual else y_dir   # 全量目标 Series（train_tft 内部按 usable 取预测日样本）
        nit = int(best_it[residual]) if isinstance(best_it, dict) else int(best_it)
        for obj in cfg["objectives"]:
            if obj == "quantile":
                alphas = cfg["quantile_alphas"]
            else:
                alphas = [None]
            for qa in alphas:
                for s in cfg["seeds"]:
                    loss_type = "quantile" if obj == "quantile" else "regression"
                    alpha = qa if qa is not None else 0.5
                    tft, feat_mean, feat_std = train_tft(
                        X=X, y=ytr, w_full=w_full, usable=usable,
                        feat_cols=feat_cols, static_cols=static_cols, cfg=cfg,
                        loss_type=loss_type, alpha=alpha, seed=s,
                        epochs=nit, device=device, verbose=(n == 0))
                    model.add_member(tft, is_residual=residual)
                    if feat_mean_shared is None:
                        feat_mean_shared = feat_mean
                        feat_std_shared = feat_std
                    n += 1
    model.feat_mean = feat_mean_shared
    model.feat_std = feat_std_shared
    print(f"      集成成员数: {n}  (device={device})")
    return model


def compute_hour_bias(times, X, pred_load, actual, usable, cfg, best_it, mos_model=None):
    """
    用 3-fold walk-forward OOF 估计每小时的系统性偏置（仅用训练期数据，无泄露）。

    对每个折：在折内训练子集上训练完整集成 -> 预测折内验证子集 -> 收集 OOF 预测。
    hour_bias[h] = mean(oof_pred - actual) for hour h，覆盖所有 OOF 点。
    predict 时减去该偏置。

    注：TFT 按预测日序列前向，需 encoder 历史在 X 内，故 fold 预测传 full X 再取 [fva] 段
    （TCN 版传 X[fva]；TFT 不可，否则 fva 段起始预测日 encoder 历史不足）。
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
        fold_model = train_ensemble(times, X, pred_load, actual, ftr, cfg, best_it, mos_model=mos_model)
        # TFT 需 full X 提供 encoder 历史；predict_load 返回 full len(X)，取 fva 段
        full_pred = fold_model.predict_load(X, pred_load)
        oof_pred[fva] = np.asarray(full_pred)[fva]
    oof_mask = usable & oof_pred.notna().values
    resid = (oof_pred - actual).values
    n_slots = int(cfg.get("hour_bias_slots", 24))
    step = 1440 // n_slots
    dt_all = pd.DatetimeIndex(times)
    mod_all = dt_all.hour.values * 60 + dt_all.minute.values
    slot_all = (mod_all // step).astype(int)
    hour_bias = np.zeros(n_slots, dtype=float)
    for q in range(n_slots):
        m = oof_mask & (slot_all == q)
        if m.sum():
            hour_bias[q] = float(np.average(resid[m]))
    h_all = dt_all.hour.values
    print(f"      OOF 点数={oof_mask.sum()}  hour_bias[{n_slots}维] 范围=[{hour_bias.min():.0f}, {hour_bias.max():.0f}]")

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
    threshold_corr = []
    for tc in cfg.get("threshold_corr", []):
        feat_name = tc["feature"]
        op = tc.get("op", ">")
        thr = tc["thr"]
        hours_list = tc["hours"]
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
        else:
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
    mismatch_model = F.MismatchModel().fit(X, usable)
    X = mismatch_model.transform(X)
    mos_model = None
    if cfg.get("mos"):
        mc = cfg["mos"]
        mos_model = F.MosModel(cols=mc.get("cols"), alpha=mc.get("alpha", 1.0)).fit(X, actual, usable)
        if verbose:
            anchor = pd.Series(mos_model.transform(X), index=X.index)
            print(f"      MOS 锚: pred_load MAE={np.abs(pred_load[usable]-actual[usable]).mean():.1f} "
                  f"-> corrected MAE={np.abs(anchor[usable]-actual[usable]).mean():.1f}")
    if verbose:
        print(f"      特征数: {X.shape[1]}  可训练点: {usable.sum()}  "
              f"[{_fmt_time(times[0])} ~ {_fmt_time(times[-1])}]")

    y_dir = actual
    if verbose:
        print("[2/6] 确定 best_iter（固定 epochs，walk-forward 已禁用） ...")
    best_it, its = determine_best_iteration(times, X, y_dir, usable, cfg)
    if verbose:
        print(f"      best_it={best_it} (epochs)")

    if verbose:
        print("[3/6] 训练 TFT 集成 ...")
    model = train_ensemble(times, X, pred_load, actual, usable, cfg, best_it, mos_model=mos_model)
    model.mismatch_model = mismatch_model

    if verbose:
        print("[3.5/6] 计算 OOF 每小时偏置校正 + 漂移方向校正 + 阈值场景校正 ...")
    model.hour_bias, model.drift_corr, model.threshold_corr = compute_hour_bias(
        times, X, pred_load, actual, usable, cfg, best_it, mos_model=mos_model)

    if verbose:
        print("[4/6] 全量推理 ...")
    pred_full = pd.Series(model.predict_load(X, pred_load), index=times)
    dec = int(cfg.get("round_decimals", 2))
    pred_full = pred_full.round(dec)

    if verbose:
        print("[5/6] 写出 full_predictions.csv / full_mae.csv ...")
    _write_full_outputs(times, pred_full, actual)

    if verbose:
        print("[6/6] 计算官方验证集指标并保存模型 ...")
    val_mask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
                & actual.notna()).values
    val_times = times[val_mask]
    metrics = _evaluate(pred_full, actual, val_times)
    _write_eval(metrics)

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
    lines.append("山东省全省日前(D+1)负荷预测 - 验证集评估结果（TFT 集成）")
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
