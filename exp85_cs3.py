# -*- coding: utf-8 -*-
"""exp85_cs3 - CS3 多云午间校正的 walk-forward 跨年稳定性验证。

FDS/report.md 候选 CS3：多云午间场景 clearness∈[0.2,0.5)@11-14 残留 MAE 3854，v6 的
threshold_corr 已含该项（shrinkage=1.0），但报告质疑该偏置是否跨年稳定，还是仅在
2026 春季单一验证窗（且经大量 val 选择）上偶然。

本实验【不是】"在 val 上有没有效果"，而是：该多云午间偏置是否跨年份稳定存在、OOF 估计的
shift 能否迁移到独立 held-out 折。与 exp84 完全一致的 walk-forward 稳定性判据。

设计（与 exp84 同构的 ablation）：
  对每个 fold（train≤te，held-out val=[vs,ve]）：
    1. 就地按 train 区域重拟合 mismatch + MOS（无泄露，同 exp84 Part B）。
    2. 生成该 fold 训练区域内的 OOF 子折（官方折用 cfg 原 3 折；内折用 make_subfolds）。
    3. 复用生产 compute_hour_bias（cfg.best_it_folds 临时替换为 fold 子折）-> 在训练区域
       OOF 残差上估计全部校正量（hour_bias/drift/4 项 threshold，含多云午间 shift）。无泄露。
    4. 训练终集成于 fold 训练区域。
       treatment = predict（含多云午间校正）  baseline = predict（移除多云午间项，其余校正不变）
    5. Δ = treat_MAE − base_MAE（负=多云午间校正有益）；另报 shift_est vs val 实际偏置。

合规：阈值固定为生产值（clearness∈[0.2,0.5)@11-14, shrinkage=1.0），不重新搜索阈值；
所有 shift 由训练期 OOF 估计，不使用官方验证窗选择参数；不修改生产代码/模型；不写产物。
判定：跨年稳定 = 各折 Δ 一致≤0 且 shift_est 与 val 实际偏置同号近似。
"""
from __future__ import annotations
import io, sys, copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from load_pred import config as C, data_loader as dl, features as F, train as T

# v6 生产多云午间校正项的标识（固定阈值，不重搜）
CN_FEATURE, CN_OP, CN_THR, CN_HOURS = "clearness", "range", [0.2, 0.5], [11, 12, 13, 14]

FOLDS = [
    ("spring25", "2025-02-28 23:45:00", "2025-03-01 00:00:00", "2025-05-31 23:45:00"),
    ("autumn25", "2025-08-31 23:45:00", "2025-09-01 00:00:00", "2025-11-30 23:45:00"),
    ("winter26", "2025-12-31 23:45:00", "2026-01-01 00:00:00", "2026-02-28 23:45:00"),
    ("official", C.TRAIN_END,            C.VAL_START,            C.VAL_END),
]
DEC = int(C.TRAIN_CONFIG.get("round_decimals", 2))


def mae(pred, act):
    p = np.round(np.asarray(pred, dtype=float), DEC)
    return float(np.abs(p - act).mean())


def is_cloudynoon(tc):
    return tc["feature"] == CN_FEATURE and tc.get("op") == CN_OP


def make_subfolds(train_end, n=3):
    """在 [train_start, train_end] 内生成 n 个不重叠 ~3 月 OOF 子折：(train_cutoff, val_start, val_end)。
    每子折用 train_cutoff 之前数据训练、验证 [val_start, val_end]；训练数据不足 6 月则止。"""
    te = pd.Timestamp(train_end)
    ts0 = pd.Timestamp(C.TRAIN_CONFIG["train_start"])
    min_train = pd.DateOffset(months=6)
    folds, end = [], te
    for _ in range(n):
        val_start = end - pd.DateOffset(months=3) + pd.Timedelta(minutes=15)
        train_cutoff = val_start - pd.Timedelta(minutes=15)
        if train_cutoff < ts0 + min_train:
            break
        folds.append((train_cutoff, val_start, end))
        end = train_cutoff - pd.Timedelta(minutes=15)
    return list(reversed(folds))


def cloudynoon_val_mask(X, times, vmask):
    dt = pd.DatetimeIndex(times[vmask])
    hrs = dt.hour.values
    clr = X["clearness"].values[vmask]
    return (clr >= CN_THR[0]) & (clr < CN_THR[1]) & np.isin(hrs, CN_HOURS)


def run_fold(name, te, vs, ve, times, X_base, pred_load, actual, usable, cfg, best_it):
    """单折：训练区域重拟合 + fold 子折 OOF 估计校正量 + treatment/baseline 对比。"""
    te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
    train_mask = usable if name == "official" else (usable & (times <= te))
    vmask = ((times >= vs) & (times <= ve) & actual.notna() & pred_load.notna()).values

    # 1) fold 训练区域 mismatch + MOS
    mm = F.MismatchModel().fit(X_base, train_mask); X = mm.transform(X_base)
    mos = F.MosModel().fit(X, actual, train_mask)

    # 2)+3) fold 子折 OOF 估计全部校正量（临时替换 best_it_folds；无泄露）
    if name == "official":
        subfolds = [(pd.Timestamp(a), pd.Timestamp(b), pd.Timestamp(c))
                    for a, b, c in cfg["best_it_folds"]]
    else:
        subfolds = make_subfolds(te, n=3)
    cfg_f = dict(cfg); cfg_f["best_it_folds"] = subfolds
    print(f"\n--- fold={name}  subfolds={len(subfolds)}  train≤{str(te)[:10]}  val={vs.strftime('%m-%d')}~{ve.strftime('%m-%d')}", flush=True)
    hour_bias, drift_corr, threshold_corr = T.compute_hour_bias(
        times, X, pred_load, actual, train_mask, cfg_f, best_it, mos_model=mos)

    cn_item = next((tc for tc in threshold_corr if is_cloudynoon(tc)), None)
    shift_est = float(cn_item["shift"]) if cn_item else 0.0

    # 4) 终集成
    model = T.train_ensemble(times, X, pred_load, actual, train_mask, cfg, best_it, mos_model=mos)
    model.mismatch_model = mm
    model.hour_bias = hour_bias
    model.drift_corr = drift_corr
    act_v = actual[vmask].values

    # treatment：含多云午间校正
    model.threshold_corr = threshold_corr
    pred_T = model.predict_load(X[vmask], pred_load[vmask])
    mae_T = mae(pred_T, act_v)

    # baseline：移除多云午间项（其余校正不变）
    model.threshold_corr = [tc for tc in threshold_corr if not is_cloudynoon(tc)]
    pred_B = model.predict_load(X[vmask], pred_load[vmask])
    mae_B = mae(pred_B, act_v)

    # 多云午间 val 实际偏置（baseline 残差在该场景的均值 = 校正目标）
    cn_v = cloudynoon_val_mask(X, times, vmask)
    bias_val = float(np.mean(pred_B[cn_v] - act_v[cn_v])) if cn_v.any() else float("nan")
    return dict(name=name, n_cn=int(cn_v.sum()), shift_est=shift_est, bias_val=bias_val,
                mae_B=mae_B, mae_T=mae_T, delta=mae_T - mae_B)


def main():
    cfg = C.TRAIN_CONFIG; best_it = cfg["best_it_fixed"]
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)
    X_base = F.build_features(times, pred_load, weather)
    usable = T.usable_mask(times, pred_load, actual)
    print(f"CS3 多云午间校正跨年稳定性验证  特征数={X_base.shape[1]}  best_it={best_it}", flush=True)
    print(f"固定阈值: {CN_FEATURE} {CN_OP} {CN_THR} @h{CN_HOURS}  shrinkage=1.0 (生产值，不重搜)", flush=True)

    rows = []
    for name, te, vs, ve in FOLDS:
        rows.append(run_fold(name, te, vs, ve, times, X_base, pred_load, actual, usable, cfg, best_it))

    # ================ 汇总 ================
    print("\n==== 各折结果 ====", flush=True)
    print(f"{'折':<10}{'n_cn':>6}{'shift_est':>11}{'bias_val':>10}{'base_MAE':>10}{'treat_MAE':>11}{'Δ':>9}", flush=True)
    for r in rows:
        print(f"{r['name']:<10}{r['n_cn']:>6}{r['shift_est']:>+11.1f}{r['bias_val']:>+10.1f}"
              f"{r['mae_B']:>10.1f}{r['mae_T']:>11.1f}{r['delta']:>+9.1f}", flush=True)

    # 官方折 treatment 应复现 v6=1445.62（sanity）
    off = next(r for r in rows if r["name"] == "official")
    print(f"\n[Sanity] 官方折 treat_MAE={off['mae_T']:.2f} (生产 v6≈1445.62)  base_MAE={off['mae_B']:.2f}", flush=True)

    # ================ 判定 ================
    print("\n================ CS3 判定 ================", flush=True)
    deltas = [r["delta"] for r in rows]
    # 跨年稳定性：内 3 折（独立 held-out）的 Δ 符号 + shift/bias 同号
    inner = [r for r in rows if r["name"] != "official"]
    inner_help = sum(1 for r in inner if r["delta"] < 0)
    inner_hurt = sum(1 for r in inner if r["delta"] > 1)
    sign_consistent = all((r["shift_est"] < 0) == (r["bias_val"] < 0) for r in rows
                          if np.isfinite(r["bias_val"]) and abs(r["shift_est"]) > 50)
    print(f"各折 Δ = {[round(d,1) for d in deltas]}", flush=True)
    print(f"内 3 折: 有益(Δ<0)={inner_help}/3  明显有害(Δ>+1)={inner_hurt}/3", flush=True)
    print(f"shift_est 与 val 实际偏置同号(|shift|>50): {sign_consistent}", flush=True)
    if inner_hurt == 0 and inner_help >= 2 and sign_consistent:
        verdict = "VALIDATED 跨年稳定--多云午间偏置跨年份稳定存在，OOF shift 可迁移，校正具有稳定收益"
    elif inner_hurt == 0 and inner_help >= 1:
        verdict = "MARGINAL 部分迁移--偏置存在但收益微弱/不稳，价值有限"
    else:
        verdict = "REJECTED 不跨年稳定--多云午间偏置在独立 held-out 折上不稳定或反向，系 val 窗偶然，不建议依赖"
    print(f"结论: {verdict}", flush=True)


if __name__ == "__main__":
    main()
