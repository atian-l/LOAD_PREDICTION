# -*- coding: utf-8 -*-
"""FDS/ds_prep.py - 预测诊断数据准备（只读，不修改任何生产代码/模型/训练流程）。

加载已训练的 v6 模型（EnsembleModel.load），在官方验证窗 2026/03/01-06/15 上生成预测，
并拼接全部诊断所需列（误差 / 外部预测误差 / 气象 / 日历 / 派生特征），存为 parquet 供后续分析。
合规：actual 仅作评估目标与误差计算（eval-only，#1）；不新增特征、不改训练、不改模型。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd

# 只读导入生产模块（不修改）
from load_pred import config as C, data_loader as dl, features as F, train as T
from load_pred.model import EnsembleModel

OUT = Path(__file__).parent / "output"
OUT.mkdir(parents=True, exist_ok=True)


def season_of(m: int) -> str:
    return {12: "冬", 1: "冬", 2: "冬", 3: "春", 4: "春", 5: "春", 6: "夏",
            7: "夏", 8: "夏", 9: "秋", 10: "秋", 11: "秋"}[m]


def build_val_predictions():
    """复用生产数据流（只读），生成验证窗预测 + 全诊断列。"""
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    weather = dl.load_weather_dedup(run_time=None)  # 诊断用全历史起报（与 val 评估同条件）
    X = F.build_features(times, pred_load, weather)
    usable = T.usable_mask(times, pred_load, actual)
    mm = F.MismatchModel().fit(X, usable)
    X = mm.transform(X)
    # v6 模型已含 MOS；load 即可（bundle 自带 mos_model/mismatch_model/hour_bias/drift/threshold）
    model = EnsembleModel.load(C.MODEL_BUNDLE)

    vmask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    Xv = X[vmask]
    pred = model.predict_load(Xv, pred_load[vmask])
    idx = times[vmask]

    d = pd.DataFrame(index=idx)
    d["pred"] = pred
    d["actual"] = actual[vmask].values
    d["pred_load"] = pred_load[vmask].values
    d["error"] = d["pred"] - d["actual"]            # 模型最终误差（正=高估）
    d["abs_error"] = d["error"].abs()
    d["ext_error"] = d["pred_load"] - d["actual"]   # 外部预测自身误差
    d["model_corr"] = d["pred"] - d["pred_load"]    # 模型相对外部预测的校正量
    d["pct_error"] = d["error"] / d["actual"] * 100.0

    # 气象（X 中已有）
    for c in ["temp", "irrad", "wind", "precip", "solar_wind", "clearness",
              "cloud_deficit", "hdd", "cdd", "clear_sky", "pl_weather_residual",
              "solar_mismatch", "pl_dip_96", "irrad_anom_672"]:
        if c in Xv.columns:
            d[c] = Xv[c].values
    # 日历
    dt = pd.DatetimeIndex(idx)
    d["hour"] = dt.hour.values
    d["minute"] = dt.minute.values
    d["slot"] = (dt.hour.values * 60 + dt.minute.values) // 15  # 0-95
    d["dow"] = dt.dayofweek.values  # 0=Mon
    d["month"] = dt.month.values
    d["doy"] = dt.dayofyear.values
    d["is_weekend"] = np.asarray(dt.dayofweek >= 5, dtype=int)
    d["season"] = [season_of(m) for m in dt.month.values]
    for c in ["is_holiday", "is_day_before_holiday", "is_midday", "is_daytime"]:
        if c in Xv.columns:
            d[c] = Xv[c].values
    # 爬坡（15min）
    d["actual_ramp"] = d["actual"].diff()
    d["pred_ramp"] = d["pred"].diff()
    d["ext_ramp"] = d["pred_load"].diff()
    d["ramp_error"] = d["pred_ramp"] - d["actual_ramp"]
    return d, Xv, model, X, times, pred_load, actual, usable


def build_train_oof(X, times, pred_load, actual, usable, cfg, best_it, mos_model):
    """训练期 3 折 walk-forward OOF 预测（in-distribution 残差，用于对比 val）。
    复用 train.compute_hour_bias 的折划分逻辑（只读调用 train_ensemble）。"""
    oof = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        fm = T.train_ensemble(times, X, pred_load, actual, ftr, cfg, best_it, mos_model=mos_model)
        oof[fva] = fm.predict_load(X[fva], pred_load[fva])
    m = usable & oof.notna().values
    d = pd.DataFrame(index=times[m])
    d["pred"] = oof[m].values
    d["actual"] = actual[m].values
    d["pred_load"] = pred_load[m].values
    d["error"] = d["pred"] - d["actual"]
    d["abs_error"] = d["error"].abs()
    d["ext_error"] = d["pred_load"] - d["actual"]
    dt = pd.DatetimeIndex(times[m])
    d["hour"] = dt.hour.values
    d["slot"] = (dt.hour.values * 60 + dt.minute.values) // 15
    d["month"] = dt.month.values
    return d


def main():
    d, Xv, model, X, times, pred_load, actual, usable = build_val_predictions()
    print(f"val 点数={len(d)}  MAE={d['abs_error'].mean():.2f}  "
          f"Bias={d['error'].mean():.2f}  RMSE={np.sqrt((d['error']**2).mean()):.2f}", flush=True)
    print(f"外部预测 MAE={d['ext_error'].abs().mean():.2f}  "
          f"模型校正后改善={(d['ext_error'].abs()-d['abs_error']).mean():.2f}", flush=True)

    # 训练期 OOF（in-distribution 对比）
    cfg = C.TRAIN_CONFIG
    # 重建 mos_model（诊断用，与生产同构；model bundle 已自带但 OOF 折需独立 fit）
    mos = F.MosModel().fit(X, actual, usable)
    oof = build_train_oof(X, times, pred_load, actual, usable, cfg, cfg["best_it_fixed"], mos)
    print(f"train OOF 点数={len(oof)}  MAE={oof['abs_error'].mean():.2f}  "
          f"Bias={oof['error'].mean():.2f}", flush=True)

    try:
        d.to_parquet(OUT / "diag_val.parquet")
        oof.to_parquet(OUT / "diag_oof.parquet")
        Xv.to_parquet(OUT / "X_val.parquet")
        print("saved parquet", flush=True)
    except Exception as e:
        d.to_csv(OUT / "diag_val.csv", encoding="utf-8-sig")
        oof.to_csv(OUT / "diag_oof.csv", encoding="utf-8-sig")
        Xv.to_csv(OUT / "X_val.csv", encoding="utf-8-sig")
        print(f"parquet failed ({e}), saved csv", flush=True)
    print("ds_prep done.", flush=True)


if __name__ == "__main__":
    main()
