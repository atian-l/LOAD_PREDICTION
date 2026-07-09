# -*- coding: utf-8 -*-
"""D+2 可达 MAE 估算: pred_load[D+2] 缺失(用 shift(96) 模拟"最近外部预测是 1 天前"),
分别测有/无 D+2 气象两档。复用 v6 全套管线。不写产物。"""
import sys, copy, numpy as np, pandas as pd
sys.path.insert(0, "e:/01/python/SdPproject/load_prediction/load_pred_v6")
from load_pred import config as C, data_loader as dl, features as F
from load_pred import train as T


def run_d2(weather_mode, label):
    """
    注释：
    weather_mode: "real" or "none"
    label: str, 用于打印输出
    函数功能：
    1. 加载负荷数据和气象数据，根据 weather_mode 决定
    2. 构建特征矩阵 X
    3. 根据配置训练模型，计算最佳迭代次数
    4. 训练集成模型，并计算小时偏差、漂移校正
    5. 预测负荷，计算验证集上的 MAE、R2、
偏差等指标，并打印输出
    6. 返回验证集上的 MAE
    """
    load_df = dl.load_load_data().set_index(C.COL_TIME)
    times = dl.full_time_index()
    pl_real = load_df[C.COL_PRED_LOAD].reindex(times)
    actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
    pl_d2 = pl_real.shift(96)  # D+2 信息态: 最近外部预测 = 1 天前(D+1)

    if weather_mode == "real":
        weather = dl.load_weather_dedup(run_time=None)
    else:  # 无 D+2 气象 -> 全 NaN
        weather = dl.load_weather_dedup(run_time=None).iloc[:0].copy()

    X = F.build_features(times, pl_d2, weather)
    usable = T.usable_mask(times, pl_d2, actual)
    cfg = copy.deepcopy(C.TRAIN_CONFIG)# cfg是一个字典，包含训练配置参数

    mm = None; mos = None
    if weather_mode == "real":
        mm = F.MismatchModel().fit(X, usable)
        X = mm.transform(X)
        if cfg.get("mos"):
            mc = cfg["mos"]
            mos = F.MosModel(cols=mc.get("cols"), alpha=mc.get("alpha", 1.0)).fit(X, actual, usable)
    else:  # 无气象 -> 无 pl_weather_residual, 关闭 drift/threshold 校正
        cfg["drift_corr"] = None
        cfg["threshold_corr"] = []

    y_dir = actual
    best_it, its = T.determine_best_iteration(times, X, y_dir, usable, cfg)
    model = T.train_ensemble(times, X, pl_d2, actual, usable, cfg, best_it, mos_model=mos)
    if mm is not None:
        model.mismatch_model = mm
    model.hour_bias, model.drift_corr, model.threshold_corr = T.compute_hour_bias(
        times, X, pl_d2, actual, usable, cfg, best_it, mos_model=mos)

    pred_full = pd.Series(model.predict_load(X, pl_d2), index=times)
    val_mask = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
                & actual.notna()).values
    metrics = T._evaluate(pred_full, actual, times[val_mask])
    raw = (pl_d2 - actual).abs()[val_mask].mean()
    base = (pl_real - actual).abs()[val_mask].mean()  # 正常 D+1 外部预测基线
    print("\n=== %s ===" % label)
    print("  正常 D+1 外部预测基线 MAE        = %.1f" % base)
    print("  D+2 裸 proxy |pl[T-96]-actual| MAE = %.1f" % raw)
    print("  D+2 优化模型 val MAE              = %.1f  (R2=%.4f, Bias=%.1f, n=%d)" %
          (metrics["MAE"], metrics["R2"], metrics["Bias"], metrics["N_points"]))
    return metrics["MAE"]


if __name__ == "__main__":
    print("building D+2 (有气象, 乐观上界) ...")
    run_d2("real", "D+2 乐观 (假设 D+2 气象可得)")
    print("\nbuilding D+2 (无气象, 现实) ...")
    run_d2("none", "D+2 现实 (D+2 气象不可得)")
    print("\n参照: v6 正常 D+1 模型 val MAE = 1445.6")
