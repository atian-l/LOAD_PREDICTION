# -*- coding: utf-8 -*-
"""
TCN 拟合诊断（exp_fit_diag）—— 调优门控脚本（首要）。

用户点名的"在纯训练集上输出 MAE 判欠/过拟合"诊断。三类预测：
  - train_raw : 模型在自己训练的数据上预测（含已见，乐观下界）
  - OOF       : 3 折 walk-forward 无泄露预测（训练数据的真实可预测性）
  - val_raw   : 官方验证集预测（无校正，泛化能力）
  - val_full  : 官方验证集预测（含 OOF 校正，生产管线读数）

判定：
  train_raw − OOF  大(>400) -> 过拟合；小(<150) -> 欠拟合；中间 -> 轻度
  OOF vs val_raw   |差|<200 -> 泛化一致(数据信号弱/结构性)；OOF<<val -> 漂移；OOF>>val -> 训练期更难
  TCN train_raw vs LightGBM v6 train_raw -> 拟合能力差异(模型 vs 数据)，若 lightgbm 可用

注：选择/判定仅用训练期 OOF，官方 val 仅读数（不参与选择，防泄露）。
运行：python -m load_pred_tcn.exp_fit_diag   （4090 上约 15~45 min，全量 40 成员）
"""
from __future__ import annotations
import sys
import time
import warnings

import numpy as np
import pandas as pd

from . import config as C
from . import exp_common as ec


def _hour_breakdown(pred_tr, a_tr, times_tr, pred_va, a_va, times_va):
    """分时段 train vs val raw MAE 表。"""
    h_tr = pd.DatetimeIndex(times_tr).hour.values
    h_va = pd.DatetimeIndex(times_va).hour.values
    err_tr = np.asarray(pred_tr) - np.asarray(a_tr)
    err_va = np.asarray(pred_va) - np.asarray(a_va)
    print(f"{'时段':>8} {'train':>9} {'val':>9} {'Δ(val-tr)':>10}")
    for lo, hi, n in [(0, 6, "00-06"), (6, 11, "06-11"), (11, 15, "11-14"),
                      (15, 18, "15-18"), (18, 24, "18-24")]:
        mt = (h_tr >= lo) & (h_tr < hi)
        mv = (h_va >= lo) & (h_va < hi)
        tmae = float(np.mean(np.abs(err_tr[mt]))) if mt.sum() else float("nan")
        vmae = float(np.mean(np.abs(err_va[mv]))) if mv.sum() else float("nan")
        print(f"{n:>8} {tmae:>9.0f} {vmae:>9.0f} {vmae - tmae:>+10.0f}")


def _lgb_compare(times, X, pred_load, actual, usable, val_m, mos_model,
                 tcn_train_raw, tcn_val_raw):
    """可选 LightGBM v6 同条件对比（判模型拟合能力 vs 泛化/数据）。lightgbm 不可用则跳过。"""
    print("\n[3] LightGBM v6 配置同条件对比（若 lightgbm 可用）...")
    try:
        import load_pred.train as lgb_train          # 生产 LightGBM 管线
        from load_pred import config as lgb_cfg
        cfg = lgb_cfg.TRAIN_CONFIG
        # 用 load_pred 自带 config（v6）训练；mos_model 鸭子类型可复用
        lgb_model = lgb_train.train_ensemble(times, X, pred_load, actual, usable, cfg,
                                             cfg["best_it_fixed"], mos_model)
        feat_cols = list(X.columns)

        def _lgb_raw(X_sub):
            av = (mos_model.transform(X_sub) if mos_model is not None
                  else pred_load[X_sub.index].values)
            av = np.asarray(av, dtype=float)
            mp = np.empty((len(lgb_model.members), len(X_sub)), dtype=float)
            for i, (booster, is_res) in enumerate(zip(lgb_model.members, lgb_model.member_residual)):
                raw = booster.predict(X_sub[feat_cols])
                mp[i] = av + raw if is_res else raw
            return av + cfg["shrinkage"] * (np.median(mp, axis=0) - av)

        lgb_train_raw = _lgb_raw(X[usable])
        lgb_val_raw = _lgb_raw(X[val_m])
        a_tr = actual[usable].values
        a_va = actual[val_m].values
        m_lgb_tr = ec._mae(lgb_train_raw, a_tr)
        m_lgb_va = ec._mae(lgb_val_raw, a_va)
        r2_lgb_tr = ec._r2(lgb_train_raw, a_tr)
        r2_lgb_va = ec._r2(lgb_val_raw, a_va)
        m_tcn_tr = ec._mae(tcn_train_raw, a_tr)
        m_tcn_va = ec._mae(tcn_val_raw, a_va)
        r2_tcn_tr = ec._r2(tcn_train_raw, a_tr)
        r2_tcn_va = ec._r2(tcn_val_raw, a_va)
        print(f"  {'':14} {'train_raw':>10} {'val_raw':>9}  {'R²_tr':>7} {'R²_val':>7}")
        print(f"  {'TCN (cur)':14} {m_tcn_tr:>10.1f} {m_tcn_va:>9.1f}  {r2_tcn_tr:>7.4f} {r2_tcn_va:>7.4f}")
        print(f"  {'LightGBM v6':14} {m_lgb_tr:>10.1f} {m_lgb_va:>9.1f}  {r2_lgb_tr:>7.4f} {r2_lgb_va:>7.4f}")
        d_tr = m_tcn_tr - m_lgb_tr
        d_va = m_tcn_va - m_lgb_va
        print(f"  差(TCN−LGB): train_raw {d_tr:+.1f}, val_raw {d_va:+.1f}")
        if d_tr > 50:
            print("  -> TCN train_raw 显著高于 LightGBM：拟合能力更弱 -> 模型问题(算法拟合能力差)")
        else:
            print("  -> 两者 train_raw 相近：拟合能力相当，val 差异来自泛化/数据，非 TCN 特有拟合不足")
    except ImportError:
        print("  (lightgbm 未安装，跳过对比；仅 TCN 自对比)")
    except Exception as e:  # noqa: BLE001
        print(f"  (LightGBM 对比失败: {type(e).__name__}: {str(e)[:120]})")


def main() -> int:
    t0 = time.perf_counter()
    print("=" * 78)
    print("TCN 拟合诊断: train_raw / OOF / val_raw / val_full   "
          f"(vs v6={ec.V6_VAL_MAE}, TCN基线={ec.TCN_BASE_MAE})")
    print("=" * 78)

    d = ec.build_cached()
    times, X, pred_load, actual, usable, val_m, mos_model, feat_cols = (
        d["times"], d["X"], d["pred_load"], d["actual"], d["usable"],
        d["val_m"], d["mos_model"], d["feat_cols"])
    print(f"特征数={len(feat_cols)}  训练点={int(usable.sum())}  val点={int(val_m.sum())}")

    # ---- 标准化 sanity ----
    print("\n[0] 标准化 sanity（feat_std / target_std 分布）...")
    sanity = ec.train_ens({"seeds": [42], "objectives": ["regression"],
                           "residual_modes": [False, True]}, verbose=False)
    fs = sanity.feat_std
    print(f"  feat_std: min={fs.min():.2f} max={fs.max():.2f} mean={fs.mean():.2f} "
          f"(应无 0/inf；pred_load 列 ~3e3)")
    for i, (tcn, is_res) in enumerate(zip(sanity.members, sanity.member_residual)):
        tm = float(tcn.target_mean); tsd = float(tcn.target_std)
        tag = "residual" if is_res else "direct"
        print(f"  member{i}[{tag}]: target_mean={tm:.1f} target_std={tsd:.1f} "
              f"(direct~6e4; residual 应远小)")
        if i >= 3:
            break

    # ---- [1] 全量训练（usable）+ 3 折 OOF 校正 ----
    print("\n[1] 训练 TCN 40 成员（usable）+ 3 折 OOF 校正估计 ...")
    model = ec.train_ens(verbose=False)  # 全量 40 成员，生产配置
    oof = ec.compute_oof(verbose=False)
    print(f"   成员数={len(model.members)}  OOF点={int(np.asarray(oof['oof_mask']).sum())}  "
          f"折MAE={[f'{x:.0f}' for x in oof['fold_mae']]}")

    # ---- [2] 四类预测 ----
    print("\n[2] 四类预测 (train_raw / OOF / val_raw / val_full) ...")
    a_tr = actual[usable].values
    a_va = actual[val_m].values
    t_tr = times[usable]
    t_va = times[val_m]

    train_raw = ec.ensemble_raw(model, X[usable], pred_load[usable])
    val_raw = ec.ensemble_raw(model, X[val_m], pred_load[val_m])
    ec.apply_corrections(model, oof["hour_bias"], oof["drift_corr"], oof["threshold_corr"])
    train_full = ec.ensemble_raw(model, X[usable], pred_load[usable])
    val_full = ec.ensemble_raw(model, X[val_m], pred_load[val_m])

    oof_mask = np.asarray(oof["oof_mask"])
    oof_p = oof["oof_pred"].values[oof_mask]
    oof_a = actual.values[oof_mask]

    m_train_raw = ec._mae(train_raw, a_tr)
    m_oof = ec._mae(oof_p, oof_a)
    m_val_raw = ec._mae(val_raw, a_va)
    m_train_full = ec._mae(train_full, a_tr)
    m_val_full = ec._mae(val_full, a_va)
    r2_train = ec._r2(train_raw, a_tr)
    r2_val = ec._r2(val_raw, a_va)

    print("\n" + "=" * 78)
    print("拟合诊断结果（TCN 当前配置）")
    print("=" * 78)
    print(f"{'':30} {'MAE':>9} {'R²':>8}")
    print(f"{'train (raw, 含已见乐观)':30} {m_train_raw:>9.1f} {r2_train:>8.4f}")
    print(f"{'OOF (无泄露训练误差)':30} {m_oof:>9.1f} {'':>8}")
    print(f"{'val (raw)':30} {m_val_raw:>9.1f} {r2_val:>8.4f}")
    print(f"{'train (full, 含校正)':30} {m_train_full:>9.1f}")
    print(f"{'val (full, 含校正)':30} {m_val_full:>9.1f}   (vs v6 {ec.V6_VAL_MAE}, TCN基线 {ec.TCN_BASE_MAE})")
    print("-" * 78)
    gap_to = m_train_raw - m_oof
    ov = m_oof - m_val_raw
    print(f"train_raw − OOF   = {gap_to:+.1f}   (>400 过拟合; <150 欠拟合; 中间 轻度)")
    print(f"OOF − val_raw     = {ov:+.1f}   (|<200| 泛化一致/数据信号弱; >200 训练期更难; <-200 漂移)")

    print("\n分时段 MAE (raw):")
    _hour_breakdown(train_raw, a_tr, t_tr, val_raw, a_va, t_va)

    # ---- [3] 可选 LightGBM v6 对比 ----
    _lgb_compare(times, X, pred_load, actual, usable, val_m, mos_model,
                 train_raw, val_raw)

    # ---- 判定 ----
    print("\n" + "=" * 78)
    print("判定：")
    if gap_to > 400:
        print(f"  train_raw({m_train_raw:.0f}) << OOF({m_oof:.0f})：能拟合训练数据但泛化差 -> 过拟合倾向")
        print("  -> 优先 exp_regularize(+dropout/wd) + exp_capacity(减通道/层) + exp_train_dyn(减 epoch)")
    elif gap_to < 150:
        print(f"  train_raw({m_train_raw:.0f}) ≈ OOF({m_oof:.0f})：连训练数据都拟合不好 -> 欠拟合(能力不足)")
        print("  -> 优先 exp_capacity(增通道/层/RF) + exp_train_dyn(增 epoch/lr)")
    else:
        print(f"  train_raw({m_train_raw:.0f}) vs OOF({m_oof:.0f}) gap={gap_to:.0f}：轻度过拟合")
        print("  -> exp_regularize + exp_train_dyn(找 epoch 最优点)")
    if abs(ov) < 200:
        print(f"  OOF({m_oof:.0f}) ≈ val_raw({m_val_raw:.0f})：泛化已一致，误差来自数据(信号弱/噪声) -> 结构性落后倾向")
        print("  -> 若同时 train_raw 能拟合：TCN 在本特征集结构性落后（同 CatBoost 结论），建议停止单独调优/转异质集成")
    elif ov > 200:
        print(f"  OOF({m_oof:.0f}) > val_raw({m_val_raw:.0f})：训练期更难预测(数据更噪/季节差异)")
    else:
        print(f"  OOF({m_oof:.0f}) < val_raw({m_val_raw:.0f})：验证期显著更差 -> 漂移/泛化失败")
    print("=" * 78)
    print(f"\n耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
