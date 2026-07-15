# -*- coding: utf-8 -*-
"""v8.1 P0 残差溯源：验证"多阶段任务阶梯"的核心假设。

核心问题：任务 formulation 是否改变跨年可迁移性？
  - 幅度回归（v6 已知不可迁移，OOF 改善不迁移）vs 风险/方向分类（新角度）
  - 若分类任务跨年可迁移而回归不可迁移 -> 多阶段阶梯（风险->类型->幅度->Shape）有价值 -> P2
  - 若全部不可迁移 -> 信息上限硬约束，任务 formulation 无关 -> v8.1 多阶段 NO-GO

方法：用 v6 OOF 残差（3 折 walk-forward，全在训练期 <=TRAIN_END）作溯源对象。
  - 训练折 = 2025 春+秋 OOF 段；测试折 = 2026 冬 OOF 段（跨年，最接近 val）
  - 对 3 个子任务在训练折上训 LightGBM -> 测测试折跨年指标：
    [1] 幅度回归  target=resid            指标 R2/MAE  (baseline，预期 R2<0)
    [2] 风险二分类 target=(|resid|>P75)   指标 AUC     (>0.55=可迁移)
    [3] 方向三分类 target=sign(resid)分级 指标 acc     (>0.40=可迁移, baseline 0.33)

合规（不变量）:
  - actual 仅作 target/resid 计算（#1）；特征 X 不含 actual（#1）
  - OOF 全在训练期 best_it_folds（#5）；val（>=2026-03-01）eval-only，不参与
  - 复用 build_dataset/train_ensemble/compute_hour_bias 的 OOF 机制（#5）
  - throwaway 纯 stdout，不写产物；不修改生产脚本
"""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score

from . import config as C
from .train import build_dataset, usable_mask, train_ensemble
from .features import MismatchModel, MosModel


def _compute_oof_residual(times, X, pred_load, actual, usable, cfg, best_it, mos_model):
    """复用 compute_hour_bias 的 3 折 OOF 循环，返回 (oof_pred, resid, oof_mask)。
    每折在 ftr=usable&times<=te 上训完整集成 -> 预测 fva -> 收集 OOF。无泄露。"""
    oof_pred = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        fold_model = train_ensemble(times, X, pred_load, actual, ftr, cfg, best_it,
                                    mos_model=mos_model)
        oof_pred[fva] = fold_model.predict_load(X[fva], pred_load[fva])
    oof_mask = usable & oof_pred.notna().values
    resid = (oof_pred - actual).values
    return oof_pred, resid, oof_mask


def _subtask_lgb(X, feat_cols, tr_mask, te_mask, target, task, params):
    """训练一个 LightGBM 子任务模型并返回测试折预测。task: 'regression'/'binary'/'multiclass'。"""
    Xtr = X.loc[np.asarray(tr_mask), feat_cols]
    Xte = X.loc[np.asarray(te_mask), feat_cols]
    ytr = np.asarray(target)[tr_mask]
    dtr = lgb.Dataset(Xtr, label=ytr)
    if task == "multiclass":
        nclass = int(np.max(ytr)) + 1
        params_c = dict(params); params_c["num_class"] = nclass
        params_c["objective"] = "multiclass"
        params_c["metric"] = "multi_logloss"
    elif task == "binary":
        params_c = dict(params); params_c["objective"] = "binary"; params_c["metric"] = "auc"
    else:
        params_c = dict(params); params_c["objective"] = "regression"; params_c["metric"] = "rmse"
    model = lgb.train(params_c, dtr, num_boost_round=params_c.pop("num_iterations", 150),
                      callbacks=[])
    return model.predict(Xte)


def main() -> int:
    print("=" * 72)
    print("v8.1 P0 残差溯源：任务 formulation 是否改变跨年可迁移性？")
    print("=" * 72)

    print("\n[0] 构建数据集 + v6 配置 ...")
    times, X, pred_load, actual = build_dataset()
    usable = usable_mask(times, pred_load, actual)
    mismatch_model = MismatchModel().fit(X, usable)
    X = mismatch_model.transform(X)
    mc = C.TRAIN_CONFIG["mos"]
    mos_model = MosModel(cols=mc["cols"], alpha=mc["alpha"]).fit(X, actual, usable)
    # 2 成员快速 OOF（P0 诊断趋势；残差可迁移性由数据特性决定，与成员数无关，2 成员足够且快 5x）
    cfg = copy.deepcopy(C.TRAIN_CONFIG)
    cfg["objectives"] = ["regression"]
    cfg["residual_modes"] = [False, True]
    cfg["seeds"] = [42]
    best_it = int(cfg["best_it_fixed"])
    print(f"    特征数={X.shape[1]}  usable={int(usable.sum())}  best_it={best_it}  (2 成员 OOF)")

    print("\n[1] 构造 v6 OOF 残差（3 折 walk-forward，训练期内）...")
    oof_pred, resid, oof_mask = _compute_oof_residual(
        times, X, pred_load, actual, usable, cfg, best_it, mos_model)
    abs_resid = np.abs(resid)
    print(f"    OOF 点数={int(oof_mask.sum())}  resid 范围=[{np.nanmin(resid):.0f},{np.nanmax(resid):.0f}]"
          f"  |resid| 中位={np.nanmedian(abs_resid):.0f}  P75={np.nanquantile(abs_resid,0.75):.0f}"
          f"  P90={np.nanquantile(abs_resid,0.9):.0f}")

    # ---- 训练折(2025 春+秋) vs 测试折(2026 冬，跨年) ----
    folds = cfg["best_it_folds"]
    sp0, sp1 = pd.Timestamp(folds[0][1]), pd.Timestamp(folds[0][2])  # 2025-03-01..05-31
    au0, au1 = pd.Timestamp(folds[1][1]), pd.Timestamp(folds[1][2])  # 2025-09-01..11-30
    wi0, wi1 = pd.Timestamp(folds[2][1]), pd.Timestamp(folds[2][2])  # 2026-01-01..02-28
    tarr = np.asarray(times)
    train_oof = oof_mask & (((tarr >= sp0) & (tarr <= sp1)) | ((tarr >= au0) & (tarr <= au1)))
    test_oof = oof_mask & ((tarr >= wi0) & (tarr <= wi1))
    print(f"    训练折(2025春+秋)={int(train_oof.sum())}  测试折(2026冬)={int(test_oof.sum())}")

    feat_cols = list(X.columns)
    # 保守子任务参数（诊断用，防过拟合喧宾夺主）
    params = {"num_iterations": 150, "learning_rate": 0.05, "num_leaves": 63,
              "min_data_in_leaf": 100, "lambda_l2": 2.0, "feature_fraction": 0.8,
              "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1}

    # ---- [1] 幅度回归（baseline，预期不可迁移）----
    print("\n[2] 子任务[1] 幅度回归: target=resid（v6 已知不可迁移，baseline）")
    pred_amp = _subtask_lgb(X, feat_cols, train_oof, test_oof, resid, "regression", params)
    yte = resid[test_oof]
    r2_amp = r2_score(yte, pred_amp)
    mae_amp = np.mean(np.abs(yte - pred_amp))
    # 对照：不预测（pred=0，即接受 v6 原预测）的 MAE
    mae_zero = np.mean(np.abs(yte))
    print(f"    跨年 R2={r2_amp:.4f}  MAE(校正)={mae_amp:.1f}  MAE(不校正=0)={mae_zero:.1f}"
          f"  校正{'改善' if mae_amp<mae_zero else '恶化'} {mae_zero-mae_amp:+.1f}")

    # ---- [2] 风险二分类（|resid|>P75）----
    print("\n[3] 子任务[2] 风险二分类: target=(|resid|>P75)（哪些点会大误差）")
    thr = float(np.nanquantile(abs_resid[train_oof], 0.75))  # 训练折 P75（无泄露）
    y_risk = (abs_resid > thr).astype(int)
    pred_risk = _subtask_lgb(X, feat_cols, train_oof, test_oof, y_risk, "binary", params)
    yte_risk = y_risk[test_oof]
    auc_risk = roc_auc_score(yte_risk, pred_risk)
    base_risk = yte_risk.mean()
    print(f"    阈值 |resid|>{thr:.0f}  测试折大误差占比={base_risk:.3f}  跨年 AUC={auc_risk:.4f}"
          f"  ({'可迁移' if auc_risk>0.55 else '不可迁移'}，>0.55)")

    # ---- [3] 方向三分类（高估/准/低估）----
    print("\n[4] 子任务[3] 方向三分类: target=sign(resid)分级（高估/准/低估）")
    dir_gap = 200.0  # MW：|resid|<200 视为"准"
    y_dir = np.where(resid < -dir_gap, 0, np.where(resid > dir_gap, 2, 1)).astype(int)  # 0高估/1准/2低估
    pred_dir = _subtask_lgb(X, feat_cols, train_oof, test_oof, y_dir, "multiclass", params)
    pred_dir_lbl = np.argmax(pred_dir, axis=1)
    yte_dir = y_dir[test_oof]
    acc_dir = accuracy_score(yte_dir, pred_dir_lbl)
    # 各类 baseline（按多数类的 acc）
    base_dir = max(np.bincount(yte_dir, minlength=3)) / len(yte_dir)
    # macro AUC（一对其余）
    try:
        from sklearn.metrics import roc_auc_score as _auc
        macro_auc = _auc(yte_dir, pred_dir, multi_class="ovr", average="macro", labels=[0, 1, 2])
    except Exception:
        macro_auc = float("nan")
    print(f"    分级阈值+/-{dir_gap:.0f}MW  测试折分布={np.bincount(yte_dir,minlength=3)}"
          f"  跨年 acc={acc_dir:.4f}  macroAUC={macro_auc:.4f}"
          f"  ({'可迁移' if acc_dir>0.40 else '不可迁移'}，>0.40；多数类baseline={base_dir:.4f})")

    # ---- 判定 ----
    print("\n" + "=" * 72)
    print("P0 残差溯源结论")
    print("=" * 72)
    print(f"  [1] 幅度回归  : 跨年 R2={r2_amp:+.4f}  校正dMAE={mae_zero-mae_amp:+.1f}  "
          f"({'不可迁移' if r2_amp<0 else '可迁移?'})")
    print(f"  [2] 风险分类  : 跨年 AUC={auc_risk:.4f}  ({'可迁移' if auc_risk>0.55 else '不可迁移'})")
    print(f"  [3] 方向分类  : 跨年 acc={acc_dir:.4f}  ({'可迁移' if acc_dir>0.40 else '不可迁移'})")
    print("-" * 72)
    class_mig = (auc_risk > 0.55) or (acc_dir > 0.40)
    amp_mig = r2_amp > 0
    if class_mig and not amp_mig:
        print("  判定: 分类任务(风险/方向)可迁移而幅度回归不可迁移 -> 任务 formulation 改变可迁移性")
        print("        -> 多阶段阶梯（先分类筛选可迁移子集，再处理幅度）有理论价值 -> 进 P2 原型")
        return 0
    elif class_mig and amp_mig:
        print("  判定: 幅度回归也跨年可迁移 -> 应已有 v6 OOF 校正收益，但 v6 OOF 改善不迁移（矛盾）")
        print("        -> 需复查（可能测试折太短或子任务过拟合），暂缓 P2")
        return 2
    else:
        print("  判定: 全部子任务跨年不可迁移（幅度+风险+方向）")
        print("        -> 信息上限是硬约束，任务 formulation 无关；v6 OOF 改善不迁移的根因是")
        print("           ext_error 跨年结构变化（FDS 78.5% unlearnable），而非'任务错配'")
        print("        -> v8.1 多阶段任务阶梯 NO-GO，v6 1445.62 确认为最终基线")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
