# -*- coding: utf-8 -*-
"""
Phase 1：CatBoost vs v6 LightGBM 公平 A/B（同特征矩阵 + 同 OOF 校正流程，仅换 booster）

目标（见 catboost_migration_plan.md §7 Phase 1）：
  - 用与 v6 完全相同的特征矩阵（build_features + MismatchModel）、相同的 40 成员结构
    ({regression, quantile 0.45/0.5/0.55} × {direct, residual} × 5 seeds)、相同的 MOS 锚 +
    median + λ 收缩 + OOF 校正流程，仅把 LightGBM 换成 CatBoost（GPU, Plain, rsm=1.0）。
  - best_iter 由 CatBoost 自己的 walk-forward 早停重定（不照搬 LightGBM 的 80）。
  - OOF 校正量（hour_bias / drift_corr / threshold_corr）在 CatBoost OOF 残差上重新估计。
  - 输出 6 指标 vs v6（1445.62）。

Phase 0 已确认：rsm on GPU 报错->不用；quantile on GPU OK；Ordered 5.36x 慢->本阶段用 Plain。

合规：
  - 不修改任何生产脚本；仅 import train.build_dataset / train._time_weights / train.usable_mask
    / features.MismatchModel / features.MosModel 复用。
  - actual 仅作 target / MOS 目标 / 评估；训练仅 usable mask(<=TRAIN_END)；val eval-only。
  - 6 条泄露不变量全保持。

运行：python -m load_pred.exp_catboost_ab   （4090 上约 4-6 min）
"""
from __future__ import annotations
import sys
import time
import warnings

import numpy as np
import pandas as pd

try:
    import catboost as cb
    from catboost import CatBoostRegressor, Pool
except ImportError:
    print("[FATAL] 未安装 catboost：pip install catboost")
    sys.exit(1)

from . import config as C
from .train import build_dataset, usable_mask, _time_weights
from .features import MismatchModel, MosModel

V6_VAL_MAE = 1445.62  # 生产基线（含 40 成员 + OOF 校正）


# --------------------------------------------------------------------------- #
# CatBoost 参数 / 训练原语
# --------------------------------------------------------------------------- #
def _cb_params(loss: str, seed: int, iters: int, eval_set: bool = False) -> dict:
    """CatBoost 基础参数（映射自 v6 LightGBM；rsm 因 GPU 不支持而弃用，见 Phase 0）。"""
    p = dict(
        task_type="GPU", devices="0",
        loss_function=loss,
        learning_rate=0.03,
        depth=8,                # ≈ LightGBM num_leaves=255（2^8=256 叶）
        l2_leaf_reg=4.0,        # = lambda_l2
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        boosting_type="Plain",  # Ordered 5.36x 慢，留 Phase 3
        random_seed=seed,
        verbose=0,
        allow_writing_files=False,
        iterations=iters,
    )
    if eval_set:
        p["eval_metric"] = "MAE"
        p["od_type"] = "Iter"
        p["od_wait"] = 50
    return p


def _cb_fit(Xtr, ytr, wtr, loss, seed, iters, eval_set=None):
    pool = Pool(Xtr, label=ytr, weight=wtr)
    m = CatBoostRegressor(**_cb_params(loss, seed, iters, eval_set=bool(eval_set)))
    if eval_set is not None:
        evX, evy = eval_set
        m.fit(pool, eval_set=Pool(evX, label=evy))
    else:
        m.fit(pool)
    return m


def _arr(df, cols) -> np.ndarray:
    """DataFrame -> numpy（规避中文特征名；为 cat_features 位置索引前向兼容）。"""
    return df[cols].to_numpy(dtype=np.float64, copy=False)


# --------------------------------------------------------------------------- #
# walk-forward best_iter（CatBoost 早停；镜像 train._walk_forward_best_iters）
# --------------------------------------------------------------------------- #
def _cb_walk_forward_best_iters(times, X, y_dir, usable, pred_load, cfg, feat_cols):
    cap = int(cfg.get("cb_best_it_cap", 3000))
    its = []
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & (times <= te)
        fva = usable & (times >= vs) & (times <= ve)
        Xtr = _arr(X[ftr], feat_cols); ytr = y_dir[ftr].to_numpy(np.float64)
        wtr = _time_weights(times, ftr, cfg["alpha_w"], pred_load=pred_load,
                            load_gamma=cfg.get("weight_load_gamma", 0.0))
        Xva = _arr(X[fva], feat_cols); yva = y_dir[fva].to_numpy(np.float64)
        m = _cb_fit(Xtr, ytr, wtr, "RMSE", 42, cap, eval_set=(Xva, yva))
        bi = m.get_best_iteration()
        if bi is None:
            bi = m.tree_count_
        its.append(int(bi))
        print(f"      fold {te.date()}~{ve.date()}: best_iter={int(bi)}")
    return its


# --------------------------------------------------------------------------- #
# 40 成员集成训练（镜像 train.train_ensemble）
# --------------------------------------------------------------------------- #
def _cb_train_ensemble(X, actual, anchor, mask, cfg, best_it, feat_cols, tag=""):
    """返回 [(catboost_model, is_residual), ...]，共 40 成员。"""
    y_res = actual - anchor
    Xtr = _arr(X[mask], feat_cols)
    wtr = _time_weights(times_global, mask, cfg["alpha_w"], pred_load=pred_load_global,
                        load_gamma=cfg.get("weight_load_gamma", 0.0))
    ytr_dir = actual[mask].to_numpy(np.float64)
    ytr_res = y_res[mask].to_numpy(np.float64)
    members = []
    n = 0
    for residual in cfg["residual_modes"]:
        y = ytr_res if residual else ytr_dir
        for obj in cfg["objectives"]:
            alphas = cfg["quantile_alphas"] if obj == "quantile" else [None]
            for qa in alphas:
                loss = f"Quantile:alpha={qa}" if obj == "quantile" else "RMSE"
                for s in cfg["seeds"]:
                    m = _cb_fit(Xtr, y, wtr, loss, s, best_it)
                    members.append((m, bool(residual)))
                    n += 1
    if tag:
        print(f"      [{tag}] 集成成员数: {n}  best_it={best_it}")
    return members


# --------------------------------------------------------------------------- #
# 推理（镜像 model.EnsembleModel.predict_load，仅 booster.predict 换 CatBoost）
# --------------------------------------------------------------------------- #
def _ensemble_raw(members, X, anchor_vals, feat_cols, shrinkage) -> np.ndarray:
    """anchor + λ*(median(member_preds) - anchor)，不含 OOF 校正（用于 OOF 估计）。"""
    Xarr = _arr(X, feat_cols)
    mp = np.empty((len(members), len(X)), dtype=float)
    for i, (m, is_res) in enumerate(members):
        raw = m.predict(Xarr)
        mp[i] = anchor_vals + raw if is_res else raw
    ens = np.median(mp, axis=0)
    return anchor_vals + shrinkage * (ens - anchor_vals)


def _predict_load(members, X, anchor_vals, feat_cols, shrinkage,
                  hour_bias, drift_corr, threshold_corr) -> np.ndarray:
    """完整预测：ensemble_raw + hour_bias + drift_corr + threshold_corr + clip。"""
    pred = _ensemble_raw(members, X, anchor_vals, feat_cols, shrinkage)
    dt = pd.DatetimeIndex(X.index)
    hours = dt.hour.values.astype(int)
    if hour_bias is not None:
        n = len(hour_bias)
        mod = hours * 60 + dt.minute.values
        idx = ((mod * n) // 1440).astype(int)
        pred = pred - hour_bias[idx]
    # drift_corr：符号 +=（bug#2 已验，勿改；详见 model.py 注释）
    for feat_name, beta in drift_corr:
        beta = np.asarray(beta, dtype=float)
        pred = pred + beta[hours] * X[feat_name].values.astype(float)
    # threshold_corr
    for tc in threshold_corr:
        fv = X[tc["feature"]].values.astype(float)
        op = tc.get("op", ">")
        thr = tc["thr"]
        if op == "range":
            sel = (fv >= thr[0]) & (fv < thr[1])
        elif op == ">=":
            sel = fv >= thr
        elif op == "<":
            sel = fv < thr
        elif op == "<=":
            sel = fv <= thr
        else:
            sel = fv > thr
        hl = tc.get("hours")
        if hl is not None:
            sel = sel & np.isin(hours, list(hl))
        shift = tc.get("shift", 0.0)
        if shift != 0.0:
            pred[sel] = pred[sel] - shift
    return np.clip(pred, 0.0, None)


# --------------------------------------------------------------------------- #
# OOF 校正估计（镜像 train.compute_hour_bias，换 CatBoost 集成）
# --------------------------------------------------------------------------- #
def _cb_compute_hour_bias(times, X, pred_load, actual, usable, anchor, cfg, best_it, feat_cols):
    oof_pred = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0:
            continue
        members = _cb_train_ensemble(X, actual, anchor, ftr, cfg, best_it, feat_cols, tag=f"OOF fold {ve.date()}")
        oof_pred[fva] = _ensemble_raw(members, X[fva], anchor[fva].values, feat_cols, cfg["shrinkage"])
    oof_mask = usable & oof_pred.notna().values
    resid = (oof_pred - actual).values

    # hour_bias（slot 粒度由 config hour_bias_slots 决定；默认 96=逐 15min）
    n_slots = int(cfg.get("hour_bias_slots", 96))
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
    print(f"      OOF 点数={int(oof_mask.sum())}  hour_bias[{n_slots}] 范围=[{hour_bias.min():.0f},{hour_bias.max():.0f}]")

    # drift_corr β（午间 11-14 逐小时；无泄露）
    drift_corr = []
    dc = cfg.get("drift_corr")
    if dc:
        fn = dc["feature"]; hs = set(dc["hours"])
        feat = X[fn].values.astype(float)
        beta = np.zeros(24, dtype=float)
        for h in range(24):
            if h not in hs:
                continue
            m = oof_mask & (h_all == h)
            f = feat[m]; e = resid[m]
            good = np.isfinite(f) & np.isfinite(e)
            d = float(np.dot(f[good], f[good]))
            if d > 0:
                beta[h] = float(np.dot(f[good], e[good]) / d)
        drift_corr.append((fn, beta))
        print(f"      drift_corr[{fn}] 午间β={[(h, round(beta[h],4)) for h in sorted(hs)]}")

    # threshold_corr shift（OOF 残差均值 × shrinkage；无泄露）
    threshold_corr = []
    for tc in cfg.get("threshold_corr", []):
        fn = tc["feature"]; op = tc.get("op", ">"); thr = tc["thr"]
        hl = tc["hours"]; shrink = float(tc["shrinkage"])
        feat = X[fn].values.astype(float)
        if op == "range":
            m = oof_mask & (feat >= thr[0]) & (feat < thr[1])
        elif op == ">=":
            m = oof_mask & (feat >= thr)
        elif op == "<":
            m = oof_mask & (feat < thr)
        elif op == "<=":
            m = oof_mask & (feat <= thr)
        else:
            m = oof_mask & (feat > thr)
        if hl is not None:
            m = m & np.isin(h_all, list(hl))
        shift = float(np.average(resid[m])) * shrink if m.sum() else 0.0
        threshold_corr.append({"feature": fn, "op": op, "thr": thr, "hours": hl, "shift": shift})
        td = thr if op != "range" else f"[{thr[0]},{thr[1]})"
        print(f"      threshold_corr[{fn}{op}{td}, {'全天' if hl is None else f'h{hl}'}] n={int(m.sum())} shift={shift:+.1f}")
    return hour_bias, drift_corr, threshold_corr, oof_pred, oof_mask


# --------------------------------------------------------------------------- #
# 评估 / 6 指标
# --------------------------------------------------------------------------- #
def _metrics(pred, actual_idx, times_idx):
    a = actual_idx; p = pred
    m = a.notna().values
    p, a = p[m], a[m]
    err = p - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    # 午间 11-14（小时 11,12,13,14 = 11:00-14:59，与 config drift_corr.hours=[11,12,13,14] 同口径）
    h = pd.DatetimeIndex(times_idx).hour.values[m]
    mid = (h >= 11) & (h <= 14)
    mid_mae = float(np.mean(np.abs(err[mid]))) if mid.sum() else float("nan")
    # 分时段
    bands = {}
    for lo, hi, name in [(0,6,"00-06"),(6,11,"06-11"),(11,15,"11-14"),(15,18,"15-18"),(18,24,"18-24")]:
        bm = (h>=lo)&(h<hi)
        bands[name] = float(np.mean(np.abs(err[bm]))) if bm.sum() else float("nan")
    # Top10 误差日
    df = pd.DataFrame({"date": pd.DatetimeIndex(times_idx)[m].normalize(),
                       "abs_err": np.abs(err)})
    top10 = df.groupby("date")["abs_err"].mean().sort_values(ascending=False).head(10)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Bias": bias, "midday_MAE": mid_mae,
            "bands": bands, "top10": top10}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
times_global = None
pred_load_global = None


def main() -> int:
    global times_global, pred_load_global
    t_start = time.perf_counter()
    print("=" * 64)
    print(f"Phase 1: CatBoost vs v6 公平 A/B   catboost {cb.__version__}")
    print("=" * 64)

    print("[1/6] 构建数据集（train.build_dataset + MismatchModel + MOS）...")
    times, X, pred_load, actual = build_dataset()
    times_global, pred_load_global = times, pred_load
    usable = usable_mask(times, pred_load, actual)
    mm = MismatchModel().fit(X, usable)
    X = mm.transform(X)
    mos_model = MosModel(cols=C.TRAIN_CONFIG["mos"]["cols"],
                         alpha=C.TRAIN_CONFIG["mos"]["alpha"]).fit(X, actual, usable)
    anchor = pd.Series(mos_model.transform(X), index=X.index)
    feat_cols = list(X.columns)
    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    print(f"      特征数={len(feat_cols)}  训练点={int(usable.sum())}  val点={int(val_m.sum())}")

    print("[2/6] walk-forward 3 折定 CatBoost best_iter（早停 MAE）...")
    cfg = C.TRAIN_CONFIG
    try:
        its = _cb_walk_forward_best_iters(times, X, actual, usable, pred_load, cfg, feat_cols)
        best_it = int(np.mean(its))
        print(f"      各折 best_iter={its} -> 均值 {best_it}")
    except Exception as e:
        ename = type(e).__name__
        print(f"      walk-forward 早停失败({ename}): {e} -> 回退 best_it=200")
        best_it, its = 200, []
    # 软上限：bound 运行时间（2.8s/800iter@4090；160 模型 ×1000iter ≈ 9 min）
    cap_apply = int(cfg.get("cb_best_it_apply_cap", 1000))
    if best_it > cap_apply:
        print(f"      best_iter={best_it} > {cap_apply}，截断为 {cap_apply}（防早停失效致运行时间失控；Phase 3 可放开）")
        best_it = cap_apply

    print("[3/6] 训练 CatBoost 40 成员集成（官方 fold, full train）...")
    members = _cb_train_ensemble(X, actual, anchor, usable, cfg, best_it, feat_cols, tag="official")

    print("[4/6] OOF 估计 hour_bias + drift_corr + threshold_corr（3 折重训 CatBoost）...")
    hour_bias, drift_corr, threshold_corr, oof_pred, oof_mask = _cb_compute_hour_bias(
        times, X, pred_load, actual, usable, anchor, cfg, best_it, feat_cols)

    print("[5/6] 官方 val 全量推理 + 评估 ...")
    pred_val = _predict_load(members, X[val_m], anchor[val_m].values, feat_cols,
                             cfg["shrinkage"], hour_bias, drift_corr, threshold_corr)
    actual_val = actual[val_m]
    mt = _metrics(pd.Series(pred_val, index=times[val_m]), actual_val, times[val_m])

    # walk-forward 折 MAE（从 OOF）
    fold_maes = {}
    for te, vs, ve in cfg["best_it_folds"]:
        vs, ve = pd.Timestamp(vs), pd.Timestamp(ve)
        fm = usable & (times >= vs) & (times <= ve) & oof_pred.notna().values
        if fm.sum():
            fold_maes[f"{vs.date()}~{ve.date()}"] = float(
                np.mean(np.abs(oof_pred[fm].values - actual[fm].values)))

    print("[6/6] 输出 6 指标 ...")
    dt = time.perf_counter() - t_start
    # 折间稳定性
    fold_arr = np.array(list(fold_maes.values())) if fold_maes else np.array([])
    fold_cv = float(fold_arr.std() / fold_arr.mean()) if len(fold_arr) >= 2 and fold_arr.mean() > 0 else float("nan")
    delta = mt['MAE'] - V6_VAL_MAE
    competitive = mt['MAE'] <= V6_VAL_MAE + 60   # 60 MW 内视为可竞争（值得 Phase 3 超参搜索）
    stable = (not np.isnan(fold_cv)) and fold_cv < 0.06

    print("\n" + "=" * 64)
    print("Phase 1 汇总：CatBoost vs v6（同特征矩阵 / 同 OOF 校正流程 / 仅换 booster）")
    print("=" * 64)
    print(f"耗时: {dt:.1f}s   best_iter={best_it} (各折 {its})")
    print(f"\n[指标1] 官方 val (n={int(val_m.sum())}, 窗口 {C.VAL_START} ~ {C.VAL_END})")
    print(f"  CatBoost  MAE={mt['MAE']:.2f}  R²={mt['R2']:.6f}  RMSE={mt['RMSE']:.2f}  Bias={mt['Bias']:.2f}")
    print(f"  v6 基线   MAE={V6_VAL_MAE:.2f}  R²≈0.9292")
    print(f"  ΔMAE(CatBoost−v6) = {delta:+.2f} MW  -> {'CatBoost 优' if mt['MAE']<V6_VAL_MAE else 'v6 优'}")
    print(f"\n[指标2] walk-forward 3 折 OOF MAE + 稳定性")
    for k, v in fold_maes.items():
        print(f"  {k}: {v:.2f}")
    print(f"  折间 std/mean = {fold_cv:.3f}  (越小越稳；<0.06 视为稳定)")
    print(f"\n[指标3] 午间(小时 11-14, 11:00-14:59) val MAE = {mt['midday_MAE']:.2f}"
          f"  (v6 同口径参考约 2835，FDS_midday 诊断口径可能略异)")
    print(f"\n[指标4] Bias = {mt['Bias']:.2f}")
    print(f"\n[指标5] 分时段 val MAE")
    for k, v in mt["bands"].items():
        print(f"  {k}: {v:.2f}")
    print(f"\n[指标6] Top10 误差日")
    for d, v in mt["top10"].items():
        print(f"  {d.date()}: {v:.1f}")
    print("\n" + "-" * 64)
    print(f"Phase 3 准入: competitive={competitive} (val≤{V6_VAL_MAE+60:.1f})  stable={stable}")
    if competitive and stable:
        print("  -> 建议进 Phase 3：超参搜索(depth/lr/l2/bagging_temp) + Ordered 单点验证 + 集成层列子采样")
    elif mt['MAE'] >= V6_VAL_MAE + 60:
        print("  -> CatBoost 明显落后(>+60MW)：若 best_iter 过大(val 远差于 OOF 折)->Phase 3 试 fixed-best_it/Ordered；")
        print("     若 OOF 折本身已差->CatBoost 在本特征集上无优势，考虑放弃或仅作异质集成成员(Phase 4)")
    else:
        print("  -> 接近但折间不稳：Phase 3 先稳住(固定 best_it / 降 depth)再超参搜索")
    print("-" * 64)
    print("\n请将以上汇总贴回，我据此定 Phase 3 配置。")

    try:
        with open("exp_catboost_ab_result.txt", "w", encoding="utf-8") as f:
            f.write(f"catboost_version={cb.__version__}\n")
            f.write(f"best_iter={best_it} folds={its}\n")
            f.write(f"val_MAE={mt['MAE']:.4f} v6={V6_VAL_MAE}\n")
            f.write(f"val_R2={mt['R2']:.6f} val_Bias={mt['Bias']:.4f}\n")
            f.write(f"midday_MAE={mt['midday_MAE']:.4f}\n")
            f.write(f"fold_maes={fold_maes}\n")
            f.write(f"bands={mt['bands']}\n")
            f.write(f"elapsed_seconds={dt:.1f}\n")
        print("(已写 exp_catboost_ab_result.txt)")
    except Exception as e:
        print(f"(写结果失败: {e})")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
