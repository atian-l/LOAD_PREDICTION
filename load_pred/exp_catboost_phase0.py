# -*- coding: utf-8 -*-
"""
Phase 0：CatBoost 欠拟合可解性 go/no-go 诊断（catboost_underfit_plan.md §4）

目的：验证核心假设"拟合改善能否迁移到 val"。
  - 测各配置 train_raw（拟合程度）vs val_raw（泛化）趋势
  - 不需 OOF 校正估计（3 折重训是最慢环节，省掉 -> 比 hp 快约 4x），只测 raw + debiased(val)

判定矩阵（plan §4）：
  train_raw 随 best_it 降 但 val_raw/debiased 不降 -> 拟合不迁移 -> 停止(欠拟合不可解)
  train_raw 降 且 val_raw 跟降                       -> 拟合迁移   -> Phase 1
  Lossguide train_raw <1300                          -> leaf-wise 解决欠拟合 -> Phase 1-B
  Lossguide train_raw 仍高(>1400)                    -> leaf-wise 也拟合不好 -> 对称树路径
  bootstrap=No train_raw 显著降                      -> bootstrap 是根因 -> Phase 1-C

参考基准（fit_diag）：
  CatBoost l2_8 best_it=80  train_raw=1517 / val_raw=1533 / debiased=1483
  LightGBM  v6  best_it=80  train_raw=1147 / val_raw=1512 ; v6 val(含校正)=1445.62

合规：不修改任何生产脚本；仅 import train/features 复用；actual 仅 target/eval；6 不变量保持。
运行：python -m load_pred.exp_catboost_phase0   （4090 上约 10~15 min，无 OOF 故快于 hp）
"""
from __future__ import annotations
import sys
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from . import config as C
from .train import build_dataset, usable_mask, _time_weights
from .features import MismatchModel, MosModel
from .exp_catboost_ab import _arr, V6_VAL_MAE
from . import exp_catboost_ab as ab

MIN_DATA_IN_LEAF = 200  # 对齐 v6 / hp

# (tag, depth, lr, l2, bootstrap, bagging_temp, subsample, grow_policy, max_leaves, best_it)
CONFIGS = [
    ("sym_bi80",    8, 0.03, 8.0, "Bayesian", 1.0, None, "SymmetricTree", None, 80),
    ("sym_bi160",   8, 0.03, 8.0, "Bayesian", 1.0, None, "SymmetricTree", None, 160),
    ("sym_bi300",   8, 0.03, 8.0, "Bayesian", 1.0, None, "SymmetricTree", None, 300),
    ("sym_bi500",   8, 0.03, 8.0, "Bayesian", 1.0, None, "SymmetricTree", None, 500),
    ("loss_bi80",   8, 0.03, 8.0, "Bayesian", 1.0, None, "Lossguide", 255, 80),
    ("loss_bi300",  8, 0.03, 8.0, "Bayesian", 1.0, None, "Lossguide", 255, 300),
    ("no_bs80",     8, 0.03, 8.0, "No", None, None, "SymmetricTree", None, 80),
]


# --------------------------------------------------------------------------- #
# 参数化训练原语（支持 bootstrap_type / grow_policy；自包含，不改 hp）
# --------------------------------------------------------------------------- #
def _params(loss, seed, iters, hp):
    p = dict(
        task_type="GPU", devices="0", loss_function=loss,
        learning_rate=hp["lr"], depth=hp["depth"], l2_leaf_reg=hp["l2"],
        bootstrap_type=hp["bootstrap"], boosting_type="Plain",
        random_seed=seed, verbose=0, allow_writing_files=False, iterations=iters,
        grow_policy=hp["grow_policy"], min_data_in_leaf=MIN_DATA_IN_LEAF,
    )
    if hp["bootstrap"] == "Bayesian":
        p["bagging_temperature"] = hp["bagging_temp"]
    elif hp["bootstrap"] in ("Bernoulli", "MVS"):
        p["subsample"] = hp["subsample"]
    # bootstrap_type="No" 无 bagging 参数
    if hp["grow_policy"] == "Lossguide":
        p["max_leaves"] = hp["max_leaves"]
    return p


def _fit(Xtr, ytr, wtr, loss, seed, iters, hp):
    pool = Pool(Xtr, label=ytr, weight=wtr)
    m = CatBoostRegressor(**_params(loss, seed, iters, hp))
    m.fit(pool)
    return m


def _train_ensemble(X, actual, anchor, mask, cfg, best_it, feat_cols, hp):
    y_res = actual - anchor
    Xtr = _arr(X[mask], feat_cols)
    wtr = _time_weights(ab.times_global, mask, cfg["alpha_w"],
                        pred_load=ab.pred_load_global,
                        load_gamma=cfg.get("weight_load_gamma", 0.0))
    ytr_dir = actual[mask].to_numpy(np.float64)
    ytr_res = y_res[mask].to_numpy(np.float64)
    members = []
    for residual in cfg["residual_modes"]:
        y = ytr_res if residual else ytr_dir
        for obj in cfg["objectives"]:
            alphas = cfg["quantile_alphas"] if obj == "quantile" else [None]
            for qa in alphas:
                loss = f"Quantile:alpha={qa}" if obj == "quantile" else "RMSE"
                for s in cfg["seeds"]:
                    m = _fit(Xtr, y, wtr, loss, s, best_it, hp)
                    members.append((m, bool(residual)))
    return members


def _ensemble_raw(members, X, anchor_vals, feat_cols, shrinkage):
    Xarr = _arr(X, feat_cols)
    mp = np.empty((len(members), len(X)), dtype=float)
    for i, (m, is_res) in enumerate(members):
        raw = m.predict(Xarr)
        mp[i] = anchor_vals + raw if is_res else raw
    ens = np.median(mp, axis=0)
    return anchor_vals + shrinkage * (ens - anchor_vals)


def _mae(pred, actual):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(a) & np.isfinite(p)
    return float(np.mean(np.abs(p[m] - a[m]))) if m.sum() else float("nan")


def _r2(pred, actual):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    m = np.isfinite(a) & np.isfinite(p)
    a, p = a[m], p[m]
    ss_res = float(np.sum((p - a) ** 2))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _run_config(tag, hp, best_it, X, actual, anchor, usable, cfg, feat_cols, val_m):
    ts = time.perf_counter()
    members = _train_ensemble(X, actual, anchor, usable, cfg, best_it, feat_cols, hp)
    train_raw = _ensemble_raw(members, X[usable], anchor[usable].values, feat_cols, cfg["shrinkage"])
    val_raw = _ensemble_raw(members, X[val_m], anchor[val_m].values, feat_cols, cfg["shrinkage"])
    tr_a = actual[usable].values
    va_a = actual[val_m].values
    m_tr = _mae(train_raw, tr_a)
    m_va = _mae(val_raw, va_a)
    r2_tr = _r2(train_raw, tr_a)
    r2_va = _r2(val_raw, va_a)
    err = val_raw - va_a
    debiased = float(np.mean(np.abs(err - err.mean())))
    dt = time.perf_counter() - ts
    return {"tag": tag, "best_it": best_it, "hp": hp, "train_raw": m_tr,
            "val_raw": m_va, "debiased": debiased, "R2_tr": r2_tr, "R2_va": r2_va,
            "gap": m_tr - m_va, "dt": dt}


# --------------------------------------------------------------------------- #
def main() -> int:
    t0 = time.perf_counter()
    print("=" * 78)
    print("Phase 0: CatBoost 欠拟合可解性 go/no-go (train_raw vs val_raw 趋势; 无 OOF)")
    print("  参考: CatBoost l2_8 bi80 train_raw=1517 / val_raw=1533 / debiased=1483")
    print(f"        LightGBM v6 bi80 train_raw=1147 / val_raw=1512 ; v6 val(含校正)={V6_VAL_MAE}")
    print("=" * 78)

    print("[1] 构建数据集...")
    times, X, pred_load, actual = build_dataset()
    ab.times_global, ab.pred_load_global = times, pred_load
    usable = usable_mask(times, pred_load, actual)
    mm = MismatchModel().fit(X, usable)
    X = mm.transform(X)
    mos_model = MosModel(cols=C.TRAIN_CONFIG["mos"]["cols"],
                         alpha=C.TRAIN_CONFIG["mos"]["alpha"]).fit(X, actual, usable)
    anchor = pd.Series(mos_model.transform(X), index=X.index)
    feat_cols = list(X.columns)
    val_m = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
             & actual.notna()).values
    cfg = C.TRAIN_CONFIG
    print(f"    特征数={len(feat_cols)}  训练点={int(usable.sum())}  val点={int(val_m.sum())}  配置数={len(CONFIGS)}")

    print(f"\n[2] 逐配置训练 40 成员 + 测 train_raw / val_raw (无校正) ...")
    rows = []
    for tag, depth, lr, l2, bs, bt, ss, gp, ml, best_it in CONFIGS:
        hp = {"depth": depth, "lr": lr, "l2": l2, "bootstrap": bs,
              "bagging_temp": bt, "subsample": ss, "grow_policy": gp, "max_leaves": ml}
        try:
            r = _run_config(tag, hp, best_it, X, actual, anchor, usable, cfg, feat_cols, val_m)
            rows.append(r)
            print(f"  {tag:12s} bi={best_it:>3} train_raw={r['train_raw']:7.1f} val_raw={r['val_raw']:7.1f} "
                  f"debiased={r['debiased']:7.1f} gap={r['gap']:+6.1f} R²_tr={r['R2_tr']:.4f} ({r['dt']:.0f}s)")
        except Exception as e:
            ename = type(e).__name__
            msg = str(e).splitlines()[0][:90]
            print(f"  {tag:12s} FAIL ({ename}: {msg})")

    if not rows:
        print("\n所有配置失败。")
        return 1

    # ---- 对比表 ----
    print("\n" + "=" * 78)
    print("Phase 0 对比表（raw 无校正；参考 v6 train_raw≈1147 / val_raw≈1512）")
    print("=" * 78)
    print(f"{'tag':12} {'best_it':>7} {'train_raw':>10} {'val_raw':>9} {'debiased':>9} {'gap':>7} {'R²_tr':>7} {'policy':>13}")
    for r in rows:
        print(f"{r['tag']:12} {r['best_it']:>7} {r['train_raw']:>10.1f} {r['val_raw']:>9.1f} "
              f"{r['debiased']:>9.1f} {r['gap']:>+7.1f} {r['R2_tr']:>7.4f} {r['hp']['grow_policy']:>13}")

    # ---- 趋势分析（核心判定）----
    print("\n" + "-" * 78)
    print("趋势分析（核心 go/no-go）：")
    sym = [r for r in rows if r["tag"].startswith("sym_bi")]
    sym_sorted = sorted(sym, key=lambda r: r["best_it"])
    verdict_sym = "数据不足"
    if len(sym_sorted) >= 2:
        first, last = sym_sorted[0], sym_sorted[-1]
        d_tr = last["train_raw"] - first["train_raw"]
        d_va = last["val_raw"] - first["val_raw"]
        d_db = last["debiased"] - first["debiased"]
        print(f"  SymmetricTree best_it {first['best_it']}->{last['best_it']}:")
        print(f"    train_raw {first['train_raw']:.0f}->{last['train_raw']:.0f} (Δ{d_tr:+.0f})")
        print(f"    val_raw   {first['val_raw']:.0f}->{last['val_raw']:.0f} (Δ{d_va:+.0f})")
        print(f"    debiased  {first['debiased']:.0f}->{last['debiased']:.0f} (Δ{d_db:+.0f})")
        train_descends = d_tr < -50
        val_follows = d_va < -20
        if train_descends and not val_follows:
            verdict_sym = "拟合不迁移"
            print(f"    -> train_raw 降({d_tr:+.0f}) 但 val_raw 不跟降({d_va:+.0f}) -> 【拟合不迁移】")
        elif train_descends and val_follows:
            verdict_sym = "拟合迁移"
            print(f"    -> train_raw 降({d_tr:+.0f}) 且 val_raw 跟降({d_va:+.0f}) -> 【拟合迁移】")
        else:
            verdict_sym = "拟合未改善"
            print(f"    -> train_raw 未显著降({d_tr:+.0f}) -> 容量增大未改善拟合（CatBoost 拟合能力本质受限）")

    # Lossguide vs sym
    loss = [r for r in rows if r["tag"].startswith("loss_bi")]
    sym80 = next((r for r in rows if r["tag"] == "sym_bi80"), None)
    verdict_loss = "N/A"
    if loss and sym80:
        loss80 = next((r for r in loss if r["best_it"] == 80), None)
        if loss80:
            d = loss80["train_raw"] - sym80["train_raw"]
            print(f"\n  Lossguide vs SymmetricTree (bi80): train_raw {loss80['train_raw']:.0f} vs {sym80['train_raw']:.0f} (Δ{d:+.0f})")
            if loss80["train_raw"] < 1300:
                verdict_loss = "leaf-wise 解决欠拟合"
                print(f"    -> Lossguide train_raw<1300 -> leaf-wise 显著改善拟合 -> 【Phase 1-B 路径】")
            elif d < -100:
                verdict_loss = "leaf-wise 部分改善"
                print(f"    -> Lossguide train_raw 降{d:+.0f} 但未<1300 -> 部分改善 -> Phase 1-B 可试加强正则")
            else:
                verdict_loss = "leaf-wise 无改善"
                print(f"    -> Lossguide train_raw 未降({d:+.0f}) -> leaf-wise 也拟合不好 -> 对称树路径")

    # bootstrap=No
    nobs = next((r for r in rows if r["tag"] == "no_bs80"), None)
    verdict_bs = "N/A"
    if nobs and sym80:
        d = nobs["train_raw"] - sym80["train_raw"]
        print(f"\n  bootstrap=No vs Bayesian (bi80): train_raw {nobs['train_raw']:.0f} vs {sym80['train_raw']:.0f} (Δ{d:+.0f})")
        if d < -100:
            verdict_bs = "bootstrap 是根因"
            print(f"    -> bootstrap=No train_raw 降{d:+.0f} -> Bayesian bootstrap 是欠拟合根因 -> 【Phase 1-C 路径】")
        else:
            verdict_bs = "bootstrap 非根因"
            print(f"    -> bootstrap=No train_raw 未降({d:+.0f}) -> bootstrap 非根因")

    # ---- 总判定 ----
    print("\n" + "=" * 78)
    print("总判定（go/no-go）：")
    print(f"  对称树拟合迁移: {verdict_sym}")
    print(f"  Lossguide 拟合: {verdict_loss}")
    print(f"  bootstrap 根因: {verdict_bs}")
    print("-" * 78)
    go = (verdict_sym == "拟合迁移"
          or verdict_loss == "leaf-wise 解决欠拟合"
          or verdict_bs == "bootstrap 是根因")
    if not go and verdict_sym == "拟合不迁移":
        print("  -> 【GO/NO-GO = NO-GO】拟合改善不迁移到 val，无结构/bootstrap 杠杆解决欠拟合")
        print("     欠拟合不可解 -> 维持 CatBoost 路径终止结论，回 v6 + 等新数据")
        print("     （catboost_underfit_plan.md Phase 0 判定矩阵：拟合不迁移 -> 停止）")
    elif go:
        print("  -> 【GO/NO-GO = GO】存在拟合迁移路径 -> 进 Phase 1 优化")
        if verdict_loss == "leaf-wise 解决欠拟合":
            print("     优先 Phase 1-B: Lossguide + 强正则控过拟合")
        if verdict_bs == "bootstrap 是根因":
            print("     优先 Phase 1-C: bootstrap=No/MVS + best_it 增大")
        if verdict_sym == "拟合迁移":
            print("     Phase 1-A: 增大 best_it + recency 校正控 bias 漂移")
    else:
        print("  -> 拟合未改善（容量受限）-> CatBoost 拟合能力本质弱于 LightGBM")
        print("     欠拟合难逆 -> 倾向 NO-GO，但可 cheap 试 Phase 1 单点确认")
    print("=" * 78)

    try:
        with open("exp_catboost_phase0_result.txt", "w", encoding="utf-8") as f:
            f.write(f"v6={V6_VAL_MAE} min_data_in_leaf={MIN_DATA_IN_LEAF}\n")
            f.write("ref: CatBoost l2_8 bi80 train_raw=1517/val_raw=1533/debiased=1483; LGB train_raw=1147/val_raw=1512\n")
            f.write("tag\tbest_it\ttrain_raw\tval_raw\tdebiased\tgap\tR2_tr\tpolicy\tbootstrap\n")
            for r in rows:
                f.write(f"{r['tag']}\t{r['best_it']}\t{r['train_raw']:.4f}\t{r['val_raw']:.4f}\t"
                        f"{r['debiased']:.4f}\t{r['gap']:.4f}\t{r['R2_tr']:.4f}\t"
                        f"{r['hp']['grow_policy']}\t{r['hp']['bootstrap']}\n")
            f.write(f"\nverdict_sym={verdict_sym}\nverdict_loss={verdict_loss}\nverdict_bs={verdict_bs}\n")
        print("(已写 exp_catboost_phase0_result.txt)")
    except Exception as e:
        print(f"(写结果失败: {e})")
    print(f"\n总耗时 {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        sys.exit(main())
