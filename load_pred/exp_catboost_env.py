# -*- coding: utf-8 -*-
"""
Phase 0：CatBoost GPU 环境打通 + 单模型耗时校准 + 3 个 GPU 坑验证（rsm / quantile / ordered）

目标（见 catboost_migration_plan.md §7 Phase 0）：
  1. 确认 catboost 安装、GPU 可用（task_type="GPU" 能跑通）；
  2. 校准单模型训练耗时（full config: depth=8, 800 iter, RMSE, direct, full train）；
  3. 验证 3 个 CatBoost GPU 关键坑：
       - rsm（列子采样）：GPU 是否生效 / 被忽略 / 报错；
       - quantile loss：GPU 是否可用（分位成员必需）；
       - ordered boosting：GPU 是否可用 + 相对 Plain 的耗时比；
  4. 输出单成员 val MAE（仅 sanity，非 v6 对比——v6=1445.62 含 40 成员+OOF 校正）。

合规：
  - 不修改任何生产脚本；仅 import train.build_dataset / train._time_weights / features.MismatchModel 复用。
  - actual 仅作 target；训练仅 usable mask（<=TRAIN_END=2026-02-28）；val eval-only。
  - 6 条泄露不变量全部保持（特征矩阵与生产完全一致，仅 booster 换 CatBoost）。

运行：python -m load_pred.exp_catboost_env   （从项目根目录，需 data/ 两 CSV 在位）
"""
from __future__ import annotations
import sys
import time
import warnings
import subprocess
import platform

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# 1. catboost 可用性
# --------------------------------------------------------------------------- #
try:
    import catboost as cb
    from catboost import CatBoostRegressor, Pool
    _CB_VER = cb.__version__
except ImportError:
    print("[FATAL] 未安装 catboost。")
    print("        云端安装：pip install catboost  （Linux + CUDA；Python 建议 3.11/3.12）")
    print("        注意：CatBoost GPU 在 Windows 上支持很差，必须用 Linux 云实例。")
    sys.exit(1)

from . import config as C
from .train import build_dataset, usable_mask, _time_weights
from .features import MismatchModel


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _gpu_info() -> None:
    """打印 nvidia-smi 摘要（若可用）。"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            stderr=subprocess.STDOUT, text=True, timeout=10,
        )
        print(f"[env] GPU: {out.strip()}")
    except Exception as e:
        print(f"[env] nvidia-smi 不可用: {e}")


def _gpu_sanity() -> bool:
    """tiny 50-iter GPU fit 探路；失败则早期退出，不浪费重模型时间。"""
    print("[0] GPU sanity（tiny 50-iter fit）...")
    try:
        rng = np.random.default_rng(42)
        m = CatBoostRegressor(
            iterations=50, depth=4, learning_rate=0.1, loss_function="RMSE",
            task_type="GPU", devices="0", verbose=0, allow_writing_files=False,
        )
        m.fit(rng.standard_normal((1000, 10)), rng.standard_normal(1000))
        print("    GPU OK")
        return True
    except Exception as e:
        print(f"    GPU 不可用: {type(e).__name__}: {e}")
        print("    排查：1) nvidia-smi 是否有 GPU；2) CUDA 版本是否与 catboost wheel 匹配；")
        print("          3) 必须 Linux；4) pip install catboost（GPU 含在 Linux wheel，无需编译）。")
        return False


def _check_data() -> None:
    """检查 data/ 两 CSV 是否在位。"""
    for p in (C.LOAD_CSV, C.WEATHER_CSV):
        ok = p.exists()
        print(f"[env] {'OK' if ok else 'MISSING'}  {p}")


def _val_mask(times, actual) -> np.ndarray:
    return ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END))
            & actual.notna()).values


def _fit_one(pool: Pool, params: dict, n_iter: int, seed: int = 42, tag: str = ""):
    """训练单个 CatBoost 模型，返回 (model, seconds, caught_warnings)。"""
    p = dict(params)
    p["iterations"] = n_iter
    p["random_seed"] = seed
    p["verbose"] = 0
    p["allow_writing_files"] = False
    print(f"  [{tag}] 训练中 (iter={n_iter}) ...", flush=True)
    model = CatBoostRegressor(**p)
    caught: list[str] = []
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        model.fit(pool)
        caught = [str(w.message) for w in wlist]
    dt = time.perf_counter() - t0
    print(f"  [{tag}] 完成  耗时={dt:.1f}s", flush=True)
    for c in caught:
        low = c.lower()
        if any(k in low for k in ("rsm", "gpu", "feature", "bootstrap", "grow",
                                   "quantile", "ordered", "device")):
            print(f"      ! {c[:220]}")
    return model, dt, caught


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 64)
    print("Phase 0: CatBoost GPU 环境打通 + rsm/quantile/ordered 验证")
    print(f"catboost {_CB_VER}   python {platform.python_version()}")
    print("=" * 64)

    _gpu_info()
    _check_data()
    if not _gpu_sanity():
        return 1

    # ---- 构建数据集（复用生产管线，保证特征矩阵与 v6 完全一致）----
    print("\n[1] 构建数据集（train.build_dataset + MismatchModel）...")
    times, X, pred_load, actual = build_dataset()
    usable = usable_mask(times, pred_load, actual)
    # 拟合错配模型（仅训练期），加入 pl_weather_residual / solar_mismatch 等需拟合特征
    mm = MismatchModel().fit(X, usable)
    X = mm.transform(X)
    feat_cols = list(X.columns)
    val_m = _val_mask(times, actual)

    # 用 numpy 数组喂 CatBoost（特征名含中文，CatBoost 对特征名有正则限制；位置索引规避，
    # 且为 Phase 2 cat_features=[位置] 前向兼容）
    Xtr = X[usable][feat_cols].to_numpy(dtype=np.float64, copy=False)
    ytr = actual[usable].to_numpy(dtype=np.float64, copy=False)
    wtr = _time_weights(times, usable, C.TRAIN_CONFIG["alpha_w"],
                        pred_load=pred_load,
                        load_gamma=C.TRAIN_CONFIG.get("weight_load_gamma", 0.0))
    Xval = X[val_m][feat_cols].to_numpy(dtype=np.float64, copy=False)
    yval = actual[val_m].to_numpy(dtype=np.float64, copy=False)
    pool = Pool(Xtr, label=ytr, weight=wtr)
    print(f"    特征数={len(feat_cols)}  训练点={int(usable.sum())}  val点={int(val_m.sum())}")

    # CatBoost 基础参数（映射自 v6 LightGBM 配置，见计划 §3.3）
    base_gpu = dict(
        task_type="GPU", devices="0",
        loss_function="RMSE",
        learning_rate=0.03,
        depth=8,                   # ≈ LightGBM num_leaves=255 (2^8=256 叶)
        l2_leaf_reg=4.0,           # = lambda_l2
        bootstrap_type="Bayesian", # GPU 支持 Bayesian/Poisson（Bernoulli 不支持）
        bagging_temperature=1.0,   # ≈ bagging_fraction 机制（需重调，此处先取 1.0）
        boosting_type="Plain",
    )

    # ---- 2. 主计时：full config 800 iter ----
    print("\n[2] 主计时模型（full config, 800 iter, RMSE, direct, Plain）...")
    m_main, t_main, _ = _fit_one(pool, base_gpu, n_iter=800, tag="main-RMSE-800")
    val_pred = m_main.predict(Xval)
    val_mae = float(np.mean(np.abs(val_pred - yval)))
    print(f"    单成员 val MAE = {val_mae:.1f} MW  (仅 sanity；v6=1445.62 含 40 成员+OOF 校正，不可直接比)")

    # ---- 3. rsm 探针：同 seed 下 rsm=0.5 vs rsm=1.0，预测相同->被忽略；不同->生效 ----
    print("\n[3] rsm 探针（300 iter；rsm=0.5 vs rsm=1.0）...")
    m_rsm10, _, _ = _fit_one(pool, dict(base_gpu, rsm=1.0), n_iter=300, tag="rsm=1.0")
    pr1 = m_rsm10.predict(Xval)
    rsm_status: str
    rsm_diff = float("nan")
    try:
        m_rsm05, _, _ = _fit_one(pool, dict(base_gpu, rsm=0.5), n_iter=300, tag="rsm=0.5")
        pr5 = m_rsm05.predict(Xval)
        rsm_diff = float(np.mean(np.abs(pr5 - pr1)))
        if rsm_diff > 1.0:
            rsm_status = f"生效（预测差异 {rsm_diff:.2f} MW）"
        else:
            rsm_status = f"被忽略（预测差异仅 {rsm_diff:.4f} MW，等同 rsm=1.0）"
    except Exception as e:
        rsm_status = f"GPU 报错({type(e).__name__}) -> 需 grow_policy=Lossguide 或集成层手动列子采样"
        print(f"      rsm=0.5 失败: {e}")
    print(f"    => rsm on GPU: {rsm_status}")

    # ---- 4. quantile 探针 ----
    print("\n[4] quantile 探针（Quantile:alpha=0.45, 300 iter, GPU）...")
    quantile_ok = False
    try:
        m_q, _, _ = _fit_one(pool, dict(base_gpu, loss_function="Quantile:alpha=0.45"),
                             n_iter=300, tag="quantile-0.45")
        pq = m_q.predict(Xval)
        quantile_ok = bool(np.isfinite(pq).all() and (pq.max() - pq.min()) > 100.0)
        print(f"    => quantile GPU: {'OK' if quantile_ok else '异常'}  "
              f"val pred range=[{pq.min():.0f}, {pq.max():.0f}]")
    except Exception as e:
        print(f"    => quantile GPU 失败: {type(e).__name__}: {e}")

    # ---- 5. ordered boosting 探针 ----
    print("\n[5] ordered boosting 探针（300 iter；Plain vs Ordered 计时）...")
    m_plain, t_plain, _ = _fit_one(pool, dict(base_gpu, boosting_type="Plain"),
                                   n_iter=300, tag="Plain-300")
    ordered_ok = False
    ratio = float("nan")
    try:
        m_ord, t_ord, _ = _fit_one(pool, dict(base_gpu, boosting_type="Ordered"),
                                   n_iter=300, tag="Ordered-300")
        ordered_ok = True
        ratio = t_ord / max(t_plain, 1e-6)
        print(f"    => Ordered GPU: OK  耗时比 Ordered/Plain = {ratio:.2f}x "
              f"(Ordered={t_ord:.1f}s, Plain={t_plain:.1f}s)")
    except Exception as e:
        print(f"    => Ordered GPU 失败: {type(e).__name__}: {e}")

    # ---- 汇总 ----
    est_serial = t_main * 160 / 60.0
    est_par8 = t_main * 160 / 8 / 60.0
    print("\n" + "=" * 64)
    print("Phase 0 汇总")
    print("=" * 64)
    print(f"catboost {_CB_VER}   task_type=GPU  可用")
    print(f"主模型(full 800iter RMSE direct): {t_main:.1f}s/模型")
    print(f"  -> 40 成员 × 4 折 = 160 模型：串行 ~{est_serial:.0f} min，8 路并发 ~{est_par8:.0f} min")
    print(f"rsm 列子采样 on GPU: {rsm_status}")
    print(f"quantile loss on GPU: {'OK' if quantile_ok else '失败'}")
    print(f"ordered boosting on GPU: {'OK' if ordered_ok else '失败'}  耗时比={ratio:.2f}x")
    print(f"单成员 val MAE(sanity) = {val_mae:.1f} MW  (v6=1445.62，不可直接比)")
    print("=" * 64)
    print("\n请将以上输出（或 exp_catboost_env_result.txt）贴回，以决定 Phase 1 配置。")

    # ---- 写结果文件 ----
    try:
        with open("exp_catboost_env_result.txt", "w", encoding="utf-8") as f:
            f.write(f"catboost_version={_CB_VER}\n")
            f.write(f"python_version={platform.python_version()}\n")
            f.write(f"main_train_seconds={t_main:.2f}\n")
            f.write(f"single_member_val_mae={val_mae:.2f}\n")
            f.write(f"rsm_status={rsm_status}\n")
            f.write(f"rsm_pred_diff_mw={rsm_diff:.4f}\n")
            f.write(f"quantile_on_gpu_ok={quantile_ok}\n")
            f.write(f"ordered_on_gpu_ok={ordered_ok}\n")
            f.write(f"ordered_plain_ratio={ratio:.3f}\n")
            f.write(f"feat_cols={len(feat_cols)} train_n={int(usable.sum())} val_n={int(val_m.sum())}\n")
            f.write(f"est_full_pipeline_min_serial={est_serial:.1f}\n")
            f.write(f"est_full_pipeline_min_par8={est_par8:.1f}\n")
        print("(已写 exp_catboost_env_result.txt)")
    except Exception as e:
        print(f"(写结果文件失败: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
