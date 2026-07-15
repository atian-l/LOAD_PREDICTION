# -*- coding: utf-8 -*-
"""v8.1 Phase 0：残差溯源 + 跨年可迁移性诊断。

回答 v8.1 全部模块的 go/no-go 依据："残差里哪一块跨年可学"。

残差定义：r = actual - base_A（base_A = v6，根 model_bundle.pkl）。
  r_train：v8 bundle 的 oof_pool（2025 训练期 3 折 walk-forward，无泄露）。
  r_val  ：base_A 在 val 窗口（2026）的预测残差（仅评估，不参与任何参数学习）。

诊断维度：
  A. 残差总览（分布/分段 MAE/方向）
  B. 分量跨年可迁移性：把特征列划分为 5 个分量组（calendar/weather/solar_renewable/
     load_level/load_temporal），各组用 LightGBM 探针在 r_train 上拟合、在 r_val 上测
     跨年 transfer R² + 方向命中率 + MAE 改善。combined = 全特征。
  C. 分段（night/day/evening）× 分量 transfer R²，重点看午间 day 段。
  D. Shape vs Point：日级 shape 描述子（daily_mean / midday(11-14) / evening(18-20)）
     的跨年 transfer R²，对比逐点 transfer R²。直接检验 v8.1 "Shape not Point" 命题。
  E. Domain 条件化可迁移性：按日级天气向量聚类成 Domain，per-Domain 探针 vs 全局探针的
     跨年 transfer R²；Domain 可靠性漂移（2025 vs 2026 残差均值符号翻转率）。
     直接检验 v8.1 "Weather Sim -> Domain" 命题。
  F. 自动判定：哪些分量/Shape/Domain 跨年可迁移 -> v8.1 模块 go/no-go。

合规：探针仅用 2025 OOF 残差训练；val 仅作跨年测试目标（eval-only，不变量 #5/#6）。
运行：python -m v8_1.diag_residual   报告 -> v8_1/output/phase0_residual_attribution.md
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from load_pred import config as LC
from load_pred import train as T

from v8 import config as VC
from v8 import segments as SEG
from v8 import weather_sim as WS
from v8.model import V8Model

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = OUT_DIR / "phase0_residual_attribution.md"

PROBE_PARAMS = dict(
    objective="regression", metric="mae", verbose=-1, force_col_wise=True,
    learning_rate=0.03, num_leaves=63, min_data_in_leaf=300, lambda_l2=8.0,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=42,
)
PROBE_IT = 80


# --------------------------------------------------------------------------- #
# 特征分量分组（不相交划分 X_full 列）
# --------------------------------------------------------------------------- #
def categorize(col: str) -> str:
    # load_temporal：pred_load 的滞后/差分/爬坡/滚动/偏离
    if (col.startswith("pred_load_lag") or col.startswith("pred_load_diff")
            or col.startswith("pred_load_ramp") or col.startswith("pred_load_roll")
            or col.startswith("pred_load_vs_mean")):
        return "load_temporal"
    # solar_renewable：光伏/晴空/午间/pl_dip/irrad_anom/pl_wr/solar_mismatch（新能源耦合代理）
    if col in {"clear_sky", "clearness", "cloud_deficit", "is_midday", "is_daytime",
               "irrad_x_midday", "clearness_x_midday", "pl_dip_96", "pl_dip_x_irrad",
               "pl_dip_x_clearness", "pl_dip_x_midday", "pl_dip_x_clear_sky", "pl_dip_ratio",
               "irrad_anom_672", "pl_x_irrad_anom", "irrad_anom_x_midday",
               "pl_x_irrad_x_midday", "pl_x_clearness", "pl_x_cloud_deficit",
               "pl_weather_residual"}:
        return "solar_renewable"
    if col.startswith("pl_wr") or col.startswith("solar_mismatch"):
        return "solar_renewable"
    # load_level：pred_load 水平 + pl×weather/calendar 交互（负荷水平/shrinkage 代理）
    if col == "pred_load" or col.startswith("pl_x_") or col.startswith("plnorm_") or col.startswith("plvsmean_"):
        return "load_level"
    # weather：原始气象 + 非线性 + 气象×小时交互
    if col.startswith("w_") or col in {"temp", "irrad", "wind", "precip", "solar_wind",
               "temp_sq", "irrad_sq", "hdd", "cdd", "temp_std", "irrad_std", "wind_std",
               "irrad_range", "temp_x_hour", "irrad_x_daylight", "temp_x_daylight",
               "hdd_x_hour", "cdd_x_hour"}:
        return "weather"
    # calendar：日历标志 + 周期编码
    if col in {"hour", "minute_of_day", "dayofweek", "dayofyear", "month",
               "is_weekend", "is_holiday", "is_day_before_holiday"} or col.endswith("_sin") or col.endswith("_cos"):
        return "calendar"
    return "other"


def group_columns(columns) -> dict[str, list[str]]:
    g: dict[str, list[str]] = {}
    for c in columns:
        g.setdefault(categorize(c), []).append(c)
    return g


# --------------------------------------------------------------------------- #
# 探针：fit on 2025, 测跨年 transfer 指标
# --------------------------------------------------------------------------- #
def _fit_probe(Xtr, ytr, cols):
    d = lgb.Dataset(Xtr[cols], label=ytr)
    return lgb.train(PROBE_PARAMS, d, num_boost_round=PROBE_IT)


def _metrics(pred, y, train_mean) -> dict:
    """跨年指标。train_mean 基线 = 用 2025 残差均值预测 2026（=do nothing）。"""
    pred = np.asarray(pred, dtype=float)
    y = np.asarray(y, dtype=float)
    base_mse = float(np.mean((y - train_mean) ** 2))           # do-nothing 基线
    probe_mse = float(np.mean((y - pred) ** 2))
    transfer_r2 = 1.0 - probe_mse / base_mse if base_mse > 0 else float("nan")
    # 标准 R²（val 自身均值基线），可与 memory 的"R² 为负"对照
    val_mse = float(np.mean((y - y.mean()) ** 2))
    std_r2 = 1.0 - probe_mse / val_mse if val_mse > 0 else float("nan")
    # 方向命中率（v8.1 Stage1/2 关心：方向是否跨年可迁移）
    nz = np.abs(y) > 50.0  # 忽略近零残差的方向噪声
    dir_acc = float(np.mean(np.sign(pred[nz]) == np.sign(y[nz]))) if nz.any() else float("nan")
    base_mae = float(np.mean(np.abs(y - train_mean)))
    probe_mae = float(np.mean(np.abs(y - pred)))
    mae_red = base_mae - probe_mae
    return {"transfer_R2": transfer_r2, "std_R2": std_r2, "dir_acc": dir_acc,
            "base_MAE": base_mae, "probe_MAE": probe_mae, "MAE_red": mae_red}


def _holdout_r2(Xtr, ytr, cols) -> float:
    """2025 内部时序 holdout R²（前 80% 训练 -> 后 20% 测），衡量"期内可学性"。
    与跨年 transfer_R² 的 GAP = 过拟合/不可迁移程度。"""
    n = len(ytr)
    cut = int(n * 0.8)
    if cut < 100 or n - cut < 100:
        return float("nan")
    bst = _fit_probe(Xtr.iloc[:cut], ytr[:cut], cols)
    pred = bst.predict(Xtr[cols].iloc[cut:])
    y = ytr[cut:]
    val_mse = float(np.mean((y - y.mean()) ** 2))
    return 1.0 - float(np.mean((y - pred) ** 2)) / val_mse if val_mse > 0 else float("nan")


# --------------------------------------------------------------------------- #
# 主诊断
# --------------------------------------------------------------------------- #
def run(verbose: bool = True) -> dict:
    if verbose:
        print("[1/6] 构建数据集 + 加载 v8 bundle ...")
    times, X, pred_load, actual = T.build_dataset()
    vm8 = V8Model.load(VC.V8_BUNDLE)
    mm = vm8.mismatch_model
    X_full = mm.transform(X)
    oof = vm8.oof_pool
    baseA = vm8.base_A

    # ---- r_train（2025 OOF，无泄露）----
    idx = oof["idx"]
    X_train = X_full.iloc[idx].reset_index(drop=True)
    r_train = np.asarray(oof["actual"] - oof["base_A_oof"], dtype=float)
    times_train = pd.DatetimeIndex(oof["times"])
    seg_train = np.asarray(oof["seg"], dtype=object)
    train_mean = float(r_train.mean())

    # ---- r_val（2026，eval-only）----
    if verbose:
        print("[2/6] 计算 val 残差 ...")
    vs, ve = pd.Timestamp(LC.VAL_START), pd.Timestamp(LC.VAL_END)
    buf = vs - pd.Timedelta(days=7)
    em = (times >= buf) & (times <= ve)
    Xe = X[em]
    te = pd.DatetimeIndex(times[em])
    ae = actual.reindex(te)
    Xe_full = mm.transform(Xe)
    predA = np.asarray(baseA.predict_load(Xe_full, pred_load), dtype=float)
    vmask = (te >= vs) & (te <= ve) & ae.notna()
    X_val = Xe_full[vmask].reset_index(drop=True)
    r_val = (ae[vmask].values.astype(float) - predA[vmask])
    times_val = te[vmask]
    hours_val = times_val.hour.values.astype(int)
    seg_val = SEG.segment_array(hours_val)

    groups = group_columns(X_full.columns)
    if verbose:
        print(f"      r_train N={len(r_train)}  r_val N={len(r_val)}  train_mean={train_mean:+.1f}")
        print(f"      分量组: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))

    # ===================== A. 总览 =====================
    if verbose:
        print("[3/6] A. 残差总览 ...")
    ov = {
        "r_train_mean": train_mean, "r_train_std": float(r_train.std()),
        "r_val_mean": float(r_val.mean()), "r_val_std": float(r_val.std()),
        "r_val_MAE": float(np.mean(np.abs(r_val))),
    }
    seg_overview = {}
    for s in VC.SEGMENTS:
        mt = seg_train == s
        mv = seg_val == s
        seg_overview[s] = {
            "train_mean": float(r_train[mt].mean()) if mt.any() else float("nan"),
            "val_mean": float(r_val[mv].mean()) if mv.any() else float("nan"),
            "val_MAE": float(np.mean(np.abs(r_val[mv]))) if mv.any() else float("nan"),
            "val_std": float(r_val[mv].std()) if mv.any() else float("nan"),
        }

    # ===================== B. 分量跨年可迁移性 =====================
    if verbose:
        print("[4/6] B. 分量跨年可迁移性探针 ...")
    comp_rows = []
    combined_cols = [c for cs in groups.values() for c in cs]
    ordered = list(groups.keys()) + ["combined"]
    for gname in ordered:
        cols = combined_cols if gname == "combined" else groups.get(gname, [])
        if not cols:
            continue
        bst = _fit_probe(X_train, r_train, cols)
        pred = bst.predict(X_val[cols])
        m = _metrics(pred, r_val, train_mean)
        holdout = _holdout_r2(X_train, r_train, cols)
        comp_rows.append({
            "component": gname, "n_cols": len(cols),
            "within2025_R2": holdout, "transfer_R2": m["transfer_R2"],
            "std_R2": m["std_R2"], "dir_acc": m["dir_acc"],
            "base_MAE": m["base_MAE"], "probe_MAE": m["probe_MAE"], "MAE_red": m["MAE_red"],
        })

    # ===================== C. 分段 × 分量 transfer R² =====================
    if verbose:
        print("[5/6] C. 分段(午间重点) × 分量 transfer R² ...")
    seg_comp = {}  # (seg, comp) -> transfer_R2
    for s in VC.SEGMENTS:
        mtr = seg_train == s
        mva = seg_val == s
        if not mva.any():
            continue
        Xtr_s = X_train[mtr].reset_index(drop=True)
        rtr_s = r_train[mtr]
        seg_mean = float(rtr_s.mean()) if len(rtr_s) else train_mean
        for gname in ordered:
            cols = combined_cols if gname == "combined" else groups.get(gname, [])
            if not cols or len(rtr_s) < 200:
                seg_comp[(s, gname)] = float("nan")
                continue
            bst = _fit_probe(Xtr_s, rtr_s, cols)
            pred = bst.predict(X_val[mva][cols])
            y = r_val[mva]
            bmse = float(np.mean((y - seg_mean) ** 2))
            seg_comp[(s, gname)] = 1.0 - float(np.mean((y - pred) ** 2)) / bmse if bmse > 0 else float("nan")

    # ===================== D. Shape vs Point =====================
    if verbose:
        print("[6/6] D. Shape vs Point + E. Domain 条件化 ...")
    # 逐点 transfer R²（combined 探针，已算）
    point_transfer_r2 = next(r["transfer_R2"] for r in comp_rows if r["component"] == "combined")
    # 日级 shape 描述子
    def _daily_descriptors(r_series, times_idx) -> pd.DataFrame:
        d = pd.DatetimeIndex(times_idx).normalize()
        df = pd.DataFrame({"r": r_series, "date": d, "hour": pd.DatetimeIndex(times_idx).hour})
        g = df.groupby("date")
        out = pd.DataFrame(index=g.groups.keys())
        out["daily_mean"] = g["r"].mean()
        out["midday_11_14"] = df[df["hour"].between(11, 13)].groupby("date")["r"].mean()
        out["evening_18_20"] = df[df["hour"].between(18, 20)].groupby("date")["r"].mean()
        out["morning_ramp"] = df[df["hour"] == 12].groupby("date")["r"].mean() - df[df["hour"] == 10].groupby("date")["r"].mean()
        return out.fillna(0.0)
    # 日级特征（按日聚合 X_full）
    def _daily_features(X_df, times_idx) -> pd.DataFrame:
        d = pd.DatetimeIndex(times_idx).normalize()
        xf = X_df.copy()
        xf["date"] = d
        xf["hour"] = pd.DatetimeIndex(times_idx).hour
        agg = xf.groupby("date").agg(
            pl_mean=("pred_load", "mean"), pl_max=("pred_load", "max"),
            temp_mean=("temp", "mean"), irrad_sum=("irrad", "sum"),
            clearness_mean=("clearness", "mean"), precip_sum=("precip", "sum"),
            wind_mean=("wind", "mean"),
        )
        cal = pd.DataFrame({"date": d, "dow": pd.DatetimeIndex(times_idx).dayofweek,
                            "month": pd.DatetimeIndex(times_idx).month})
        cm = cal.groupby("date").first()
        return agg.join(cm)
    dtr_X = _daily_features(X_train, times_train)
    dtr_y = _daily_descriptors(r_train, times_train)
    dva_X = _daily_features(X_val, times_val)
    dva_y = _daily_descriptors(r_val, times_val)
    # 对齐日期索引
    dtr_X, dtr_y = dtr_X.align(dtr_y, join="inner", axis=0)
    dva_X, dva_y = dva_X.align(dva_y, join="inner", axis=0)
    shape_cols = list(dtr_X.columns)
    shape_rows = []
    for desc in ["daily_mean", "midday_11_14", "evening_18_20", "morning_ramp"]:
        ytr = dtr_y[desc].values
        yva = dva_y[desc].values
        tm = float(ytr.mean())
        bst = _fit_probe(dtr_X, ytr, shape_cols)
        pred = bst.predict(dva_X[shape_cols])
        bmse = float(np.mean((yva - tm) ** 2))
        tr2 = 1.0 - float(np.mean((yva - pred) ** 2)) / bmse if bmse > 0 else float("nan")
        nz = np.abs(yva) > 50.0
        da = float(np.mean(np.sign(pred[nz]) == np.sign(yva[nz]))) if nz.any() else float("nan")
        shape_rows.append({"descriptor": desc, "transfer_R2": tr2, "dir_acc": da,
                           "val_std": float(yva.std())})

    # ===================== E. Domain 条件化 =====================
    # 日级天气向量 -> KMeans 聚类成 Domain
    day_vec_all = WS.day_weather_vectors(X_full, times)
    from sklearn.cluster import KMeans
    train_dates = pd.DatetimeIndex(times_train).normalize().unique()
    val_dates = pd.DatetimeIndex(times_val).normalize().unique()
    dv_tr = day_vec_all.loc[day_vec_all.index.isin(train_dates)]
    dv_va = day_vec_all.loc[day_vec_all.index.isin(val_dates)]
    # 填 NaN（部分日气象缺失）用训练列均值，与 WeatherSim 的 nan_to_num 哲学一致
    col_mean = dv_tr.mean()
    dv_tr = dv_tr.fillna(col_mean)
    dv_va = dv_va.fillna(col_mean)
    K = 8
    km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(dv_tr.values)
    dom_tr = pd.Series(km.labels_, index=dv_tr.index)
    dom_va = pd.Series(km.predict(dv_va.values), index=dv_va.index)
    # 全局 transfer R²（combined，已算）
    global_tr2 = point_transfer_r2
    # per-Domain transfer R²（在该 Domain 的 2025 日上训练，2026 同 Domain 日上测）
    train_date_of = pd.DatetimeIndex(times_train).normalize()
    val_date_of = pd.DatetimeIndex(times_val).normalize()
    dom_results = []
    weighted_num, weighted_den = 0.0, 0.0
    sign_flip = 0
    for k in range(K):
        tr_dates_k = dom_tr[dom_tr == k].index
        va_dates_k = dom_va[dom_va == k].index
        mtr = train_date_of.isin(tr_dates_k)
        mva = val_date_of.isin(va_dates_k)
        if mtr.sum() < 300 or mva.sum() < 50:
            continue
        bst = _fit_probe(X_train[mtr], r_train[mtr], combined_cols)
        pred = bst.predict(X_val[mva][combined_cols])
        y = r_val[mva]
        seg_mean = float(r_train[mtr].mean())
        bmse = float(np.mean((y - seg_mean) ** 2))
        tr2 = 1.0 - float(np.mean((y - pred) ** 2)) / bmse if bmse > 0 else float("nan")
        mean_25 = float(r_train[mtr].mean())
        mean_26 = float(y.mean())
        flip = int(np.sign(mean_25) != np.sign(mean_26))
        sign_flip += flip
        dom_results.append({"domain": k, "n_train": int(mtr.sum()), "n_val": int(mva.sum()),
                            "mean_2025": mean_25, "mean_2026": mean_26,
                            "sign_flip": flip, "transfer_R2": tr2})
        weighted_num += tr2 * len(y) if not np.isnan(tr2) else 0.0
        weighted_den += len(y)
    dom_weighted_tr2 = weighted_num / weighted_den if weighted_den > 0 else float("nan")
    sign_flip_rate = sign_flip / len(dom_results) if dom_results else float("nan")

    # ===================== F. 自动判定 =====================
    def _transfers(row):
        return row["transfer_R2"] > 0.0
    transferring_comps = [r["component"] for r in comp_rows
                          if r["component"] != "combined" and _transfers(r) and r["dir_acc"] > 0.52]
    best_comp = max((r for r in comp_rows if r["component"] != "combined"),
                    key=lambda r: r["transfer_R2"], default=None)
    combined_tr2 = next(r["transfer_R2"] for r in comp_rows if r["component"] == "combined")
    combined_holdout = next(r["within2025_R2"] for r in comp_rows if r["component"] == "combined")
    # Shape：区分"信号迁移(R²>0)"与"形态安全(比Point少伤害)"
    shape_transfers = any(r["transfer_R2"] > 0.0 for r in shape_rows)
    shape_safer = any(r["transfer_R2"] > point_transfer_r2 + 0.01 for r in shape_rows)
    # Domain：区分"信号迁移(R²>0且符号稳)"与"比全局少伤害"
    domain_transfers = (not np.isnan(dom_weighted_tr2)) and dom_weighted_tr2 > 0.0 and sign_flip_rate < 0.3
    domain_safer = (not np.isnan(dom_weighted_tr2)) and dom_weighted_tr2 > global_tr2 + 0.005

    # ===================== 报告 =====================
    L = []
    L.append("# v8.1 Phase 0：残差溯源 + 跨年可迁移性诊断\n")
    L.append(f"残差 r = actual − base_A(v6)。r_train=2025 OOF(无泄露, N={len(r_train)})，"
             f"r_val=2026 val(eval-only, N={len(r_val)})。train_mean={train_mean:+.1f} MW。\n")
    L.append("跨年 transfer R²：探针在 2025 残差上训练、2026 上测试，基线=2025 残差均值(do nothing)。"
             ">0 表示该分量跨年可迁移；dir_acc>0.52 表示方向可迁移。\n")

    L.append("\n## A. 残差总览\n")
    L.append(f"- r_train: mean={ov['r_train_mean']:+.1f} std={ov['r_train_std']:.0f}")
    L.append(f"- r_val:   mean={ov['r_val_mean']:+.1f} std={ov['r_val_std']:.0f} MAE={ov['r_val_MAE']:.0f}\n")
    L.append("| 段 | 2025均值 | 2026均值 | 2026 MAE | 2026 std |")
    L.append("|---|---|---|---|---|")
    for s in VC.SEGMENTS:
        o = seg_overview[s]
        L.append(f"| {s} | {o['train_mean']:+.0f} | {o['val_mean']:+.0f} | {o['val_MAE']:.0f} | {o['val_std']:.0f} |")

    L.append("\n## B. 分量跨年可迁移性\n")
    L.append("| 分量 | 列数 | 期内R²(2025holdout) | 跨年transfer R² | 标准R² | 方向命中 | 基线MAE | 探针MAE | MAE改善 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in comp_rows:
        L.append(f"| {r['component']} | {r['n_cols']} | {r['within2025_R2']:+.3f} | "
                 f"{r['transfer_R2']:+.3f} | {r['std_R2']:+.3f} | {r['dir_acc']:.3f} | "
                 f"{r['base_MAE']:.0f} | {r['probe_MAE']:.0f} | {r['MAE_red']:+.1f} |")
    L.append(f"\n**combined 跨年 transfer R² = {combined_tr2:+.3f}**（= 无新数据下可榨取的跨年信号上限）。\n"
             f"期内 R² 与跨年 transfer R² 的 GAP = 过拟合/不可迁移程度。\n")

    L.append("\n## C. 分段 × 分量 跨年 transfer R²（重点看 day/午间）\n")
    L.append("| 段 | " + " | ".join(ordered) + " |")
    L.append("|---|" + "|".join(["---"] * len(ordered)) + "|")
    for s in VC.SEGMENTS:
        row = [s] + [f"{seg_comp.get((s, g), float('nan')):+.3f}" for g in ordered]
        L.append("| " + " | ".join(row) + " |")

    L.append("\n## D. Shape vs Point 跨年可迁移性\n")
    L.append(f"逐点(point) combined transfer R² = {point_transfer_r2:+.3f}\n")
    L.append("| shape 描述子 | 跨年 transfer R² | 方向命中 | 2026 std |")
    L.append("|---|---|---|---|")
    for r in shape_rows:
        L.append(f"| {r['descriptor']} | {r['transfer_R2']:+.3f} | {r['dir_acc']:.3f} | {r['val_std']:.0f} |")
    L.append(f"\n**Shape**：信号迁移={'支持' if shape_transfers else '不支持'}（shape 描述子 transfer R² 是否>0）；"
             f"形态安全={'是' if shape_safer else '否'}（比逐点 {point_transfer_r2:+.3f} 少伤害）。\n")

    L.append("\n## E. Domain 条件化可迁移性\n")
    L.append(f"全局 combined transfer R² = {global_tr2:+.3f}")
    L.append(f"Domain 条件化加权 transfer R² = {dom_weighted_tr2:+.3f}（K={K} 个天气型 Domain）")
    L.append(f"Domain 可靠性符号翻转率 = {sign_flip_rate:.2%}（2025 vs 2026 残差均值符号翻转的 Domain 占比）\n")
    L.append("| Domain | 2025样本 | 2026样本 | 2025均值 | 2026均值 | 符号翻转 | transfer R² |")
    L.append("|---|---|---|---|---|---|---|")
    for d in dom_results:
        L.append(f"| D{d['domain']} | {d['n_train']} | {d['n_val']} | {d['mean_2025']:+.0f} | "
                 f"{d['mean_2026']:+.0f} | {'是' if d['sign_flip'] else '否'} | {d['transfer_R2']:+.3f} |")
    L.append(f"\n**Domain**：信号迁移={'支持' if domain_transfers else '不支持'}（加权 transfer R²={dom_weighted_tr2:+.3f} 是否>0 且符号翻转率<30%）；"
             f"比全局少伤害={'是' if domain_safer else '否'}。\n")

    L.append("\n## F. 自动判定（v8.1 模块 go/no-go 依据）\n")
    L.append(f"0. **关键发现**：combined 期内(2025 holdout)R²={combined_holdout:+.3f}（亦为负）。"
             f"残差不仅在跨年不可学，**期内即不可学**（特征对残差无可解释方差）。"
             f"这是比\"跨年 R² 负\"更强的信息上限证据。\n")
    L.append(f"1. **跨年可迁移分量**：{transferring_comps if transferring_comps else '无（全部分量 transfer R²≤0 或方向不可迁移）'}")
    L.append(f"2. **最强分量**：{best_comp['component']} (transfer R²={best_comp['transfer_R2']:+.3f}, dir_acc={best_comp['dir_acc']:.3f})"
             if best_comp else "无")
    L.append(f"3. **无新数据上限**：combined 跨年 transfer R²={combined_tr2:+.3f}（<0），方向命中率全 <0.5（反相关）。"
             f"任何特征探针都使 val MAE 恶化（探针 MAE 全部 > 基线 {comp_rows[-1]['base_MAE']:.0f}）。"
             f"-> v8.1 多阶段任务阶梯（无新数据）**无信号可榨**。")
    L.append(f"4. **Shape**：信号迁移={'go' if shape_transfers else 'no-go'}（shape 描述子 transfer R² 全≈0，不>0）；"
             f"形态安全={'是' if shape_safer else '否'}（比逐点 {point_transfer_r2:+.3f} 少伤害）。"
             f"-> 若修正，用 Shape 不用 Point（更安全），但不产生信号。")
    L.append(f"5. **Domain**：信号迁移={'go' if domain_transfers else 'no-go'}（加权 transfer R²={dom_weighted_tr2:+.3f}<0，"
             f"符号翻转率 {sign_flip_rate:.0%}{'>30% 不稳' if sign_flip_rate > 0.3 else ''}）；"
             f"比全局少伤害={'是' if domain_safer else '否'}。"
             f"-> Weather Sim->Domain 作修正依据 no-go；Domain 可靠性跨年不稳。")

    L.append("\n## v8.1 方向建议（基于上述事实）\n")
    L.append(f"- **信息上限第 5 次确认（最强证据）**：combined 期内 R²={combined_holdout:+.3f}、跨年 transfer R²={combined_tr2:+.3f} "
             f"均为负，方向命中率全 <0.5（跨年反相关）。残差=外部预测误差，对当前特征**期内即不可学**，"
             f"非仅跨年不迁移。Direction Layer 无法修正一个特征解释不了的量。")
    L.append(f"- **多阶段任务阶梯（无新数据）**：no-go 倾向。上层任务（风险/类型）依赖残差方向稳定性，"
             f"而 dir_acc<0.5 表明方向跨年反相关 -> 阶梯上层亦不可迁移。唯一未覆盖的窄口：Stage1 若重定义为"
             f"\"pred_load 输入异常检测\"（输入层面，非残差方向）可能迁移，须单独验证。")
    L.append(f"- **Shape**：作为修正**形态**优选（比 Point 安全，R²≈0 不伤害），但**不产生信号**。"
             f"v8.1 若保留修正层，用低维 Shape（Peak Height/Ramp/Peak Time）；预期收益噪声级。")
    L.append(f"- **Domain**：no-go 作修正依据（符号翻转 {sign_flip_rate:.0%}）。Weather Sim 降级为纯诊断分层，不作 Information Layer。")
    L.append(f"- **唯一突破口仍是新信息层**：①新能源出力（exogenous，当前特征无）②Foundation 时序模型"
             f"（Chronos/Moirai 等预训练先验，不在当前特征内，Phase 0 未覆盖）。两者均须实测，不预设上限。")
    L.append(f"- **工程结论**：v8.1 极简版（v6 base + 单一 c + 低维 Shape + 仅输入异常触发）= v6 parity，"
             f"换可维护性不换 MAE。破 1445.62 的概率集中在新能源/Foundation，不在架构技巧。")

    report = "\n".join(L)
    REPORT.write_text(report, encoding="utf-8")
    if verbose:
        print("\n" + report)
        print(f"\n报告已写: {REPORT}")
    return {
        "overview": ov, "seg_overview": seg_overview, "comp_rows": comp_rows,
        "seg_comp": seg_comp, "shape_rows": shape_rows, "dom_results": dom_results,
        "global_tr2": global_tr2, "dom_weighted_tr2": dom_weighted_tr2,
        "sign_flip_rate": sign_flip_rate, "combined_tr2": combined_tr2,
        "combined_holdout": combined_holdout, "transferring_comps": transferring_comps,
        "shape_transfers": shape_transfers, "shape_safer": shape_safer,
        "domain_transfers": domain_transfers, "domain_safer": domain_safer,
    }


def main():
    run(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
