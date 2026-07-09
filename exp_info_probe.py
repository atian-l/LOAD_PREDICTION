# -*- coding: utf-8 -*-
"""exp_info_probe.py - 新信息源预测力探针（只读诊断，不修改生产代码/模型/特征）。

为 /goal 的"是否值得进入实验阶段"判断提供数据依据。用生产同款 LightGBM（非 Ridge，避免线性失真）
测试两类"新可预测信息"对【外部预测误差 ext_error = pred_load - actual】的增量预测力：
  [FB] 历史预测误差反馈（Part 4）：ext_error 过去值（actual-pred_load，已实现历史）
  [IC] 辐照变化（Part 2）：irrad 差分/滚动std（从现有 irrad 派生，无需新数据）
对照 baseline=现有特征。比较 baseline / +FB / +IC / +FB+IC。

泄露时效（D+1 部署，09:00 day D 预测 day D+1）：运行日 day(T)-1 全天 actual 09:00 尚不可得；
  => 任何 actual 派生反馈特征须 lag >= 2 天（相对目标 T）。FB 一律用 T-2d 及更早。IC 用预报辐照，无泄露。

walk-forward 4 折（春25/秋25/冬26/官方val26），逐折 train(<=折前)/test(折内)。
跨折一致降低 MAE 才算可迁移；仅官方窗改善=过拟合，不值得进入实验阶段。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from load_pred import config as C, data_loader as dl, features as F, train as T

ROOT = Path(__file__).resolve().parent
MID_HOURS = list(range(11, 14))
LGB_PARAMS = dict(objective="regression", metric="mae", learning_rate=0.05, num_leaves=63,
                  min_data_in_leaf=200, lambda_l2=4.0, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, verbose=-1, n_estimators=300)

print("building features ...", flush=True)
load_df = dl.load_load_data().set_index(C.COL_TIME)
times = dl.full_time_index()
pred_load = load_df[C.COL_PRED_LOAD].reindex(times)
actual = load_df[C.COL_ACTUAL_LOAD].reindex(times)
weather = dl.load_weather_dedup(run_time=None)
X = F.build_features(times, pred_load, weather)
mm = F.MismatchModel().fit(X, T.usable_mask(times, pred_load, actual))
X = mm.transform(X)
ext_err = (pred_load - actual)
eligible = actual.notna() & pred_load.notna()  # 不用 usable（其截断到 TRAIN_END，会排除 val）

BASE_COLS = [c for c in ["clearness", "temp", "irrad", "precip", "wind", "clear_sky",
                         "pl_weather_residual", "solar_mismatch", "cloud_deficit",
                         "pred_load", "hdd", "cdd", "irrad_anom_672", "pl_dip_96",
                         "hour", "slot", "month", "dow"] if c in X.columns]
ee = ext_err.copy()
fb = pd.DataFrame(index=times)
fb["fb_lag2d"] = ee.shift(2 * 96)
fb["fb_lag7d"] = ee.shift(7 * 96)
fb["fb_mean7d"] = ee.shift(2 * 96).rolling(7 * 96, min_periods=3 * 96).mean()
fb["fb_std7d"] = ee.shift(2 * 96).rolling(7 * 96, min_periods=3 * 96).std()
FB_COLS = list(fb.columns)
ir = X["irrad"].copy() if "irrad" in X.columns else weather.reindex(times)["光伏_辐照度"]
ic = pd.DataFrame(index=times)
ic["ic_d15"] = ir.diff(1); ic["ic_d1h"] = ir.diff(4)
ic["ic_std96"] = ir.rolling(96, min_periods=24).std(); ic["ic_d1d"] = ir.diff(96)
IC_COLS = list(ic.columns)

feat = X[BASE_COLS].copy()
for c in FB_COLS: feat[c] = fb[c].values
for c in IC_COLS: feat[c] = ic[c].values
VARIANTS = [("baseline", BASE_COLS), ("+FB", BASE_COLS + FB_COLS),
            ("+IC", BASE_COLS + IC_COLS), ("+FB+IC", BASE_COLS + FB_COLS + IC_COLS)]

# ext_error 自相关（前置证据）
day_ee = ext_err.resample("D").mean().dropna()
print(f"\n=== ext_error 日均自相关: ACF(lag1)={day_ee.autocorr(1):.3f}  ACF(lag7)={day_ee.autocorr(7):.3f} ===", flush=True)

FOLDS = [("春25", "2025-02-28", "2025-03-01", "2025-05-31"),
         ("秋25", "2025-08-31", "2025-09-01", "2025-11-30"),
         ("冬26", "2025-12-31", "2026-01-01", "2026-02-28"),
         ("官方val26", "2026-02-28", "2026-03-01", "2026-06-15")]


def run_fold(name, te, vs, ve):
    te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
    tr = eligible & np.asarray(times <= te)
    va = eligible & np.asarray(times >= vs) & np.asarray(times <= ve)
    y = ext_err.values
    res = {"折": name, "N_tr": int(tr.sum()), "N_te": int(va.sum())}
    for vname, cols in VARIANTS:
        cols2 = [c for c in cols if c in feat.columns]
        M = feat[cols2].astype(float)
        mask = eligible & M.notna().all(axis=1).values & ext_err.notna().values
        trm = mask & np.asarray(times <= te)
        vam = mask & np.asarray(times >= vs) & np.asarray(times <= ve)
        if trm.sum() == 0 or vam.sum() == 0:
            res[vname] = None; continue
        mdl = lgb.LGBMRegressor(**LGB_PARAMS)
        mdl.fit(M.loc[trm], y[trm])
        pr = mdl.predict(M.loc[vam])
        e = pr - y[vam]
        hrs = pd.DatetimeIndex(times[vam]).hour
        mid_m = np.isin(hrs, MID_HOURS)
        thr = np.quantile(np.abs(y[vam]), 0.95)
        hi = np.abs(y[vam]) >= thr
        r2 = 1 - np.sum((y[vam] - pr) ** 2) / np.sum((y[vam] - y[vam].mean()) ** 2)
        res[vname] = {"MAE": float(np.abs(e).mean()),
                      "mid_MAE": float(np.abs(e[mid_m]).mean()) if mid_m.any() else float("nan"),
                      "hi_MAE": float(np.abs(e[hi]).mean()) if hi.any() else float("nan"),
                      "R2": float(r2), "Bias": float(e.mean())}
    return res


print("\n=== walk-forward: LightGBM 预测 ext_error ===", flush=True)
all_res = []
for name, te, vs, ve in FOLDS:
    r = run_fold(name, te, vs, ve)
    all_res.append(r)
    print(f"\n[{name}] Ntr={r['N_tr']} Nte={r['N_te']}", flush=True)
    for vn in ["baseline", "+FB", "+IC", "+FB+IC"]:
        d = r.get(vn)
        if d: print(f"  {vn:9s} MAE={d['MAE']:7.1f} mid={d['mid_MAE']:7.1f} hi={d['hi_MAE']:7.1f} "
                    f"R2={d['R2']:+.3f} Bias={d['Bias']:+.1f}", flush=True)

print("\n=== 跨折稳定性（春25/秋25/冬26 三内折一致降低 MAE 才算可迁移）===", flush=True)
inner = [r for r in all_res if r["折"] != "官方val26"]
for vn in ["+FB", "+IC", "+FB+IC"]:
    ds, dsm = [], []
    for r in inner:
        if r.get(vn) and r.get("baseline"):
            ds.append(r[vn]["MAE"] - r["baseline"]["MAE"])
            dsm.append(r[vn]["mid_MAE"] - r["baseline"]["mid_MAE"])
    ok = all(d < 0 for d in ds) if ds else False
    print(f"  {vn:9s} overall ΔMAE={[round(x,1) for x in ds]}  mid ΔMAE={[round(x,1) for x in dsm]}  "
          f"-> {'一致改善(可迁移)' if ok else '不一致(不可迁移)'}", flush=True)

rv = [r for r in all_res if r["折"] == "官方val26"][0]
print("\n=== 官方val26（仅参考）===", flush=True)
for vn in ["baseline", "+FB", "+IC", "+FB+IC"]:
    d = rv.get(vn)
    if d: print(f"  {vn:9s} MAE={d['MAE']:7.1f} mid={d['mid_MAE']:7.1f} hi={d['hi_MAE']:7.1f} R2={d['R2']:+.3f}", flush=True)

rows = []
for r in all_res:
    for vn in ["baseline", "+FB", "+IC", "+FB+IC"]:
        if r.get(vn): rows.append({"折": r["折"], "variant": vn, **r[vn]})
pd.DataFrame(rows).to_csv(ROOT / "exp_info_probe_result.csv", encoding="utf-8-sig", index=False)
print("\nsaved exp_info_probe_result.csv\nexp_info_probe done.", flush=True)
