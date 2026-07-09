# -*- coding: utf-8 -*-
"""exp_midday_model_probe.py - 午间专项模型结构性探针（只读诊断，不改生产）。

评估 /goal 第三部分方案A（午间/非午间拆分模型）：用现有特征训练"午间专用"LightGBM，
walk-forward 检验其午间 ext_error 预测是否【跨折稳定优于】全天统一模型。
- 机制先验：午间专用模型用的是同一批特征，无法创造新信息；且午间偏置跨年符号翻转
  （2025春+1971 -> 2026春-572），午间专用模型大概率过拟合2025、误判2026。
- 本探针直接验证：midday-only 训练 vs all-day 训练，在午间 test 上的 MAE 跨折是否一致更低。
不比较全天（拆分模型非午间部分等价全天模型，全天不会退化；关键是午间是否真改善且可迁移）。
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
P = dict(objective="regression", metric="mae", learning_rate=0.05, num_leaves=63,
         min_data_in_leaf=200, lambda_l2=4.0, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, verbose=-1, n_estimators=300)

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
eligible = actual.notna() & pred_load.notna()
hrs = pd.DatetimeIndex(times).hour
is_mid = np.isin(hrs.values, MID_HOURS)

BASE_COLS = [c for c in ["clearness", "temp", "irrad", "precip", "wind", "clear_sky",
                         "pl_weather_residual", "solar_mismatch", "cloud_deficit",
                         "pred_load", "hdd", "cdd", "irrad_anom_672", "pl_dip_96",
                         "hour", "slot", "month", "dow"] if c in X.columns]
M = X[BASE_COLS].astype(float)
mask_all = eligible & M.notna().all(axis=1).values & ext_err.notna().values
y = ext_err.values

FOLDS = [("春25", "2025-02-28", "2025-03-01", "2025-05-31"),
         ("秋25", "2025-08-31", "2025-09-01", "2025-11-30"),
         ("冬26", "2025-12-31", "2026-01-01", "2026-02-28"),
         ("官方val26", "2026-02-28", "2026-03-01", "2026-06-15")]

print("\n=== 午间专用模型 vs 全天模型（午间 test ext_error MAE）===", flush=True)
rows = []
for name, te, vs, ve in FOLDS:
    te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
    tr = mask_all & np.asarray(times <= te)
    va = mask_all & np.asarray(times >= vs) & np.asarray(times <= ve)
    va_mid = va & is_mid
    # 全天模型
    m1 = lgb.LGBMRegressor(**P).fit(M.loc[tr], y[tr])
    mid_allday = np.abs(m1.predict(M.loc[va_mid]) - y[va_mid]).mean()
    # 午间专用模型（仅用午间训练样本）
    tr_mid = tr & is_mid
    m2 = lgb.LGBMRegressor(**P).fit(M.loc[tr_mid], y[tr_mid])
    mid_specific = np.abs(m2.predict(M.loc[va_mid]) - y[va_mid]).mean()
    d = mid_specific - mid_allday
    rows.append({"折": name, "N_mid_train": int(tr_mid.sum()), "N_mid_test": int(va_mid.sum()),
                 "全天_midMAE": mid_allday, "午间专用_midMAE": mid_specific, "Δ(专用-全天)": d})
    print(f"  {name:10s} 全天={mid_allday:7.1f}  午间专用={mid_specific:7.1f}  Δ={d:+.1f}", flush=True)

inner = [r for r in rows if r["折"] != "官方val26"]
ds = [r["Δ(专用-全天)"] for r in inner]
ok = all(d < 0 for d in ds)
print(f"\n三内折 Δ(午间专用 - 全天) = {[round(x,1) for x in ds]}", flush=True)
print(f"-> {'午间专用一致更优(可迁移)' if ok else '不一致/多更差 -> 午间专项模型无可迁移收益（过拟合风险）'}", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "exp_midday_model_probe_result.csv", encoding="utf-8-sig", index=False)
print("saved exp_midday_model_probe_result.csv\ndone.", flush=True)
