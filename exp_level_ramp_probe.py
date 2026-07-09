# -*- coding: utf-8 -*-
"""exp_level_ramp_probe.py - level+ramp 建模探针（只读诊断，不改生产）。

评估 /goal 第三部分方案B（level 模型 + ramp 模型）。机制：ramp（变化量）可能比 level（水平）更平稳可预测。
泄露安全 operationalization（D+1 @ 09:00 day D 预测 day D+1）：
  anchor(t) = actual(t-2d)（day D-1，已实现可得，lag>=2d 合规）。
  ramp target r(t) = actual(t) - actual(t-2d)。
  重构：actual_hat(t) = anchor(t) + r_hat(t)。
对照三变体（同现有特征，walk-forward 4 折，目标为实际负荷 actual）：
  V1 baseline      : LGBM(actual ~ 特征)
  V2 baseline+anchor: LGBM(actual ~ 特征 + anchor)   —— 隔离"加 anchor"的收益
  V3 ramp          : LGBM(r ~ 特征)，actual_hat = anchor + r_hat —— level+ramp 方案
若 V3 不能跨折稳定优于 V1（且优于 V2），则 level+ramp 无可迁移收益。
输出：午间 MAE / 全天 MAE / Bias / 跨年稳定性。
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
eligible = actual.notna() & pred_load.notna()
anchor = actual.shift(2 * 96)          # actual(t-2d)，已实现，lag>=2d 合规
ramp = actual - anchor                  # ramp target
hrs = pd.DatetimeIndex(times).hour
is_mid = np.isin(hrs.values, MID_HOURS)

BASE_COLS = [c for c in ["clearness", "temp", "irrad", "precip", "wind", "clear_sky",
                         "pl_weather_residual", "solar_mismatch", "cloud_deficit",
                         "pred_load", "hdd", "cdd", "irrad_anom_672", "pl_dip_96",
                         "hour", "slot", "month", "dow"] if c in X.columns]
Fb = X[BASE_COLS].astype(float).copy()
Fb["anchor"] = anchor.values            # V2/V3 用
y_act = actual.values
y_ramp = ramp.values

FOLDS = [("春25", "2025-02-28", "2025-03-01", "2025-05-31"),
         ("秋25", "2025-08-31", "2025-09-01", "2025-11-30"),
         ("冬26", "2025-12-31", "2026-01-01", "2026-02-28"),
         ("官方val26", "2026-02-28", "2026-03-01", "2026-06-15")]


def mae_by(pred, true, midmask):
    return (float(np.abs(pred - true).mean()),
            float(np.abs(pred - true)[midmask].mean()) if midmask.any() else float("nan"))


print("\n=== level+ramp 探针（目标=actual，walk-forward 4 折）===", flush=True)
rows = []
for name, te, vs, ve in FOLDS:
    te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
    # 测试集要求 anchor 可得（ramp 重构需要）；训练集同理
    base_mask = eligible & Fb[BASE_COLS].notna().all(axis=1).values & anchor.notna().values & actual.notna().values
    tr = base_mask & np.asarray(times <= te)
    va = base_mask & np.asarray(times >= vs) & np.asarray(times <= ve)
    va = va.values if hasattr(va, "values") else va
    mid_va = is_mid[va]
    Xm = Fb[BASE_COLS]
    Xa = Fb[BASE_COLS + ["anchor"]]
    # V1 baseline
    m1 = lgb.LGBMRegressor(**P).fit(Xm.loc[tr], y_act[tr])
    p1 = m1.predict(Xm.loc[va]); v1_all, v1_mid = mae_by(p1, y_act[va], mid_va)
    # V2 baseline+anchor
    m2 = lgb.LGBMRegressor(**P).fit(Xa.loc[tr], y_act[tr])
    p2 = m2.predict(Xa.loc[va]); v2_all, v2_mid = mae_by(p2, y_act[va], mid_va)
    # V3 ramp: 预测 r，重构 actual=anchor+r_hat
    m3 = lgb.LGBMRegressor(**P).fit(Xm.loc[tr], y_ramp[tr])
    p3 = anchor.values[va] + m3.predict(Xm.loc[va]); v3_all, v3_mid = mae_by(p3, y_act[va], mid_va)
    rows.append({"折": name, "N_te": int(va.sum()),
                 "V1_base_all": v1_all, "V1_base_mid": v1_mid,
                 "V2_anch_all": v2_all, "V2_anch_mid": v2_mid,
                 "V3_ramp_all": v3_all, "V3_ramp_mid": v3_mid})
    print(f"\n[{name}] Nte={int(va.sum())}", flush=True)
    print(f"  V1 baseline      all={v1_all:7.1f} mid={v1_mid:7.1f}", flush=True)
    print(f"  V2 baseline+anch all={v2_all:7.1f} mid={v2_mid:7.1f}  ΔvsV1_all={v2_all-v1_all:+.1f}", flush=True)
    print(f"  V3 level+ramp    all={v3_all:7.1f} mid={v3_mid:7.1f}  ΔvsV1_all={v3_all-v1_all:+.1f}  ΔvsV1_mid={v3_mid-v1_mid:+.1f}", flush=True)

print("\n=== 跨折稳定性（V3 ramp vs V1 baseline，三内折一致降低 MAE 才算可迁移）===", flush=True)
inner = [r for r in rows if r["折"] != "官方val26"]
d_all = [r["V3_ramp_all"] - r["V1_base_all"] for r in inner]
d_mid = [r["V3_ramp_mid"] - r["V1_base_mid"] for r in inner]
ok = all(d < 0 for d in d_all) and all(d < 0 for d in d_mid)
print(f"  全天 ΔMAE={[round(x,1) for x in d_all]}  午间 ΔMAE={[round(x,1) for x in d_mid]}", flush=True)
print(f"  -> {'一致改善(可迁移)' if ok else '不一致/多更差 -> level+ramp 无可迁移收益'}", flush=True)
rv = [r for r in rows if r["折"] == "官方val26"][0]
print(f"  官方val: V1={rv['V1_base_all']:.1f}(mid {rv['V1_base_mid']:.1f})  "
      f"V3={rv['V3_ramp_all']:.1f}(mid {rv['V3_ramp_mid']:.1f})  "
      f"Δall={rv['V3_ramp_all']-rv['V1_base_all']:+.1f} Δmid={rv['V3_ramp_mid']-rv['V1_base_mid']:+.1f}", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "exp_level_ramp_probe_result.csv", encoding="utf-8-sig", index=False)
print("\nsaved exp_level_ramp_probe_result.csv\ndone.", flush=True)
