# -*- coding: utf-8 -*-
"""exp76 — 气象起报时间分布诊断 + 9AM 部署可用性评估。

用户指出部署在每天 ~9:00 AM 运行（非 21:00）。诊断：
  1. 起报时间(_issue) 的小时分布 / 频率 / 预报提前期 (forecast - issue)。
  2. 对验证集每日 D+1，其覆盖气象的起报时间是否 <= 9:00 AM on D（部署可得）。
  3. 比较 run_time=None vs run_time=9AM 两种去重下的气象覆盖率与差异。
仅诊断，不写产物。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from load_pred import config as C, data_loader as dl

w = pd.read_csv(C.WEATHER_CSV, encoding="utf-8-sig")
w.columns = [c.strip() for c in w.columns]
w[C.WCOL_ISSUE] = pd.to_datetime(w[C.WCOL_ISSUE])
w[C.WCOL_FORECAST] = pd.to_datetime(w[C.WCOL_FORECAST])

print(f"气象总行数={len(w)}  唯一起报时间数={w[C.WCOL_ISSUE].nunique()}  唯一预测时间数={w[C.WCOL_FORECAST].nunique()}", flush=True)

# 1. 起报时间的小时分布
iss = w.drop_duplicates(subset=[C.WCOL_ISSUE])[C.WCOL_ISSUE]
print(f"\n起报时间跨度: {iss.min()} ~ {iss.max()}", flush=True)
print(f"起报时间 hour 分布:\n{iss.dt.hour.value_counts().sort_index()}", flush=True)
print(f"起报时间 minute 分布:\n{iss.dt.minute.value_counts().sort_index().head()}", flush=True)

# 2. 每个起报版本覆盖的预测时间范围 & 提前期
cov = w.groupby(C.WCOL_ISSUE)[C.WCOL_FORECAST].agg(["min", "max", "count"])
cov["lead_min_h"] = (cov["min"] - cov.index).dt.total_seconds() / 3600
cov["lead_max_h"] = (cov["max"] - cov.index).dt.total_seconds() / 3600
print(f"\n每个起报版本覆盖点数: min={cov['count'].min()} max={cov['count'].max()} (96=1天)", flush=True)
print(f"提前期(预报-起报) 小时: lead_min [{cov['lead_min_h'].min():.1f}, {cov['lead_min_h'].max():.1f}], lead_max [{cov['lead_max_h'].min():.1f}, {cov['lead_max_h'].max():.1f}]", flush=True)
print(f"\n起报版本覆盖的 forecast 日期数 (96点=1天则=1):", flush=True)
print((cov["count"] == 96).mean(), "= frac covering exactly 1 day", flush=True)

# 3. 验证集 D+1 部署可用性：对每个验证日 D+1，覆盖其 96 点的起报时间，
#    判断该起报时间是否 <= (D+1 - 1天 + 9:00) = D 日 09:00（9AM 部署可得）
print("\n=== 验证集 9AM 部署可用性 ===", flush=True)
val_days = pd.date_range(C.VAL_START, C.VAL_END, freq="D")
# 取 run_time=None 的去重（每个 forecast time 的唯一起报）
w_all = w.sort_values(C.WCOL_ISSUE).drop_duplicates(subset=[C.WCOL_FORECAST], keep="last").set_index(C.WCOL_FORECAST)
avail_9am = 0; total = 0; lead_at_deploy = []
for d1 in val_days:
    d = d1 - pd.Timedelta(days=1)
    deploy = d + pd.Timedelta(hours=9)  # 9 AM on D
    # D+1 全天 96 点
    t_idx = pd.date_range(d1, d1 + pd.Timedelta(days=1) - pd.Timedelta(minutes=15), freq=C.FREQ)
    sub = w_all.reindex(t_idx)
    # 每个预测时间对应的起报时间
    iss_times = w.drop_duplicates(subset=[C.WCOL_FORECAST]).set_index(C.WCOL_FORECAST).loc[:, C.WCOL_ISSUE].reindex(t_idx)
    has = sub.notna().all(axis=1)  # 该 forecast time 在去重后有数据
    avail = (iss_times <= deploy) & has  # 起报时间 <= 9AM on D 且有数据
    avail_9am += int(avail.sum()); total += len(t_idx)
    if has.any():
        lead = (t_idx[has] - iss_times[has]).dt.total_seconds() / 3600
        lead_at_deploy.extend(lead.values.tolist())
print(f"验证集 D+1 总点数={total}", flush=True)
full_idx_val = pd.date_range(pd.Timestamp(C.VAL_START), pd.Timestamp(C.VAL_END) + pd.Timedelta(days=1) - pd.Timedelta(minutes=15), freq=C.FREQ)
print(f"  run_time=None 全覆盖点数={int(w_all.reindex(full_idx_val).notna().all(axis=1).sum())}/{len(full_idx_val)}", flush=True)
print(f"  9AM 部署可得点数 (起报<=9AM on D)={avail_9am}  ({100*avail_9am/total:.1f}%)", flush=True)
if lead_at_deploy:
    la = np.array(lead_at_deploy)
    print(f"  部署时预报提前期(预报-起报) 小时: mean={la.mean():.1f} min={la.min():.1f} max={la.max():.1f}", flush=True)

# 4. 21:00 vs 9:00 部署对比
print("\n=== 21:00 vs 9:00 部署可得性 ===", flush=True)
for hr, label in [(21, "21:00"), (9, "9:00")]:
    cnt = 0
    for d1 in val_days:
        d = d1 - pd.Timedelta(days=1)
        deploy = d + pd.Timedelta(hours=hr)
        t_idx = pd.date_range(d1, d1 + pd.Timedelta(days=1) - pd.Timedelta(minutes=15), freq=C.FREQ)
        iss_times = w.drop_duplicates(subset=[C.WCOL_FORECAST]).set_index(C.WCOL_FORECAST).loc[:, C.WCOL_ISSUE].reindex(t_idx)
        sub = w_all.reindex(t_idx)
        has = sub.notna().all(axis=1)
        cnt += int(((iss_times <= deploy) & has).sum())
    print(f"  {label} 部署: {cnt}/{total} ({100*cnt/total:.1f}%) 点可得", flush=True)
