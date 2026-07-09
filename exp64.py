# -*- coding: utf-8 -*-
"""exp64: 综合误差结构诊断（当前生产模型 1493.66）。

目标：找到从 1493.66 → <1300 的最大可压缩误差块。
输出：
  1. pred_load-only 基线 MAE（headroom）。
  2. 当前模型 val 残差按多维度子群的分布（均值偏置 / MAE / 贡献）。
  3. OOF 已校正残差按子群的偏置（可迁移的新校正候选）。
仅诊断，不写产物。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import numpy as np
import pandas as pd
from load_pred import config as C, features as F, train as T


def main():
    times, X, pred_load, actual = T.build_dataset()
    usable = T.usable_mask(times, pred_load, actual)
    mm = F.MismatchModel().fit(X, usable); X = mm.transform(X)
    val = ((times >= pd.Timestamp(C.VAL_START)) & (times <= pd.Timestamp(C.VAL_END)) & actual.notna()).values
    cfg = dict(C.TRAIN_CONFIG); cfg["best_it_fixed"] = 80

    # 生产模型（含 hour_bias/drift_corr/threshold_corr）
    model = T.train_ensemble(times, X, pred_load, actual, usable, cfg, 80)
    model.hour_bias, model.drift_corr, model.threshold_corr = T.compute_hour_bias(
        times, X, pred_load, actual, usable, cfg, 80)
    pred_v = model.predict_load(X[val], pred_load[val])
    act_v = actual[val].values
    resid_v = pred_v - act_v
    mae_v = np.abs(resid_v).mean()
    print(f"=== 总体 ===", flush=True)
    print(f"  模型 val MAE = {mae_v:.2f}  bias={resid_v.mean():.1f}", flush=True)
    pl_v = pred_load[val].values
    print(f"  pred_load-only MAE = {np.abs(pl_v - act_v).mean():.2f}  (headroom: 模型已降 {np.abs(pl_v-act_v).mean()-mae_v:.1f})", flush=True)
    print(f"  目标 1300 → 需再降 {mae_v-1300:.1f} MW", flush=True)
    print(f"  q50/q90/q95/q99 = {np.quantile(np.abs(resid_v),[.5,.9,.95,.99]).round(1)}", flush=True)
    print(flush=True)

    # OOF（折内原始集成），用于迁移检查
    oof = pd.Series(np.nan, index=times)
    for te, vs, ve in cfg["best_it_folds"]:
        te, vs, ve = pd.Timestamp(te), pd.Timestamp(vs), pd.Timestamp(ve)
        ftr = usable & np.asarray(times <= te)
        fva = usable & np.asarray(times >= vs) & np.asarray(times <= ve)
        if fva.sum() == 0: continue
        fm = T.train_ensemble(times, X, pred_load, actual, ftr, cfg, 80)
        oof[fva] = fm.predict_load(X[fva], pred_load[fva])
    oof_mask = usable & oof.notna().values
    # OOF 已校正残差（应用生产 hour_bias/drift/threshold 后的残差）—— 新校正应基于此
    h_all = pd.DatetimeIndex(times).hour.values.astype(int)
    oof_raw = oof.values
    oof_cor = oof_raw - model.hour_bias[h_all]
    for fn, beta in model.drift_corr:
        oof_cor = oof_cor + np.asarray(beta, dtype=float)[h_all] * X[fn].values.astype(float)
    for fn, thr, hrs, sh in model.threshold_corr:
        fv = X[fn].values.astype(float); sel = fv > thr
        if hrs is not None: sel = sel & np.isin(h_all, hrs)
        oof_cor[sel] = oof_cor[sel] - sh
    oof_resid = oof_cor - actual.values  # 训练期 OOF 已校正残差

    # 子群扫描
    dt = pd.DatetimeIndex(times)
    feats = {
        "hour": dt.hour.values.astype(int),
        "month": dt.month.values.astype(int),
        "dow": dt.dayofweek.values.astype(int),
        "clearness": np.nan_to_num(X["clearness"].values.astype(float), nan=0.0),
        "precip": np.nan_to_num(X["precip"].values.astype(float), nan=0.0),
        "temp": np.nan_to_num(X.get("光伏_温度", pd.Series(0,index=X.index)).values.astype(float) if "光伏_温度" in X.columns else np.zeros(len(X)), nan=0.0),
        "irrad": np.nan_to_num(X["irrad"].values.astype(float) if "irrad" in X.columns else (X["光伏_辐照度"].values.astype(float) if "光伏_辐照度" in X.columns else np.zeros(len(X))), nan=0.0),
        "pl_level": pl_v if False else pred_load.values.astype(float),
    }
    # 节假日
    hol = np.zeros(len(times), dtype=int)
    if "is_holiday" in X.columns:
        hol = X["is_holiday"].values.astype(int)
    elif "holiday" in X.columns:
        hol = X["holiday"].values.astype(int)

    def scan(name, key, bins=None, is_int=False):
        print(f"=== 按 {name} ===", flush=True)
        k = key[val] if len(key)==len(times) else key
        kk_all = key
        rows = []
        if bins is not None:
            bk = np.digitize(k, bins)
            bk_all = np.digitize(kk_all, bins)
            labels = [f"<{bins[0]}"] + [f"{bins[i]}~{bins[i+1]}" for i in range(len(bins)-1)] + [f">={bins[-1]}"]
            for b in range(len(bins)+1):
                m = bk == b
                m_all = (oof_mask) & (bk_all == b)
                if m.sum() < 30: continue
                vr = resid_v[m]; oor = oof_resid[m_all]
                rows.append((labels[b], m.sum(), vr.mean(), np.abs(vr).mean(), oor.mean() if len(oor)>20 else np.nan))
        else:
            for v in sorted(set(k)):
                m = k == v
                m_all = (oof_mask) & (kk_all == v)
                if m.sum() < 30: continue
                vr = resid_v[m]; oor = oof_resid[m_all]
                rows.append((v, m.sum(), vr.mean(), np.abs(vr).mean(), oor.mean() if len(oor)>20 else np.nan))
        # 按 MAE 贡献排序
        rows.sort(key=lambda r: -r[1]*r[3])
        print(f"  {'grp':>10} {'n':>6} {'val_bias':>9} {'val_MAE':>8} {'oof_bias':>9}  (oof_bias 可迁移候选)", flush=True)
        for r in rows:
            print(f"  {str(r[0]):>10} {r[1]:>6} {r[2]:>9.1f} {r[3]:>8.1f} {r[4]:>9.1f}", flush=True)
        print(flush=True)

    scan("hour", feats["hour"])
    scan("month", feats["month"])
    scan("dow", feats["dow"])
    scan("clearness", feats["clearness"], bins=[0.2, 0.5, 0.8])
    scan("precip", feats["precip"], bins=[0.01])
    scan("temp", feats["temp"], bins=[5, 15, 25])
    scan("irrad", feats["irrad"], bins=[100, 400, 800])
    scan("pl_level", feats["pl_level"], bins=np.quantile(pred_load.values.astype(float)[val], [.25,.5,.75]))
    # 节假日
    print("=== 按 is_holiday ===", flush=True)
    for v in [0,1]:
        m = (hol[val]==v); m_all = (oof_mask)&(hol==v)
        if m.sum()<30: continue
        vr=resid_v[m]; oor=oof_resid[m_all]
        print(f"  holiday={v} n={m.sum()} val_bias={vr.mean():.1f} val_MAE={np.abs(vr).mean():.1f} oof_bias={oor.mean() if len(oor)>20 else np.nan:.1f}", flush=True)
    print(flush=True)

    # 误差最大的天（top 15）—— 找异常日
    df = pd.DataFrame({"d": pd.DatetimeIndex(times[val]).date, "r": resid_v})
    daily = df.groupby("d")["r"].agg(["mean","count"])
    daily["abs"] = daily["mean"].abs()
    print("=== 偏置最大的 15 天 ===", flush=True)
    print(daily.sort_values("abs", ascending=False).head(15).round(0), flush=True)


if __name__ == "__main__":
    main()
