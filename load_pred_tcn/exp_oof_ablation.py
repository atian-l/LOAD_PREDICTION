# -*- coding: utf-8 -*-
"""
exp_oof_ablation：OOF 校正消融（5 档）+ hour_bias 粒度扫。

调优元素：hour_bias / drift_corr / threshold_corr 三档 OOF 校正对 TCN 的边际贡献，
以及 hour_bias_slots ∈ {24,48,96} 粒度。判断 OOF 校正在 TCN 上是否仍正向
（v6 LGB 上 +67MW；TCN 上待验）。

两块（共用一次 OOF + 一次全量训练，post-hoc 即时）：
  Block 1：5 档消融（none / +hour_bias / +drift / +threshold / +all），val MAE/Bias/午间。
  Block 2：hour_bias_slots ∈ {24,48,96}（同一 OOF 残差重估 hour_bias），仅 +hour_bias 的 val MAE。

成员：6 成员（3 seeds × regression × {direct,residual}），兼顾速度与残差稳定性。
OOF 校正逻辑复用 train.compute_hour_bias（经 exp_common.estimate_corrections），无 train/serve skew。
运行：python -m load_pred_tcn.exp_oof_ablation
"""
from __future__ import annotations
import numpy as np

from . import exp_common as E

# 6 成员：3 seeds × regression × {direct,residual}
MEM_OVERRIDE = {"seeds": [42, 7, 123], "objectives": ["regression"], "residual_modes": [False, True]}


def main():
    d = E.build_cached()
    times, X, pred_load, actual, val_m = (d["times"], d["X"], d["pred_load"],
                                          d["actual"], d["val_m"])
    va = actual[val_m].values
    vt = times[val_m]
    print(f"\n[exp_oof_ablation] 数据: 特征{X.shape[1]} 可用{d['usable'].sum()} val{val_m.sum()}")
    print("计算 OOF（3 折，6 成员）+ 全量训练 ...")
    oof = E.compute_oof(MEM_OVERRIDE)
    hb, dr, tc = oof["hour_bias"], oof["drift_corr"], oof["threshold_corr"]
    model = E.train_ens(MEM_OVERRIDE)
    print(f"  OOF MAE={E._mae(oof['oof_pred'].values[oof['oof_mask']], actual.values[oof['oof_mask']]):.1f}  "
          f"WF-CV={oof['wfcv_mean']:.1f}±{oof['wfcv_std']:.1f}")

    # ---- Block 1：5 档消融 ----
    print("\n" + "=" * 70)
    print("[Block 1] OOF 校正消融 (val)   生产=+all")
    print("=" * 70)
    stages = [
        ("none",      None, [], []),
        ("+hour_bias", hb,  [], []),
        ("+drift",     hb,  dr, []),
        ("+threshold", hb,  dr, tc),
        ("+all",       hb,  dr, tc),   # = +threshold
    ]
    print(f"{'stage':>12} {'MAE':>8} {'Bias':>8} {'午间':>6} {'Δvs_none':>9}")
    base = None
    for tag, h, drc, trc in stages:
        E.apply_corrections(model, h, drc, trc)
        pred = E.ensemble_raw(model, X[val_m], pred_load[val_m])
        m = E._metrics(pred, va, vt)
        if tag == "none":
            base = m["MAE"]
        print(f"{tag:>12} {m['MAE']:>8.1f} {m['Bias']:>+8.1f} {m['midday']:>6.0f} {m['MAE'] - base:+9.1f}")
    print(f"  全校正相对 none 的 Δ = TCN 上 OOF 校正总贡献（v6 LGB 为 +67MW）")

    # ---- Block 2：hour_bias 粒度扫 ----
    print("\n" + "=" * 70)
    print("[Block 2] hour_bias_slots 扫 (仅 +hour_bias, 同一 OOF 残差重估)")
    print("=" * 70)
    print(f"{'slots':>6} {'val_MAE':>9} {'Δvs_96':>8}  hour_bias范围")
    base96 = None
    for slots in (24, 48, 96):
        cfg_s = E._cfg({**MEM_OVERRIDE, "hour_bias_slots": slots})
        hb_s, _, _ = E.estimate_corrections(oof["oof_pred"], oof["oof_mask"], cfg_s,
                                            times, X, actual)
        E.apply_corrections(model, hb_s, [], [])
        pred = E.ensemble_raw(model, X[val_m], pred_load[val_m])
        mae = E._mae(pred, va)
        if slots == 96:
            base96 = mae
        dlt = "" if base96 is None else f"{mae - base96:+8.1f}"
        print(f"{slots:>6} {mae:>9.1f} {dlt}  [{hb_s.min():+.0f},{hb_s.max():+.0f}]")
    print(f"  生产=96（v6 exp75: 24->1461.63, 96->1459.06）；TCN 上粒度趋势待验")


if __name__ == "__main__":
    main()
