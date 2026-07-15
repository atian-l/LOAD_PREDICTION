# -*- coding: utf-8 -*-
"""v8.1 P-Foundation：Chronos zero-shot 探针（无外部数据路径的最后一牌）。

问题：Foundation 预训练先验能否携带当前特征没有的信号，尤其符号/方向？
背景：符号通道不可迁移（Phase0），幅度已饱和（气象不确定探针），无外部新能源出力。
Foundation 是唯一不在当前特征内且不依赖外部数据的路径。

合规设计（不变量#2：lag 仅 pred_load，不用 actual）：
  - 变体 A（COMPLIANT）：context = past pred_load -> 预测 D+1。可部署。测 Foundation 先验
    能否改进外部预测（pred_load 序列动力学）。
  - 变体 B（UPPER-BOUND 诊断，非合规）：context = past actual -> 预测 D+1 actual。**仅诊断**，
    actual 作模型输入违#2，不可部署。设上界：若连 actual 历史都打不过 v6 / 抓不到符号方向，
    则任何 load-series 模型都补不了符号 -> 1445 终判硬上限。

Chronos 单变量（无天气/日历），故预期弱；但 B 是"符号是否本质封闭"的决定性上界测试。

运行：python -m v8_1.diag_foundation  报告 -> v8_1/output/p_foundation.md
需 chronos-forecasting + torch（已装）。模型 amazon/chronos-bolt-tiny（CPU）。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from load_pred import config as LC
from load_pred import train as T
from v8 import config as VC
from v8.model import V8Model

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = OUT_DIR / "p_foundation.md"

V6_VAL_MAE = 1445.62
CONTEXT_DAYS = 14
PRED_LEN = 96


def _load_chronos():
    """优先 bolt-tiny（快+新），失败回退 t5-tiny。返回 (pipeline, max_context)。"""
    import torch
    from chronos import ChronosPipeline
    for name, maxctx in [("amazon/chronos-bolt-tiny", 2048), ("amazon/chronos-t5-tiny", 512)]:
        try:
            p = ChronosPipeline.from_pretrained(name, device="cpu", dtype=torch.float32)
            return p, name, maxctx
        except Exception as e:
            print(f"    load {name} 失败: {type(e).__name__}: {str(e)[:120]}")
    return None, None, None


def _forecast_day(pipe, context_series, pred_len):
    """Chronos 预测 pred_len 点，返回中位数点预测 (pred_len,)。context_series 已去 NaN。"""
    import torch
    ctx = torch.tensor([np.asarray(context_series, dtype=np.float32).tolist()])
    fc = pipe.predict(context=ctx, prediction_length=pred_len)  # (1, Q_or_S, pred_len)
    fc = fc.squeeze(0).cpu().numpy()  # (Q_or_S, pred_len)
    return np.median(fc, axis=0)  # 中位数 = 点预测


def run(verbose: bool = True) -> dict:
    if verbose:
        print("[1/4] 构建数据 + 加载 v6（取 val v6 残差对照）...")
    times, X, pred_load, actual = T.build_dataset()
    tidx = pd.DatetimeIndex(times)
    pl = pred_load.reindex(tidx).astype(float)
    ac = actual.reindex(tidx).astype(float)

    vm8 = V8Model.load(VC.V8_BUNDLE)
    mm = vm8.mismatch_model
    baseA = vm8.base_A
    # v6 val 预测 + 残差（对照）
    vs, ve = pd.Timestamp(LC.VAL_START), pd.Timestamp(LC.VAL_END)
    buf = vs - pd.Timedelta(days=CONTEXT_DAYS + 2)
    em = (times >= buf) & (times <= ve)
    Xe = X[em]
    te = pd.DatetimeIndex(times[em])
    Xe_full = mm.transform(Xe)
    predA = np.asarray(baseA.predict_load(Xe_full, pred_load), dtype=float)
    ae = ac.reindex(te)
    vmask = (te >= vs) & (te <= ve) & ae.notna()
    v6_pred = pd.Series(predA, index=te).reindex(tidx)
    v6_resid = (ac - v6_pred)  # 全时间线，val 段非 NaN

    if verbose:
        print("[2/4] 加载 Chronos ...")
    pipe, mname, maxctx = _load_chronos()
    if pipe is None:
        print("    Chronos 全部加载失败 -> 无法运行 P-Foundation")
        REPORT.write_text("# P-Foundation\n\nChronos 加载失败（版本不兼容），探针无法运行。\n"
                           "需修复 chronos-forecasting/transformers 版本三角后再试。\n", encoding="utf-8")
        return {"blocked": True}
    ctx_pts = min(CONTEXT_DAYS * PRED_LEN, maxctx)
    if verbose:
        print(f"    用 {mname}, context={ctx_pts} 点 ({ctx_pts//PRED_LEN} 天)")

    # ---- val 逐日预测 ----
    if verbose:
        print("[3/4] val 逐日 Chronos zero-shot 预测（A=pred_load 合规, B=actual 上界诊断）...")
    val_dates = pd.date_range(vs.normalize(), ve.normalize(), freq="D")
    rows_A, rows_B, idxs = [], [], []
    for i, d in enumerate(val_dates):
        day_start = d
        day_end = d + pd.Timedelta("1D")
        ctx_end = day_start  # 不含当日
        ctx_start = ctx_end - pd.Timedelta(minutes=15 * ctx_pts)
        ctx_pl = pl.loc[ctx_start:ctx_end - pd.Timedelta(minutes=15)]
        ctx_ac = ac.loc[ctx_start:ctx_end - pd.Timedelta(minutes=15)]
        if len(ctx_pl) < ctx_pts * 0.8 or len(ctx_ac) < ctx_pts * 0.8:
            continue
        # 去NaN（ffill+bfill）
        ctx_pl = ctx_pl.ffill().bfill().values[-ctx_pts:]
        ctx_ac = ctx_ac.ffill().bfill().values[-ctx_pts:]
        day_actual = ac.loc[day_start:day_end - pd.Timedelta(minutes=15)]
        day_v6 = v6_pred.loc[day_start:day_end - pd.Timedelta(minutes=15)]
        day_pl = pl.loc[day_start:day_end - pd.Timedelta(minutes=15)]
        m = day_actual.notna() & day_v6.notna() & day_pl.notna()
        if m.sum() < 24:
            continue
        try:
            fcA = _forecast_day(pipe, ctx_pl, PRED_LEN)[:m.sum()] if m.sum() == PRED_LEN \
                else _forecast_day(pipe, ctx_pl, PRED_LEN)
            fcB = _forecast_day(pipe, ctx_ac, PRED_LEN)
        except Exception as e:
            if verbose:
                print(f"    day {d.date()} predict 失败: {e}")
            continue
        # 对齐到 m
        fa = pd.Series(fcA, index=day_actual.index).reindex(day_actual.index)[m]
        fb = pd.Series(fcB, index=day_actual.index).reindex(day_actual.index)[m]
        ya = day_actual[m]
        rows_A.append((fa.values, ya.values))
        rows_B.append((fb.values, ya.values))
        idxs.append((day_actual.index[m], day_v6[m].values, day_pl[m].values))

    if not rows_A:
        print("    无有效预测日")
        return {"blocked": True}

    fa = np.concatenate([r[0] for r in rows_A])
    ya = np.concatenate([r[1] for r in rows_A])
    fb = np.concatenate([r[0] for r in rows_B])
    yb = np.concatenate([r[1] for r in rows_B])
    # idxs = list of (day_index, v6_values, pl_values)
    v6_arr = np.concatenate([np.asarray(r[1]) for r in idxs])
    pl_arr = np.concatenate([np.asarray(r[2]) for r in idxs])

    mae_A = float(np.mean(np.abs(ya - fa)))
    mae_B = float(np.mean(np.abs(yb - fb)))
    mae_v6 = float(np.mean(np.abs(ya - v6_arr)))
    mae_pl = float(np.mean(np.abs(ya - pl_arr)))

    # 符号分析（B 上界）：chronos_B 残差方向 vs 外部预测误差方向
    resid_B = yb - fb                       # actual - chronos_B
    ext_err = ya - pl_arr                   # actual - pred_load（外部误差方向）
    nz = np.abs(ext_err) > 200.0
    sign_agree = float(np.mean(np.sign(resid_B[nz]) == np.sign(ext_err[nz]))) if nz.any() else float("nan")
    # 互补性：chronos_B 残差 vs v6 残差
    resid_v6 = ya - v6_arr
    from scipy.stats import spearmanr
    comp = float(spearmanr(resid_B, resid_v6).correlation)

    # ===================== 报告 =====================
    if verbose:
        print("[4/4] 报告 ...")
    L = []
    L.append("# v8.1 P-Foundation：Chronos zero-shot 探针\n")
    L.append(f"> 无外部数据路径最后一牌。Chronos 单变量 zero-shot（{mname}, CPU, context={ctx_pts}点).\n"
             f"> A=合规(context pred_load) | B=上界诊断(context actual, **非合规仅诊断**, 违#2不可部署).\n")
    L.append(f"val 评估点 N={len(ya)}（{len(rows_A)} 天）。\n")
    L.append("\n## A. val MAE 对比\n")
    L.append("| 预测源 | val MAE | 说明 |")
    L.append("|---|---|---|")
    L.append(f"| raw pred_load（外部）| {mae_pl:.0f} | 外部预测基线 |")
    L.append(f"| **v6（生产）** | {mae_v6:.0f} | 生产基线（memory 1445.62）|")
    L.append(f"| Chronos A（合规, pred_load context）| {mae_A:.0f} | Foundation 先验改进外部预测? |")
    L.append(f"| Chronos B（上界诊断, actual context）| {mae_B:.0f} | **非合规**，actual 历史上界 |")
    L.append(f"\n**关键对比**：Chronos B（actual 历史，非合规上界）val MAE = {mae_B:.0f} vs v6 {mae_v6:.0f}。"
             f"{'B < v6 -> Foundation 先验有潜力（但非合规，需重设计才能部署）' if mae_B < mae_v6 else 'B >= v6 -> 即便用 actual 历史，Foundation 单变量也打不过 v6 -> 符号通道对 load-series 模型本质封闭'}.\n")

    L.append("\n## B. 符号方向（B 上界）\n")
    L.append(f"- Chronos_B 残差方向 vs 外部误差方向（|ext|>200）符号一致率 = **{sign_agree:.3f}**"
             f"（>0.5=Foundation 抓到外部误差方向/符号；~0.5=未抓到）")
    L.append(f"- Chronos_B 残差 vs v6 残差 Spearman = **{comp:+.3f}**（低=互补信息，高=冗余）")
    L.append(f"\n符号一致率{'<0.5' if sign_agree < 0.5 else '>=0.5'} -> Foundation {'未' if sign_agree < 0.5 else ''}抓到符号方向。"
             f"{'符号通道对 load-series Foundation 本质封闭（无新能源数据不可补）' if sign_agree < 0.5 else '符号或有 load-dynamics 成分，但 B 非合规需重设计'}.\n")

    L.append("\n## C. 判定\n")
    b_beats_v6 = mae_B < mae_v6
    a_beats_pl = mae_A < mae_pl
    sign_captured = sign_agree >= 0.5
    if a_beats_pl or b_beats_v6 or sign_captured:
        verdict = "GO（有信号）"
        L.append(f"- Chronos A 改进外部预测: {'是' if a_beats_pl else '否'}（MAE {mae_A:.0f} vs pred_load {mae_pl:.0f}）")
        L.append(f"- Chronos B（上界）打过 v6: {'是' if b_beats_v6 else '否'}（{mae_B:.0f} vs {mae_v6:.0f}）")
        L.append(f"- 符号方向抓到: {'是' if sign_captured else '否'}（一致率 {sign_agree:.3f}）")
        L.append(f"- -> **{verdict}**：Foundation 先验携带当前特征没有的信号。**但 B 非合规（actual 输入）**，"
                 f"可部署路径须重设计（如 Foundation 作特征提取器 + 合规输入），且仍不保证破 1445。")
    else:
        verdict = "NO-GO（符号通道本质封闭）"
        L.append(f"- Chronos A 改进外部预测: 否（MAE {mae_A:.0f} >= pred_load {mae_pl:.0f}）")
        L.append(f"- Chronos B（actual 上界）打过 v6: 否（{mae_B:.0f} >= {mae_v6:.0f}）")
        L.append(f"- 符号方向抓到: 否（一致率 {sign_agree:.3f} < 0.5）")
        L.append(f"- -> **{verdict}**：即便用 actual 历史（非合规上界），Foundation 单变量既打不过 v6 也抓不到符号方向。"
                 f"**符号通道对 load-series 模型本质封闭**--符号由外部新能源出力误差驱动，无实际出力数据则任何"
                 f"load-series 模型（Foundation 亦然）都看不到。")
        L.append(f"- -> **1445.62 在当前数据下终判为硬上限**（无外部新能源出力不可破）。"
                 f"无新数据路径已穷尽：有符号值不可迁移 / 幅度饱和 / 气象不确定冗余 / Foundation 符号封闭。"
                 f"破 1445 唯一出路=获得外部实际新能源出力（或云量）。生产回 v6 1445.62。")

    report = "\n".join(L)
    REPORT.write_text(report, encoding="utf-8")
    if verbose:
        print("\n" + report)
        print(f"\n报告已写: {REPORT}")
    return {"mae_A": mae_A, "mae_B": mae_B, "mae_v6": mae_v6, "mae_pl": mae_pl,
            "sign_agree": sign_agree, "complementarity": comp,
            "verdict": verdict, "model": mname}


def main():
    run(verbose=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
