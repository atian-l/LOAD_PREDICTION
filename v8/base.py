# -*- coding: utf-8 -*-
"""第二层：基础预测层 + Adaptive Model Selection。

base A = v6（加载根 models/model_bundle.pkl，= 目前最优版本，不重训）。
base B = reg_only LightGBM（10 成员，diversity 备选；全量训练 + OOF 校正）。
adaptive selection：OOF 池按天气型分桶比较 A/B，B 优超 margin 且样本足才切换；默认 A。
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from load_pred import train as T
from load_pred.model import EnsembleModel
from . import config as VC
from ._io import load_booster


def load_base_A() -> EnsembleModel:
    """加载根 model_bundle.pkl（v6 完整：40 成员 + MOS + hour_bias/drift/threshold）。

    根 boosters 被 git autocrlf=true 污染（LF->CRLF），`EnsembleModel.load` 内部
    `lgb.Booster(model_file=...)` 解析失败；故这里读入 bundle 后用 v8._io.load_booster
    （CRLF->LF 规范化 + model_str）加载成员。逻辑等价 EnsembleModel.load，不修改根文件。
    """
    path = Path(VC.BASE_A_BUNDLE)
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    # 兼容旧版 tuple 格式 threshold_corr：(feature, thr, hours, shift) -> dict(op=">")
    tc = bundle.get("threshold_corr") or []
    tc_norm = []
    for entry in tc:
        if isinstance(entry, dict):
            tc_norm.append(entry)
        else:
            feat_name, thr, hours_list, shift = entry
            tc_norm.append({"feature": feat_name, "op": ">", "thr": thr,
                            "hours": hours_list, "shift": shift})
    obj = EnsembleModel(
        feature_cols=bundle["feature_cols"],
        shrinkage=bundle.get("shrinkage", 1.0),
        train_meta=bundle.get("train_meta", {}),
        hour_bias=bundle.get("hour_bias"),
        mismatch_model=bundle.get("mismatch_model"),
        drift_corr=bundle.get("drift_corr"),
        threshold_corr=tc_norm,
        aggregation=bundle.get("aggregation", "median"),
        trim_frac=bundle.get("trim_frac", 0.2),
        mos_model=bundle.get("mos_model"),
    )
    for p in bundle["booster_paths"]:
        obj.members.append(load_booster(p))
    obj.member_residual = list(bundle["member_residual"])
    return obj


def train_base_B_full(times, X_full, pred_load, actual, usable, cfg_B, best_it,
                      full_mos, mismatch_model, corr_B) -> EnsembleModel:
    """全量训练 base B（reg_only 10 成员）+ 应用 OOF 估的校正参数。"""
    model = T.train_ensemble(times, X_full, pred_load, actual, usable, cfg_B, best_it, mos_model=full_mos)
    model.mismatch_model = mismatch_model
    hb_B, dc_B, tc_B = corr_B
    model.hour_bias = hb_B
    model.drift_corr = dc_B
    model.threshold_corr = tc_B
    return model


# --------------------------------------------------------------------------- #
# Adaptive Model Selection：天气型分桶 + OOF 偏好
# --------------------------------------------------------------------------- #
def weather_type(day_vec_row) -> str:
    """日级天气型：clearness_day_mean × temp_day_mean 9 宫格 + precip 雨型。"""
    if float(day_vec_row["precip_day_sum"]) > 0.1:
        return "rain"
    c = float(day_vec_row["clearness_day_mean"])
    t = float(day_vec_row["temp_day_mean"])
    cb = 0 if c < VC.CLEARNESS_BINS[0] else (1 if c < VC.CLEARNESS_BINS[1] else 2)
    tb = 0 if t < VC.TEMP_BINS[0] else (1 if t < VC.TEMP_BINS[1] else 2)
    return f"c{cb}_t{tb}"


def adaptive_preference(oof_pool: dict, day_vec_pool: pd.DataFrame) -> dict[tuple, str]:
    """OOF 池按 (天气型, 段) 分桶比较 base A vs B 的 OOF MAE，返回 {(天气型, 段): 'A'|'B'}。

    细化到段级：B 仅在其真正优于 A 的段被选用，避免"全日选 B"误伤夜间/晚间
    （按日选 B 会把 B 的午间收益抵消在夜间/晚间损失上，实测 overall +0.43）。
    B 优 A 超过 ADAPTIVE_MIN_MARGIN 且桶样本≥ADAPTIVE_MIN_N 才偏好 B；否则 A（保守）。
    天气型日级 + OOF 历史统计（非单日误差），避免偶然切换；val 零参与。
    """
    dates = pd.DatetimeIndex(oof_pool["dates"]).normalize()
    types = np.array([weather_type(day_vec_pool.loc[d]) for d in dates], dtype=object)
    segs = oof_pool["seg"]
    pref = {}
    for t in np.unique(types):
        for seg in VC.SEGMENTS:
            m = (types == t) & (segs == seg)
            if int(m.sum()) < VC.ADAPTIVE_MIN_N:
                pref[(str(t), seg)] = "A"
                continue
            mae_A = float(np.abs(oof_pool["base_A_oof"][m] - oof_pool["actual"][m]).mean())
            mae_B = float(np.abs(oof_pool["base_B_oof"][m] - oof_pool["actual"][m]).mean())
            pref[(str(t), seg)] = "B" if mae_B < mae_A * (1.0 - VC.ADAPTIVE_MIN_MARGIN) else "A"
    return pref


def select_base(d1_type: str, d1_seg: str, preference: dict) -> str:
    """部署时选 base：查 (天气型, 段) 偏好表，默认 A。"""
    return preference.get((d1_type, d1_seg), "A")
