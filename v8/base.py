# -*- coding: utf-8 -*-
"""第二层：基础预测层 + Adaptive Model Selection。

base A = v6（加载根 models/model_bundle.pkl，= 目前最优版本，不重训）。
base B = reg_only LightGBM（10 成员，diversity 备选；全量训练 + OOF 校正）。
adaptive selection：OOF 池按天气型分桶比较 A/B，B 优超 margin 且样本足才切换；默认 A。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from load_pred import train as T
from load_pred.model import EnsembleModel
from . import config as VC


def load_base_A() -> EnsembleModel:
    """加载根 model_bundle.pkl（v6 完整：40 成员 + MOS + hour_bias/drift/threshold）。"""
    return EnsembleModel.load(VC.BASE_A_BUNDLE)


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


def adaptive_preference(oof_pool: dict, day_vec_pool: pd.DataFrame) -> dict[str, str]:
    """OOF 池按天气型分桶比较 base A vs B 的 OOF MAE，返回 {天气型: 'A'|'B'}。

    B 优 A 超过 ADAPTIVE_MIN_MARGIN 且桶样本≥ADAPTIVE_MIN_N 才偏好 B；否则 A（保守）。
    天气型日级 + OOF 历史统计（非单日误差），避免偶然切换。
    """
    dates = pd.DatetimeIndex(oof_pool["dates"]).normalize()
    types = np.array([weather_type(day_vec_pool.loc[d]) for d in dates], dtype=object)
    pref = {}
    for t in np.unique(types):
        m = types == t
        if int(m.sum()) < VC.ADAPTIVE_MIN_N:
            pref[str(t)] = "A"
            continue
        mae_A = float(np.abs(oof_pool["base_A_oof"][m] - oof_pool["actual"][m]).mean())
        mae_B = float(np.abs(oof_pool["base_B_oof"][m] - oof_pool["actual"][m]).mean())
        pref[str(t)] = "B" if mae_B < mae_A * (1.0 - VC.ADAPTIVE_MIN_MARGIN) else "A"
    return pref


def select_base(d1_type: str, preference: dict) -> str:
    """部署时选 base：查偏好表，默认 A。"""
    return preference.get(d1_type, "A")
