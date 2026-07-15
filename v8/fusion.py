# -*- coding: utf-8 -*-
"""第五层：Dynamic Fusion w（动态，非固定）。

w = 邻居 OOF 中"修正有效"的加权比例 × 幅度可信度。
w 语义 = 修正幅度可信度（邻居中修正后改善越多 -> w 越大；恶化则 w->0）。
Trigger off 时 w=0；与 α（方向可信度）依据不同，非冗余。
"""
from __future__ import annotations
import numpy as np


def estimate_w(base_err: np.ndarray, corr_err: np.ndarray, weights: np.ndarray) -> float:
    """计算融合权重 w ∈ [0,1]。

    base_err / corr_err：邻居 OOF 点的 |base-actual| / |base+α·corr-actual|。
    weights：邻居点相似度权重（归一化）。
    w = clip(Σ w_i · max(0, 1 - corr_err_i/base_err_i), 0, 1)。
    """
    be = np.asarray(base_err, dtype=float) + 1e-6
    ce = np.asarray(corr_err, dtype=float)
    wv = np.maximum(0.0, 1.0 - ce / be)
    wt = np.asarray(weights, dtype=float)
    s = wt.sum()
    if s <= 0:
        return 0.0
    return float(np.clip(np.average(wv, weights=wt), 0.0, 1.0))
