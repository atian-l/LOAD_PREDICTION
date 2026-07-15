# -*- coding: utf-8 -*-
"""第一层：分段建模。按 hour 划分 night/day/evening 三段。

每段独立 correction model + trigger/α/w 参数。base 全天共享（adaptive 可日级选 A/B）。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as VC


def segment_mask(hours: np.ndarray) -> dict[str, np.ndarray]:
    """返回 {seg_name: bool mask}，hours 为 int 数组。"""
    h = np.asarray(hours, dtype=int)
    masks = {}
    for name, (lo, hi) in VC.SEGMENT_HOURS.items():
        masks[name] = (h >= lo) & (h < hi)
    return masks


def segment_array(hours: np.ndarray) -> np.ndarray:
    """返回每点的段名数组（object/str）。"""
    h = np.asarray(hours, dtype=int)
    out = np.empty(len(h), dtype=object)
    for name, (lo, hi) in VC.SEGMENT_HOURS.items():
        out[(h >= lo) & (h < hi)] = name
    return out
