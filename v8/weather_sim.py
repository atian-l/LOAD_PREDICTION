# -*- coding: utf-8 -*-
"""统一天气相似度（服务于 adaptive/trigger/α/w 四层）。

日级天气向量（跨年稳定物理量），标准化欧氏距离 + KNN + softmax 权重。
查询池 = 训练期 OOF 日（≤ TRAIN_END），不含 val。物理量跨年稳定 -> 跨年泛化。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as VC


def day_weather_vectors(X: pd.DataFrame, times: pd.DatetimeIndex) -> pd.DataFrame:
    """从特征矩阵提取日级天气向量（按时点列按日聚合）。

    返回 DataFrame，index=date，列=WEATHER_SIM_COLS：
      temp_day_mean（时点 temp 按日 mean）
      irrad_day_sum（时点 irrad 按日 sum，反映日总辐照）
      clearness_day_mean（时点 clearness 按日 mean）
      precip_day_sum（时点 precip 按日 sum）
      temp_day_range（日 max - min）
    注：build_features 不调 day_level_features，故从时点 temp/irrad/clearness/precip 聚合。
    """
    dates = pd.DatetimeIndex(times).normalize()
    df = pd.DataFrame({
        "date": dates,
        "temp": X["temp"].values.astype(float),
        "irrad": X["irrad"].values.astype(float),
        "clearness": X["clearness"].values.astype(float),
        "precip": X["precip"].values.astype(float),
    })
    g = df.groupby("date")
    day = pd.DataFrame({
        "temp_day_mean": g["temp"].mean(),
        "irrad_day_sum": g["irrad"].sum(),
        "clearness_day_mean": g["clearness"].mean(),
        "precip_day_sum": g["precip"].sum(),
        "temp_day_range": g["temp"].max() - g["temp"].min(),
    })
    return day[VC.WEATHER_SIM_COLS]


class WeatherSim:
    """日级天气 KNN 相似度。

    fit: 用训练期日向量估计标准化统计 + 存查询池。
    query: 给预测日向量，返回 K 个邻居日在池中的位置 + softmax 权重。
    """

    def __init__(self, k: int = VC.WEATHER_SIM_K, tau: float = 1.0):
        self.k = int(k)
        self.tau = float(tau)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.pool_vec_: np.ndarray | None = None   # (n_days, D) standardized
        self.pool_dates_: np.ndarray | None = None  # datetime64[D]

    def fit(self, day_vec: pd.DataFrame, train_mask: np.ndarray | None = None) -> "WeatherSim":
        v = day_vec.values.astype(float)
        if train_mask is not None:
            v_train = v[train_mask]
        else:
            v_train = v
        self.mean_ = np.nanmean(v_train, axis=0)
        self.std_ = np.nanstd(v_train, axis=0) + 1e-6
        self.pool_vec_ = (v - self.mean_) / self.std_
        # NaN 填 0（标准化后，避免距离爆炸）
        self.pool_vec_ = np.nan_to_num(self.pool_vec_, nan=0.0)
        self.pool_dates_ = np.asarray(day_vec.index.values).astype("datetime64[D]")  # 统一到 [D]
        return self

    def query(self, q_vec: np.ndarray, exclude_date=None) -> tuple[np.ndarray, np.ndarray]:
        """返回 (邻居在池中的位置索引, softmax 权重)。exclude_date 排除当日（防自泄露）。"""
        q = (np.asarray(q_vec, dtype=float) - self.mean_) / self.std_
        q = np.nan_to_num(q, nan=0.0)
        d = np.sqrt(((self.pool_vec_ - q) ** 2).sum(axis=1))
        # 排除当日（OOF 模拟时防自泄露）
        if exclude_date is not None:
            d_excl = d.copy()
            excl = np.datetime64(pd.Timestamp(exclude_date)).astype("datetime64[D]")
            d_excl[self.pool_dates_ == excl] = np.inf
        else:
            d_excl = d
        k = min(self.k, np.sum(np.isfinite(d_excl)))
        if k <= 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        idx = np.argpartition(d_excl, k - 1)[:k]
        # 过滤 inf（被排除的）
        idx = idx[np.isfinite(d_excl[idx])]
        idx = idx[np.argsort(d_excl[idx])]
        w = np.exp(-d_excl[idx] / max(self.tau, 1e-6))
        s = w.sum()
        if s <= 0:
            return idx, np.full(len(idx), 1.0 / max(len(idx), 1))
        return idx, w / s

    def pool_matrix(self) -> np.ndarray:
        """返回标准化后的查询池矩阵（供批量操作）。"""
        return self.pool_vec_
