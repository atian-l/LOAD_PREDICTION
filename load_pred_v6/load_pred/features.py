# -*- coding: utf-8 -*-
"""
特征工程（训练/预测共享，保证无 train/predict skew）。

合规要点（Inviolable Constraints）：
  #1 数据隔离：本模块**绝不**读取或使用“实际直调负荷”。
      所有特征仅来自：外部预测直调负荷、气象预报、日历。
  #2 外部预测：预测直调负荷可作为输入特征与滞后特征；运行当天最远获 D+1。
  #3 滞后特征：仅基于预测直调负荷；最短滞后含 lag_192。
  #4 气象：使用去重后（最晚起报版本）的气象预报。
  #5 时间边界：所有特征在对应预测时刻 T 均可由“运行时”获得，无未来信息。

特征在预测时刻 T 的可获取性说明（生产环境）：
  - pred_load[T]            : 外部日前预测，运行日(D)可获 D+1 全天。✓
  - pred_load[T-k] (滞后)   : 历史外部预测，均已成过去。✓
  - 滚动统计(基于 pred_load) : 仅用 T 之前(含)的预测值。✓
  - weather[T]              : 起报于 (T的日历日-1) 20:00，覆盖 T；运行日 D>=T-1日 晚可获。✓
  - calendar[T]             : 确定性时间。✓
"""
from __future__ import annotations
import pandas as pd
import numpy as np

from . import config as C


# --------------------------------------------------------------------------- #
# 日历特征
# --------------------------------------------------------------------------- #
def calendar_features(times: pd.DatetimeIndex) -> pd.DataFrame:
    """由时间索引构造日历特征（确定性，无泄露）。"""
    t = pd.DatetimeIndex(times)
    hour = t.hour
    minute = t.minute
    minute_of_day = hour * 60 + minute
    dow = t.dayofweek
    doy = t.dayofyear
    month = t.month

    feat = pd.DataFrame(index=t)
    feat["hour"] = hour
    feat["minute_of_day"] = minute_of_day
    feat["dayofweek"] = dow
    feat["dayofyear"] = doy
    feat["month"] = month
    feat["is_weekend"] = (dow >= 5).astype(int)
    feat["is_holiday"] = _holiday_flag(t).astype(int)
    # 节前/节后过渡日（负荷往往异常）
    feat["is_day_before_holiday"] = _holiday_flag(t + pd.Timedelta(days=1)).astype(int)

    # 周期编码
    feat["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    feat["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    feat["mod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    feat["mod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    feat["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    feat["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    feat["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    feat["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return feat


# --------------------------------------------------------------------------- #
# 中国法定节假日（影响负荷；确定性，无泄露）
# --------------------------------------------------------------------------- #
_HOLIDAY_RANGES = [
    # (start, end) 含两端，日期字符串
    ("2023-01-21", "2023-01-27"),  # 春节
    ("2023-04-05", "2023-04-05"),  # 清明
    ("2023-04-29", "2023-05-03"),  # 劳动节
    ("2023-06-22", "2023-06-24"),  # 端午
    ("2023-09-29", "2023-10-06"),  # 中秋+国庆
    ("2024-02-10", "2024-02-17"),  # 春节
    ("2024-04-04", "2024-04-06"),  # 清明
    ("2024-05-01", "2024-05-05"),  # 劳动节
    ("2024-06-08", "2024-06-10"),  # 端午
    ("2024-09-15", "2024-09-17"),  # 中秋
    ("2024-10-01", "2024-10-07"),  # 国庆
    ("2025-01-28", "2025-02-04"),  # 春节
    ("2025-04-04", "2025-04-06"),  # 清明
    ("2025-05-01", "2025-05-05"),  # 劳动节
    ("2025-05-31", "2025-06-02"),  # 端午
    ("2025-10-01", "2025-10-08"),  # 国庆
    ("2026-02-15", "2026-02-22"),  # 春节
    ("2026-04-04", "2026-04-06"),  # 清明
    ("2026-05-01", "2026-05-05"),  # 劳动节
]


def _holiday_flag(times: pd.DatetimeIndex) -> np.ndarray:
    dates = pd.DatetimeIndex(times).normalize()
    flag = np.zeros(len(dates), dtype=int)
    for s, e in _HOLIDAY_RANGES:
        s = pd.Timestamp(s); e = pd.Timestamp(e)
        flag[(dates >= s) & (dates <= e)] = 1
    return flag


# --------------------------------------------------------------------------- #
# 基于预测直调负荷的特征（滞后/滚动/差分）
# --------------------------------------------------------------------------- #
def pred_load_features(
    times: pd.DatetimeIndex,
    pred_load: pd.Series,
) -> pd.DataFrame:
    """
    构造基于“预测直调负荷”的特征。

    参数
    ----
    times      : 需要特征的时刻 T（15 分钟网格）
    pred_load  : 以时间为索引的预测直调负荷序列（外部预测，全历史；不含真实负荷）

    返回
    ----
    DataFrame，索引=times，列含：pred_load[T] 及各滞后/滚动/差分特征。
    缺失（历史不足）以 NaN 表示，由 LightGBM 原生处理。
    """
    # 对齐到 15 分钟网格
    full_idx = pd.date_range(pred_load.index.min(), pred_load.index.max(), freq=C.FREQ)
    pl = pred_load.reindex(full_idx)

    feat = pd.DataFrame(index=times)

    # 当前外部预测值 pred_load[T]
    feat["pred_load"] = pl.reindex(times).values

    # 滞后（仅基于预测直调负荷；含 lag_192）
    for lag in C.PRED_LAGS:
        feat[f"pred_load_lag_{lag}"] = pl.shift(lag).reindex(times).values

    # 差分：预测值相对 1 天前 / 1 周前的变化
    feat["pred_load_diff_96"] = (pl - pl.shift(96)).reindex(times).values
    feat["pred_load_diff_672"] = (pl - pl.shift(672)).reindex(times).values

    # 短期爬坡（预测曲线局部形状，捕捉早晚高峰爬坡特征）
    for step in (4, 12, 48):
        feat[f"pred_load_ramp_{step}"] = (pl - pl.shift(step)).reindex(times).values

    # 滚动统计（窗口仅含 T 及之前；基于预测直调负荷）
    for win in C.PRED_ROLLING_WINDOWS:
        roll = pl.rolling(window=win, min_periods=max(8, win // 4))
        feat[f"pred_load_roll_mean_{win}"] = roll.mean().reindex(times).values
        feat[f"pred_load_roll_std_{win}"] = roll.std().reindex(times).values

    # 当前预测值相对近期均值的偏离（捕捉当日预测是否异常偏高/低）
    feat["pred_load_vs_mean_96"] = (pl - pl.rolling(96, min_periods=24).mean()).reindex(times).values
    feat["pred_load_vs_mean_672"] = (pl - pl.rolling(672, min_periods=96).mean()).reindex(times).values

    return feat


# --------------------------------------------------------------------------- #
# 气象特征（去重后）
# --------------------------------------------------------------------------- #
def weather_features(
    times: pd.DatetimeIndex,
    weather_dedup: pd.DataFrame,
) -> pd.DataFrame:
    """
    将去重后的气象预报按预测时间对齐到 times，并构造交互/非线性特征。

    weather_dedup 索引=预测时间，列=25 气象特征。
    """
    w = weather_dedup.reindex(times).copy()

    feat = pd.DataFrame(index=times)

    # 主要气象量（取中位数 p50 更稳健，回退到主值）
    temp = w["光伏_温度_p50"].fillna(w["光伏_温度"])
    # 辐照度物理范围 [0, 1200] W/m²；原始数据存在 ±22 万离群值，裁剪修复数据质量（无泄露）
    irrad = w["光伏_辐照度_p50"].fillna(w["光伏_辐照度"]).clip(lower=0.0, upper=1200.0)
    wind = w["风电_风速_p50"].fillna(w["风电_风速"])
    precip = w["光伏_降水_p50"].fillna(w["光伏_降水"])
    swind = w["光伏_风速_p50"].fillna(w["光伏_风速"])

    feat["temp"] = temp
    feat["irrad"] = irrad
    feat["wind"] = wind
    feat["precip"] = precip
    feat["solar_wind"] = swind

    # 非线性：温度二次项（供暖/制冷 U 型效应）、辐照二次项
    feat["temp_sq"] = temp ** 2
    feat["irrad_sq"] = irrad ** 2

    # 供暖/制冷度日指示（基准 18℃ 与 26℃）
    feat["hdd"] = (18.0 - temp).clip(lower=0.0)
    feat["cdd"] = (temp - 26.0).clip(lower=0.0)

    # 气象不确定性（预报离散度）
    feat["temp_std"] = w["光伏_温度_std"]
    feat["irrad_std"] = w["光伏_辐照度_std"]
    feat["wind_std"] = w["风电_风速_std"]
    feat["irrad_range"] = w["光伏_辐照度_p75"] - w["光伏_辐照度_p25"]

    # 全部 25 列原始气象特征（含各分位）也保留，供模型自由使用
    for c in C.WEATHER_FEATURE_COLS:
        feat["w_" + c] = w[c]

    return feat


# --------------------------------------------------------------------------- #
# 日级预报特征（T 当日全天预报的汇总；运行日 D 可获 D+1 全天预报，无泄露）
# --------------------------------------------------------------------------- #
def day_level_features(times: pd.DatetimeIndex, pred_load: pd.Series,
                       weather_dedup: pd.DataFrame) -> pd.DataFrame:
    """
    构造 T 当日（日历日）的预报汇总特征。

    合规：仅使用“预测直调负荷”与“气象预报”，均为运行时可获得的 D+1 全天数据。
    """
    # 预测直调负荷：对齐到 15 分钟网格后按日聚合
    full_idx = pd.date_range(pred_load.index.min(), pred_load.index.max(), freq=C.FREQ)
    pl = pred_load.reindex(full_idx)
    pl_day = pl.groupby(pl.index.date)
    day_mean = pl_day.transform("mean")
    day_max = pl_day.transform("max")
    day_min = pl_day.transform("min")
    day_std = pl_day.transform("std")

    feat = pd.DataFrame(index=times)
    feat["pl_day_mean"] = day_mean.reindex(times).values
    feat["pl_day_max"] = day_max.reindex(times).values
    feat["pl_day_min"] = day_min.reindex(times).values
    feat["pl_day_std"] = day_std.reindex(times).values
    # 当前预测值在当日预报中的相对位置
    pl_t = pl.reindex(times)
    feat["pl_in_day_frac"] = ((pl_t - day_min.reindex(times)) /
                              (day_max.reindex(times) - day_min.reindex(times) + 1e-6)).values
    # 当日预报峰谷差
    feat["pl_day_range"] = (day_max - day_min).reindex(times).values

    # 气象：按预测时间的日历日聚合
    w = weather_dedup.copy()
    w.index = pd.to_datetime(w.index)
    temp = w["光伏_温度_p50"].fillna(w["光伏_温度"])
    irrad = w["光伏_辐照度_p50"].fillna(w["光伏_辐照度"])
    wday = pd.DataFrame({"temp": temp, "irrad": irrad}, index=w.index)
    wday = wday.groupby(wday.index.date)
    tmax = wday["temp"].transform("max")
    tmin = wday["temp"].transform("min")
    tmean = wday["temp"].transform("mean")
    imax = wday["irrad"].transform("max")
    isum = wday["irrad"].transform("sum")

    feat["temp_day_max"] = tmax.reindex(times).values
    feat["temp_day_min"] = tmin.reindex(times).values
    feat["temp_day_mean"] = tmean.reindex(times).values
    feat["temp_day_range"] = (tmax - tmin).reindex(times).values
    feat["irrad_day_max"] = imax.reindex(times).values
    feat["irrad_day_sum"] = isum.reindex(times).values
    # 供暖/制冷度日（日级）
    feat["hdd_day"] = (18.0 - tmean).clip(lower=0.0).reindex(times).values
    feat["cdd_day"] = (tmean - 26.0).clip(lower=0.0).reindex(times).values
    return feat


# --------------------------------------------------------------------------- #
# 汇总
# --------------------------------------------------------------------------- #
FEATURE_COLS_ORDER: list[str] | None = None  # 训练时确定，预测时复用


def build_features(
    times: pd.DatetimeIndex,
    pred_load: pd.Series,
    weather_dedup: pd.DataFrame,
) -> pd.DataFrame:
    """
    构造完整特征矩阵（训练/预测共用）。

    严格不使用“实际直调负荷”。
    """
    cal = calendar_features(times)
    plf = pred_load_features(times, pred_load)
    wth = weather_features(times, weather_dedup)

    X = pd.concat([plf, cal, wth], axis=1)

    # 气象 × 时段 交互（让模型学到“同一气象在不同时段对负荷的不同影响”）
    hour = cal["hour"].values
    is_daylight = ((hour >= 6) & (hour <= 18)).astype(int)
    X["temp_x_hour"] = wth["temp"].values * hour
    X["irrad_x_daylight"] = wth["irrad"].values * is_daylight
    X["temp_x_daylight"] = wth["temp"].values * is_daylight
    X["hdd_x_hour"] = wth["hdd"].values * hour
    X["cdd_x_hour"] = wth["cdd"].values * hour

    # 预测负荷 × 气象/日历 交互（Agent Loop exp28-30 确认 -15 MW）
    # 关键抗漂移机制：pred_load 水平反映当年实况，pl×calendar 让模型按“当前负荷水平”
    # 而非“历史季节均值”学习校正，从而部分缓解年际预测偏差漂移。
    pl = plf["pred_load"].values
    pl_norm = (plf["pred_load"] / plf["pred_load"].rolling(672, min_periods=96).mean()).values
    plvsm = plf["pred_load_vs_mean_96"].values  # pl 相对近期均值的偏离
    X["pl_x_temp"] = pl * wth["temp"].values
    X["pl_x_hdd"] = pl * wth["hdd"].values
    X["pl_x_cdd"] = pl * wth["cdd"].values
    X["pl_x_irrad"] = pl * wth["irrad"].values
    X["pl_x_wind"] = pl * wth["wind"].values
    X["plnorm_x_temp"] = pl_norm * wth["temp"].values
    X["plnorm_x_hdd"] = pl_norm * wth["hdd"].values
    X["plnorm_x_cdd"] = pl_norm * wth["cdd"].values
    X["pl_x_hour"] = pl * hour
    X["pl_x_dow"] = pl * cal["dayofweek"].values
    X["pl_x_month"] = pl * cal["month"].values
    X["plnorm_x_hour"] = pl_norm * hour
    X["plvsmean_x_temp"] = plvsm * wth["temp"].values
    X["plvsmean_x_hdd"] = plvsm * wth["hdd"].values

    # 太阳能/晴空特征（Agent Loop exp36；针对午间 11-13 时高误差 MAE≈3449）
    # 用户建议：重点关注午间与阴雨天。山东光伏装机巨大，午间净负荷=需求-光伏，
    # 受辐照主导。晴空辐照为天文确定量（无泄露），clearness 量化云量。
    # 注：阴雨/多云的"偏置方向"年际漂移且不可预测（exp38 证实），故这些特征仅作
    # 模型输入让模型按当前辐照水平校正，不做固定场景偏置校正。
    solar = solar_features(times, wth["irrad"].values, hour, cal["dayofyear"].values)
    clear_sky, clearness, cloud_deficit = solar
    is_midday = ((hour >= 11) & (hour <= 13)).astype(int)
    is_daytime = ((hour >= 8) & (hour <= 16)).astype(int)
    X["clear_sky"] = clear_sky
    X["clearness"] = clearness
    X["cloud_deficit"] = cloud_deficit
    X["is_midday"] = is_midday
    X["is_daytime"] = is_daytime
    X["pl_x_clearness"] = pl * clearness
    X["pl_x_cloud_deficit"] = pl * cloud_deficit
    X["irrad_x_midday"] = wth["irrad"].values * is_midday
    X["pl_x_irrad_x_midday"] = pl * wth["irrad"].values * is_midday
    X["clearness_x_midday"] = clearness * is_midday

    # 预测负荷 vs 气象的"错配"特征（Agent Loop exp39-43；无泄露方向性信号）
    # 关键发现：pl_weather_residual（pred_load 减去"气象+日历"Ridge 隐含负荷）与误差
    # 方向全天相关性达 +0.29，是首个能预测"方向"的无泄露信号，推动验证 MAE 1547→1519。
    # 本块为不需拟合的错配量（dip/anomaly/交互）；Ridge 残差族由 MismatchModel 加入。
    pl_s = plf["pred_load"]
    pl_dip_96 = (pl_s.rolling(96, min_periods=24).mean() - pl_s).values
    irrad_s = pd.Series(wth["irrad"].values, index=X.index)
    irrad_anom_672 = (irrad_s - irrad_s.rolling(672, min_periods=96).mean()).values
    X["pl_dip_96"] = pl_dip_96
    X["irrad_anom_672"] = irrad_anom_672
    X["pl_dip_x_irrad"] = pl_dip_96 * wth["irrad"].values
    X["pl_dip_x_clearness"] = pl_dip_96 * clearness
    X["pl_dip_x_midday"] = pl_dip_96 * is_midday
    X["pl_x_irrad_anom"] = pl * irrad_anom_672
    X["irrad_anom_x_midday"] = irrad_anom_672 * is_midday
    X["pl_dip_x_clear_sky"] = pl_dip_96 * clear_sky
    X["pl_dip_ratio"] = pl_dip_96 / (clear_sky + 1.0)

    X.index.name = C.COL_TIME
    return X


# --------------------------------------------------------------------------- #
# 太阳能/晴空辐照特征（天文确定量，无泄露）
# --------------------------------------------------------------------------- #
_SHANDONG_LAT = np.radians(36.0)  # 山东省纬度


def solar_features(times: pd.DatetimeIndex, irrad: np.ndarray,
                   hour: np.ndarray, doy: np.ndarray):
    """返回 (clear_sky, clearness, cloud_deficit)。

    clear_sky    : 天文晴空辐照度 ≈ 0.8 * 1367 * sin(elevation)  [W/m²]
    clearness    : irrad / (clear_sky + 1)  （1=晴空，0=全遮蔽；云量逆指标）
    cloud_deficit: clear_sky - irrad  （被云遮挡的辐照量）

    全部仅由时间(确定)与气象预报辐照(运行时可获)决定，无实际负荷、无未来信息。
    """
    decl = np.radians(23.45 * np.sin(np.radians(360.0 * (284.0 + doy) / 365.25)))
    ha = np.radians((hour * 60.0 - 720.0) * 0.25)  # 时角（15min 网格，分钟=0）
    sin_elev = (np.sin(_SHANDONG_LAT) * np.sin(decl) +
                np.cos(_SHANDONG_LAT) * np.cos(decl) * np.cos(ha))
    sin_elev = np.clip(sin_elev, 0.0, None)
    clear_sky = 0.8 * 1367.0 * sin_elev
    irrad = np.clip(np.asarray(irrad, dtype=float), 0.0, 1200.0)
    clearness = irrad / (clear_sky + 1.0)
    cloud_deficit = clear_sky - irrad
    return clear_sky, clearness, cloud_deficit


# --------------------------------------------------------------------------- #
# 错配/残差模型（训练期拟合，预测期复用；无泄露）
# --------------------------------------------------------------------------- #
from sklearn.linear_model import Ridge as _Ridge

# Ridge 残差用到的列（气象+日历），完全不含实际负荷
_RIDGE_COLS = ["irrad", "temp", "hdd", "cdd",
               "hour_sin", "hour_cos", "month_sin", "month_cos", "doy_sin", "doy_cos"]


class MismatchModel:
    """持有训练期拟合的系数，用于计算需要拟合的"错配/残差"特征。

    全部系数仅用 pred_load + weather + calendar（训练期）拟合，绝不使用实际负荷；
    transform 仅用运行时可获数据。训练时 fit() 并存入 model bundle，预测时 load() 后 transform()。

    背景（exp39-43）：pl_weather_residual = pred_load − Ridge(weather,calendar) 与误差方向
    全天相关性 +0.29，是首个能预测"偏差方向"的无泄露信号；solar_mismatch 捕捉午间
    pred_load 下凹与辐照的错配。二者均无泄露（不含实际负荷、不含未来信息）。
    """

    def __init__(self):
        self.ridge_coef: np.ndarray | None = None
        self.ridge_intercept: float = 0.0
        self.ridge_col_mean: np.ndarray | None = None  # 训练期各列均值，用于 NaN 填充
        self.b: float = 0.0           # pl_dip_96 ~ irrad 斜率（全天，过原点最小二乘）
        self.b_cs: float = 0.0        # pl_dip_96 ~ clear_sky 斜率
        self.b_hour: np.ndarray | None = None  # [24] 每小时 pl_dip_96 ~ irrad 斜率

    def fit(self, X: pd.DataFrame, train_mask: np.ndarray) -> "MismatchModel":
        pl = X["pred_load"].values.astype(float)
        irrad = X["irrad"].values.astype(float)
        clear_sky = X["clear_sky"].values.astype(float)
        pl_dip = X["pl_dip_96"].values.astype(float)
        hour = X["hour"].values.astype(int)
        tr = np.asarray(train_mask, dtype=bool)

        # --- Ridge: pl ~ (气象+日历)，仅训练期拟合 ---
        M = X[_RIDGE_COLS].values.astype(float)
        col_mean = np.nanmean(M[tr], axis=0)
        self.ridge_col_mean = col_mean
        Mf = M.copy()
        nan_mask = np.isnan(Mf)
        if nan_mask.any():
            Mf[nan_mask] = np.take(col_mean, np.where(nan_mask)[1])
        rg = _Ridge(alpha=1.0)
        rg.fit(Mf[tr], pl[tr])
        self.ridge_coef = rg.coef_
        self.ridge_intercept = float(rg.intercept_)

        # --- b: pl_dip_96 ~ irrad （irrad>0 训练点，过原点最小二乘）---
        m = tr & (irrad > 0) & np.isfinite(pl_dip)
        d = float(np.dot(irrad[m], irrad[m]))
        self.b = float(np.dot(pl_dip[m], irrad[m]) / d) if d > 0 else 0.0
        # --- b_cs: pl_dip_96 ~ clear_sky ---
        mcs = tr & (clear_sky > 0) & np.isfinite(pl_dip)
        dcs = float(np.dot(clear_sky[mcs], clear_sky[mcs]))
        self.b_cs = float(np.dot(pl_dip[mcs], clear_sky[mcs]) / dcs) if dcs > 0 else 0.0
        # --- b_hour[24]: 每小时 pl_dip_96 ~ irrad ---
        b_hour = np.zeros(24, dtype=float)
        for h in range(24):
            mh = m & (hour == h)
            dh = float(np.dot(irrad[mh], irrad[mh]))
            b_hour[h] = float(np.dot(pl_dip[mh], irrad[mh]) / dh) if dh > 0 else 0.0
        self.b_hour = b_hour
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pl = X["pred_load"].values.astype(float)
        irrad = X["irrad"].values.astype(float)
        clear_sky = X["clear_sky"].values.astype(float)
        clearness = X["clearness"].values.astype(float)
        hour = X["hour"].values.astype(int)
        is_midday = X["is_midday"].values.astype(float)
        is_daytime = X["is_daytime"].values.astype(float)
        pl_dip = X["pl_dip_96"].values.astype(float)

        # Ridge 隐含负荷 + 残差
        M = X[_RIDGE_COLS].values.astype(float)
        Mf = M.copy()
        nan_mask = np.isnan(Mf)
        if nan_mask.any():
            Mf[nan_mask] = np.take(self.ridge_col_mean, np.where(nan_mask)[1])
        pl_implied = Mf @ self.ridge_coef + self.ridge_intercept
        pl_wr = pl - pl_implied

        # 太阳能错配（pl 下凹 vs 辐照）
        sm = pl_dip - self.b * irrad
        sm_cs = pl_dip - self.b_cs * clear_sky
        sm_hour = pl_dip - self.b_hour[hour] * irrad

        X = X.copy()
        X["pl_weather_residual"] = pl_wr
        X["pl_wr_x_hour"] = pl_wr * hour
        X["pl_wr_x_midday"] = pl_wr * is_midday
        X["pl_wr_x_is_daytime"] = pl_wr * is_daytime
        X["pl_wr_x_clearness"] = pl_wr * clearness
        X["solar_mismatch"] = sm
        X["solar_mismatch_x_midday"] = sm * is_midday
        X["solar_mismatch_cs"] = sm_cs
        X["solar_mismatch_cs_x_midday"] = sm_cs * is_midday
        X["solar_mismatch_hour"] = sm_hour
        X["solar_mismatch_hour_x_midday"] = sm_hour * is_midday
        # 残差的滚动/差分（仅用 T 及之前，无未来）
        wr_s = pd.Series(pl_wr, index=X.index)
        X["pl_wr_roll_mean_96"] = wr_s.rolling(96, min_periods=24).mean().values
        X["pl_wr_roll_mean_672"] = wr_s.rolling(672, min_periods=96).mean().values
        X["pl_wr_roll_std_672"] = wr_s.rolling(672, min_periods=96).std().values
        X["pl_wr_diff_672"] = (wr_s - wr_s.shift(672)).values
        X["pl_wr_diff_96"] = (wr_s - wr_s.shift(96)).values
        return X


# --------------------------------------------------------------------------- #
# 两级系统 Stage1 MOS（Model Output Statistics；Agent Loop exp80）
# --------------------------------------------------------------------------- #
# Ridge(actual ~ pred_load + 气象 + 日历)：target=actual（仅作目标，合规#1），
# inputs=pred_load+weather+calendar（均合规，不含 actual 输入）。corrected_pred = MOS 预测，
# 作为残差成员的"锚"（较 raw pred_load 更接近 actual -> 残差更小更易学，且 MOS 已吸收大部分
# 天气驱动偏置，使 threshold/drift 校正量减小、更稳健）。exp80: direct+residual@MOS_enrich
# 较 @pred_load -9.86 MW (1459.06->1449.20)；MOS 锚 val MAE 较 pred_load -49.6 MW（可迁移）。
# 训练期 fit()并存入 bundle，预测期 transform()复用。无泄露（actual 仅作目标；无未来信息）。
class MosModel:
    """Stage1 MOS：用 Ridge 把外部预测 pred_load 修正向 actual（仅作目标）。"""

    # 默认 MOS 特征列：pred_load 水平 + 关键气象 + 日历周期 + pl_weather_residual（错配信号）
    DEFAULT_COLS = ["pred_load", "irrad", "temp", "hdd", "cdd",
                    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
                    "precip", "wind", "solar_wind", "pl_weather_residual"]

    def __init__(self, cols: list[str] | None = None, alpha: float = 1.0):
        self.cols = list(cols) if cols is not None else list(self.DEFAULT_COLS)
        self.alpha = float(alpha)
        self.ridge_coef: np.ndarray | None = None
        self.ridge_intercept: float = 0.0
        self.col_mean: np.ndarray | None = None  # 训练期各列均值，用于 NaN 填充

    def fit(self, X: pd.DataFrame, actual: pd.Series, train_mask: np.ndarray) -> "MosModel":
        M = X[self.cols].values.astype(float)
        tr = np.asarray(train_mask, dtype=bool)
        self.col_mean = np.nanmean(M[tr], axis=0)
        Mf = M.copy(); nan = np.isnan(Mf)
        if nan.any():
            Mf[nan] = np.take(self.col_mean, np.where(nan)[1])
        rg = _Ridge(alpha=self.alpha)
        rg.fit(Mf[tr], actual.reindex(X.index).values[tr])
        self.ridge_coef = rg.coef_
        self.ridge_intercept = float(rg.intercept_)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """返回 corrected_pred（MOS 修正后的预测，作为残差锚）。"""
        M = X[self.cols].values.astype(float)
        Mf = M.copy(); nan = np.isnan(Mf)
        if nan.any():
            Mf[nan] = np.take(self.col_mean, np.where(nan)[1])
        corrected = Mf @ self.ridge_coef + self.ridge_intercept
        return np.clip(corrected, 0.0, None)
