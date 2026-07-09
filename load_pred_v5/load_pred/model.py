# -*- coding: utf-8 -*-
"""
模型封装（LightGBM 集成）。

设计：
  - EnsembleModel 持有一组 LightGBM booster 成员，每个成员标注是否为残差模式
    （残差模式：预测 = pred_load[T] + resid_hat；直接模式：预测 = raw）。
  - 最终预测 = 各成员预测的中位数（对离群成员更稳健）；可选收缩 λ：
    pred = pred_load + λ*(ens - pred_load)。
  - 训练阶段确定 成员配置 / best_iter / λ，保存后预测模式仅加载推理（Constraint #6 解耦）。

合规：
  - 残差目标 = actual - pred_load（pred_load 为外部预测，允许使用；actual 仅作目标）。
  - 全程不使用真实负荷作为输入特征。
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as C


class EnsembleModel:
    """LightGBM 集成回归模型。"""

    def __init__(self, feature_cols: list[str], shrinkage: float = 1.0,
                 train_meta: dict | None = None,
                 hour_bias: np.ndarray | None = None,
                 mismatch_model=None,
                 drift_corr: list | None = None,
                 threshold_corr: list | None = None):
        self.feature_cols = feature_cols
        self.shrinkage = float(shrinkage)   # λ
        self.train_meta = train_meta or {}
        # 小时偏置校正（来自训练期 3-fold OOF，仅用训练数据估计；无泄露）。
        # 按 len 自适应索引：24=逐小时，48=逐半小时，96=逐 15min slot（exp75 确认 96 维 -2.57 MW）。
        # predict 时减去 hour_bias[slot]，消除模型系统性的时段偏置。
        self.hour_bias = None if hour_bias is None else np.asarray(hour_bias, dtype=float)
        # 错配/残差模型（features.MismatchModel）：训练期拟合，预测期 transform 出
        # pl_weather_residual / solar_mismatch 等需拟合特征。无泄露（仅用 pred_load+weather+calendar）。
        self.mismatch_model = mismatch_model
        # 漂移方向校正：[(特征名, β[24])]，β 由 3 折 OOF 残差逐小时估计（无泄露）。
        # predict 时叠加 β[hour]·feat_value（仅午间等指定小时非零）。exp47-49 确认午间 β·pl_wr -13 MW。
        self.drift_corr = drift_corr or []
        # 阈值场景校正：[(特征名, thr, hours_list|None, shift)]，shift 由 3 折 OOF 残差估计（无泄露）。
        # predict 时对 (feat>thr [且 hour∈hours]) 的点 pred -= shift。exp58-61 确认晴天午间/阴雨天修正。
        self.threshold_corr = threshold_corr or []
        self.members: list[lgb.Booster] = []
        self.member_residual: list[bool] = []

    # ---------------- 增删成员 ---------------- #
    def add_member(self, booster: lgb.Booster, is_residual: bool) -> None:
        self.members.append(booster)
        self.member_residual.append(bool(is_residual))

    # ---------------- 推理 ---------------- #
    def ensemble_raw(self, X: pd.DataFrame) -> np.ndarray:
        """返回各成员最终预测（已按 residual 还原）的中位数。"""
        if not self.members:
            raise RuntimeError("集成无成员。")
        # pred_load_at_T 由调用方在 predict_load 传入；此处成员原始输出需在外部还原。
        raise NotImplementedError("使用 predict_load。")

    def predict_load(self, X: pd.DataFrame, pred_load_at_T: pd.Series) -> np.ndarray:
        """
        返回最终“直调负荷”预测（含收缩 λ）。
        ens = median_i ( residual_i ? pred_load[T]+raw_i : raw_i )
        pred = pred_load[T] + λ * (ens - pred_load[T])
        中位数聚合对个别过拟合/离群成员更稳健（实验确认较均值降低验证 MAE）。
        """
        if not self.members:
            raise RuntimeError("集成无成员。")
        pl = pred_load_at_T.reindex(X.index).values.astype(float)
        member_preds = np.empty((len(self.members), len(X)), dtype=float)
        for i, (booster, is_res) in enumerate(zip(self.members, self.member_residual)):
            raw = booster.predict(X[self.feature_cols])
            member_preds[i] = pl + raw if is_res else raw
        ens = np.median(member_preds, axis=0)
        pred = pl + self.shrinkage * (ens - pl)
        hours = pd.DatetimeIndex(X.index).hour.values.astype(int)
        # 小时偏置校正（OOF 估计；按 hour_bias 长度自适应索引：24=逐小时,48=逐半小时,96=逐15min）
        if self.hour_bias is not None:
            n = len(self.hour_bias)
            if n == 24:
                idx = hours
            else:  # 48/96 等按分钟 slot 索引
                mod = hours * 60 + pd.DatetimeIndex(X.index).minute.values
                idx = (mod // (1440 // n)).astype(int)
            pred = pred - self.hour_bias[idx]
        # 漂移方向校正（OOF 估计的 β·feat，仅指定小时非零；无泄露）
        for feat_name, beta in self.drift_corr:
            beta = np.asarray(beta, dtype=float)
            pred = pred + beta[hours] * X[feat_name].values.astype(float)
        # 阈值场景校正（OOF 估计的 shift；晴午间高估/阴雨天/低温/多云低估；无泄露）
        # 每项为 dict: {feature, op, thr, hours, shift}；op ∈ >/>=/</<=/range(range 时 thr=[lo,hi))。
        for tc in self.threshold_corr:
            feat_name = tc["feature"]
            fv = X[feat_name].values.astype(float)
            op = tc.get("op", ">")
            thr = tc["thr"]
            if op == "range":
                lo, hi = thr
                sel = (fv >= lo) & (fv < hi)
            elif op == ">=":
                sel = fv >= thr
            elif op == "<":
                sel = fv < thr
            elif op == "<=":
                sel = fv <= thr
            else:  # ">"（默认）
                sel = fv > thr
            hours_list = tc.get("hours")
            if hours_list is not None:
                sel = sel & np.isin(hours, list(hours_list))
            shift = tc.get("shift", 0.0)
            if shift != 0.0:
                pred[sel] = pred[sel] - shift
        return np.clip(pred, 0.0, None)

    # ---------------- 持久化 ---------------- #
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        booster_dir = path.parent / "boosters"
        booster_dir.mkdir(parents=True, exist_ok=True)
        booster_paths = []
        for i, bst in enumerate(self.members):
            p = booster_dir / f"member_{i:03d}.txt"
            bst.save_model(str(p))
            booster_paths.append(str(p))
        bundle = {
            "feature_cols": self.feature_cols,
            "shrinkage": self.shrinkage,
            "train_meta": self.train_meta,
            "member_residual": self.member_residual,
            "booster_paths": booster_paths,
            "hour_bias": None if self.hour_bias is None else self.hour_bias.tolist(),
            "mismatch_model": self.mismatch_model,
            "drift_corr": [(n, np.asarray(b, dtype=float).tolist()) for n, b in self.drift_corr],
            "threshold_corr": [dict(tc) for tc in self.threshold_corr],
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)

    @classmethod
    def load(cls, path: Path) -> "EnsembleModel":
        path = Path(path)
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        # 兼容旧版 tuple 格式 threshold_corr：(feature, thr, hours, shift) → dict(op=">")
        tc = bundle.get("threshold_corr") or []
        tc_norm = []
        for entry in tc:
            if isinstance(entry, dict):
                tc_norm.append(entry)
            else:  # 旧版 tuple (feature, thr, hours, shift)
                feat_name, thr, hours_list, shift = entry
                tc_norm.append({"feature": feat_name, "op": ">", "thr": thr,
                                "hours": hours_list, "shift": shift})
        obj = cls(feature_cols=bundle["feature_cols"],
                  shrinkage=bundle.get("shrinkage", 1.0),
                  train_meta=bundle.get("train_meta", {}),
                  hour_bias=bundle.get("hour_bias"),
                  mismatch_model=bundle.get("mismatch_model"),
                  drift_corr=bundle.get("drift_corr"),
                  threshold_corr=tc_norm)
        for p in bundle["booster_paths"]:
            obj.members.append(lgb.Booster(model_file=p))
        obj.member_residual = list(bundle["member_residual"])
        return obj


# ---------------- 兼容旧名 ---------------- #
LoadModel = EnsembleModel
