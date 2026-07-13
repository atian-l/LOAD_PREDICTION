# -*- coding: utf-8 -*-
"""
模型封装（TCN 集成）。

与 load_pred/model.py（LightGBM）公共接口与后处理逐行一致，仅将 booster 换为 TCN（PyTorch）：
  - EnsembleModel 持有一组 TCN 成员，每个成员标注是否为残差模式
    （残差模式：预测 = anchor[T] + resid_hat；直接模式：预测 = raw）。
  - 最终预测 = 各成员预测的中位数（对离群成员更稳健）；可选收缩 λ：
    pred = anchor + λ*(ens - anchor)。
  - 训练阶段确定 成员配置 / epochs / λ / OOF 校正，保存后预测模式仅加载推理（Constraint #6 解耦）。
  - predict_load 中除"成员 raw 预测"一行外，锚/聚合/收缩/hour_bias/drift_corr/threshold_corr/clip
    全部与 LightGBM 版相同（模型无关）。

合规：
  - 残差目标 = actual - anchor（anchor 为 MOS 修正或 pred_load，均为外部预测；actual 仅作目标）。
  - 全程不使用真实负荷作为输入特征。
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config as C
from .tcn import TCN, predict_tcn, get_device


class EnsembleModel:
    """TCN 集成回归模型（接口与 LightGBM 版一致）。"""

    def __init__(self, feature_cols: list[str], shrinkage: float = 1.0,
                 train_meta: dict | None = None,
                 hour_bias: np.ndarray | None = None,
                 mismatch_model=None,
                 drift_corr: list | None = None,
                 threshold_corr: list | None = None,
                 aggregation: str = "median",
                 trim_frac: float = 0.2,
                 mos_model=None,
                 feat_mean: np.ndarray | None = None,
                 feat_std: np.ndarray | None = None,
                 tcn_config: dict | None = None,
                 device: str = "auto"):
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
        # 阈值场景校正：[{feature,op,thr,hours,shift}]，shift 由 3 折 OOF 残差估计（无泄露）。
        # predict 时对 (feat op thr [且 hour∈hours]) 的点 pred -= shift。exp58-61 确认晴天午间/阴雨天修正。
        self.threshold_corr = threshold_corr or []
        # 集成聚合方式（exp78 确认 trimmed-mean 较 median raw -11.56 MW，但 exp79 全管线 median 更优）：
        #   "median"=中位数（默认，对离群成员稳健）；"mean"=均值；"trimmed"=去两端各 trim_frac/2 后均值。
        self.aggregation = aggregation
        self.trim_frac = float(trim_frac)
        # 两级系统 Stage1 MOS（features.MosModel）：训练期拟合 Ridge(actual~pred_load+weather+calendar)，
        # 预测期 transform 出 corrected_pred 作为残差成员的"锚"（较 raw pred_load 更接近 actual）。
        # exp80: direct+residual@MOS_enrich 较 @pred_load -9.86 MW。无泄露（actual 仅作 MOS 目标）。
        self.mos_model = mos_model
        # ---- TCN 专用 ----
        # feat_mean/feat_std：训练期特征列均值/标准差，用于推理时 NaN 填充 + 标准化
        # （神经网对尺度敏感；NaN 填 feat_mean 后标准化为 0。输入预处理，无泄露）。
        self.feat_mean = None if feat_mean is None else np.asarray(feat_mean, dtype=np.float32)
        self.feat_std = None if feat_std is None else np.asarray(feat_std, dtype=np.float32)
        # tcn_config：重建 TCN 架构所需 {num_channels, kernel_size, dropout}（load 时用）。
        self.tcn_config = tcn_config or {}
        self.device = get_device(device)
        self.members: list[TCN] = []
        self.member_residual: list[bool] = []

    def _aggregate(self, member_preds: np.ndarray) -> np.ndarray:
        """按 self.aggregation 聚合各成员预测（行=成员，列=样本）。"""
        if self.aggregation == "mean":
            return np.mean(member_preds, axis=0)
        if self.aggregation == "trimmed":
            n = member_preds.shape[0]
            k = int(np.floor(n * self.trim_frac / 2))
            if k > 0 and (n - 2 * k) > 0:
                Ms = np.sort(member_preds, axis=0)
                return np.mean(Ms[k:n - k], axis=0)
            return np.mean(member_preds, axis=0)
        return np.median(member_preds, axis=0)  # "median"（默认）

    # ---------------- 增删成员 ---------------- #
    def add_member(self, tcn: TCN, is_residual: bool) -> None:
        self.members.append(tcn)
        self.member_residual.append(bool(is_residual))

    # ---------------- 推理 ---------------- #
    def predict_load(self, X: pd.DataFrame, pred_load_at_T: pd.Series) -> np.ndarray:
        """
        返回最终“直调负荷”预测（含收缩 λ）。
        ens = median_i ( residual_i ? anchor[T]+raw_i : raw_i )
        pred = anchor[T] + λ * (ens - anchor[T])
        中位数聚合对个别过拟合/离群成员更稳健（实验确认较均值降低验证 MAE）。

        除成员 raw 预测（TCN 前向）外，锚/聚合/收缩/hour_bias/drift_corr/threshold_corr/clip
        与 LightGBM 版逐行相同（模型无关）。
        """
        if not self.members:
            raise RuntimeError("集成无成员。")
        pl = pred_load_at_T.reindex(X.index).values.astype(float)
        # 残差锚：两级系统 Stage1 MOS 的 corrected_pred（较 raw pred_load 更接近 actual；exp80 -9.86 MW）。
        # 无 MOS 时回退到 raw pred_load（向后兼容）。直接成员不受锚影响（预测 actual 本身）。
        anchor = self.mos_model.transform(X) if self.mos_model is not None else pl
        # TCN 输入：X[feature_cols] -> [T, C] 数组（predict_tcn 内部用 feat_mean/feat_std 标准化、
        # NaN 填均值；输出用成员 target_mean/target_std 反标准化回原始量纲）
        X_arr = X[self.feature_cols].to_numpy(dtype=np.float32)
        member_preds = np.empty((len(self.members), len(X)), dtype=float)
        for i, (tcn, is_res) in enumerate(zip(self.members, self.member_residual)):
            raw = predict_tcn(tcn, X_arr, self.feat_mean, self.feat_std, self.device)   # [T]
            member_preds[i] = anchor + raw if is_res else raw
        ens = self._aggregate(member_preds)
        pred = anchor + self.shrinkage * (ens - anchor)
        hours = pd.DatetimeIndex(X.index).hour.values.astype(int)
        # 小时偏置校正（OOF 估计；按 hour_bias 长度自适应 slot 索引：24=逐小时,48=逐半小时,96=逐15min）
        if self.hour_bias is not None:
            n = len(self.hour_bias)
            # (mod*n)//1440 对任意 n 恒为 0..n-1（n 整除 1440 时等价于 mod//(1440//n)；n=24 时等价于
            # hour）。避免 n 不整除 1440 时 mod//(1440//n) 越界（bug#7：当前 n=96 安全，但留隐患）。
            mod = hours * 60 + pd.DatetimeIndex(X.index).minute.values
            idx = ((mod * n) // 1440).astype(int)
            pred = pred - self.hour_bias[idx]
        # 漂移方向校正（OOF 估计的 β·feat，仅指定小时非零；无泄露）
        # 符号说明（bug#2 已实测验证，勿改）：此处为 pred += β·feat，β=<feat, oof_pred-actual>/<feat²>。
        # 数学上"去偏"似应为 pred -= β·feat，但 pl_weather_residual 与误差的方向关系在训练(OOF)与
        # 验证集间发生符号翻转（FDS 午间诊断：跨年符号翻转），故 += 恰好匹配验证集方向；改为 -= 反而
        # +105 MW（exp_drift_sign.py：+=1445.62 vs -=1551.12）。切勿"修正"为 -=。
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
        for i, tcn in enumerate(self.members):
            p = booster_dir / f"member_{i:03d}.pt"
            torch.save(tcn.state_dict(), str(p))
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
            "aggregation": self.aggregation,
            "trim_frac": self.trim_frac,
            "mos_model": self.mos_model,
            # TCN 专用
            "feat_mean": None if self.feat_mean is None else self.feat_mean.tolist(),
            "feat_std": None if self.feat_std is None else self.feat_std.tolist(),
            "tcn_config": self.tcn_config,
            "device": "auto",
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)

    @classmethod
    def load(cls, path: Path) -> "EnsembleModel":
        path = Path(path)
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        # 兼容旧版 tuple 格式 threshold_corr：(feature, thr, hours, shift) -> dict(op=">")
        tc = bundle.get("threshold_corr") or []
        tc_norm = []
        for entry in tc:
            if isinstance(entry, dict):
                tc_norm.append(entry)
            else:  # 旧版 tuple (feature, thr, hours, shift)
                feat_name, thr, hours_list, shift = entry
                tc_norm.append({"feature": feat_name, "op": ">", "thr": thr,
                                "hours": hours_list, "shift": shift})
        tcn_config = bundle.get("tcn_config", {})
        obj = cls(feature_cols=bundle["feature_cols"],
                  shrinkage=bundle.get("shrinkage", 1.0),
                  train_meta=bundle.get("train_meta", {}),
                  hour_bias=bundle.get("hour_bias"),
                  mismatch_model=bundle.get("mismatch_model"),
                  drift_corr=bundle.get("drift_corr"),
                  threshold_corr=tc_norm,
                  aggregation=bundle.get("aggregation", "median"),
                  trim_frac=bundle.get("trim_frac", 0.2),
                  mos_model=bundle.get("mos_model"),
                  feat_mean=bundle.get("feat_mean"),
                  feat_std=bundle.get("feat_std"),
                  tcn_config=tcn_config,
                  device="auto")
        n_feat = len(obj.feature_cols)
        for p in bundle["booster_paths"]:
            tcn = TCN(n_feat,
                      tcn_config.get("num_channels", [64, 64, 64, 64]),
                      tcn_config.get("kernel_size", 7),
                      tcn_config.get("dropout", 0.1))
            # PyTorch 2.6+ 起 torch.load 默认 weights_only=True；state_dict 仅含张量，
            # 显式指定以明示"只加载权重"且版本安全（2.8.0 / 2.12 均验证）。
            state = torch.load(p, map_location=obj.device, weights_only=True)
            tcn.load_state_dict(state)
            tcn.to(obj.device)
            tcn.eval()
            obj.members.append(tcn)
        obj.member_residual = list(bundle["member_residual"])
        return obj


# ---------------- 兼容旧名 ---------------- #
LoadModel = EnsembleModel
