# -*- coding: utf-8 -*-
"""
模型封装（TFT 集成）。

与 load_pred/model.py（LightGBM）、load_pred_tcn/model.py（TCN）公共接口与后处理一致，
仅将成员换为 TFT（PyTorch 序列模型），且 predict_load 内部由"逐时刻前向"改为"按预测日
序列前向、拼回逐时刻"（TFT 是 multi-horizon 序列模型）：
  - EnsembleModel 持有 N 个 TFT 成员，每个标注是否残差模式
    （残差：预测 = anchor[T] + resid_hat；直接：预测 = raw）。残差目标 = actual - MOS_anchor。
  - 最终预测 = 各成员预测的中位数；收缩 λ：pred = anchor + λ*(ens - anchor)。
  - 后处理（hour_bias/drift_corr/threshold_corr/clip）逐时刻，与 LightGBM/TCN 版逐行相同（模型无关）。

合规：
  - 残差目标 = actual - anchor（anchor=MOS corrected pred_load；actual 仅作目标，#1）。
  - encoder 历史负荷用 pred_load（#1/#2）；decoder 未来用 weather+calendar（#4）。
  - 全程不使用真实负荷作为输入特征。
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config as C
from .tft import TFT, predict_tft, get_device


class EnsembleModel:
    """TFT 集成回归模型（接口与 LightGBM/TCN 版一致）。"""

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
                 static_cols: list | None = None,
                 tft_config: dict | None = None,
                 feat_clip: float | None = 10.0,
                 device: str = "auto"):
        self.feature_cols = feature_cols
        self.shrinkage = float(shrinkage)
        self.train_meta = train_meta or {}
        self.hour_bias = None if hour_bias is None else np.asarray(hour_bias, dtype=float)
        self.mismatch_model = mismatch_model
        self.drift_corr = drift_corr or []
        self.threshold_corr = threshold_corr or []
        self.aggregation = aggregation
        self.trim_frac = float(trim_frac)
        self.mos_model = mos_model
        # ---- TFT 专用 ----
        # feat_mean/feat_std：训练期特征列均值/标准差（usable 行估计，各成员共享）。
        # 推理时 predict_tcn 用其标准化 X（NaN 填均值后标准化为 0）；输入预处理，无泄露。
        self.feat_mean = None if feat_mean is None else np.asarray(feat_mean, dtype=np.float32)
        self.feat_std = None if feat_std is None else np.asarray(feat_std, dtype=np.float32)
        # static_cols：日级日历列（TFT static metadata），推理 predict_tft 需要。
        self.static_cols = list(static_cols) if static_cols else list(C.STATIC_COLS)
        # tft_config：重建 TFT 架构 {n_feat,n_static,hidden_size,num_heads,num_lstm_layers,
        # dropout,encoder_len,decoder_len}（load 时用）。
        self.tft_config = tft_config or {}
        self.feat_clip = feat_clip  # 标准化空间 clip（与 train_tft 一致，防 train/serve skew）
        self.device = get_device(device)
        self.members: list[TFT] = []
        self.member_residual: list[bool] = []

    def _aggregate(self, member_preds: np.ndarray) -> np.ndarray:
        """按 self.aggregation 聚合（行=成员，列=样本）；NaN 传播（前段无 encoder 历史处为 NaN）。"""
        if self.aggregation == "mean":
            return np.nanmean(member_preds, axis=0)
        if self.aggregation == "trimmed":
            n = member_preds.shape[0]
            k = int(np.floor(n * self.trim_frac / 2))
            if k > 0 and (n - 2 * k) > 0:
                Ms = np.sort(member_preds, axis=0)
                return np.nanmean(Ms[k:n - k], axis=0)
            return np.nanmean(member_preds, axis=0)
        return np.nanmedian(member_preds, axis=0)  # "median"（默认）

    def add_member(self, tft: TFT, is_residual: bool) -> None:
        self.members.append(tft)
        self.member_residual.append(bool(is_residual))

    def predict_load(self, X: pd.DataFrame, pred_load_at_T: pd.Series) -> np.ndarray:
        """返回最终"直调负荷"预测（含收缩 λ + OOF 后处理），长度 = len(X)。

        TFT 按预测日序列前向：predict_tft 对 X 内每个预测日（日界 96 步）输出 96 点残差/直接
        预测，拼回逐时刻 [len(X)]（前 encoder_len 段无足够历史 -> NaN，与 v6 前段无特征一致）。
        后处理（hour_bias/drift_corr/threshold_corr/clip）逐时刻，与 LightGBM/TCN 版逐行相同。

        注意：TFT 需要每个预测日的前 encoder_len 步历史，故调用方应传足够长的 X（≥ encoder_len
        + 96）。compute_hour_bias / predict_d1 均传 full 或 14 天回看窗口，满足此要求。
        """
        if not self.members:
            raise RuntimeError("集成无成员。")
        pl = pred_load_at_T.reindex(X.index).values.astype(float)
        anchor = self.mos_model.transform(X) if self.mos_model is not None else pl
        member_preds = np.empty((len(self.members), len(X)), dtype=float)
        for i, (tft, is_res) in enumerate(zip(self.members, self.member_residual)):
            raw = predict_tft(tft, X, self.feature_cols, self.static_cols,
                              self.feat_mean, self.feat_std, self.device,
                              feat_clip=self.feat_clip)  # [len(X)]
            member_preds[i] = anchor + raw if is_res else raw
        ens = self._aggregate(member_preds)
        pred = anchor + self.shrinkage * (ens - anchor)

        hours = pd.DatetimeIndex(X.index).hour.values.astype(int)
        # 小时偏置校正（OOF 估计；按 hour_bias 长度自适应 slot 索引：24/48/96）
        if self.hour_bias is not None:
            n = len(self.hour_bias)
            mod = hours * 60 + pd.DatetimeIndex(X.index).minute.values
            idx = ((mod * n) // 1440).astype(int)
            pred = pred - self.hour_bias[idx]
        # 漂移方向校正（OOF 估计的 β·feat，仅指定小时非零；符号 += 勿改，见 tcn/model.py bug#2）
        for feat_name, beta in self.drift_corr:
            beta = np.asarray(beta, dtype=float)
            pred = pred + beta[hours] * X[feat_name].values.astype(float)
        # 阈值场景校正（OOF 估计的 shift；晴午间/阴雨天/低温/多云；无泄露）
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
            else:  # ">"
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
        for i, tft in enumerate(self.members):
            p = booster_dir / f"member_{i:03d}.pt"
            torch.save(tft.state_dict(), str(p))
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
            # TFT 专用
            "feat_mean": None if self.feat_mean is None else self.feat_mean.tolist(),
            "feat_std": None if self.feat_std is None else self.feat_std.tolist(),
            "static_cols": self.static_cols,
            "tft_config": self.tft_config,
            "feat_clip": self.feat_clip,
            "device": "auto",
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)

    @classmethod
    def load(cls, path: Path) -> "EnsembleModel":
        path = Path(path)
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        tc = bundle.get("threshold_corr") or []
        tc_norm = []
        for entry in tc:
            if isinstance(entry, dict):
                tc_norm.append(entry)
            else:  # 旧版 tuple
                feat_name, thr, hours_list, shift = entry
                tc_norm.append({"feature": feat_name, "op": ">", "thr": thr,
                                "hours": hours_list, "shift": shift})
        tft_config = bundle.get("tft_config", {})
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
                  static_cols=bundle.get("static_cols"),
                  tft_config=tft_config,
                  feat_clip=bundle.get("feat_clip", 10.0),
                  device="auto")
        for p in bundle["booster_paths"]:
            tft = TFT(
                n_feat=int(tft_config.get("n_feat", len(obj.feature_cols))),
                n_static=int(tft_config.get("n_static", len(obj.static_cols))),
                hidden_size=int(tft_config.get("hidden_size", 64)),
                num_heads=int(tft_config.get("num_heads", 2)),
                num_lstm_layers=int(tft_config.get("num_lstm_layers", 1)),
                dropout=float(tft_config.get("dropout", 0.1)),
                encoder_len=int(tft_config.get("encoder_len", 288)),
                decoder_len=int(tft_config.get("decoder_len", 96)),
            )
            # PyTorch 2.6+ 起 torch.load 默认 weights_only=True；state_dict 含张量+buffer，显式指定。
            state = torch.load(p, map_location=obj.device, weights_only=True)
            tft.load_state_dict(state)
            tft.to(obj.device)
            tft.eval()
            obj.members.append(tft)
        obj.member_residual = list(bundle["member_residual"])
        return obj


# ---------------- 兼容旧名 ---------------- #
LoadModel = EnsembleModel
