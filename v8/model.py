# -*- coding: utf-8 -*-
"""V8Model：五层架构组装 + predict + save/load。

持有：base A(v6 加载) / base B(reg_only) / correction_models(3段) /
      weather_sim / DynamicEstimator / adaptive_pref / mismatch_model / full_mos。
predict：分段 -> adaptive 选 base -> 天气相似度 KNN -> trigger/α/w -> correction -> fusion。
save/load：base A 不存（部署加载根 bundle）；其余全存 v8_bundle.pkl + boosters txt。
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from load_pred.model import EnsembleModel
from . import config as VC
from . import segments as SEG
from . import weather_sim as WS
from . import base as BASE
from . import correction as CORR
from ._io import load_booster


class V8Model:
    def __init__(self, feat_cols: list, cfg: dict, mismatch_model, full_mos,
                 base_A: EnsembleModel, base_B: EnsembleModel,
                 correction_models: dict, weather_sim, dynamic, adaptive_pref: dict,
                 corr_B, oof_pool: dict, corr_oof: np.ndarray, day_vec_pool: pd.DataFrame):
        self.feat_cols = feat_cols
        self.cfg = cfg
        self.mismatch_model = mismatch_model
        self.full_mos = full_mos
        self.base_A = base_A
        self.base_B = base_B
        self.correction_models = correction_models
        self.ws = weather_sim
        self.dynamic = dynamic
        self.adaptive_pref = adaptive_pref
        self.corr_B = corr_B
        self.oof_pool = oof_pool
        self.corr_oof = corr_oof
        self.day_vec_pool = day_vec_pool

    # --------------------------------------------------------------------------- #
    # 推理：五层串联
    # --------------------------------------------------------------------------- #
    def predict(self, X_raw: pd.DataFrame, pred_load: pd.Series, times) -> np.ndarray:
        """对 times 各点输出最终预测。

        X_raw：build_features 输出（未 mismatch transform）。
        流程：mismatch transform -> base A/B 全天预测 -> 逐段 adaptive 选 base +
              KNN 估 α/w/trigger -> final = base + w·α·residual（trigger on 时）。
        """
        X = self.mismatch_model.transform(X_raw)
        times = pd.DatetimeIndex(times)
        hours = times.hour.values.astype(int)
        dates = times.normalize()
        seg_arr = SEG.segment_array(hours)

        base_A_pred = self.base_A.predict_load(X, pred_load)
        base_B_pred = self.base_B.predict_load(X, pred_load)

        # 日级天气向量 + adaptive base 选择（(天气型, 段) 级，B 仅在其更优的段选用）
        day_vec = WS.day_weather_vectors(X, times)
        uniq_dates = np.unique(dates)
        date_wt = {}  # date -> 天气型
        for d in uniq_dates:
            d = pd.Timestamp(d).normalize()
            date_wt[d] = BASE.weather_type(day_vec.loc[d]) if d in day_vec.index else None

        final = np.empty(len(times), dtype=float)
        # 逐段处理：correction 预测按段批量，α/w/trigger 按 (date,seg) 缓存
        for seg in VC.SEGMENTS:
            m = seg_arr == seg
            if not m.any():
                continue
            idx_seg = np.where(m)[0]
            X_seg = X.iloc[idx_seg][self.feat_cols]
            corr_seg = CORR.correction_predict(self.correction_models[seg], X_seg)
            dates_seg = dates[idx_seg]
            # (date,seg) 参数缓存（传入当日天气向量 q_vec，使 val/未来日也能在训练池找邻居）
            ds_cache = {}
            for d in pd.DatetimeIndex(np.unique(dates_seg)).normalize():
                q_vec = day_vec.loc[d].values if d in day_vec.index else None
                ds_cache[d] = self.dynamic.params(d, seg, q_vec=q_vec)
            for j, d in enumerate(dates_seg):
                d = pd.Timestamp(d).normalize()
                wt = date_wt.get(d)
                sel = BASE.select_base(wt, seg, self.adaptive_pref) if wt is not None else "A"
                base_val = base_A_pred[idx_seg[j]] if sel == "A" else base_B_pred[idx_seg[j]]
                a, w, trig = ds_cache[d]
                if trig and a > 0.0 and w > 0.0:
                    final[idx_seg[j]] = base_val + w * a * corr_seg[j]
                else:
                    final[idx_seg[j]] = base_val
        return np.clip(final, 0.0, None)

    # --------------------------------------------------------------------------- #
    # 持久化
    # --------------------------------------------------------------------------- #
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        corr_dir = path.parent / "correction_boosters"
        bB_dir = path.parent / "base_B_boosters"
        corr_dir.mkdir(parents=True, exist_ok=True)
        bB_dir.mkdir(parents=True, exist_ok=True)

        corr_paths = {}
        for seg, bs in self.correction_models.items():
            corr_paths[seg] = []
            for i, b in enumerate(bs):
                p = corr_dir / f"{seg}_{i:02d}.txt"
                b.save_model(str(p))
                corr_paths[seg].append(str(p))

        bB_paths = []
        for i, b in enumerate(self.base_B.members):
            p = bB_dir / f"member_{i:03d}.txt"
            b.save_model(str(p))
            bB_paths.append(str(p))

        bundle = {
            "feat_cols": self.feat_cols,
            "cfg": self.cfg,
            "mismatch_model": self.mismatch_model,
            "full_mos": self.full_mos,
            "weather_sim": self.ws,
            "adaptive_pref": self.adaptive_pref,
            "trig_frac": self.dynamic.trig_frac,
            "min_gain": self.dynamic.min_gain,
            "corr_B": self.corr_B,
            "base_B_member_residual": list(self.base_B.member_residual),
            "base_B_booster_paths": bB_paths,
            "base_B_feature_cols": list(self.base_B.feature_cols),
            "base_B_shrinkage": float(self.base_B.shrinkage),
            "base_B_aggregation": self.base_B.aggregation,
            "base_B_trim_frac": float(self.base_B.trim_frac),
            "correction_booster_paths": corr_paths,
            "oof_pool": self.oof_pool,
            "corr_oof": np.asarray(self.corr_oof, dtype=float),
            "day_vec_pool": self.day_vec_pool,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)

    @classmethod
    def load(cls, path) -> "V8Model":
        path = Path(path)
        with open(path, "rb") as f:
            bundle = pickle.load(f)

        base_A = BASE.load_base_A()

        base_B = EnsembleModel(
            feature_cols=bundle["base_B_feature_cols"],
            shrinkage=bundle["base_B_shrinkage"],
            aggregation=bundle.get("base_B_aggregation", "median"),
            trim_frac=bundle.get("base_B_trim_frac", 0.2),
            mos_model=bundle["full_mos"],
            mismatch_model=bundle["mismatch_model"],
        )
        hb_B, dc_B, tc_B = bundle["corr_B"]
        base_B.hour_bias = np.asarray(hb_B, dtype=float)
        base_B.drift_corr = [(n, np.asarray(b, dtype=float)) for n, b in dc_B]
        base_B.threshold_corr = [dict(tc) for tc in tc_B]
        for p in bundle["base_B_booster_paths"]:
            base_B.members.append(load_booster(p))
        base_B.member_residual = list(bundle["base_B_member_residual"])

        correction_models = {}
        for seg, paths in bundle["correction_booster_paths"].items():
            correction_models[seg] = [load_booster(p) for p in paths]

        ws = bundle["weather_sim"]
        oof_pool = bundle["oof_pool"]
        corr_oof = bundle["corr_oof"]
        day_vec_pool = bundle["day_vec_pool"]

        # 重建 DynamicEstimator（不 fit，用已存 trig_frac/min_gain）
        dynamic = CORR.DynamicEstimator(ws, oof_pool, corr_oof, day_vec_pool)
        dynamic.trig_frac = bundle["trig_frac"]
        dynamic.min_gain = bundle["min_gain"]
        dynamic.restore()  # 重建 _ds_table/_oof_final（evaluate 取真实 OOF trigger 命中率）

        return cls(
            feat_cols=bundle["feat_cols"], cfg=bundle["cfg"],
            mismatch_model=bundle["mismatch_model"], full_mos=bundle["full_mos"],
            base_A=base_A, base_B=base_B, correction_models=correction_models,
            weather_sim=ws, dynamic=dynamic, adaptive_pref=bundle["adaptive_pref"],
            corr_B=bundle["corr_B"], oof_pool=oof_pool, corr_oof=corr_oof,
            day_vec_pool=day_vec_pool,
        )
