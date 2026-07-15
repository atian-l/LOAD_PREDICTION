# -*- coding: utf-8 -*-
"""第三/四层：Correction Trigger + 动态 Shrink α + 残差模型。

残差模型：3 段各一 LightGBM（3 种子 median），target = actual - base_A_OOF，OOF 池训练。
correction_OOF：嵌套 2 折（OOF 池内）产无泄露 correction 预测，供 trigger/α/w 估计。
DynamicEstimator：用天气相似度 KNN 邻居，逐 (date,seg) 估 α（grid）/w（公式）/trigger（grid 阈值）。
所有参数来自训练期 OOF，val 零参与；跨年泛化（物理量相似度 + 局部估计）。
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb

from . import config as VC
from . import segments as SEG
from . import fusion


def _corr_lgb_params(seed: int) -> dict:
    p = dict(VC.CORR_CFG)
    return {
        "objective": "regression", "metric": "mae", "verbose": -1,
        "force_col_wise": True, "seed": int(seed),
        "learning_rate": p["learning_rate"], "num_leaves": p["num_leaves"],
        "min_data_in_leaf": p["min_data_in_leaf"], "lambda_l2": p["lambda_l2"],
        "feature_fraction": p["feature_fraction"], "bagging_fraction": p["bagging_fraction"],
        "bagging_freq": p["bagging_freq"],
    }


def train_correction_models(oof_pool: dict, X_full: pd.DataFrame, feat_cols: list, cfg=VC.CORR_CFG) -> dict:
    """3 段各训练 len(seeds) 个残差 LightGBM。target = actual - base_A_oof。"""
    boosters = {}
    for seg in VC.SEGMENTS:
        m = oof_pool["seg"] == seg
        idx = oof_pool["idx"][m]
        y = oof_pool["actual"][m] - oof_pool["base_A_oof"][m]
        Xtr = X_full.iloc[idx][feat_cols]
        bs = []
        for s in cfg["seeds"]:
            bst = lgb.train(_corr_lgb_params(s), lgb.Dataset(Xtr, label=y),
                            num_boost_round=int(cfg["best_it_fixed"]))
            bs.append(bst)
        boosters[seg] = bs
    return boosters


def correction_predict(seg_boosters: list, X_df: pd.DataFrame) -> np.ndarray:
    """3 种子 median。"""
    preds = np.array([b.predict(X_df) for b in seg_boosters])
    return np.median(preds, axis=0)


def correction_oof_nested(oof_pool: dict, X_full: pd.DataFrame, feat_cols: list, cfg=VC.CORR_CFG) -> np.ndarray:
    """嵌套 2 折（OOF 池内，按时间中点）产无泄露 correction_OOF。

    每点由未见过它的 fold 预测，保证 trigger/α/w 估计无泄露。
    """
    n = len(oof_pool["idx"])
    corr_oof = np.full(n, np.nan)
    for seg in VC.SEGMENTS:
        pos = np.where(oof_pool["seg"] == seg)[0]
        if len(pos) < 4:
            continue
        t_seg = pd.DatetimeIndex(oof_pool["times"][pos]).astype(np.int64)
        mid = np.median(t_seg)
        for tr_m, va_m in [(t_seg <= mid, t_seg > mid), (t_seg > mid, t_seg <= mid)]:
            tr_pos = pos[tr_m]; va_pos = pos[va_m]
            if len(tr_pos) == 0 or len(va_pos) == 0:
                continue
            y_tr = oof_pool["actual"][tr_pos] - oof_pool["base_A_oof"][tr_pos]
            Xtr = X_full.iloc[oof_pool["idx"][tr_pos]][feat_cols]
            Xva = X_full.iloc[oof_pool["idx"][va_pos]][feat_cols]
            preds = []
            for s in cfg["seeds"]:
                bst = lgb.train(_corr_lgb_params(s), lgb.Dataset(Xtr, label=y_tr),
                                num_boost_round=int(cfg["best_it_fixed"]))
                preds.append(bst.predict(Xva))
            corr_oof[va_pos] = np.median(preds, axis=0)
    return corr_oof


# --------------------------------------------------------------------------- #
# DynamicEstimator：trigger / α / w 的 KNN 局部估计 + grid search
# --------------------------------------------------------------------------- #
class DynamicEstimator:
    """逐 (date, seg) 用天气相似度 KNN 邻居估 α（grid）/w（公式）/trigger（grid 阈值）。

    fit：OOF 池上 grid search trig_frac/min_gain 使 OOF MAE 最优（val 不参与）。
    params：部署时对 D+1 (date,seg) 返回 (alpha, w, trigger)。
    """

    def __init__(self, weather_sim, oof_pool: dict, corr_oof: np.ndarray,
                 day_vec_pool: pd.DataFrame, fold_windows=None):
        self.ws = weather_sim
        self.oof = oof_pool
        self.corr_oof = np.asarray(corr_oof, dtype=float)
        self.day_vec_pool = day_vec_pool
        # 3 折 walk-forward 的 (te, vs, ve) 评估窗，用于 minimax 跨季稳定性选 trigger。
        # 跨年泛化代理：要求 trigger 在每个季节折上都不恶化（max-fold MAE 最小），
        # 而非平均 OOF MAE 最优（后者会为单季收益过拟合、跨年翻转 -> val 过度修正）。
        # val 零参与（仅用 2025 训练期 3 折）。
        self.fold_windows = fold_windows
        self.trig_frac = 0.6
        self.min_gain = 30.0
        self._oof_final = None
        self._oof_mae = None
        self._oof_worst = None
        self._ds_table = None
        self._build_index()

    def _fold_ids(self) -> np.ndarray:
        """每个 OOF 池点所属的 walk-forward 折 id（按评估窗 vs..ve 归属）；无归属为 -1。"""
        if not self.fold_windows:
            return np.full(len(self.oof["idx"]), -1, dtype=int)
        t = pd.DatetimeIndex(self.oof["times"])
        fid = np.full(len(t), -1, dtype=int)
        for k, fw in enumerate(self.fold_windows):
            _, vs, ve = fw
            m = (t >= pd.Timestamp(vs)) & (t <= pd.Timestamp(ve))
            fid[m] = k
        return fid

    def _build_index(self):
        d = defaultdict(list)
        dates_norm = pd.DatetimeIndex(self.oof["dates"]).normalize()
        for i, s in enumerate(self.oof["seg"]):
            d[(dates_norm[i], s)].append(i)
        self.pool_by_ds = {k: np.array(v) for k, v in d.items()}
        self.dates_norm = dates_norm

    def _neighbor_points(self, q_date, seg, q_vec=None) -> tuple[np.ndarray, np.ndarray]:
        """对 (date, seg)，返回邻居 OOF 池点位置 + 归一化权重（排除当日防自泄露）。

        q_vec：查询日的日级天气向量。OOF 模拟时不传（q_date 在训练池内，自查 day_vec_pool）；
        部署/val 时由调用方传入（q_date 可能不在训练池内——如 2026 val 或 D+1 未来日，
        此时 day_vec_pool.loc[q_date] 会 KeyError，旧实现静默返回空 -> trigger 永不命中，
        val_trigger=0 的根因）。传 q_vec 后任意日期都能在训练池中找天气邻居，实现跨年迁移。
        """
        q_date = pd.Timestamp(q_date).normalize()
        if q_vec is None:
            if q_date not in self.day_vec_pool.index:
                return np.array([], dtype=int), np.array([], dtype=float)
            q = self.day_vec_pool.loc[q_date].values.astype(float)
        else:
            q = np.asarray(q_vec, dtype=float)
        nb_idx, nb_w = self.ws.query(q, exclude_date=q_date)
        if len(nb_idx) == 0:
            return np.array([], dtype=int), np.array([], dtype=float)
        nb_dates = pd.DatetimeIndex(self.ws.pool_dates_[nb_idx]).normalize()
        pts_all, w_all = [], []
        for j, nd in enumerate(nb_dates):
            key = (nd, seg)
            if key not in self.pool_by_ds:
                continue
            pts = self.pool_by_ds[key]
            npts = len(pts)
            pts_all.append(pts)
            w_all.append(np.full(npts, nb_w[j] / npts))
        if not pts_all:
            return np.array([], dtype=int), np.array([], dtype=float)
        pts = np.concatenate(pts_all)
        w = np.concatenate(w_all)
        s = w.sum()
        if s <= 0:
            return pts, np.ones(len(pts)) / len(pts)
        return pts, w / s

    def _estimate_alpha_w_frac_gain(self, q_date, seg, q_vec=None) -> dict:
        """对 (date,seg) 估 α（grid）/w/frac/gain。q_vec 透传给 _neighbor_points（部署/val 用）。"""
        pts, w = self._neighbor_points(q_date, seg, q_vec=q_vec)
        if len(pts) == 0:
            return {"alpha": 0.0, "w": 0.0, "frac": 0.0, "gain": 0.0}
        base = self.oof["base_A_oof"][pts]
        corr = self.corr_oof[pts]
        act = self.oof["actual"][pts]
        be = np.abs(base - act) + 1e-6
        best_a, best_mae = 0.0, np.inf
        for a in VC.ALPHA_GRID:
            mae = np.average(np.abs(base + a * corr - act), weights=w)
            if mae < best_mae:
                best_mae = mae; best_a = a
        ce = np.abs(base + best_a * corr - act)
        w_val = fusion.estimate_w(be, ce, w)
        frac = float(np.average((ce < be).astype(float), weights=w))
        gain = float(np.average(be - ce, weights=w))
        return {"alpha": best_a, "w": w_val, "frac": frac, "gain": gain}

    def fit(self, verbose=True) -> "DynamicEstimator":
        # 1) 逐 (date,seg) 估 α/w/frac/gain（与 trig_frac/min_gain 无关，预计算一次）
        uniq = list(dict.fromkeys(zip(self.dates_norm, self.oof["seg"])))
        self._ds_table = {}
        for d, s in uniq:
            self._ds_table[(d, s)] = self._estimate_alpha_w_frac_gain(d, s)
        if verbose:
            n_trig = sum(1 for v in self._ds_table.values() if v["gain"] > 0)
            print(f"  [Dyn] (date,seg) 组={len(self._ds_table)}  gain>0={n_trig}")
        # 2) grid search min_gain：minimax 跨季稳定性（max-fold MAE 最小），val 零参与。
        #    trigger 可信度门槛 TRIG_MIN_FRAC 固定（跨年保守，见 config 注释），仅 grid 选 min_gain。
        fid = self._fold_ids()
        actual = self.oof["actual"]
        use_minimax = self.fold_windows is not None and (fid >= 0).any()

        def _grid_obj(final):
            err = np.abs(final - actual)
            if use_minimax:
                folds = [k for k in range(len(self.fold_windows)) if (fid == k).any()]
                worst = max(float(err[fid == k].mean()) for k in folds)
                avg = float(err.mean())
                return worst, avg
            return float(err.mean()), float(err.mean())

        best = None  # (worst, avg, mg)
        for mg in VC.MIN_GAIN_GRID:
            final = self._simulate(mg)
            worst, avg = _grid_obj(final)
            if best is None or worst < best[0] - 1e-6 or (abs(worst - best[0]) <= 1e-6 and avg < best[1]):
                best = (worst, avg, mg)
        self.min_gain = best[2]
        self.trig_frac = VC.TRIG_MIN_FRAC  # 固定可信度门槛（记录用）
        self._oof_final = self._simulate(self.min_gain)
        self._oof_mae = float(np.mean(np.abs(self._oof_final - actual)))
        if use_minimax:
            folds = [k for k in range(len(self.fold_windows)) if (fid == k).any()]
            err = np.abs(self._oof_final - actual)
            self._oof_worst = max(float(err[fid == k].mean()) for k in folds)
        else:
            self._oof_worst = self._oof_mae
        if verbose:
            n_on = int(np.sum(self._oof_final != self.oof["base_A_oof"]))
            base_mae = float(np.mean(np.abs(self.oof["base_A_oof"] - actual)))
            print(f"  [Dyn] 可信度门槛 frac>={VC.TRIG_MIN_FRAC} grid 选 min_gain={self.min_gain} "
                  f"(minimax) -> OOF MAE {base_mae:.2f}->{self._oof_mae:.2f} "
                  f"(Δ{self._oof_mae-base_mae:+.2f}, OOF trigger命中率={n_on/len(actual):.3f})")
        return self

    def restore(self) -> "DynamicEstimator":
        """加载后重建 _ds_table + _oof_final（用已存 trig_frac/min_gain，不重新 grid search）。

        fit() 的 _ds_table/_oof_final 不入存档；加载后须调用本方法才能让 evaluate 取到真实
        OOF trigger 命中率（否则 _oof_final=None，`None != array` 会被广播成全 True -> 假 1.0）。
        """
        uniq = list(dict.fromkeys(zip(self.dates_norm, self.oof["seg"])))
        self._ds_table = {}
        for d, s in uniq:
            self._ds_table[(d, s)] = self._estimate_alpha_w_frac_gain(d, s)
        self._oof_final = self._simulate(self.trig_frac, self.min_gain)
        self._oof_mae = float(np.mean(np.abs(self._oof_final - self.oof["actual"])))
        self._oof_worst = self._oof_mae  # 加载路径无 fold_windows（evaluate 不用此值）
        return self

    def _simulate(self, mg: float, tf: float | None = None) -> np.ndarray:
        """OOF 模拟：逐点用其 (date,seg) 的 α/w/trigger 应用 correction_OOF。
        trigger = frac >= tf（默认 TRIG_MIN_FRAC 固定门槛）且 gain >= mg。
        tf 可显式传入（仅诊断扫描用，生产用默认）。"""
        if tf is None:
            tf = VC.TRIG_MIN_FRAC
        n = len(self.oof["idx"])
        final = self.oof["base_A_oof"].copy()
        for i in range(n):
            d = self.dates_norm[i]
            s = self.oof["seg"][i]
            e = self._ds_table[(d, s)]
            if e["frac"] >= tf and e["gain"] >= mg:
                final[i] = self.oof["base_A_oof"][i] + e["w"] * e["alpha"] * self.corr_oof[i]
        return final

    def params(self, d1_date, d1_seg, q_vec=None, tf: float | None = None) -> tuple[float, float, bool]:
        """部署时：对 D+1 (date,seg) 返回 (alpha, w, trigger)。

        q_vec：D+1 日级天气向量（调用方算好传入）。不传则回退查 day_vec_pool（仅训练期日期可用）。
        tf：可信度门槛（默认 TRIG_MIN_FRAC，仅诊断扫描显式传入）。
        """
        if tf is None:
            tf = VC.TRIG_MIN_FRAC
        e = self._estimate_alpha_w_frac_gain(d1_date, d1_seg, q_vec=q_vec)
        trig = (e["frac"] >= tf) and (e["gain"] >= self.min_gain)
        return e["alpha"], e["w"], trig
