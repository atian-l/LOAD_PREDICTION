# -*- coding: utf-8 -*-
"""FDS/a11_feature.py - (十一) 特征贡献分析 (Gain / Permutation / SHAP / PDP / ICE)。

只读诊断：用 v6 模型的 booster 计算 gain 重要性、SHAP(pred_contrib)、全管线置换重要性、
PDP/ICE。不训练新模型、不改生产。
"""
from __future__ import annotations
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from load_pred import config as C
from load_pred.model import EnsembleModel
from diag_lib import load_val, load_Xval, save_fig, save_table


def main():
    d = load_val()
    Xv = load_Xval()
    model = EnsembleModel.load(C.MODEL_BUNDLE)
    fcols = model.feature_cols
    X = Xv[fcols]
    y = d["actual"].values
    base_pred = d["pred"].values
    base_mae = float(np.abs(base_pred - y).mean())
    print("== (十一) 特征贡献 ==", flush=True)
    print(f"基线 MAE={base_mae:.2f}  特征数={len(fcols)}", flush=True)

    # 1) Gain importance (平均 40 成员)
    gain = np.zeros(len(fcols))
    for bst in model.members:
        g = bst.feature_importance(importance_type="gain")
        gain += g
    gain /= len(model.members)
    gi = pd.Series(gain, index=fcols).sort_values(ascending=False)
    save_table(gi.head(25).to_frame("gain"), "11_gain_importance")
    print("Gain Top10:\n", gi.head(10).round(1).to_string(), flush=True)

    # 2) SHAP (pred_contrib)，对全部成员取平均 |SHAP|；抽样 3000 点加速
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), size=min(3000, len(X)), replace=False)
    Xs = X.iloc[idx]
    shap_abs = np.zeros(len(fcols))
    for bst in model.members:
        contrib = bst.predict(Xs, pred_contrib=True)  # (n, n_feat+1)
        shap_abs += np.abs(contrib[:, :-1]).mean(axis=0)
    shap_abs /= len(model.members)
    si = pd.Series(shap_abs, index=fcols).sort_values(ascending=False)
    save_table(si.head(25).to_frame("mean_abs_shap"), "11_shap_importance")
    print("SHAP Top10:\n", si.head(10).round(1).to_string(), flush=True)

    # 3) 置换重要性（全管线 predict_load，top 15 特征）
    top15 = gi.head(15).index.tolist()
    perm = {}
    for f in top15:
        Xp = Xv.copy()
        Xp[f] = Xp[f].sample(frac=1, random_state=42).values
        pp = model.predict_load(Xp[fcols], d["pred_load"])
        perm[f] = float(np.abs(pp - y).mean()) - base_mae
    pi = pd.Series(perm).sort_values(ascending=False)
    save_table(pi.to_frame("perm_mae_increase"), "11_permutation_importance")
    print("置换重要性 Top(全管线, MAE上升):\n", pi.round(1).to_string(), flush=True)

    # 4) PDP + ICE for top 4 特征
    top4 = gi.head(4).index.tolist()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, f in zip(axes.flat, top4):
        vals = X[f].values
        qs = np.quantile(vals, np.linspace(0.02, 0.98, 25))
        qs = np.unique(qs)
        avg = []
        ice = []  # 抽 40 条
        ice_idx = rng.choice(len(X), size=40, replace=False)
        Xb = X.iloc[ice_idx].copy()
        for v in qs:
            Xc = X.copy(); Xc[f] = v
            # 用成员直接预测（直方图近似，单成员够看趋势）-> 用全部成员 median
            preds = np.empty((len(model.members), len(Xc)))
            for i, bst in enumerate(model.members):
                preds[i] = bst.predict(Xc[fcols])
            pred_med = np.median(preds, axis=0)
            avg.append(pred_med.mean())
            Xbb = Xb.copy(); Xbb[f] = v
            preds_b = np.empty((len(model.members), len(Xbb)))
            for i, bst in enumerate(model.members):
                preds_b[i] = bst.predict(Xbb[fcols])
            ice.append(np.median(preds_b, axis=0))
        ice = np.array(ice).T
        ax.plot(qs, avg, "r-", lw=2.5, label="PDP(均值)")
        for row in ice:
            ax.plot(qs, row, color="tab:blue", alpha=0.12, lw=0.6)
        ax.set_xlabel(f); ax.set_ylabel("预测负荷")
        ax.set_title(f"PDP/ICE - {f}"); ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "11_pdp_ice")

    # 重要性对比图
    fig, ax = plt.subplots(figsize=(9, 7))
    cmp = pd.DataFrame({"gain(归一)": gi.head(15) / gi.head(15).max(),
                        "SHAP(归一)": si.reindex(gi.head(15).index) / si.reindex(gi.head(15).index).max(),
                        "置换(归一)": pi.reindex(gi.head(15).index).fillna(0) / pi.reindex(gi.head(15).index).fillna(0).max()})
    cmp.plot(kind="barh", ax=ax); ax.invert_yaxis()
    ax.set_title("特征重要性对比 (top15 by gain, 归一化)")
    fig.tight_layout()
    save_fig(fig, "11_importance_compare")


if __name__ == "__main__":
    main()
