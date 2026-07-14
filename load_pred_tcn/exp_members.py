# -*- coding: utf-8 -*-
"""
exp_members：集成成员分析（逐成员 / 分组贡献 / 多样性 / LOO / 剪枝）。

调优元素：40 成员集成的内部结构诊断--谁拖后腿、谁冗余、可否剪枝缩成员。
（v6 LGB 成员结构固定为最优；TCN 上成员贡献分布待验。）

在单一 16 成员模型上做全 post-hoc 分析（训练一次，分析即时）：
  - 逐成员 val MAE（direct 成员预测 actual；residual 成员预测 anchor+resid）
  - 分组贡献：direct/residual、regression/quantile、各 seed 子集中位数 MAE
  - 多样性：成员预测两两相关矩阵（均值/最大；>0.98 视为冗余）
  - LOO：去某成员后集成 val MAE 变化（Δ>0=该成员有帮助；Δ<0=拖后腿）
  - 剪枝：移除所有 Δ<0 成员后的剪枝集成 MAE vs 全量

成员标签由 train_ensemble 的成员循环顺序重建（residual→obj→alpha→seed），与 model.members 对齐。
运行：python -m load_pred_tcn.exp_members
"""
from __future__ import annotations
import numpy as np

from . import exp_common as E
from .tcn import predict_tcn

# 16 成员：2 seeds × 全目标(regression+quantile×3) × {direct,residual}
MEM_OVERRIDE = {"seeds": [42, 7]}


def _member_preds(model, X_sub, pred_load_sub, mos_model):
    X_arr = X_sub[model.feature_cols].to_numpy(dtype=np.float32)
    pl = pred_load_sub.reindex(X_sub.index).values.astype(float)
    anchor = mos_model.transform(X_sub) if mos_model is not None else pl
    mp = np.empty((len(model.members), len(X_sub)), dtype=float)
    for i, (tcn, is_res) in enumerate(zip(model.members, model.member_residual)):
        raw = predict_tcn(tcn, X_arr, model.feat_mean, model.feat_std, model.device)
        mp[i] = anchor + raw if is_res else raw
    return mp


def _tags(cfg):
    """重建成员标签（顺序与 train_ensemble 的循环一致）。"""
    tags = []
    for residual in cfg["residual_modes"]:
        for obj in cfg["objectives"]:
            alphas = cfg["quantile_alphas"] if obj == "quantile" else [None]
            for qa in alphas:
                for s in cfg["seeds"]:
                    tags.append({"residual": bool(residual), "obj": obj, "alpha": qa, "seed": s})
    return tags


def main():
    d = E.build_cached()
    X, pred_load, actual, val_m, mos_model = d["X"], d["pred_load"], d["actual"], d["val_m"], d["mos_model"]
    va = actual[val_m].values
    cfg = E._cfg(MEM_OVERRIDE)
    print(f"\n[exp_members] 数据: 特征{X.shape[1]} 可用{d['usable'].sum()} val{val_m.sum()}")
    print(f"训练 {np.prod([len(cfg[k]) if k!='objectives' else (len(cfg['quantile_alphas'])+1) for k in ['residual_modes','seeds']])}..."
          f" 16 成员模型（seeds={cfg['seeds']} 全目标）...")
    model = E.train_ens(MEM_OVERRIDE)
    tags = _tags(cfg)
    assert len(tags) == len(model.members), f"标签数{len(tags)} != 成员数{len(model.members)}"
    mp = _member_preds(model, X[val_m], pred_load[val_m], mos_model)   # [n_mem, T]
    n_mem = mp.shape[0]

    full_pred = np.median(mp, axis=0)
    full_mae = E._mae(full_pred, va)
    print(f"  全量 {n_mem} 成员 median 集成 val MAE = {full_mae:.1f}\n")

    # ---- 逐成员 MAE ----
    print("=" * 78)
    print("[逐成员] val MAE")
    print("=" * 78)
    print(f"{'idx':>3} {'res':>5} {'obj':>10} {'alpha':>6} {'seed':>5} {'MAE':>8} {'Δvs_full':>9}")
    per_mae = np.array([E._mae(mp[i], va) for i in range(n_mem)])
    for i, t in enumerate(tags):
        res = "res" if t["residual"] else "dir"
        al = "-" if t["alpha"] is None else f"{t['alpha']}"
        print(f"{i:>3} {res:>5} {t['obj']:>10} {al:>6} {t['seed']:>5} {per_mae[i]:>8.1f} {per_mae[i] - full_mae:+9.1f}")

    # ---- 分组贡献（子集中位数 MAE）----
    print("\n" + "=" * 78)
    print("[分组贡献] 各子集中位数 val MAE（越低=该子集单独越强）")
    print("=" * 78)
    def _group_mae(sel):
        sel = np.asarray(sel, dtype=bool)
        if sel.sum() == 0:
            return float("nan")
        return E._mae(np.median(mp[sel], axis=0), va)
    groups = [
        ("direct",    [not t["residual"] for t in tags]),
        ("residual",  [t["residual"] for t in tags]),
        ("regression",[t["obj"] == "regression" for t in tags]),
        ("quantile",  [t["obj"] == "quantile" for t in tags]),
    ]
    for s in cfg["seeds"]:
        groups.append((f"seed{s}", [t["seed"] == s for t in tags]))
    for name, sel in groups:
        print(f"  {name:>10}: n={int(np.sum(sel)):>2}  MAE={_group_mae(sel):.1f}")

    # ---- 多样性：两两相关 ----
    print("\n" + "=" * 78)
    print("[多样性] 成员预测两两 Pearson 相关")
    print("=" * 78)
    corr = np.corrcoef(mp)
    off = corr[~np.eye(n_mem, dtype=bool)]
    print(f"  均值={off.mean():.4f}  最大={off.max():.4f}  最小={off.min():.4f}")
    redun = []
    for i in range(n_mem):
        for j in range(i + 1, n_mem):
            if corr[i, j] > 0.98:
                redun.append((i, j, corr[i, j]))
    if redun:
        print(f"  >0.98 冗余对 ({len(redun)}):")
        for i, j, c in redun[:10]:
            print(f"    [{i}]<->[{j}]  r={c:.4f}")
    else:
        print("  无 >0.98 冗余对（成员多样性充分）")

    # ---- LOO ----
    print("\n" + "=" * 78)
    print("[LOO] 去单成员后集成 val MAE（Δ=LOO−full；Δ>0=成员有帮助，Δ<0=拖后腿）")
    print("=" * 78)
    print(f"{'idx':>3} {'res':>5} {'obj':>10} {'seed':>5} {'LOO_MAE':>9} {'Δ':>8}  判定")
    loo = np.empty(n_mem)
    for i in range(n_mem):
        loo[i] = E._mae(np.median(np.delete(mp, i, axis=0), axis=0), va)
    order = np.argsort(loo - full_mae)  # 最负(最拖后腿)在前
    for i in order:
        t = tags[i]
        res = "res" if t["residual"] else "dir"
        dl = loo[i] - full_mae
        verdict = "拖后腿" if dl < 0 else ("有帮助" if dl > 0 else "中性")
        print(f"{i:>3} {res:>5} {t['obj']:>10} {t['seed']:>5} {loo[i]:>9.1f} {dl:>+8.1f}  {verdict}")

    # ---- 剪枝：移除所有 Δ<0 成员 ----
    hurt = np.where(loo - full_mae < 0)[0]
    keep = np.where(loo - full_mae >= 0)[0]
    print("\n" + "=" * 78)
    print(f"[剪枝] 拖后腿成员(Δ<0): {hurt.tolist()}  保留成员: {keep.tolist()}")
    print("=" * 78)
    if len(keep) >= 2:
        pruned_mae = E._mae(np.median(mp[keep], axis=0), va)
        print(f"  全量 {n_mem} 成员 MAE = {full_mae:.1f}")
        print(f"  剪枝 {len(keep)} 成员 MAE = {pruned_mae:.1f}  (Δ={pruned_mae - full_mae:+.1f})")
    else:
        print(f"  保留成员<2，无法剪枝集成")
    best1 = int(np.argmin(per_mae))
    print(f"  最佳单成员 [{best1}] MAE = {per_mae[best1]:.1f}  (单成员<<集成说明集成增益显著)")


if __name__ == "__main__":
    main()
