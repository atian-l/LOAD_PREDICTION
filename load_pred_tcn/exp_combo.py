"""exp_combo：组合 Phase-2 三个最佳发现，测 TCN 确切最佳 val MAE (全 RAW，关 OOF 校正)。

组合（全 RAW，因 exp_oof_ablation 证 OOF 校正在 TCN 上 +694 有毒）：
  - λ 向 MOS 锚收缩（exp_ensemble: λ=0.5->1554.9 vs λ=1->1638）
  - wd1e-3 正则（exp_regularize: wd1e-3 明显优于当前 wd1e-4）
  - 剪枝拖后腿成员（exp_members: 16->10, 1637.8->1616.4, −21MW）

post-hoc（λ/剪枝/关校正）免费；只有训练配置轴重训 3 个 16 成员模型。
运行：python -m load_pred_tcn.exp_combo   (~8 分钟 @4090)
"""
from __future__ import annotations
import numpy as np

from . import exp_common as E
from .exp_ensemble import _member_preds, _lam_pred

SEEDS = [42, 7]
LAMS = [0.3, 0.4, 0.5, 0.6, 1.0]
PRIOR_BEST = 1554.9  # exp_ensemble λ=0.5 RAW 16成员

# 3 个训练配置：base 复现 1554.9；wd3 隔离 weight_decay 效应；wd3_d0.4 = exp_regularize 最佳 raw
TRAIN_CFGS = [
    ("base_d0.2_w1e-4", {"seeds": SEEDS}),                                       # = exp_ensemble 基线
    ("wd3_d0.2",        {"seeds": SEEDS, "weight_decay": 1e-3}),                  # 隔离 wd 效应
    ("wd3_d0.4",        {"seeds": SEEDS, "dropout": 0.4, "weight_decay": 1e-3}),  # exp_regularize 最佳 raw
]


def _loo_keep(mp, va):
    """λ=1 下 LOO，返回 Δ>=0 的成员索引（保留集，与 exp_members 一致）。"""
    n = mp.shape[0]
    full = E._mae(np.median(mp, axis=0), va)
    loo = np.array([E._mae(np.median(np.delete(mp, i, axis=0), axis=0), va) for i in range(n)])
    return np.where(loo - full >= 0)[0]


def main():
    d = E.build_cached()
    X, pred_load, actual, val_m, mos_model = (d["X"], d["pred_load"], d["actual"],
                                               d["val_m"], d["mos_model"])
    va = actual[val_m].values
    Xv = X[val_m]
    print(f"\n[exp_combo] 数据: 特征{X.shape[1]} 可用{d['usable'].sum()} val{val_m.sum()}")
    print(f"  组合 λ×剪枝×正则, 全 RAW(关OOF校正) | v6={E.V6_VAL_MAE} TCN基线={E.TCN_BASE_MAE} 前期最优={PRIOR_BEST}\n")

    best = None  # (mae, desc)
    for tag, override in TRAIN_CFGS:
        print(f"训练 16 成员 [{tag}] ...")
        model = E.train_ens(override)
        mp, anchor = _member_preds(model, Xv, pred_load[val_m], mos_model)
        n_mem = mp.shape[0]
        keep = _loo_keep(mp, va)
        sel_sets = [("full16", np.arange(n_mem))]
        if len(keep) >= 2 and len(keep) < n_mem:
            sel_sets.append((f"pruned{len(keep)}", keep))
        print(f"  成员={n_mem} 剪枝保留={keep.tolist()}\n")
        print(f"  {'sel':>9} {'λ':>5} {'val_MAE':>9} {'Δvs_v6':>9} {'Δvs_1554.9':>11}")
        for sel_tag, sel in sel_sets:
            for lam in LAMS:
                mae = E._mae(_lam_pred(mp[sel], anchor, lam, "median", 0.2), va)
                print(f"  {sel_tag:>9} {lam:>5.1f} {mae:>9.1f} "
                      f"{mae - E.V6_VAL_MAE:>+9.1f} {mae - PRIOR_BEST:>+11.1f}")
                if best is None or mae < best[0]:
                    best = (mae, f"{tag}/{sel_tag}/λ{lam}")
        if len(keep) >= 2:
            pr_l1 = E._mae(_lam_pred(mp[keep], anchor, 1.0, "median", 0.2), va)
            tag_s = "  (对齐 exp_members base 1616.4)" if tag.startswith("base") else ""
            print(f"\n  [sanity] {tag} pruned{len(keep)} λ=1 RAW = {pr_l1:.1f}{tag_s}\n")

    print("=" * 72)
    print(f"TCN 确切最佳: {best[1]}  val MAE = {best[0]:.1f}  "
          f"(Δvs v6 {E.V6_VAL_MAE} = {best[0]-E.V6_VAL_MAE:+.1f})")
    print("=" * 72)


if __name__ == "__main__":
    main()
