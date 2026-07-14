# -*- coding: utf-8 -*-
"""
exp_train_dyn：TCN 训练动力学扫（单变量，不全组合爆炸）。

调优元素（每轴围绕 Tier2 基线 60ep/lr1e-3/cosine/bs64/gc5.0 各扫几档）：
  - epochs(best_it_fixed) ∈ {40, 60, 80, 100}
  - learning_rate ∈ {3e-4, 1e-3, 3e-3}
  - lr_schedule ∈ {none, cosine}
  - batch_size ∈ {32, 64, 128}
  - grad_clip ∈ {0(off), 5.0(on)}

基线在各轴重复出现一次以横向对照。选择信号：WF-CV（1 成员 Stage A -> top-K 2 成员 Stage B）。
val 仅读数、不参与选择。
运行：python -m load_pred_tcn.exp_train_dyn
"""
from __future__ import annotations

from . import exp_common as E

# 单变量扫：每项 (tag, override)。基线 = {}（Tier2 当前值）
CONFIGS = [
    # epochs 轴
    ("ep40",  {"best_it_fixed": 40}),
    ("ep60",  {}),                      # 基线
    ("ep80",  {"best_it_fixed": 80}),
    ("ep100", {"best_it_fixed": 100}),
    # learning_rate 轴
    ("lr3e-4", {"learning_rate": 3e-4}),
    ("lr1e-3", {}),                     # 基线
    ("lr3e-3", {"learning_rate": 3e-3}),
    # lr_schedule 轴
    ("sched_none",  {"lr_schedule": "none"}),
    ("sched_cosine", {}),               # 基线
    # batch_size 轴
    ("bs32",  {"batch_size": 32}),
    ("bs64",  {}),                       # 基线
    ("bs128", {"batch_size": 128}),
    # grad_clip 轴
    ("gc0", {"grad_clip": 0.0}),
    ("gc5", {}),                         # 基线
]


def main():
    d = E.build_cached()
    print(f"\n[exp_train_dyn] 数据: 特征{d['X'].shape[1]} 可用{d['usable'].sum()} "
          f"val{d['val_m'].sum()}  (基线 60ep/lr1e-3/cosine/bs64/gc5)")

    # 去重基线（{}重复多次）：保留首次出现，Stage A 表内可横向对照
    seen = set()
    uniq = []
    for tag, ov in CONFIGS:
        key = tuple(sorted(ov.items()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append((tag, ov))

    E.hp_sweep(uniq, "exp_train_dyn 训练动力学扫", topk=4)


if __name__ == "__main__":
    main()
