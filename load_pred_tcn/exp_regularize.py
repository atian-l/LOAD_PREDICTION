# -*- coding: utf-8 -*-
"""
exp_regularize：TCN 正则扫（dropout × weight_decay 网格）。

调优元素：dropout（残差块内 Dropout）、weight_decay（Adam L2）。
  - dropout ∈ {0.1, 0.2, 0.3, 0.4}
  - weight_decay ∈ {1e-5, 1e-4, 5e-4, 1e-3}
  - 4×4=16 配置（Tier2 当前 = dropout0.2 / wd1e-4，在该网格内）

选择信号：WF-CV（1 成员 Stage A 廉价筛选 -> top-K 2 成员 Stage B val 读数）。
val 仅读数、不参与选择。这是 Tier2 反过拟合方向（120ep 过拟合->回 60 + dropout0.1->0.2 +
wd1e-5->1e-4）的系统化网格版，用于判定正则强度是否已达最优。
运行：python -m load_pred_tcn.exp_regularize
"""
from __future__ import annotations

from . import exp_common as E

DROPOUTS = [0.1, 0.2, 0.3, 0.4]
WEIGHT_DECAYS = [1e-5, 1e-4, 5e-4, 1e-3]


def main():
    d = E.build_cached()
    print(f"\n[exp_regularize] 数据: 特征{d['X'].shape[1]} 可用{d['usable'].sum()} "
          f"val{d['val_m'].sum()}  (基线 dropout0.2 / wd1e-4)")

    configs = []
    for dp in DROPOUTS:
        for wd in WEIGHT_DECAYS:
            tag = f"drop{dp}_wd{wd:g}"
            configs.append((tag, {"dropout": dp, "weight_decay": wd}))

    E.hp_sweep(configs, "exp_regularize 正则扫 (4x4=16)", topk=4)


if __name__ == "__main__":
    main()
