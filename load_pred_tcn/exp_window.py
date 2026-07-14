# -*- coding: utf-8 -*-
"""
exp_window：TCN 滑窗扫（seq_len × stride）。

调优元素：
  - seq_len ∈ {240, 480, 672, 960}（2.5/5/7/10 天；须 > RF≈91，均满足）
  - stride ∈ {48, 96, 192}（0.5/1/2 天；越小窗口越密、梯度步越多）
  - 含 2 个交叉点（seq672_st48 / seq960_st96）看长窗+密采样的交互

seq_len 改变单窗上下文长度（更长=更多历史，更短=更多独立样本）；
stride 改变窗口重叠度（越小样本越多但越相关，易过拟合；越大越独立但更少）。
选择信号：WF-CV（1 成员 Stage A -> top-K 2 成员 Stage B）。val 仅读数。
运行：python -m load_pred_tcn.exp_window
"""
from __future__ import annotations

from . import exp_common as E

CONFIGS = [
    # seq_len 轴（stride=96 基线）
    ("seq240_st96", {"seq_len": 240}),
    ("seq480_st96", {}),                 # 基线
    ("seq672_st96", {"seq_len": 672}),
    ("seq960_st96", {"seq_len": 960}),
    # stride 轴（seq_len=480 基线）
    ("seq480_st48",  {"stride": 48}),
    ("seq480_st192", {"stride": 192}),
    # 交叉
    ("seq672_st48", {"seq_len": 672, "stride": 48}),
    ("seq960_st48", {"seq_len": 960, "stride": 48}),
]


def main():
    d = E.build_cached()
    print(f"\n[exp_window] 数据: 特征{d['X'].shape[1]} 可用{d['usable'].sum()} "
          f"val{d['val_m'].sum()}  (基线 seq480/stride96)")

    E.hp_sweep(CONFIGS, "exp_window 滑窗扫", topk=3)


if __name__ == "__main__":
    main()
