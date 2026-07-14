# -*- coding: utf-8 -*-
"""
exp_capacity：TCN 容量扫（通道数×深度×核长 -> 感受野 RF）。

调优元素：num_channels（各残差块通道数=块数决定深度与 RF）、kernel_size。
  - 深度扫：[64]×{3,4,5,6}，k=7 -> RF=43/91/187/379（更深长程上下文，更多参数）
  - 宽度扫：{32,64,128}×4，k=7 -> 同 RF91，容量递增
  - 核长扫：[64]×4，k∈{5,7,9} -> RF=61/91/121
  - 金字塔：[32,32,64,64,128]，k=7 -> 5 层 RF187

选择信号：WF-CV（1 成员 Stage A 廉价筛选 -> top-K 2 成员 Stage B val 读数）。
val 仅读数、不参与选择。当前基线 = [64]×4 k=7 RF91（Tier2 配置）。
运行：python -m load_pred_tcn.exp_capacity
"""
from __future__ import annotations
import numpy as np

from . import exp_common as E
from .tcn import compute_receptive_field


def _rf(channels, k):
    return compute_receptive_field(list(channels), k)


def main():
    d = E.build_cached()
    print(f"\n[exp_capacity] 数据: 特征{d['X'].shape[1]} 可用{d['usable'].sum()} "
          f"val{d['val_m'].sum()}  (基线 [64]x4 k7 RF91)")

    k7 = 7
    configs = []
    # 深度扫（固定 64 通道、k=7）
    for depth in (3, 4, 5, 6):
        ch = [64] * depth
        tag = f"d{depth}_[64]x{depth}_k7_RF{_rf(ch, k7)}"
        configs.append((tag, {"num_channels": ch, "kernel_size": k7}))
    # 宽度扫（固定 4 层、k=7）
    for w in (32, 64, 128):
        ch = [w] * 4
        tag = f"w{w}_[{w}]x4_k7_RF{_rf(ch, k7)}"
        configs.append((tag, {"num_channels": ch, "kernel_size": k7}))
    # 核长扫（固定 [64]×4）
    for k in (5, 7, 9):
        ch = [64] * 4
        tag = f"k{k}_[64]x4_k{k}_RF{_rf(ch, k)}"
        configs.append((tag, {"num_channels": ch, "kernel_size": k}))
    # 金字塔
    pyr = [32, 32, 64, 64, 128]
    configs.append((f"pyr_{pyr}_k7_RF{_rf(pyr, k7)}", {"num_channels": pyr, "kernel_size": k7}))

    # 去重（基线 d4_[64]x4_k7 / w64 / k7 三者同配置，保留首个）
    seen = set()
    uniq = []
    for tag, ov in configs:
        key = (tuple(ov["num_channels"]), ov["kernel_size"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append((tag, ov))

    E.hp_sweep(uniq, "exp_capacity 容量扫", topk=4)


if __name__ == "__main__":
    main()
