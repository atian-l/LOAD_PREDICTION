# -*- coding: utf-8 -*-
"""v8.1 架构包：多阶段任务阶梯（风险->类型->幅度->Shape）+ Domain 信息层。

Phase 0 先行：diag_residual 残差溯源 + 跨年可迁移性诊断，为 v8.1 全部模块提供 go/no-go 依据。
复用 load_pred（config/data_loader/features/train/model）与 v8（base/correction/weather_sim/
segments/oof）。六条 leakage 不变量全继承；val 仅 eval-only。
"""
