# -*- coding: utf-8 -*-
"""load_pred_tft: Temporal Fusion Transformer 移植版（路线A Phase 0）。

与 load_pred（LightGBM）/ load_pred_tcn（TCN）同构：除"模型方法 -> TFT 序列建模"外，
数据/特征/集成结构/OOF 校正/6 条泄露不变量全部逐行一致。模型/输出写在本包内，
数据读共享 data/（只读）。
"""
