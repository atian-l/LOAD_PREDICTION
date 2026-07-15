# -*- coding: utf-8 -*-
"""v8: 预测->判断->修正->融合 五层动态架构。

复用 load_pred 的 config/data_loader/features/train/model（不修改 load_pred/）。
基础模型 base A = v6（加载根 models/model_bundle.pkl）；base B = reg_only LightGBM 备选。
所有动态参数（trigger/α/w/adaptive/天气相似度）来自训练期 OOF，val 零参与。
"""
