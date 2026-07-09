# load_pred 包 - 山东省全省日前（D+1）负荷预测

> 完整项目说明请见**上一级目录的 [`../README.md`](../README.md)**（含环境、数据要求、部署/训练用法、
> 模型架构、合规与性能）。本文件仅作包级索引。

## 模块

| 文件 | 职责 |
|------|------|
| `config.py` | 路径（据 `__file__` 自动推导至项目根）、时间边界、特征/模型超参数 |
| `data_loader.py` | 负荷/气象读取、气象"最晚起报"去重、时间轴 |
| `features.py` | 无泄露特征工程；**训练/预测共享单一入口 `build_features`**；含 `MismatchModel`、`MosModel` |
| `model.py` | `EnsembleModel`：40 成员 LightGBM 集成 + `hour_bias`/`drift_corr`/`threshold_corr` 事后校正 |
| `train.py` | 训练入口：训练 + 评估 + 保存模型（MAE≥1500 非零退出） |
| `predict.py` | 部署入口：加载模型推理 D+1（不重训；14 天回看窗口构造特征，skew 修复） |
| `exp*.py` / `exp_analyze.py` | Agent Loop 选参实验脚本（非生产必需） |

## 运行（从项目根 `load_pred_v7/` 执行，必须用 `-m`）

```bash
python -m load_pred.predict --run-date 2026-05-18      # 部署：预测 D+1 96 点
python -m load_pred.predict --run-date 2026-06-14 --run-hour 21   # 历史回测（复现验证集条件）
python -m load_pred.train                               # 重训 + 评估 + 保存
```

## 合规要点

实际负荷（`实际负荷`）**仅评估**，绝不入特征；滞后仅基于 `预测负荷`（最短 `lag_192`）；气象三阶段
统一去重；训练上界 `2026-02-28`，验证集 `2026-03-01~06-15` 仅评估。详见上级 README §8。
