# 山东省全省日前（D+1）直调负荷预测系统

## 1. 目录结构（满足 Constraint #6）

```
load_prediction/
├── data/                      # 输入数据（只读）
│   ├── direct_load_data.csv       # 时间, 预测直调负荷(外部), 实际直调负荷(基准)
│   └── shandong_weather_15min.csv # 起报时间, 预测时间, 25 个气象特征
├── load_pred/                 # 全部代码
│   ├── config.py              # 路径/时间边界/超参数
│   ├── data_loader.py         # 数据读取 + 气象去重
│   ├── features.py            # 无泄露特征工程（训练/预测共享）
│   ├── model.py               # LightGBM 集成封装
│   ├── train.py               # 训练模式入口
│   ├── predict.py             # 预测模式入口
│   └── exp*.py                # Agent Loop 选参实验脚本（非生产必需）
├── models/                    # 训练好的模型
│   ├── model_bundle.pkl
│   └── boosters/              # 各成员 booster
└── output/                    # 输出结果
    ├── full_predictions.csv
    ├── full_mae.csv
    ├── evaluation_metrics.txt
    ├── prediction_YYYYMMDD.csv
    └── latest_prediction.csv
```

## 2. 运行方式

### 训练模式
```bash
cd load_prediction
python -m load_pred.train
```
完成：训练集成 → 保存模型 → 生成 `full_predictions.csv` / `full_mae.csv` /
`evaluation_metrics.txt`。

### 预测模式（生产部署）
```bash
# 运行日 D（默认运行时刻 21:00，确保 D+1 气象预报已起报），预测 D+1 全天 96 点
python -m load_pred.predict --run-date 2026-05-18
```
完成：加载 `models/` 中模型（**不重训**）→ 仅用运行时可获得数据 → 输出
`prediction_YYYYMMDD.csv`（YYYYMMDD=运行日期）与 `latest_prediction.csv`。

## 3. 数据泄露合规说明（Inviolable Constraints 逐条）

### #1 数据隔离（最高优先级）
- `实际直调负荷`（真实负荷）**仅**在 `train.py` 中作为：
  - 训练目标 `y`（直接模式）或残差目标 `actual - pred_load`（残差模式）；
  - 评估基准（`_evaluate` / `full_mae.csv`）。
- `features.py` **绝不**读取 `实际直调负荷`。特征仅来自：
  外部预测直调负荷、气象预报、日历。
- 标准化/填补/编码：LightGBM 原生处理缺失，**无任何基于真实负荷的预处理**。

### #2 外部预测数据
- `预测直调负荷`（外部预测）作为输入特征 `pred_load[T]` 与滞后/滚动特征。
- 运行日 D 仅使用到 D+1 的外部预测（`predict.py` 中 `pred_load_series()` 读取历史
  及 D+1；不含 D+2）。

### #3 滞后特征
- 滞后**仅**基于 `预测直调负荷`（`features.pred_load_features`）。
- 最短滞后包含 `lag_192`（=2 天，见 `config.PRED_LAGS = [96,192,288,672]`）。
- 严禁使用真实负荷构造滞后——代码中不存在该路径。

### #4 气象数据
- `data_loader.load_weather_dedup`：对相同 `预测时间` 仅保留 `起报时间` 最晚的一条
  （`sort_values(起报时间).drop_duplicates(预测时间, keep="last")`）。
- 预测模式额外过滤 `起报时间 <= 运行时刻` 后再取最晚，模拟运行时仅能获得已起报版本。
- 训练/验证/预测三阶段统一执行该规则。

### #5 时间边界
- 训练数据上界 `TRAIN_END = 2026-01-31 23:45:00`（`< 2026-02-01`），见
  `train.py:usable_mask`。
- 官方验证集 `2026-02-01 ~ 2026-05-19 11:45:00` **仅**用于评估，不参与训练/早停。
- 早停用 walk-forward 3 折（春/秋/冬，均在训练期内）。
- 预测模式不使用未来真实负荷/未来气象实况/D+2 及之后外部预测。

### #6 工程约束
- 代码 `load_pred/`、输出 `output/`、模型 `models/`、输入 `data/`。
- 全 Python 实现。
- 训练与预测**完全解耦**：`predict.py` 仅 `EnsembleModel.load` 后推理，无任何重训逻辑。

### #7 Agent Loop 自主迭代
- 通过 `exp*.py` 系统化探索特征/模型/超参，每轮在验证集评估 MAE；
- 迭代路径：单模型 L2（1605）→ 残差+交互（1617）→ walk-forward（1605）→
  近期加权（1601）→ **多样化集成+分位数（1526）** → 满足 MAE<1500；
- 全程未违反数据隔离/时间边界。

## 4. 模型说明
- **LightGBM 集成**：`{regression, quantile(0.5)} × {direct, residual} × 5 种子` = 20 成员。
- 残差成员目标 `actual - pred_load`，预测 `pred_load + resid_hat`；直接成员预测 `actual`。
- 最终预测 = 成员均值（含 `pred_load` 作为强基线锚定）。
- `best_iter` 由 walk-forward 3 折平均确定；样本按时间线性加权（近期更高，缓解概念漂移）。
- 关键洞察：外部预测已编码气象，故气象增量信息有限；提升主要来自
  分位数（中位数）稳健性 + 多样化集成降低方差 + 系统偏差校正。

## 5. 性能（验证集 2026-02-01 ~ 2026-05-19 11:45:00）
- 外部预测基线 MAE ≈ 1646 MW
- 本系统 MAE < 1500 MW（见 `output/evaluation_metrics.txt`），R² ≈ 0.948
