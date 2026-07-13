# load_pred_tcn — TCN 移植版

山东省全省日前（D+1）直调负荷预测系统的 **TCN（时序卷积网络）变体**。

本包是 `load_pred/`（LightGBM 集成，v6 val MAE=1445.62 MW）的忠实移植：**除“模型方法
LightGBM → TCN”外，数据加载、特征构造、集成结构、OOF 校正、泄露不变量全部逐行一致**。
本包不依赖、不修改 `load_pred/` 及其之外的任何文件。

## 与 load_pred 的差异（仅这些）

| 维度 | load_pred (LightGBM) | load_pred_tcn |
|---|---|---|
| 单成员模型 | `lgb.Booster`（`lgb.train`） | `TCN`（PyTorch，因果膨胀卷积） |
| 成员持久化 | `boosters/member_NNN.txt`（`save_model`） | `boosters/member_NNN.pt`（`state_dict`） |
| 迭代控制 | `num_boost_round=best_it_fixed=80` | `epochs=best_it_fixed=60` |
| NaN 处理 | LightGBM 原生支持 NaN | 训练期列均值填充（PyTorch 卷积不支持 NaN；输入预处理，无泄露） |
| 训练超参 | num_leaves/lambda_l2/feature_fraction/bagging | num_channels/kernel_size/dropout/seq_len/stride/batch_size/lr/weight_decay |
| 模型/输出目录 | `<root>/models`、`<root>/output` | **本包内** `load_pred_tcn/models`、`load_pred_tcn/output` |

**完全不变**（逐行相同）：`data_loader.py`、`features.py`、`predict.py`、`__init__.py`；
config 的路径/时间边界/列名/气象列/`PRED_LAGS`；train 的 `build_dataset`/`usable_mask`/
`_time_weights`/`determine_best_iteration`/`compute_hour_bias`/`run_train`/`_evaluate`/`_write_*`；
model 的 `predict_load` 后处理（锚→中位数→λ 收缩→hour_bias→drift_corr→threshold_corr→clip，
含 `+= drift_corr` 符号约定）；6 条泄露不变量；40 成员集成结构
`{regression,quantile(0.45/0.5/0.55)}×{direct,residual}×5seeds`；MOS 锚；MismatchModel；
predict 的 14 天回看窗（skew 修正，同时为 TCN 提供感受野上下文）。

## 依赖

**云环境（已验证兼容：PyTorch 2.8.0 + CUDA 12.8 + Python 3.12 镜像）**

- `torch`：镜像已预装 PyTorch 2.8.0（CUDA 12.8），**无需再装**。本包仅用稳定 API
  （`nn.Conv1d`/`Adam`/`torch.save`/`torch.load(weights_only=True)`/`clip_grad_norm_` 等），
  2.8.0 完全兼容；`device="auto"` 自动检测 CUDA 12.8 GPU。
- `pandas` / `numpy` / `scikit-learn`：需自行安装。verbatim 的 `data_loader.py`/`features.py`
  无版本敏感 API，pandas 2.2+/3.0、numpy 2.x 均可；为与 load_pred 环境（pandas 3.0 / numpy 2.4 /
  scikit-learn 1.9）完全一致以保证复现，建议：

```bash
pip install "pandas>=2.2" "numpy>=2.0" "scikit-learn>=1.5"
# 无需 lightgbm。
```

- **Python 版本**：本包代码全部使用 `from __future__ import annotations`，类型注解惰性求值；
  无 3.13/3.14 专有语法。**Python 3.10+ 均可**（云镜像 3.12 已验证语法 + 冒烟通过；load_pred 本地为 3.14，两者兼容）。

GPU 训练强烈建议（40 成员 × (1 主训练 + 3 折 OOF 重训) = 160 次 TCN 训练）：
`device="auto"` 自动用 CUDA 12.8，无 GPU 时退回 CPU（会很慢）。
**4090（24GB）预计全流程 ~15–45 分钟**（卷积算子轻量；24GB 显存充裕，`batch_size` 可调大至 128/256 加速）。

> 注：`torch.load` 显式 `weights_only=True`（torch 2.6+ 安全默认；state_dict 仅含张量）。

## 用法

在项目根目录 `load_prediction/` 下以模块形式运行（**不要 cd 进 load_pred_tcn/**，相对导入要求包形式）：

```bash
# 训练 + 保存 + 评估（写 load_pred_tcn/models/、load_pred_tcn/output/）
python -m load_pred_tcn.train

# 预测 D+1 96 点（默认 D=今天，run_hour=9）
python -m load_pred_tcn.predict --run-date 2026-05-18

# 历史回测需复现 val 的 20:00-D-issue 气象条件（与 load_pred 同）
python -m load_pred_tcn.predict --run-date 2026-05-18 --run-hour 21
```

`train.main()` 在 val MAE ≥ 1500 MW 时返回非零退出码（与 load_pred 一致，作 pass/fail 闸门）。
验证指标写入 `load_pred_tcn/output/evaluation_metrics.txt`。

> **云环境运行**：本包从共享 `load_prediction/data/`（`direct_load_latest.csv`、
> `shandong_weather_15min.csv`）只读数据，模型/输出写本包内。上云时需把 `data/` 一并上传到
> 项目根目录；模型与结果会落在 `load_pred_tcn/models/`、`load_pred_tcn/output/`（不污染 load_pred）。

## TCN 设计要点

- **因果膨胀卷积**：`output[t]` 仅依赖 `input[<=t]`，无未来信息 → 满足泄露不变量 #5。
- **感受野** RF = 1 + (k−1)·Σdilations = 1 + 6·(1+2+4+8) = **91 步（≈23h）**。长程信息已由
  `PRED_LAGS=[96,192,288,672]` 滞后特征编码，TCN 在此之上做局部时序平滑。
- **训练**：滑窗 mini-batch（`seq_len=480` 即 5 天窗口、`stride=96` 即 1 天步长），
  逐时刻加权损失（regression=MSE，quantile=pinball）；窗口前 RF 步上下文不完整，不计入损失。
- **推理**：全序列因果前向，分块（块间 RF 重叠）避免长序列 OOM；predict 的 14 天回看窗
  （>672+96）使 D+1 特征与训练位等价（skew 修正），并为 TCN 提供充足回看上下文。
- **集成**：40 成员结构与 v6 完全一致；quantile 成员用 pinball 损失替代 LightGBM quantile 目标。
- **固定 epochs**：`best_it_fixed=60`（与 v6 `best_it_fixed=80` 同源哲学——walk-forward 在漂移 val
  上系统性过拟合，见 exp44，故固定保守迭代、不在 val 早停）。

## 已知适配（非逻辑变更）

1. **路径写在本包内**：为不修改 `load_pred_tcn/` 以外的文件，模型/输出目录改为本包内
   `load_pred_tcn/models`、`load_pred_tcn/output`；数据仍读共享 `load_prediction/data/`（只读）。
2. **NaN 列均值填充**：PyTorch 卷积不支持 NaN（LightGBM 原生支持）。用训练折内列均值填充
   空值，是输入预处理——不改变特征定义、不引入未来信息（NaN 位置由历史可得性决定）。
3. **TCN walk-forward 未实现**：`_walk_forward_best_iters` 在 `best_it_fixed` 启用时不被调用
   （生产路径）；仅当显式置 `best_it_fixed=None` 时会抛 `NotImplementedError` 提示需自实现
   逐 epoch 验证 MAE 早停。
4. **初始 MAE 预期**：TCN 默认超参未经 Agent Loop 调参，初始 val MAE 很可能 **高于** v6 的
   1445.62（甚至 >1500 闸门）。这是预期——本包提供与生产等价的 TCN 管线骨架，超参调优
   （num_channels/epochs/lr/seq_len/stride/dropout 等）留给后续实验。

## 文件清单

```
load_pred_tcn/
├── __init__.py        # 与 load_pred 相同
├── config.py          # 路径写包内 + TRAIN_CONFIG 换 TCN 参数（OOF 校正与 v6 一致）
├── data_loader.py     # 与 load_pred 逐行相同（verbatim）
├── features.py        # 与 load_pred 逐行相同（verbatim；特征构造模型无关）
├── tcn.py             # 新增：TCN 架构 + 滑窗训练 + 分块推理
├── model.py           # EnsembleModel 持 TCN 成员；predict_load 后处理与 v6 逐行相同
├── train.py           # train_ensemble 换 TCN 训练；其余与 load_pred 逐行相同
├── predict.py         # 与 load_pred 逐行相同（verbatim；仅 LoadModel 指向本包 TCN 版）
├── models/            # 训练后生成：model_bundle.pkl + boosters/member_NNN.pt
└── output/            # 训练后生成：full_predictions.csv / full_mae.csv / evaluation_metrics.txt / latest_prediction.csv
```
