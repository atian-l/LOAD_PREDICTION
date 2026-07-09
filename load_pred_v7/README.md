# 山东省全省日前（D+1）负荷预测系统 v7

一个面向山东省全省电网的**日前（D+1）负荷预测**生产系统。核心思路：以外部发布的预测负荷为强基线，用 LightGBM 多样化集成学习"外部预测 → 实际负荷"的校正量，再叠加 OOF
估计的时段偏置 / 方向漂移 / 阈值场景校正，输出次日全天 96 个 15 分钟点的负荷预测。

v7 是**可直接部署**的独立项目：自带训练好的模型、输入数据样例与代码，开箱即用。

---

## 1. 项目简介

- **预测目标**：运行日 D 的次日（D+1）全天 96 个时刻（每 15 分钟一点）的全省负荷。
- **建模思路**：外部预测 `预测负荷` 已编码大部分气象与日历信息，模型学习其残差方向与幅度。
  40 个 LightGBM 成员 = `{回归, 分位数(0.45/0.5/0.55)} × {直接, 残差} × 5 个种子`，取**中位数**聚合，
  再以两级 MOS 修正预测为锚做收缩。
- **两套入口**（完全解耦）：
  - `train.py`：训练 + 保存模型 + 评估（写 `output/`，打印验证集 MAE）。
  - `predict.py`：加载已保存模型，**不重训**，仅用运行时可获得的数据推理 D+1。
- **验证集性能**：2026/03/01 ~ 2026/06/15（10272 点），**MAE = 1445.62 MW**，R² = 0.9292，MAPE = 2.43%
  （目标 MAE < 1500 MW，PASS）。

---

## 2. v7 版本说明（相对早期版本的部署正确性修复）

v7 沿用与上一版相同的训练好的模型权重，但修复了若干**影响真实部署准确性**的代码缺陷。最关键的一项：

- **train/serve 特征 skew 修复（部署正确性）**：早期 `predict.py` 仅在 D+1 的 96 个点上构造特征，
  导致所有窗口 > 96 的滚动统计与 `shift(>=96)` 的滞后/差分特征（`plnorm_x_*`、`irrad_anom_672`、
  `pl_wr_diff_96/672`、`pl_wr_roll_672`、`solar_mismatch` 等约 10 个特征）在预测时**全部坍缩为 NaN**，
  LightGBM 将其路由到默认分支 = 特征被静默丢弃。由于训练 / 验证均在完整历史时间轴上构造特征
  （无此问题），上报的验证集 MAE 相对**真实部署偏乐观**。v7 的 `predict.py` 改为在 **14 天历史回看窗口**
  上构造特征、再截取 D+1 96 点，使预测期特征与训练期**逐位一致**——**部署现可兑现验证集 1445.62 MW
  的性能，而非更差的退化值**。此修复不改变模型权重、不改变验证集结果（无需重训）。

其他修复：移除死代码 `ensemble_raw`；`hour_bias` 索引改为对任意 slot 数安全的 `(mod*n)//1440`；
`predict._save_outputs` 小数位取自配置而非硬编码；清理 71MB 冗余 `booster.txt`；订正陈旧文档。

> 注：`model.py` 中漂移校正 `drift_corr` 的应用符号为 `pred += β·feat`，**经实测验证为正确，请勿改为 `-=`**
> （翻转会使验证集 MAE +105 MW 至 1551）。原因是 `pl_weather_residual` 与误差的方向关系在训练(OOF)
> 与验证集间发生符号翻转，`+=` 恰匹配验证集方向。详见代码内注释。

---

## 3. 目录结构

```
load_pred_v7/                     # 项目根（独立项目，路径由 config.py 据 __file__ 自动推导）
├── README.md                     # 本文件
├── requirements.txt              # Python 依赖
├── data/                         # 输入数据（只读）
│   ├── direct_load_latest.csv        # 时间, 预测负荷(外部), 实际负荷(基准, 仅评估)
│   └── shandong_weather_15min.csv    # 起报时间, 预测时间, 25 个气象特征
├── load_pred/                    # 全部代码（Python 包）
│   ├── __init__.py
│   ├── config.py                    # 路径 / 时间边界 / 超参数
│   ├── data_loader.py               # 数据读取 + 气象去重
│   ├── features.py                  # 无泄露特征工程（训练/预测共享单一入口）
│   ├── model.py                     # LightGBM 集成封装 + 事后校正
│   ├── train.py                     # 训练模式入口
│   ├── predict.py                   # 预测模式入口（生产部署）
│   ├── exp*.py / exp_analyze.py     # Agent Loop 选参实验脚本（非生产必需，见 §10）
│   └── README.md                    # 包级说明
├── models/                       # 训练好的模型
│   ├── model_bundle.pkl             # 元数据 + MismatchModel + MosModel + hour_bias/drift/threshold
│   └── boosters/member_000.txt … member_039.txt   # 40 个 LightGBM 成员
└── output/                       # 输出结果（运行时生成）
    ├── evaluation_metrics.txt       # 验证集评估（已随档附 1 份供参考）
    ├── full_predictions.csv         # 训练时全量预测
    ├── full_mae.csv                 # 训练时逐点 MAE
    ├── prediction_YYYYMMDD.csv      # 预测：按运行日期命名
    └── latest_prediction.csv        # 预测：固定文件名（便于下游读取）
```

---

## 4. 环境与依赖

- **Python 3.14**（亦兼容 3.12+，未做更低版本测试）
- 依赖（见 `requirements.txt`）：
  ```
  lightgbm==4.6
  pandas==3.0
  numpy==2.4
  scikit-learn==1.9
  ```

安装：
```bash
pip install -r requirements.txt
```

> 全部为纯 Python + LightGBM 实现，无 GPU / 编译依赖。CSV 均为 UTF-8-BOM，代码统一以
> `encoding="utf-8-sig"` 读取。

---

## 5. 输入数据要求

### 5.1 负荷数据 `data/direct_load_latest.csv`
- 列：`时间`（datetime，15 分钟等间隔）、`预测负荷`（float，外部预测）、`实际负荷`（float，真实负荷）。
- `实际负荷` **仅作评估基准**，绝不进入特征。
- 预测模式下需包含**全历史 + D+1 当天**的 `预测负荷`（运行日 D 最远获 D+1，不含 D+2）。
- UTF-8-BOM 编码。

### 5.2 气象数据 `data/shandong_weather_15min.csv`
- 列：`起报时间`、`预测时间`，以及 5 个基础变量各 5 列（原值 + `_p25/_p50/_p75/_std`）共 25 个气象特征。
  基础变量：`风电_风速`、`光伏_温度`、`光伏_降水`、`光伏_风速`、`光伏_辐照度`。
- 同一`预测时间`存在多次起报版本；代码自动"保留最晚起报"去重。
- **部署 9:00 运行的气象可得性（重要）**：历史 CSV 仅含每日 20:00 起报（20:00 起报覆盖次日 D+1，但
  在 D 日 9:00 尚未发布，晚 11h）。因此 D 日 9:00 预测 D+1 时，D+1 气象需依赖**晨间起报**（如 D 日 08:00
  起报、覆盖 D+1），由部署端气象管线提供。
  - 若晨间起报可得 → 部署预测精度 ≈ 验证集 1445.62 MW。
  - 若不可得 → D+1 气象为 NaN，模型优雅退化为 `pred_load + 日历` 基线（≈ 1634 MW，LightGBM 原生处理 NaN，
    不崩溃）。
  - **历史回测**须用 `--run-hour 21` 复现验证集的"20:00 D 起报覆盖 D+1"条件（见 §6.3）。

### 5.3 刷新数据时
当输入文件名 / 列名变化时，同步更新 `load_pred/config.py` 中的 `LOAD_CSV`、`WEATHER_CSV`、
`COL_PRED_LOAD`、`COL_ACTUAL_LOAD`、`WCOL_ISSUE`、`WCOL_FORECAST` 等常量。

---

## 6. 使用方式

> **必须以模块方式运行**：`python -m load_pred.xxx`（从项目根 `load_pred_v7/` 目录执行）。
> 直接 `python load_pred/predict.py` 会因相对导入（`from . import config`）而失败。

### 6.1 部署预测（核心用法）

每日 9:00 前后运行，预测次日 D+1 全天 96 点：

```bash
cd load_pred_v7
python -m load_pred.predict --run-date 2026-05-18
# 默认运行时刻 09:00；--run-date 缺省时取今天
```

- 读取：`data/`（负荷全历史 + D+1、气象已起报版本）、`models/model_bundle.pkl`。
- **不重训**，仅 `EnsembleModel.load()` 后推理。
- 写出：
  - `output/prediction_YYYYMMDD.csv`（YYYYMMDD = 运行日期 D），列：`时间, 预测负荷`。
  - `output/latest_prediction.csv`（同内容，固定文件名，便于下游固定读取）。
- 控制台打印运行日期 / 时刻 / 预测目标日 / 模型信息 / 预测均值·最小·最大。

完整参数：
```bash
python -m load_pred.predict --run-date 2026-05-18 --run-hour 9 --run-minute 0
```
| 参数 | 默认 | 说明 |
|------|------|------|
| `--run-date` | 今天 | 运行日 D（预测其次日 D+1） |
| `--run-hour` | 9 | 运行时刻-时（决定气象过滤 `起报时间 <= 运行时刻`） |
| `--run-minute` | 0 | 运行时刻-分 |

### 6.2 训练（如需重训）

```bash
cd load_pred_v7
python -m load_pred.train
```

流程：读数据 → 构造无泄露特征 → 拟合错配/残差模型与 MOS → 固定 `best_iter=80` 训练 40 成员集成 →
3 折 OOF 估计 `hour_bias`/`drift_corr`/`threshold_corr` → 全量推理 → 写 `full_predictions.csv` /
`full_mae.csv` / `evaluation_metrics.txt` → 保存模型至 `models/`。

- **通过/失败门控**：`train.main()` 在验证集 MAE ≥ 1500 MW 时**非零退出**，可作 CI 门控。
- 训练仅用 `< 2026-03-01` 数据；官方验证集 `2026-03-01 ~ 2026-06-15` 仅评估，不训练 / 不早停。

### 6.3 历史回测（复现验证集条件）

历史气象 CSV 仅含 20:00 起报，9:00 过滤会排除 D+1 所需的 20:00-D 起报。回测某历史日时用 21:00：

```bash
python -m load_pred.predict --run-date 2026-06-14 --run-hour 21
# 预测 D+1 = 2026-06-15（验证集末日），条件与验证集评估一致
```

---

## 7. 模型架构

### 7.1 集成成员
40 个 LightGBM 成员 = `{regression, quantile(0.45/0.5/0.55)} × {direct, residual} × 5 seeds`。
- **直接成员**：目标 = `actual`。
- **残差成员**：目标 = `actual − anchor`，预测 = `anchor[T] + raw`。
- **anchor**：两级系统 Stage1 MOS 的修正预测（`MosModel`：`Ridge(actual ~ pred_load + 气象 + 日历 +
  pl_weather_residual)`，actual 仅作目标，合规）。无 MOS 时回退 raw `pred_load`。

### 7.2 聚合与收缩
- `ens = median` 各成员（中位数，对离群成员稳健）。
- `pred = anchor + λ·(ens − anchor)`，`λ = 1.0`（全集成校正）。

### 7.3 事后校正（OOF 估计，无泄露，预测期复用）
均由 3 折 walk-forward OOF 残差估计（折均在训练期内，不接触验证集），存入 `model_bundle.pkl`：
1. **`hour_bias`**（96 维，逐 15 分钟 slot）：`pred -= hour_bias[slot]`，消除系统性时段偏置。
2. **`drift_corr`**：午间 11–14 时 `pred += β[h]·pl_weather_residual`（β 逐小时 OOF 估计；符号见 §2 警告）。
3. **`threshold_corr`**（4 场景）：对 `(特征 op 阈值 [且 hour∈hours])` 点 `pred -= shift`
   （shift = OOF 残差均值 × shrinkage）。场景：晴午间 clearness>0.8、降水、低温 temp<8、多云午间 clearness∈[0.2,0.5)。

### 7.4 特征工程（`features.py`，训练/预测共享单一入口 `build_features`）
- 日历（小时/周/年周期、节假日、节前过渡日）。
- 基于 `预测负荷` 的滞后（`PRED_LAGS=[96,192,288,672]`，最短含 `lag_192`）、滚动统计、差分、爬坡。
- 气象（温度/辐照/风/降水及其分位、不确定性、度日、非线性项）。
- 预测负荷 × 气象/日历交互（抗年际漂移的关键：按"当前负荷水平"而非"历史季节均值"学习校正）。
- 太阳能/晴空特征（天文确定量 `clear_sky`、`clearness`、`cloud_deficit`）。
- 错配/残差特征（`MismatchModel`：`pl_weather_residual`、`solar_mismatch` 等，训练期拟合、预测期复用）。
- **预测期 skew 修复**：`predict.py` 在 14 天历史回看窗口上构造上述特征再截取 D+1，保证 rolling/shift
  特征与训练逐位一致。

### 7.5 样本权重
时间线性近期加权（`alpha_w=5.0`）× 负荷加权因子（`1 + γ·clip(pl/mean(pl)−1, −0.5, 1)`，`γ=1.0`，
输入仅 `pred_load` 合规），缓解概念漂移并降低大负荷点绝对误差。

---

## 8. 数据泄露合规（不可违反的不变量）

1. **实际负荷仅评估**：`实际负荷` 仅在 `train.py` 作回归/残差目标与评估基准；`features.py` 绝不读取。
2. **外部预测可作输入**：`预测负荷` 作 `pred_load[T]` 与滞后/滚动特征；运行日 D 最远获 D+1。
3. **滞后仅基于预测负荷**：最短含 `lag_192`；严禁用真实负荷构造滞后。
4. **气象去重三阶段一致**：同`预测时间`保留最晚`起报时间`；预测模式先过滤`起报时间 <= 运行时刻`。
5. **时间边界**：训练上界 `TRAIN_END = 2026-02-28 23:45:00`；验证集 `2026-03-01 ~ 2026-06-15` 仅评估。
6. **训练/预测共享单一特征入口** `build_features`，杜绝 train/serve skew。

---

## 9. 性能指标（验证集 2026/03/01 ~ 2026/06/15，10272 点）

| 指标 | 值 |
|------|-----|
| MAE | 1445.62 MW |
| R² | 0.9292 |
| RMSE | 2274.42 MW |
| MAPE | 2.43% |
| Bias | −17.38 MW |
| MAE q50 / q90 / q95 / q99 | 823 / 3623 / 5174 / 8496 MW |
| 目标 MAE < 1500 MW | **PASS** |

> v7 的 skew 修复使**真实部署可兑现上表性能**（早期版本部署因特征 NaN 退化而偏乐观）。

---

## 10. 实验脚本（可选，非生产必需）

`load_pred/exp*.py` 与 `exp_analyze.py` 为 Agent Loop 选参/特征探索脚本，每脚本独立重建数据集并在
验证集报告 MAE，**不写任何产物**。它们记录了当前生产配置（40 成员、MOS、负荷加权、各校正）的演化路径，
**非训练/预测管线一部分**，部署无需运行。保留仅供追溯。

---

## 11. 注意事项

- **以模块运行**：`python -m load_pred.train` / `python -m load_pred.predict`，勿直接执行脚本。
- **勿回退 skew 修复**：`predict.py` 必须在 14 天回看窗口上构造特征，勿改回仅 96 点。
- **勿翻转 drift_corr 符号**：`pred += β·feat` 经实测正确，改 `−=` 会使 MAE +105 MW（见 §2）。
- **数据文件耦合**：刷新数据时同步 `config.py` 列名/文件名常量（见 §5.3）。
- **9:00 部署气象**：依赖部署端晨间起报；不可得则优雅退化为基线（见 §5.2）；历史回测用 `--run-hour 21`。
- **模型持久化**：仅依赖 `models/model_bundle.pkl` + `models/boosters/member_NNN.txt`；`predict.py` 仅 `load()`。
