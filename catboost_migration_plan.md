# CatBoost 迁移工程计划（LightGBM -> CatBoost）

> 状态：计划阶段，暂不输出生产代码。所有改动先走 exp_catboost_*.py 独立实验，过 walk-forward 稳定性 gate 后才考虑入生产（v8 候选）。
> 基线：v6 val MAE=1445.62 MW（R²=0.9292），val 窗口 2026-03-01~06-15。
> 训练规模：train 2024-01-01~2026-02-28 ≈ 75840 行 × 126 特征；40 成员集成；3 折 walk-forward。

---

## 1. 背景与预期管理

**前置事实（上一轮 next_phase_assessment 已证）**：误差方差 83.69% 来自外部预测误差 ext_error（预测时不可得），所有运行时可得特征对误差的可解释上限仅 21.46%，**约 78.5% 不可学习**。模型结构经 tech_evaluation 判定"非主要瓶颈"。

**因此 CatBoost 无法凭空突破信息天花板**。但它有 3 个**合理的差异化假设**值得实验：

1. **Ordered boosting 抗漂移**：LightGBM 因 2026 验证集漂移被迫用 `best_it_fixed=80`（walk-forward 系统性过拟合）。CatBoost 的 ordered boosting 专门降低 prediction shift，**或可在更多迭代下不过拟合、跨年更稳**--这是最值得验证的假设。
2. **原生分类特征**：当前 hour/dow/month/holiday 用 sin/cos 连续编码；CatBoost 的目标编码或能捕捉非平滑效应（节假日跳变、月末等）。
3. **GPU 速度红利**：4090/A100 上单模型秒级，**可支撑更大集成 + 全量超参搜索 + 更多 walk-forward 折**，用算力换泛化。

**成功 gate（硬性，三者全满才考虑入生产）**：
- 官方 val MAE < 1445.62（v6），且降幅 > 10 MW（否则在噪声范围内）
- walk-forward 3 折 MAE 均不退化（任一折 +> 15 MW 即否决）
- 午间(11-14) MAE 不退化
- 跨年符号翻转仍被 OOF 校正吸收（不引入新方向性 bug）

不满足则停留在实验档案，v6 继续生产。

---

## 2. 总体原则

1. **实验优先，生产不动**：train.py / predict.py / features.py / model.py / v6 / v7 归档在 gate 通过前**零修改**。新逻辑全部落在 `exp_catboost_*.py`（throwaway，复用现有 `build_dataset`/`build_features` 管线，仅替换 booster）。
2. **不变量守恒**：6 条泄露不变量逐条保持（见 §6）。
3. **公平 A/B**：Phase 1 必须用**与 LightGBM 完全相同的特征矩阵**对比，隔离"算法差异"与"特征差异"。
4. **OOF 校正重估**：hour_bias / drift_corr / threshold_corr 是对**模型残差**估计的，换 booster 后必须重新估计（不能复用 v6 的校正量）。
5. **best_it 不照搬**：LightGBM 的 80 是其过拟合特性下的产物；CatBoost 需重新 walk-forward 确定 iterations（预期 300-1500）。

---

## 3. 架构映射

### 3.1 成员级 1:1 映射
`EnsembleModel` 结构不变：仍持 `members: list` + `member_residual: list[bool]`，仍 {direct, residual} × {regression, quantile(0.45/0.5/0.55)} × 5 seeds = 40 成员，仍 median 聚合 + λ 收缩。

| 组件 | LightGBM (现) | CatBoost (目标) | 改动 |
|---|---|---|---|
| booster 类型 | `lgb.Booster` | `catboost.CatBoostRegressor` | model.py import + add_member 签名 |
| 训练调用 | `lgb.train(params, dtr, n)` | `CatBoostRegressor(params).fit(X, y, sample_weight=w)` | train.py train_ensemble |
| 数据容器 | `lgb.Dataset` | `catboost.Pool` (或直接 ndarray) | train.py walk-forward + OOF |
| 预测 | `booster.predict(X[cols])` | `model.predict(X[cols])` | model.py predict_load（签名几乎相同） |
| 持久化 | `bst.save_model(.txt)` / `lgb.Booster(model_file=)` | `cbm.save_model(.cbm)` / `CatBoostRegressor().load_model()` | model.py save/load |
| 早停/迭代 | `num_boost_round` + `early_stopping` | `iterations` + `od_type=Iter`/`IncToDec` | train.py determine_best_iteration |

### 3.2 模型无关部分（**不改**，直接复用）
- MOS 锚（`features.MosModel`，Ridge）--target=actual 仅作目标，与 booster 无关。
- 中位数聚合 + λ 收缩（`_aggregate` / `predict_load` 主干）。
- hour_bias / drift_corr / threshold_corr 的**应用逻辑**（`predict_load` 后段）--仅其**估计值**需重估。
- `features.build_features()` 单一入口（不变量 #5）--特征矩阵完全不变。
- `MismatchModel`（pl_weather_residual 等）--训练期拟合，与 booster 无关。

### 3.3 超参映射表（config.TRAIN_CONFIG）

| LightGBM | CatBoost | 说明 |
|---|---|---|
| `objective="regression"` (RMSE) | `loss_function="RMSE"` | 直接成员 |
| `objective="quantile", alpha=0.45` | `loss_function="Quantile:alpha=0.45"` | GPU 支持 |
| `num_leaves=255` | `depth=8` | 2^8=256 叶，近似等价 |
| `min_data_in_leaf=200` | `min_data_in_leaf=200` | 同名 |
| `lambda_l2=4.0` | `l2_leaf_reg=4.0` | 同义 |
| `feature_fraction=0.8` | `rsm=0.8` | **GPU 有限支持，见 §4** |
| `bagging_fraction=0.8, bagging_freq=1` | `bootstrap_type="Bayesian", bagging_temperature=1.0` | 机制不同，需重调 |
| `learning_rate=0.03` | `learning_rate=0.03` | 同名 |
| `best_it_fixed=80` | `iterations=TBD(300-1500)` | walk-forward 重定 |
| - | `task_type="GPU", devices="0"` | 新增 |
| - | `boosting_type="Plain"`（先 Plain，再试 Ordered） | 新增，§1 假设① |

---

## 4. CatBoost GPU 关键注意（Phase 0 必须先验证）

1. **rsm（列子采样）GPU 支持有限**：CatBoost 在 GPU 上 `rsm<1.0` 多版本下被忽略或要求 `grow_policy="Lossguide"`。**LightGBM 的 feature_fraction=0.8 不能直接迁移**。
   - 替代方案 A：GPU 上接受 rsm=1.0，靠 `bagging_temperature`（行采样）+ `depth` + `l2_leaf_reg` 正则。
   - 替代方案 B：在**集成层**做手动列子采样--每成员随机抽不同特征子集（40 成员各自 rsm），等价于 feature_fraction 的集成化实现，且不依赖单模型 GPU rsm。
   - **Phase 0 实测**：分别跑 rsm=1.0 vs Lossguide+rsm=0.8 vs 集成层列子采样，看哪个不退化。
2. **Quantile loss on GPU**：`loss_function="Quantile:alpha=..."` 在 GPU 支持（确认可用），但历史上有数值精度差异，需核对分位成员的预测分布合理。
3. **Ordered boosting on GPU**：`boosting_type="Ordered"` 比 Plain 慢 1.5-3×，GPU 上可用但部分参数受限。先 Plain 跑通基线，再单独试 Ordered 验证 §1 假设①。
4. **Linux only**：CatBoost GPU 在 Windows 上支持很差/不稳。**云实例必须用 Linux**（Ubuntu 20.04/22.04 + CUDA 11.8+ 或 12.x）。`pip install catboost` 的 Linux wheel 含 GPU 支持，无需单独编译。
5. **NaN**：CatBoost 原生处理 NaN（与 LightGBM 一致），现有 NaN 特征无需填充改动。
6. **sample_weight**：CatBoost `fit(sample_weight=)` 支持，时间+负荷联合权重（`_time_weights`）可直接传入。

---

## 5. 特征工程决策

**Phase A（公平 A/B，必做）**：`build_features()` 输出**完全相同**的 126 列连续特征矩阵喂 CatBoost。隔离算法差异。cat_features=[]（全部当数值）。

**Phase B（分类变体，可选，Phase A 达标后）**：将 hour(0-23)/dayofweek(0-6)/month(1-12)/is_holiday/is_weekend/is_day_before_holiday 作为 `cat_features`（需转为 int/category dtype）。CatBoost 目标编码可能捕捉 sin/cos 平滑编码遗漏的非连续效应。**注意**：这与 Phase A 是不同实验，不能混判；且分类编码有过拟合风险（尤其节假日样本少），必须 walk-forward 验证跨年迁移。

**不改 build_features() 本身**：Phase B 在 exp 脚本内对输出 X 做 dtype 转换 + 记录 cat_features 索引，不污染生产特征构建器。

---

## 6. 泄露不变量保持（6 条逐条）

| # | 不变量 | CatBoost 下保持方式 |
|---|---|---|
| 1 | 实际负荷仅评估 | CatBoost 训练 target=actual（direct）/ actual-anchor（residual），仅作目标；特征仍不含 actual。同 LightGBM。 |
| 2 | 滞后仅来自预测负荷 | `build_features()` 不变，PRED_LAGS=[96,192,288,672] 不变，lag_192 仍在。 |
| 3 | 气象去重三阶段一致 | `data_loader.load_weather_dedup()` 不变；CatBoost 喂同一 weather_dedup。 |
| 4 | 时间边界 | TRAIN_END=2026-02-28 不变；val 窗口 eval-only 不变；walk-forward 折仍在训练期内。 |
| 5 | 训练/预测共享特征构建器 | `build_features()` 单一入口不变；predict 仍走 14 天 lookback 窗口（skew fix 不变）。 |
| 6 | 模块结构 | CatBoost 版 model.py/train.py 作为 v8 候选独立落地，不覆盖 v6/v7；predict.py 仍只 `EnsembleModel.load()`。 |

**CatBoost 特有的新泄露风险点（需 exp 脚本自查）**：
- CatBoost 的 `cat_features` 目标编码**默认用训练集统计**，只要 fit 在训练 mask 内、predict 不重 fit，则无泄露。exp 脚本须确保 cat 编码统计仅来自训练折。
- CatBoost 的 `bootstrap_type="Bayesian"` 默认用训练集先验，无未来信息，合规。

---

## 7. 实验阶段

| 阶段 | 脚本 | 目标 | 输出 6 指标 | gate |
|---|---|---|---|---|
| **Phase 0** | exp_catboost_env.py | 环境/管线打通：CatBoost GPU 跑通单成员；验证 rsm/quantile/ordered 行为；校准单模型训练耗时 | 官 val MAE（单成员，仅看能否跑通） | 跑通即过 |
| **Phase 1** | exp_catboost_ab.py | **公平 A/B**：CatBoost 40 成员 + 同特征 + 同 OOF 校正流程 vs LightGBM v6 | 官 val MAE / 3 折 walk-forward / 午间 MAE / Bias / 高误差样本 / 跨窗稳定 | val<1445 且 3 折不退化 |
| **Phase 2** | exp_catboost_cat.py | 分类变体（hour/dow/month/holiday 为 cat_features） | 同上 | 较 Phase 1 再降且跨年稳 |
| **Phase 3** | exp_catboost_hp.py | 超参搜索（depth/lr/l2/bagging_temp/iterations/Ordered vs Plain） | 同上，加搜索轨迹 | 最优 config 锁定 |
| **Phase 4** | exp_hetero_lgbm_cb.py | **异构集成**：LightGBM 40 + CatBoost 40 混合中位数（不同算法相关性低，或真增多样性--tech_evaluation 的 0.986 是 LGBM-LGBM 自相关，LGBM-CB 不同） | 同上 | 较纯 CB/LGBM 再降 |
| **Gate** | - | Phase 1-4 任一达标且稳定 -> v8 候选 | 全 6 指标 | 过 gate 才动生产代码 |

**每阶段必输出 6 指标**（Part 1 约束）：①官 val MAE 变化 ②walk-forward 跨折 MAE ③午间(11-14) MAE ④Bias ⑤高误差样本(Top10 日)变化 ⑥跨窗稳定性。

---

## 8. GPU 选型

**工作负载特征**：75k 行 × 126 特征 = **小数据**。单模型显存 < 1GB。瓶颈不是显存而是**训练次数**（40 成员 × 4 折 = 160 次/全管线）-> **关键策略：单 GPU 上多成员并发**（24GB 可并发 8+ 个，16GB 并发 4-6 个）。

| 角色 | GPU | 显存 | 理由 | 参考价(云) |
|---|---|---|---|---|
| **首选** | **NVIDIA RTX 4090** | 24GB | 小表数据 CatBoost 性价比最优；24GB 可 8 路并发成员；CC 8.9 全面支持；国内 AutoDL/恒源云货源充足 | ¥2.5-3.5/hr (AutoDL) / $0.4-0.7/hr (RunPod) |
| **备选 1** | **NVIDIA A100 40GB**（或 80GB） | 40/80GB | 4090 缺货或 Phase 3 大规模 HP 搜索时用；单 iter 略快于 4090；40GB 可全 HP 网格并发；数据中心卡长跑更稳 | ¥6-8/hr (AutoDL) / $1.5-2.5/hr (RunPod/Lambda) |
| **备选 2** | **NVIDIA RTX 3090 24GB** 或 **L4 24GB** | 24GB | 预算型；3090 是最便宜 24GB（8 路并发仍可）；L4 是现代数据中心等效（功耗低）。75k 行足够 | 3090: ¥1.5-2/hr；L4: $0.5-0.8/hr |
| 兜底 | T4 16GB | 16GB | 最便宜；CC 7.5 仍 CatBoost 可用；但 16GB 限并发 4-6、Turing 较慢，仅 Phase 0 调试 | ¥1/hr / $0.3/hr |

**选型建议**：
- Phase 0-2：**4090 24GB** 即可（性价比最高，8 路并发覆盖 160 次训练）。
- Phase 3 大规模 HP 搜索：若 4090 排队，升 **A100 40GB** 跑全网格并发。
- 预算紧/仅调试：**3090 24GB**（仍 8 路并发，慢 1.5×）。
- **不要**为这个负载租 H100（严重过剩）；**避免**T4 做 Phase 3（太慢）。

**云实例配置建议**：
- OS：Ubuntu 22.04 LTS（CatBoost GPU 必须 Linux）
- CUDA：11.8 或 12.x（与 catboost wheel 匹配）
- Python：3.11/3.12（3.14 未必有 catboost 预编译 wheel，**建议降到 3.11/3.12**；本地生产用 3.14，云端实验用 3.12 即可，模型产物 .cbm 跨版本兼容）
- `pip install catboost pandas numpy scikit-learn lightgbm`（保留 lightgbm 供 Phase 4 异构对照）
- 数据上传：仅 `data/` 两 CSV（~几百 MB），秒传。

---

## 9. 训练时间预估

假设：depth=8, iterations≈800（CatBoost 通常多于 LightGBM 的 80，ordered 或需更多），40 成员/集成，4 集成/全管线（3 折 walk-forward + 1 官方）= 160 模型训练/全管线，8 路并发（24GB）。

**单模型训练耗时**（75k×126, depth 8, 800 iter）：
| GPU | 单模型 | 
|---|---|
| A100 40GB | ~15-25s |
| RTX 4090 | ~20-35s |
| RTX 3090 | ~35-55s |
| T4 16GB | ~80-140s |

**全管线一次**（160 模型 + OOF 校正估计 + 全量推理）：
| GPU | 并发 | 全管线耗时 |
|---|---|---|
| A100 40GB | 8 | **~15-20 min** |
| RTX 4090 | 8 | **~20-30 min** |
| RTX 3090 | 8 | ~30-40 min |
| T4 16GB | 4 | ~90-120 min |

**各阶段总耗时预估**：
| 阶段 | 工作量 | 4090 | A100 | 3090 |
|---|---|---|---|---|
| Phase 0 | 跑通 + rsm/ordered 探针(~10 模型) | ~5 min | ~3 min | ~10 min |
| Phase 1 | 1 次全管线 | ~25 min | ~18 min | ~35 min |
| Phase 2 | 1 次全管线（分类变体） | ~25 min | ~18 min | ~35 min |
| Phase 3 | HP 搜索 24 配置(单折筛选) + top3 全管线 | ~2-2.5 hr | ~1.5-2 hr | ~4-5 hr |
| Phase 4 | 异构集成(额外 40 CB 成员，与 LGBM 并) | ~25 min | ~18 min | ~35 min |
| **全流程合计** | | **~3.5-4 hr** | **~2.5-3 hr** | **~6-7 hr** |

> 以上为数量级估算，±50%。CatBoost GPU 在 75k 行小数据上可能受 per-iter 开销主导，实际以 Phase 0 校准为准。若 ordered boosting 则 ×1.5-3。

---

## 10. 风险与回退

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| CatBoost 不优于 v6（天花板假说成立） | **高** | 实验止步，不入生产 | 接受；v6 继续生产，实验归档。这正是科学结论。 |
| rsm GPU 不可用导致正则不足、过拟合 | 中 | Phase 1 退化 | 集成层手动列子采样（§4 方案 B）；或 Lossguide+rsm |
| ordered boosting 过慢 | 中 | Phase 3 耗时膨胀 | 仅对最终 config 试 Ordered，搜索阶段用 Plain |
| 分类编码跨年过拟合（节假日样本少） | 中 | Phase 2 不迁移 | walk-forward 必查；不过则弃 Phase 2 |
| catboost wheel 与 Python 3.14 不匹配 | 中 | 本地无法跑 | 云端用 3.11/3.12；.cbm 模型文件跨版本加载 |
| quantile GPU 数值差异 | 低 | 分位成员预测偏 | 核对 q45/q55 预测分布对称性 |
| 异构集成相关性仍高（Phase 4 无增益） | 中 | Phase 4 止步 | 接受；至少 Phase 1 结论独立成立 |

**回退底线**：任何阶段未过 gate，**v6 生产与 v7 归档原样不动**，零风险。

---

## 11. 里程碑与交付物

| 里程碑 | 交付物 | 判据 |
|---|---|---|
| M0 环境就绪 | 云实例 + catboost GPU 跑通 exp_catboost_env.py | 单模型训完 + 单 MAE 输出 |
| M1 公平 A/B 结论 | exp_catboost_ab.py + 结果 log | CatBoost vs v6 的 6 指标对比表 |
| M2 分类变体结论 | exp_catboost_cat.py + log | Phase 2 6 指标 |
| M3 超参锁定 | exp_catboost_hp.py + 最优 config | 搜索轨迹 + 最优 config |
| M4 异构集成结论 | exp_hetero_lgbm_cb.py + log | Phase 4 6 指标 |
| M5 v8 决策 | 决策报告（入生产 / 止步） | gate 全过 -> v8；否则归档 |

**下一步动作**：你确认计划后，我先出 **Phase 0 的 exp_catboost_env.py**（仅打通环境 + 校准耗时 + 验证 rsm/quantile/ordered 行为），你在云端跑回结果，再决定 Phase 1。
