# v8 架构级升级方案：预测->判断->修正->融合 五层动态架构

## 一、目标与约束

**目标**：在 v6/v7（val MAE 1445.62）之上构建"预测->判断->修正->融合"多层动态架构，
修正策略按天气相似度/时段/历史表现动态决定（是否修正、修正多少、融合权重），而非全天统一固定修正。
优先跨年稳定性，而非验证集局部最优。

**硬约束（不可改变）**：
1. 六条 leakage 不变量全部保持（actual 仅 eval/target；pred_load 滞后含 lag_192；weather dedup 三阶段一致；
   时间边界 TRAIN_END=2026-02-28；shared build_features；predict 14 天窗口 skew fix）
2. 不修改现有 `load_pred/` 任何脚本；v8 全部放 `v8/` 文件夹
3. 所有动态参数（trigger 阈值、α、w、adaptive 偏好、天气相似度查询池）必须来自**训练期 OOF 或训练集统计**，
   val 窗口（2026-03-01~2026-06-15）不参与任何参数学习
4. 基础模型用目前最优版本 = v6（40 成员 LightGBM 集成 + MOS anchor + OOF 三段校正）
5. 保持工程部署方式（`python -m v8.predict --run-date ...`，9 AM 时序，输出格式一致）

## 二、v8 包结构

顶层包 `v8/`（与 `load_pred/` 平级），复用 `load_pred` 的共享模块（config/data_loader/features/train/model），
自身实现五层架构。运行：`python -m v8.train` / `python -m v8.predict --run-date ...` / `python -m v8.evaluate`。

```
v8/
  __init__.py
  config.py          # v8 配置：段定义、KNN K、softmax τ、α/w 网格、trigger 阈值、base B 配置
  weather_sim.py     # 统一天气相似度（日级向量、标准化、KNN、softmax 权重）
  segments.py        # 3 时段定义与分段 mask（00-08 / 08-18 / 18-24）
  base.py            # 第二层：base A(v6) 加载 + base B(reg_only) 训练 + adaptive selection
  correction.py      # 第三/四层：3 段残差 LightGBM + Correction Trigger + 动态 Shrink α
  fusion.py          # 第五层：Dynamic Fusion w
  oof.py             # OOF 3 折引擎：fold base + fold correction，收集 OOF 池
  model.py           # V8Model：持有所有组件 + 天气相似度池 + 动态参数；predict + save/load
  train.py           # v8 训练入口（OOF 估计 + 全量 correction 训练 + 保存）
  predict.py         # v8 预测入口（D+1 五层串联，复用 14 天窗口 skew fix）
  evaluate.py        # v8 验证（vs v7：全天/午间/跨年/大偏差）
  models/            # v8 自有模型目录（v8_bundle.pkl + correction boosters）
  output/            # v8 自有输出目录
```

**依赖关系**：`v8/*` 通过 `from load_pred import config as C, data_loader as dl, features as F, train as T, model as M`
复用 v6 的 `build_features` / `train_ensemble` / `compute_hour_bias` / `EnsembleModel` / `load_weather_dedup`。
不修改 `load_pred/` 任何文件。base A 直接加载根 `models/model_bundle.pkl`（已训练 v6）。

## 三、统一天气相似度（weather_sim.py）-- 服务于四层

**设计**：日级天气向量（跨年稳定的物理量），避免时点级噪声。
- 向量维度（5 维）：[temp_day_mean, irrad_day_sum, clearness_day_mean, precip_day_sum, temp_day_range]
  （均来自 build_features 已有列，无新特征工程）
- 标准化：用**训练期**均值/标准差 z-score（统计量存 bundle，预测时复用）
- 距离：标准化欧氏距离
- KNN：对预测日 D+1，在训练期 OOF 池的**唯一日**中找 K=40 个最近邻日
- 权重：softmax(distance, temperature=τ)，τ 从训练期 OOF 估计（选使 OOF 局部 MAE 最优的 τ，grid {0.5,1,2,4}）
- 输出：邻居日索引 + 相似度权重（和为 1）

**关键合规点**：查询池 = 训练期 OOF 日（≤ TRAIN_END），不含 val。物理量跨年稳定 -> 跨年泛化。

**统一服务**：adaptive selection / trigger / α / w 四层均调用 `WeatherSim.query(d1_weather_vec)` 得同一组邻居+权重，
避免各模块各自算距离。

## 四、五层架构详细设计

### 第一层：分段建模（segments.py）
3 段（按 hour of day）：
- night:  00:00~08:00（slots 0-31）
- day:    08:00~18:00（slots 32-71，含午间 11-14 高误差区）
- evening: 18:00~24:00（slots 72-95）
每段独立的 correction model + trigger/α/w 参数。base 全天共享（v6），adaptive selection 可分段。

### 第二层：基础预测层（base.py）+ Adaptive Model Selection
- **Base A = v6**：加载根 `models/model_bundle.pkl`（40 成员 + MOS + hour_bias/drift/threshold）。
  生产部署与 OOF 评估均用 A 作为主力。`base_A_predict(X, pred_load) = EnsembleModel.predict_load`。
- **Base B = reg_only LightGBM**（新构建，diversity 备选）：
  `objectives=["regression"]`（去 quantile 成员）× {direct, residual} × 5 seeds = 10 成员，
  其余超参同 v6（nl=255, l2=4, best_it=80, MOS anchor, median）。结构差异明确（无 quantile 成员），
  CatBoost 实验启示 reg_only 在某些场景折 CV 更稳。训练用 `T.train_ensemble`（传 reg_only cfg override）。
- **Adaptive Selection**：在 OOF 池上按天气型分桶（clearness×temp 9 宫格 + precip>0 雨型），
  每桶比较 base_A_OOF_MAE vs base_B_OOF_MAE。若 B 在某桶 OOF MAE 优 A 超过 min_margin（如 1%）且桶样本数≥N_min，
  标记该桶"偏好 B"。部署时：预测日天气型 -> 查偏好表 -> 选 base。默认 A（保守，避免频繁切换）。
  天气型用日级 + OOF 历史统计（非单日误差），天然避免偶然切换。

### 第三层：修正决策层（correction.py - Trigger）
对预测样本 T（属某段），用统一天气相似度找 K 个邻居 OOF 样本（同段）：
- base_err = |base_pred - actual|，corr_err = |base_pred + α·residual_pred - actual|（在邻居 OOF 上）
- **Trigger on** 条件：邻居中 corr_err < base_err 的比例 ≥ trig_frac（如 0.6）且加权平均改善 ≥ min_gain（如 30MW）
- trig_frac / min_gain 从训练期 OOF 估计（grid search 使 OOF 全局 MAE 最优，val 不参与）
- Trigger off -> 直接输出 base，w=0，不修正（跳过 78.5% 不可学样本，防过度修正）

### 第四层：修正层（correction.py - Shrink α + 残差模型）
- **残差模型**：3 段各一个 LightGBM（direct 模式，target = actual - base_pred_OOF）。
  训练数据 = OOF 池（3 折 fva 并集，~24000 点，覆盖春夏冬三季，无泄露）。
  输入 = build_features 的 X（不含 base_pred，保持简单；base_pred 的信息已隐含在 X 的 pred_load/weather 中）。
- **动态 α**（不固定）：对预测样本 T，在邻居 OOF 样本上 grid search α ∈ {0,0.25,0.5,0.75,1.0}，
  选使邻居加权 MAE 最小的 α，再做相似度加权平滑。
  α 语义 = "修正方向可信度"（邻居中 residual_pred 与 actual-base 同号比例高 -> α 大）。
  天气越相似/历史越稳定/修正越可信 -> α 越接近 1；否则减小。防跨年残差不迁移导致过度修正。

### 第五层：动态融合层（fusion.py - w）
- **Final = (1-w)·base + w·(base + α·residual) = base + w·α·residual**
- **动态 w**（不固定）：w = 邻居 OOF 中"修正有效"的加权比例 × 修正幅度可信度。
  具体：w = clip(Σ_i weight_i · max(0, 1 - corr_err_i/base_err_i), 0, 1)。
  w 语义 = "修正幅度可信度"（邻居中修正后改善越多 -> w 越大；恶化则 w->0）。
- Trigger off 时 w=0；Trigger on 时 w>0。w 与 α 依据不同（幅度 vs 方向），非冗余。

## 五、OOF 估计流程（oof.py -- 核心，所有动态参数的来源）

3 折 walk-forward（复用 v6 `best_it_folds`：春/秋/冬 2025，全在训练期内）。每折：
1. ftr 上训练 fold base A（v6 配置，40 成员）+ fold base B（reg_only，10 成员）
2. ftr 上估计 fold 校正（hour_bias/drift/threshold，调 `T.compute_hour_bias` 但仅用 ftr 数据）
3. fva 上预测：base_A_OOF, base_B_OOF（含校正的完整 v6-style 预测）
4. 收集 fva 的 (base_A_OOF, base_B_OOF, actual, X, 段, 时间, 天气日向量) 入 OOF 池

**OOF 池** = 3 折 fva 并集（~24000 点，无重叠，无泄露，全在训练期）。
- 在 OOF 池上训练 3 段 correction model（target = actual - base_A_OOF）
- 在 OOF 池上估计：天气相似度查询池（按日去重）、adaptive 偏好表、trigger 阈值、α/w 的 KNN 映射
- correction model 自身的预测也需 OOF（避免 correction 过拟合 OOF 池）：用嵌套 2 折在 OOF 池内产生 correction_OOF，
  trigger/α/w 估计用 correction_OOF（而非 in-sample correction pred），保证 trigger/α/w 估计无泄露

**全量生产模型**（部署用）：
- base A = 加载根 model_bundle.pkl（v6）
- base B = 全训练集训练的 reg_only
- correction model = 全 OOF 池训练的 3 段 LightGBM
- 所有动态参数 = OOF 池估计的统计量
- 保存到 `v8/models/v8_bundle.pkl` + `v8/models/correction_boosters/`

## 六、训练流程（train.py）

```
[1] build_dataset（复用 T.build_dataset）+ usable_mask
[2] MismatchModel.fit + transform（复用 F.MismatchModel）
[3] MosModel.fit（复用 F.MosModel，base A/B 共享同一 MOS? 或各自 fit）
[4] OOF 3 折引擎：
    每折: train fold base A + fold base B + fold 校正 -> 预测 fva -> 入 OOF 池
[5] 在 OOF 池上:
    - 训练 3 段 correction model（嵌套 2 折产 correction_OOF）
    - 估计天气相似度标准化统计量 + τ
    - 估计 adaptive 偏好表（天气型分桶 A vs B）
    - 估计 trigger 阈值（trig_frac, min_gain）
    - 估计 α/w 的 KNN 映射
[6] 全量训练生产 correction model（全 OOF 池）+ base B（全训练集）
[7] base A 加载根 model_bundle.pkl
[8] 组装 V8Model，save 到 v8/models/v8_bundle.pkl
[9] 全量预测 + val 评估（仅报告，不参与参数）
```

## 七、预测流程（predict.py）

```
[1] run_date=D, run_dt=D 09:00
[2] pred_load = dl.pred_load_series(); weather = dl.load_weather_dedup(run_time=run_dt)
[3] 14 天回看窗口 build_features（复用 F.build_features，skew fix）+ MismatchModel.transform
[4] 提取 D+1 96 点
[5] V8Model.load(v8/models/v8_bundle.pkl)
[6] 五层串联（逐点，向量化）:
    a. 分段: 每点归 night/day/evening
    b. base: adaptive selection 选 A/B -> base_pred（v6 predict_load）
    c. 天气相似度: D+1 日向量 -> KNN 邻居 OOF 日 + 权重
    d. Trigger: 邻居 OOF 判定是否修正
    e. α: 邻居 OOF grid search -> α_T
    f. residual_hat = correction_model[seg].predict(X_T)
    g. w: 邻居 OOF 修正可信度 -> w_T
    h. final_T = base_pred_T + w_T · α_T · residual_hat_T  (Trigger off 则 = base_pred_T)
[7] clip(0,None), round, 输出 96 点（格式同 v7）
```

## 八、合规论证

| 不变量 | v8 保持方式 |
|---|---|
| #1 actual 仅 eval/target | correction target = actual - base_pred（actual 仅作目标）；features 不碰 actual |
| #2 pred_load 滞后含 lag_192 | 复用 F.build_features，PRED_LAGS 不变 |
| #3 weather dedup 三阶段一致 | 复用 dl.load_weather_dedup，run_time 过滤逻辑不变 |
| #4 时间边界 | OOF 池 ≤ TRAIN_END；val 仅评估；3 折在训练期内 |
| #5 shared build_features | v8 train/predict 均调 F.build_features |
| #6 predict skew fix | v8 predict 复用 14 天窗口构建 |

**动态参数来源**（全部训练期 OOF，val 不参与）：
- 天气相似度标准化/τ/查询池 <- OOF 池
- adaptive 偏好表 <- OOF 池按天气型分桶
- trigger 阈值 <- OOF 池 grid search
- α/w <- OOF 池 KNN 局部估计（用 correction_OOF，嵌套无泄露）
- correction model <- OOF 池训练

**跨年泛化**：天气相似度基于物理量（temp/irrad/clearness）跨年稳定；trigger 跳过不可学样本；
α/w 局部估计不依赖全局 val 调参；adaptive 用日级天气型+OOF 历史统计非单日误差。

## 九、验证方案（evaluate.py）

对比 v7（根 models/model_bundle.pkl）与 v8（v8/models/v8_bundle.pkl），同一 val 窗口：
- ① 全天 val MAE：v8 vs v7（主指标）
- ② 午间（08:00~18:00，含 11-14）MAE：v8 vs v7
- ③ 跨年泛化：用 2025 春/秋/冬 OOF 折的修正有效性外推到 2026 val 的一致性（trigger 命中率、α/w 迁移度）
- ④ 大偏差降低：val 上 |err|>3000MW 的点数与均值，v8 vs v7（修正过度是否减少）
- ⑤ 工程一致性：v8 predict 输出格式/时序/文件名与 v7 一致；同一 run_date 输出可比对
- 报告写 v8/output/v8_evaluation.md

## 十、成本与风险

**训练成本**（本地 RTX3060 6GB + CPU LightGBM）：
- OOF 3 折 × (40 base A + 10 base B) = 150 模型 ≈ v6 compute_hour_bias(120) 的 1.25 倍
- 3 段 correction model × 嵌套 2 折 = 6 小 LightGBM（fast）
- 全量 base B（10）+ correction（3）
- 预估 12-18 min（v6 全量约 6-8 min）

**风险与缓解**：
- 风险1：v6 残差 78.5% 不可学（memory），correction 收益有限
  -> 缓解：trigger 跳过不可学子集，α/w 动态保守，最坏退化为 v6（不伤害）
- 风险2：跨年残差符号翻转（memory：clear_noon 冬折 vs val 翻转）
  -> 缓解：天气相似度 KNN 只用同型历史，α 局部估计，不全局应用
- 风险3：adaptive selection 频繁切换
  -> 缓解：天气型日级 + OOF 统计 + min_margin，默认 A
- 风险4：OOF 池 correction 过拟合
  -> 缓解：嵌套 2 折产 correction_OOF 估 trigger/α/w

**诚实预期**：鉴于 memory 一致结论 v6 近信息天花板，v8 大概率小幅改善（-2~-8MW）或持平，
主要价值在"修正过度降低 + 跨年稳定性提升"而非大幅降 MAE。这符合用户"优先跨年稳定性"目标。

## 十一、实现顺序

1. v8/config.py + segments.py + weather_sim.py（基础设施）
2. v8/oof.py（OOF 引擎，最核心）
3. v8/base.py + correction.py + fusion.py（五层）
4. v8/model.py（V8Model 组装 + save/load）
5. v8/train.py（训练入口，跑通 OOF + 保存）
6. v8/predict.py（预测入口，跑通 D+1）
7. v8/evaluate.py（vs v7 验证）
8. 跑 train -> evaluate，据结果迭代（若 trigger 全 off 则 v8=v6 持平，检查 α/w 是否过保守）
