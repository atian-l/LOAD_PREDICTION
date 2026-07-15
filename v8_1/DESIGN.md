# v8.1 DESIGN — 四层架构（Information / Diagnosis / Prediction / Correction）

> 状态：**分析阶段定稿，未写实现代码。** 本文档是 v8.1 的权威设计，覆盖 Phase 0 报告
> ([output/phase0_residual_attribution.md](output/phase0_residual_attribution.md)) 中过绝对的
> "no-go" 判定措辞——Phase 0 的**数字是事实**，**判定语言已在此校准为条件结论**。
>
> 版本号 v8.1 = v8 的架构修正（非 v9）。
> 上游依据：Phase 0 残差溯源 + 用户对 Phase 0 报告的 8 点修正。
> 前序链：v6(1445.62) → v7(1445.62, skew fix) → v8(1445.62, 5 层 parity) → **v8.1(本设计)**。

---

## 1. Phase 0 的决定性结论：残差 = Feature Noise（不是 Transfer Noise）

本项目最重要的一次信息增量。以前一直认为"2025→2026 残差不能迁移"，自然怀疑"是不是 2025 能学、只是 2026 漂移了"。Phase 0 证明：

- **combined 期内 R²（2025 自身 holdout）= −0.136**——残差对当前特征**期内即不可学**，不是迁移问题。
- combined 跨年 transfer R² = −0.432；全 5 分量组（load_level / load_temporal / calendar / weather / solar_renewable）transfer R² 全负。
- 方向命中率全 < 0.5（0.44–0.49，**跨年反相关**，不是随机）。
- 所有特征探针使 val MAE 恶化（探针 MAE 1664–1815 > 基线 1491）。

**含义**：残差 ≈ Noise（相对当前特征集）。不是"Transfer Noise"（迁移问题），是"Feature Noise"（特征对残差无可解释方差）。Decision Layer 无法修正一个特征解释不了的量——这是 v8 三次加层全 inert 的根因，也是信息上限第 5 次确认（最强证据：期内即负，非仅跨年负）。

| 段 | 2025 均值 | 2026 均值 | 2026 MAE |
|---|---|---|---|
| night | −271 | +136 | 749 |
| day | −253 | −12 | **2386** |
| evening | −269 | −91 | 807 |

午间(day)是残差最重段，但 day 段 combined transfer R² = −0.544（全段最负）——最需要修的段恰好最不可学。

---

## 2. 科学谨慎表述（必须保留，不可升级为绝对结论）

- 证据支持：**"基于当前特征集，残差几乎不可学习。"**
- **非**结论：**"残差本质不可学习。"** 后者须在引入新能源出力 / Foundation 预训练先验后**仍**得到相同结论才能成立。
- 1445.62 MW = **"当前任务 + 当前特征"** 上限，非绝对上限。

---

## 3. 四层架构（v8.1 核心）

```
┌─────────────────────────────────────────────────────────┐
│ Information Layer   天气 / 新能源 / 负荷 / 节假日 /        │
│                     Foundation 预训练先验                 │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Diagnosis Layer     今天输入是否异常? OOD? 天气漂移?       │
│                     新能源异常? 预测可信?                  │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Prediction Layer    v6（40-member LightGBM ensemble）    │
└──────────────────────────┬──────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ Correction Layer    仅当 Diagnosis 判"可信"才修；          │
│                     否则直接输出 v6                       │
└─────────────────────────────────────────────────────────┘
```

### 核心思维转变

**Diagnosis 预测的是 Information 是否可靠，不是预测 Residual。**

这是与 v8 的根本分野：

| | v8（Decision Layer） | v8.1（Diagnosis Layer） |
|---|---|---|
| 预测对象 | 残差可信度（残差方向/幅度） | 输入与信息的可靠性（OOD/漂移/异常） |
| 前提 | 残差可学 | 输入分布可判 |
| Phase 0 结论 | 残差期内不可学 → inert | **未测，开放** |

v8 的 Decision Layer 之所以 inert，正因为它建立在"残差可学"前提上，而 Phase 0 证伪了该前提。v8.1 把判断对象从**残差**换成**输入/信息**，前提变成"输入分布可判"——这是 Phase 0 没测的独立问题，故为开放窄口。

工业 Forecast QA 即此模式：**先 Input Diagnosis，再 Correction**。

---

## 4. 模块决策（keep / repurpose / stop）

| 模块 | 决策 | 理由 |
|---|---|---|
| Segment（时段分段） | **keep** | 物理意义清晰，午间/夜间误差结构不同 |
| Adaptive Base A/B | **stop** | A/B 同质（corr 0.986），v8 已证 0 桶偏好 B，inert |
| Trigger / α / w | **合并为单一 Confidence c** | 三锁同落重复建模同一（已证不存在的）残差可信度信号；工业惯例单 c |
| Weather Sim | **repurpose → Domain/OOD 诊断** | 旧链"天气相似→残差相似"第二段跨年断裂；新用途"今天(temp,irrad) 是否 OOD"是不同问题（见 §7） |
| Correction 任务 | **多阶段阶梯（条件）** | 无新数据下 no-go 倾向（dir_acc<0.5）；加新能源后可能复活。不删，挂起 |
| Correction 形态 | **低维 Shape** | 作安全输出形态（不伤害），不作信号源（R²≈0） |
| Information Layer | **新增** | 新能源出力 / Foundation 预训练先验——真 Information Layer |
| Diagnosis Layer | **新增** | 输入异常检测 + 气象 OOD——本设计的中心增量 |

---

## 5. 措辞校准（8 点修正，收回过绝对表述）

Phase 0 报告的 F 节与方向建议节用了 "no-go" 绝对措辞，现校准：

1. **"no-go" → "当前特征下不可恢复"**（条件结论，非永久）。Task Hierarchy 若未来加新能源，可能立即有价值。
2. **Shape 不是"无价值"**。Phase 0 证明的是"当前特征 → Shape 不可预测"，不是"Shape 不存在"。真实链可能是 Cloud → PV → Midday Shape，缺 Cloud 故 R²≈0。
3. **Domain 不是"删除"**。改为"Weather Domain cannot explain Residual"。Domain 以后可做 OOD Detection / Confidence / Distribution Shift。
4. **Stage1 重定义为 pred_load 输入异常检测**。Residual 已证无信号，但 Input 有无信号没人测过。pred_load 突然比过去低 9000MW 是 Input 不是 Residual，Input 可能跨年稳定。

---

## 6. 剩余投入优先级（用户定）

| 优先级 | 方向 | 投入 |
|---|---|---|
| ⭐⭐⭐⭐⭐ | 新能源出力等新信息源 | **最高优先**（但卡数据可得性） |
| ⭐⭐⭐⭐☆ | Pred_load 输入异常检测（Input QA） | 值得做（本地、快） |
| ⭐⭐⭐⭐☆ | Foundation Models（Chronos / Moirai / TimesFM / Lag-Llama） | 值得做（zero-shot 可测） |
| ⭐⭐⭐☆☆ | Shape-level 修正（作输出约束，非信息源） | 可探索，不期待突破 |
| ⭐⭐☆☆☆ | 新 Decision Layer（Trigger / Gate / α / w / Domain） | **停止投入** |

**总判断**：真正的瓶颈已不再是模型，也不是修正策略，而是**信息**。后续工作围绕**增加信息**（新数据、新先验）或**提升信息质量**（输入异常检测）展开，不再增加决策层复杂度。

---

## 7. 气象相似度（用户提示，现有数据立即可用）

`config.WEATHER_BASE_VARS` 含 `光伏_温度` + `光伏_辐照度`（CSV 表头确认，另带 `_p25/_p50/_p75/_std` 集合预报分位列）= 用户所说"参考县区温度 + 辐照度"。

**分段建模 / 判断是否需修正时，用这两量加权算气象相似度。**

### 关键洞察：同一变量，两种用途，Phase 0 只否定了其中一种

| 用途 | 层 | 问题 | Phase 0 结论 |
|---|---|---|---|
| 残差预测器 | Prediction | temp/irrad 能否预测残差**值**？ | R² 负，**已证无效** |
| OOD / 天气漂移检测器 | Diagnosis | 今天 (temp,irrad) 是否相对训练分布**异常**？ | **未测，开放** |

**Prediction 失败不否定 Diagnosis 用途。** 这正是 §3 核心转变的具体体现：把 temp/irrad 从"预测残差"挪到"判断输入是否 OOD"，问的是输入新颖度而非残差可预测性。这是当前唯一一个**本地、可立即验证、Phase 0 未覆盖**的开放窄口，直接挂在 Diagnosis Layer 的 gating 上（OOD → 降低 Confidence c → 少修或不修）。

---

## 8. 待验证探针（下一步，不写生产代码）

每个探针用 **walk-forward**（非 val）判 go/no-go——val 是 deployment-realistic 基线，但跨年迁移性须用 walk-forward 折验证。

| 探针 | 内容 | 门槛 |
|---|---|---|
| **P-Diag** | Diagnosis 层原型：①pred_load 输入异常检测（相对历史/气象隐含水平偏离，跨年方向命中率）②气象 OOD（光伏_温度+辐照度加权，vs 训练分布） | dir_acc > 0.55 或 OOD 集合的 val MAE 显著低于全局 |
| **P-Renewable** | 新能源数据可得性 + 可行性（数据里目前无云量/分区/新能源出力，须先确认能否接入） | 数据可得即 go |
| **P-Foundation** | Chronos/Moirai 对 val zero-shot，看预训练先验能否把残差 R² 从负拉正 | transfer R² > 0 |

P-Diag 是 P-Diag/P-Renewable/P-Foundation 中唯一纯本地、不依赖外部数据的，建议先做。

---

## 9. 泄露不变量（必须守，6 条 + 1 工程项）

v8.1 继承全部 6 条数据泄露不变量，任何新层不得违反：

1. **Actual load is eval-only.** `实际直调负荷` 仅在 train.py 作 target / 残差 target / 评估基准。features.py / Diagnosis / Information 层均不得触碰。
2. **Lags come from external forecast only.** `PRED_LAGS=[96,192,288,672]` 来自 `预测直调负荷`，`lag_192` 必须在。
3. **Weather dedup same in 3 phases.** `load_weather_dedup` per-forecast-time 取最新 issue；predict 模式 `起报时间 <= run_time`（默认 9:00），train 模式 `run_time=None`。
4. **Time boundary.** 训练 ≤ `TRAIN_END = 2026-01-31 23:45:00`；val 窗口 `2026-02-01 ~ 2026-05-19 11:45:00` eval-only，不训练不早停。
5. **Train/predict share one feature builder.** `features.build_features()` 单入口，防 train/serve skew。
6. **Predict builds on 14-day history window.** 不在 96 点上单独建特征（skew fix，不可回退）。
7. **（工程项）autocrlf CRLF bug.** git `core.autocrlf=true` 污染 boosters（v8 发现，影响 `load_pred.predict`）。v8.1 须沿用 `_io.load_booster` 绕过，或确保 autocrlf 关闭。

---

## 10. 与历史实验的关系

| 实验 | 结论 | 对 v8.1 的意义 |
|---|---|---|
| v6 / v7 / v8 | 1445.62 parity，3 次加层全 inert | 信息上限第 1–4 次确认；加 Decision 层无效 |
| CatBoost | 异质融合 err corr 0.984，OOF 选权全 0 | 换模型族不增信息 |
| TCN | members r0.99，pruning 仅 −21MW | 换模型族不增信息 |
| FDS / midday / info-source | 残差 83.7% unlearnable，跨年 R² 负 | 残差不可学的前序证据 |
| **Phase 0** | 期内 R²=−0.136 | **第 5 次，最强：期内即负** |

v8.1 立场：**不再加 Decision 层**（上限已 5 次确认）；转向 **Information + Diagnosis** 两个未证伪方向。

---

## 11. 成功 / 失败判据

- 任何新方向须 **walk-forward 证跨年可迁移**才采纳（val 改善但不迁移 = 过拟合，拒）。
- **极简版**（v6 base + 单一 c + 低维 Shape + 仅输入异常触发）= v6 parity，换可维护性不换 MAE——这是无新数据下的兜底。
- **破 1445.62** 的概率集中在新能源出力 / Foundation 预训练先验（均不在当前特征内），不在架构技巧。
- 若 P-Diag / P-Renewable / P-Foundation 全证伪，**才**谈硬上限——不预设停止线。
