# CatBoost 欠拟合解决工程计划

> 状态：Phase 0 已执行 -> **NO-GO 铁证**，Phase 1-3 全取消
> 结论：CatBoost 非欠拟合能力不足（best_it 80->500 train_raw 1517->1016 < LGB 1147，能拟合甚至更好），但拟合改善不迁移到 val（val_raw 1533->1566 反升）-> 瓶颈是泛化+残差规律性，非欠拟合 -> 维持 CatBoost 路径终止
> 执行：exp_catboost_phase0.py @4090 466s，7 配置 raw trend
> 前置：拟合诊断（exp_catboost_fit_diag.py）已确认 CatBoost l2_8 欠拟合
> 关联：catboost_migration_plan.md / catboost-migration-progress 记忆

## 1. 问题陈述

拟合诊断结论：
- CatBoost l2_8：train_raw=1517 ≈ val_raw=1533（gap 16，**欠拟合**），train R²=0.9635
- LightGBM v6：train_raw=1147 << val_raw=1512（gap 365，**过拟合**），train R²=0.9762
- 两者 val_raw 几乎相同（1533 vs 1512，差 21）→ ~1515 是 raw val 数据天花板
- v6 的 1445 优势来自 OOF 校正：LGB 过拟合残差规律 → 校正降 66；CB 欠拟合残差随机 → 校正降 55

**目标**：探索能否解决 CatBoost 欠拟合，使其 val MAE 追近/追平 v6 1445.62。

## 2. 欠拟合根因分析

CatBoost best_it=80 train_raw=1517，LightGBM best_it=80 train_raw=1147，同样 80 轮差 370MW。可能根因（按可能性）：

| 根因 | 依据 | 可试方向 |
|---|---|---|
| **对称树表达力弱** | 对称树每层同 split，拟合难区域效率低；LGB leaf-wise 选最大增益叶 | Lossguide（已试，val 更差，待诊断 train_raw） |
| **best_it 两难** | best_it=80 欠拟合；增大→过拟合+bias 漂移（best_it=216 bias=-152） | recency 校正控漂移 + 增大 best_it |
| **Bayesian bootstrap** | Bayesian 给样本连续随机权重，可能欠拟合；LGB 用 Bernoulli 0/1 | bootstrap_type=No/MVS |
| **强正则** | l2=8 + min_data=200；但 l2=2 未改善（3-B1） | 已部分排除 |
| **内部正则机制** | CatBoost 对称树+ordered 倾向保守 | 算法本质，难调 |

**核心矛盾**：CatBoost best_it 小→欠拟合，大→过拟合漂移，无好中间点。LightGBM best_it=80 过拟合但校正有效（残差规律）；CatBoost 过拟合残差漂移（不可校正）。

## 3. 核心假设与关键障碍

**假设链**（解决欠拟合能降 val 的唯一路径）：
```
拟合改善(train_raw↓) → 残差变规律 → OOF 校正收益↑(55→66+) → val↓
```

**关键障碍**（来自已有实验，指向"不迁移"）：
1. **3-A**：val debiased 几乎不随 best_it 变（80→216 都 ~1483）。增大容量不降 val 纯误差。
2. **3-B1**：Lossguide val=1489 比 SymmetricTree 1481 更差。leaf-wise 没帮上忙。
3. **3-B1**：d10 debiased=1462 最低，但仍 > v6 1445，且 bias=-155 漂移严重。
4. **raw val 天花板 ~1515**：CatBoost/LGB raw val 几乎相同，78.5% 不可学误差是数据天花板。

**推论**：若"拟合改善不迁移到 val"成立，则欠拟合不可解--因为即使 train_raw 降到 1147，val debiased 仍 1483，校正后仍 ~1481，不改善。

**但有一个未验证的破绽**：3-A 只测了 SymmetricTree 不同 best_it 的 debiased。**不同结构（Lossguide/不同 bootstrap）的 train_raw vs val_raw 趋势未测**。若某结构能让拟合改善迁移到 val，则假设链成立。这是 Phase 0 要验证的。

## 4. Phase 0：关键诊断（go/no-go，~5-8 min）

**目的**：用最小成本验证"拟合改善能否迁移到 val"。这是整个计划的关卡--若不迁移，直接停止。

**做法**（不需 OOF 校正估计，只测 raw，所以快）：
- best_it ∈ {80, 160, 300, 500}（SymmetricTree, l2_8），测每个的 train_raw / val_raw / debiased
- Lossguide（max_leaves=255, best_it=80/300），测 train_raw / val_raw
- bootstrap_type=No（best_it=80），测 train_raw / val_raw

**判定矩阵**：

| 观测 | 结论 | 下一步 |
|---|---|---|
| train_raw 随 best_it 降，但 val_raw/debiased 不降 | **拟合不迁移** | 停止（欠拟合不可解，印证数据天花板） |
| train_raw 降且 val_raw 也降 | 拟合迁移 | Phase 1 优化 |
| Lossguide train_raw 显著低（<1300） | leaf-wise 解决欠拟合 | Phase 1-B Lossguide + 控过拟合 |
| Lossguide train_raw 仍高（>1400） | leaf-wise 也拟合不好 | 对称树路径，Phase 1-A |
| bootstrap=No train_raw 显著降 | bootstrap 是根因 | Phase 1-C |

**最可能结果**（基于 3-A/3-B1 证据）：拟合不迁移 → 停止。但 cheap，必须验证。

## 5. Phase 1：提升拟合能力（条件性，~15-20 min）

仅当 Phase 0 显示拟合可迁移或某结构 train_raw 显著降时执行。

**1-A：增大容量 + 控制漂移（对称树路径）**
- best_it 增大（160/300）+ **recency-weighted OOF 校正**控制 bias 漂移
- depth=10/12 + 强 l2（16/32）控制方差
- 目标：train_raw 降到 ~1300，val_raw 通过校正降到 ~1460

**1-B：Lossguide + 强正则（若 Lossguide 欠拟合解决）**
- Lossguide + max_leaves=255 + l2=16/32 + min_data=200/400
- 控制 leaf-wise 的过拟合（3-B1 Lossguide val 更差，可能过拟合）
- 目标：拟合好 + 不过拟合

**1-C：bootstrap 探索（若 bootstrap=No 有效）**
- bootstrap_type=No（全样本）/ MVS（基于方差采样）
- 对比 Bayesian，看拟合改善
- 目标：train_raw 降，val_raw 跟降

## 6. Phase 2：增强校正（复刻 v6 机制，~10 min）

无论 Phase 1 哪条路径，若 train_raw 降了，需增强校正来转化拟合收益。

- **recency-weighted hour_bias**：OOF 估计时给近期折（冬季 2026）更大权重，让 96-slot bias 校正更贴近 val，改善 bias 迁移（解决 3-A 的 bias 漂移）
- **per-slot recency 校正**：每个 slot 的 bias 用近期折加权估计
- **滚动/expanding bias 校正**：用时间衰减加权
- 目标：校正收益从 55 提升到 66+（v6 水平），val 降 ~11MW

**注意**：校正不能低于 debiased（oracle 上限）。所以 Phase 2 的天花板是 debiased。若 Phase 0 显示 debiased 不降，Phase 2 最多到 1483，仍 > v6。

## 7. Phase 3：集成多样性（~10 min）

若单模型仍有差距，试集成。

- **多 best_it 集成**：欠拟合（best_it=80）+ 过拟合（best_it=300）模型混合，median 聚合
- **多结构集成**：SymmetricTree + Lossguide 混合
- **多 bootstrap 集成**：Bayesian + No 混合
- 目标：多样性降方差，可能降 debiased

**风险**：corr 可能高（同特征集），增益有限（tech-eval 异质集成 corr 0.986 已评估放弃）。

## 8. 可行性评估（诚实）

**不利证据（强）**：
- 3-A debiased 不随 best_it 变 → 拟合改善大概率不迁移
- 3-B1 Lossguide val 更差 → leaf-wise 没解决
- 3-B1 d10 debiased 1462 最低仍 > v6 → 即使最优结构也追不平
- raw val 天花板 1515 + 78.5% 不可学 → 数据天花板

**有利证据（弱）**：
- 欠拟合有 370MW 理论拟合空间（train_raw 1517 vs LGB 1147）
- bootstrap_type 未试（No/MVS 可能改善拟合）
- recency 校正未试（可能改善 bias 迁移）
- v6 机制（过拟合+校正）理论上可复刻

**总体判断**：**希望不大（~15-20%），但非零**。最乐观路径（Lossguide/bootstrap 改善拟合 + recency 校正）可能到 ~1460，仍难追平 1445。最可能结果是 Phase 0 确认拟合不迁移 → 停止。

**但 Phase 0 cheap（5-8min），值得验证关键假设**，因为：
1. 若确认不迁移，彻底坐实"数据天花板"结论，关闭 CatBoost 路径
2. 若意外发现某结构拟合迁移，则打开新可能
3. bootstrap/recency 是未试杠杆，cheap 验证有信息价值

## 9. 决策 gate

| 阶段 | gate | 通过 → | 不通过 → |
|---|---|---|---|
| Phase 0 | 拟合改善迁移到 val（某结构 val_raw 随 train_raw 降） | Phase 1 | **停止**（欠拟合不可解） |
| Phase 1 | 某配置 val_full < 1470 且 train_raw < 1300 | Phase 2 | 停止 |
| Phase 2 | 校正后 val < 1460 且折间 CV < 0.06 | Phase 3 | 停止 |
| Phase 3 | val < 1450 且折稳定 | 进生产评估（异质集成或替换） | 停止，作异质集成成员 |

**硬停条件**：任何阶段若 val_full ≥ 1480（不优于现有 l2_8 1477.67）→ 停止该方向。

## 10. 与既定结论的关系

本计划是对"CatBoost 路径终止"结论的**复核性延伸**，不推翻它。即使 Phase 0-3 全部失败，也只是用更充分的证据巩固"数据天花板"结论。若意外成功，则修订 CatBoost 路径结论。

**预期最终结论（~80% 概率）**：Phase 0 确认拟合不迁移 → 欠拟合不可解 → CatBoost 路径终止结论维持，回 v6 + 等新数据。

**不投入生产**：任何阶段都不修改生产代码；全部 exp_*.py 实验，gate 通过才考虑进生产评估。

## 11. 资源预算

| 阶段 | 4090 耗时 | 累计 |
|---|---|---|
| Phase 0 | ~5-8 min | 8 min |
| Phase 1 | ~15-20 min | 28 min |
| Phase 2 | ~10 min | 38 min |
| Phase 3 | ~10 min | 48 min |

总预算 ~50 min 4090（最坏全跑）。Phase 0 后大概率提前终止。

## 12. Phase 0 执行结果（2026-07-13 @4090, 466s）

| tag | best_it | train_raw | val_raw | debiased | gap | R²_tr | policy |
|---|---|---|---|---|---|---|---|
| sym_bi80 | 80 | 1517.3 | 1532.6 | 1524.2 | -15.3 | 0.9635 | SymmetricTree |
| sym_bi160 | 160 | 1329.6 | 1562.7 | 1528.0 | -233.1 | 0.9704 | SymmetricTree |
| sym_bi300 | 300 | 1167.7 | 1564.3 | 1515.1 | -396.6 | 0.9763 | SymmetricTree |
| sym_bi500 | 500 | 1016.5 | 1566.3 | 1509.8 | -549.9 | 0.9810 | SymmetricTree |
| loss_bi80 | 80 | 1551.2 | 1574.2 | 1574.2 | -23.1 | 0.9633 | Lossguide |
| loss_bi300 | 300 | 1328.2 | 1586.7 | 1573.1 | -258.5 | 0.9697 | Lossguide |
| no_bs80 | 80 | 1519.2 | 1535.2 | 1525.6 | -15.9 | 0.9634 | SymmetricTree |

**判定**：
- 对称树拟合迁移：**拟合不迁移**（train_raw 降 501 但 val_raw 反升 34）
- Lossguide 拟合：**leaf-wise 无改善**（train_raw 1551 > 1517，更差）
- bootstrap 根因：**bootstrap 非根因**（Δ+2 无差异）
- **GO/NO-GO = NO-GO**

**关键反转（修正 fit_diag 诊断）**：CatBoost 非欠拟合能力不足。bi300 train_raw=1167 追平 LGB 1147，bi500 train_raw=1016 < LGB 1147（拟合更好）。但同拟合水平下 val_raw 1564 >> LGB 1512（泛化差 52MW）。真问题是泛化+残差规律性，非欠拟合。"解决欠拟合"本身可行但 val 不改善，作追 v6 手段无效。

**Phase 1-3 全取消**。维持 CatBoost 路径终止，回 v6 + 等新数据。
