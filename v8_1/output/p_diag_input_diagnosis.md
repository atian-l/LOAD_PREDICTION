# v8.1 P-Diag：输入异常检测 + 残差幅度可迁移性诊断

> Diagnosis 层原型探针（DESIGN §3/§8）。测 Phase 0 未覆盖的两个问题：
> ①残差**幅度** |r| 是否跨年可预测（幅度≠方向）；
> ②输入异常分数（pred_load 突变 + 天气 OOD）是否跨年识别高误差点。

r=actual−base_A(v6)。mr_train=|r_train|(2025 OOF, N=22947, mean=1715)，mr_val=|r_val|(2026 eval-only, N=10272, mean=1446)。基线=mean(|r_train|)。

对照：Phase 0 **有符号** combined 跨年 transfer R²=−0.432（方向反相关）。

> **与先前 exp_v81_p0.py 的调和**：先前 [1] 标注"幅度回归 target=resid" 实为**有符号** resid 回归（误标），R²=−0.5859 与 Phase 0 −0.432 同号一致。**先前从未测过真正 |r| 幅度回归跨年**。本 P-Diag 是首次测 |r|，得 +0.182。两边风险 AUC（0.7523 vs 0.760）一致=幅度-风险可迁移。


## A. 残差幅度总览

| 段 | 2025 mean|r| | 2026 mean|r| | 2026 median|r| |
|---|---|---|---|
| night | 826 | 749 | 490 |
| day | 2809 | 2386 | 1768 |
| evening | 1079 | 807 | 630 |

## B. 残差幅度 |r| 跨年可迁移性（核心：幅度≠方向）

| 分量 | 列数 | 期内R²(2025holdout) | 跨年transfer R² | 标准R² | risk_AUC |
|---|---|---|---|---|---|
| load_level | 15 | +0.126 | +0.156 | +0.136 | 0.734 |
| load_temporal | 15 | +0.044 | +0.229 | +0.211 | 0.740 |
| calendar | 18 | +0.175 | +0.199 | +0.180 | 0.735 |
| weather | 43 | +0.146 | +0.126 | +0.105 | 0.751 |
| solar_renewable | 35 | +0.183 | +0.196 | +0.177 | 0.749 |
| combined | 126 | +0.176 | +0.182 | +0.163 | 0.760 |

**combined 幅度跨年 transfer R² = +0.182**（期内 +0.176）。Phase 0 有符号 combined = −0.432。若幅度 transfer R²>0 -> 方向虽翻转，误差大小可预测 -> Diagnosis 层有信号。


## C. 输入异常分数 -> 高误差识别（跨年）

A1=pred_load 日级突变(标准化) | A2=光伏_温度+光伏_辐照度 日级 Mahalanobis OOD | B=幅度探针(combined)对照

| 信号 | 2025 corr(|r|) | 2026 corr(|r|) | 2025 高/全局MAE | 2026 高/全局MAE | 2026 risk_AUC | 2026 高风险MAE | 2026 低风险MAE |
|---|---|---|---|---|---|---|---|
| A1_pred_load_jump | +0.118 | +0.078 | 1.29 | 1.19 | 0.540 | 1717 | 1387 |
| A2_weather_ood | +0.103 | +0.087 | 1.34 | 0.88 | 0.551 | 1278 | 1482 |
| B_mag_probe(combined) | +0.757 | +0.533 | 2.52 | 1.89 | 0.760 | 2732 | 973 |

迁移判据：2025 corr>0 且 2026 corr>0 且 risk_AUC>0.55。ratio>1=高风险子集误差更大（信号有效）。


## D. 分段 × 分量 幅度 transfer R²（午间重点）

| 段 | load_level | load_temporal | calendar | weather | solar_renewable | combined |
|---|---|---|---|---|---|---|
| night | +0.120 | +0.127 | +0.119 | +0.115 | +0.105 | +0.146 |
| day | -0.160 | +0.071 | -0.029 | -0.160 | -0.015 | -0.044 |
| evening | +0.037 | +0.082 | +0.023 | +0.043 | +0.013 | +0.150 |

## E. 自动判定（P-Diag go/no-go）

1. **幅度跨年迁移分量**：['load_level', 'load_temporal', 'calendar', 'weather', 'solar_renewable', 'combined']
2. **幅度最强分量**：load_temporal (transfer R²=+0.229, 期内 R²=+0.044, risk_AUC=0.740)
3. **输入异常跨年迁移信号**：['B_mag_probe(combined)']
4. **P-Diag 判定**：**GO（限定）** - 残差幅度 |r| 跨年可迁移
   -> **核心发现**：有符号残差不迁移（Phase 0 −0.432），但**幅度 |r| 迁移**（+0.182，期内 +0.176 几乎无过拟合 GAP，risk_AUC 0.760）。即 v6 在哪些点会大误差、误差多大**跨年可预测**，仅"高估还是低估"不可预测。验证 DESIGN §3：Diagnosis 预测 Information 可靠性（幅度/风险）可迁移，非 Residual（有符号值）。
   -> **限定①中午段不迁移**：day 段 combined 幅度 transfer R²=-0.044（负），night/evening 为正（+0.146/+0.150）。幅度迁移集中在中等误差段，**最高误差的中午段反而不可迁移**（同 midday 诊断 R²=−0.03 信息耗尽）。
   -> **限定②A2 天气OOD 不稳定**：corr 两年弱正但高/全局 MAE 比率跨年翻转（1.34->0.88），与有符号残差同款跨年不稳，非干净信号。A1 pred_load 突变弱一致（AUC 0.540 未达 0.55）。干净信号仅幅度探针 B 本身。
   -> **MAE 含义**：Phase 0 已证有符号 Correction 无增益；幅度可迁移但符号不可迁移，故"风险标注/人工复核/触发 fallback"有价值，直接降 val MAE 仍须新能源/Foundation 作 fallback。破 1445 概率仍在新信息，但 Diagnosis 层不再是空壳。