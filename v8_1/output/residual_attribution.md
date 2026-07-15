# v8.1 工作流 A：Residual Attribution（静态归因树）

> 中心件（DESIGN §6）。统一 FDS / 午间 / 信息源 / Phase0 诊断。问题从"残差怎么修"转为"**残差为什么产生**"。
> ext_err=actual−pred_load（外部原始误差，FDS 的 ext_error）；r=actual−v6_pred（v6 修正后=天花板）。
> 合规：actual 仅作 ext_err/r 评估标签(eval-only)；cause 标签无 actual。

ext_err MAE: train=1578 / val=1660。r MAE: train=1715 / val=1446。**v6 修正量 = 215 MW**（ext_err val 1660 -> r val 1446）。


## A. 可学习 vs 不可学习（ext_err 分量 R²）

探针在 2025 ext_err 上训、2026 上测。holdout R²=期内可学；transfer R²=跨年迁移。combined holdout R²=可学习份额；1−该值=不可学习份额(天花板)。

| 通道 | 列数 | 期内R²(holdout) | 跨年transfer R² |
|---|---|---|---|
| forecast_structural | 30 | -0.146 | +0.022 |
| calendar | 18 | +0.102 | -0.016 |
| weather | 43 | +0.071 | -0.085 |
| renewable_feat | 35 | -0.203 | +0.052 |
| combined | 126 | -0.061 | -0.004 |

**combined 期内 R²=-0.061** -> 可学习份额≈-6%，**不可学习份额(天花板)≈106%**。combined 跨年 transfer R²=-0.004（实际迁移的可学习部分）。
对照 Phase0：r(=v6修正后残差) 期内 R²=−0.136（已不可学）；此处 ext_err 期内 R²=-0.061（>0，因 ext_err 含 v6 能修的可学习结构）。两者差 = v6 已移除的可学习部分。


## B. 不可学习 proxy 归因（天花板 r 的 cause 通道，corr with |r|）

天花板 r 的 cause 通道用 proxy 与 |r| 的跨年 corr/ratio/AUC 归因（非加性，重叠）。

| proxy | 2025 corr | 2026 corr | 2026 ratio | 2026 AUC |
|---|---|---|---|---|
| A1_input | +0.174 | +0.157 | 1.30 | 0.517 |
| weather_OOD | +0.224 | +0.156 | 0.87 | 0.552 |
| renewable_dev | +0.078 | -0.054 | 0.88 | 0.436 |

对照 P-Diag：A1 输入质量 GO(弱但稳 ratio 1.54/1.34)，A2 天气 OOD 翻(1.34->0.88)。此处日级复测一致。


## C. 可再生物理 proxy 排除法（关键诚实结果）

物理 PV proxy=irrad×(1−0.004·(temp−25))；wind proxy=wind³∈[3,25)。

- PV(有符号) vs ext_err(有符号) corr：2025=-0.239 / 2026=-0.043
- PV_dev(幅度) vs |r| corr 2026=-0.054
- PV(有符号) 预测 ext_err **符号**方向命中=0.495（>0.5=符号可迁移）
- 中午(11-14) irrad vs |ext_err| corr=-0.165

**排除法结论**：气象隐含可再生 proxy 与 ext_err 符号弱相关/方向命中≈0.5（不可迁移）。原因：pred_load 已含外部预测器对可再生的同源气象假设，故气象隐含可再生≈已在 pred_load 内，残差的可再生分量=可再生实际−预测器假设，**不在气象 proxy 能解释的范围内**。-> 排除"气象隐含可再生"为符号根因，收窄到"气象抓不到的可再生变率"（云局地/弃光/组件，不可观测）。与 Phase0（辐照类预测不了符号）、午间诊断（R²=−0.03）一致。


## D. 逐日 cause 标签（运行时前身，无 actual）+ 验证

优先级：holiday->Demand shift；A1 高->Input anomaly；OOD 高->Weather OOD；renewable_dev 高->Likely renewable；否则 Unknown。阈值=训练期 top-15%。

| 标签 | 日数 | mean|ext_err| | /全局 |
|---|---|---|---|
| Demand shift | 768 | 1566 | 0.94 |
| Input anomaly | 768 | 2338 | 1.41 |
| Weather OOD | 1536 | 1376 | 0.83 |
| Likely renewable | 768 | 1814 | 1.09 |
| Unknown | 6432 | 1640 | 0.99 |

全局 mean|ext_err|=1660。ratio>1=该标签日误差更大（标签有效）。此为工作流 C 运行时 QA 的 cause 标签前身（合规，无 actual）。


## E. Residual Attribution Tree

```
ext_err (val MAE=1660)
├── 可学习 (v6 修正 -> r val MAE=1446, 修正 215 MW, 份额≈-6%)
│   ├── Forecast structural (pred_load level+temporal)  transfer R²=+0.022
│   ├── Calendar (demand shift 可学习)  transfer R²=-0.016
│   └── Weather (可学习耦合)  transfer R²=-0.085
└── 不可学习 (天花板 ≈ r, 份额≈106%)
    ├── Weather OOD (novelty)       corr|r|=+0.156 (不稳, 翻)
    ├── Input anomaly (pred_load)   corr|r|=+0.157 (稳, GO flag)
    ├── Renewable (proxy)           符号命中=0.495 (排除: 气象隐含可再生不可迁移)
    └── Random/Unknown              floor (符号通道缺失)
```


## F. 与既有诊断统一

- **FDS**：ext_error 83.7% unlearnable。本处 combined 不可学习份额≈106%（同量级）。
- **Phase0**：r 期内 R²=−0.136（残差=特征噪声）。本处 ext_err 期内 R²=-0.061（>0，含 v6 可修部分）；差值=v6 已移除的可学习结构。两者一致：残差的可学习部分 v6 已榨干，剩余=特征噪声。
- **午间诊断**：中午 R²=−0.03。本处中午 irrad vs |ext_err| corr=-0.165（弱），可再生排除法印证午间不可学。
- **P-Diag/工作流B**：A1 输入质量 GO(稳)，A2 天气 OOD 翻(不稳)。本处 proxy 归因复测一致。

**统一结论**：天花板=v6 已移除全部可学习结构后剩余的特征噪声。cause 结构中，输入质量(A1)是唯一稳的可部署 flag；天气 OOD/可再生 proxy 跨年不稳或不可迁移；符号通道缺失=Random floor 的主体。破 1445 须新信息（可再生实测/Foundation），非归因能解决；归因的价值=可解释/可诊断系统能力（Forecast QA），非 MAE。
