# v8.1 工作流 C：运行时 Forecast QA（四元组）

> 把 A（归因）+ B（A1 flag）变成 predict.py 运行时输出：Prediction + Confidence + Reason + Warning。
> 合规：运行时无 actual；Confidence=2025 OOF 训练幅度探针+isotonic；cause 标签仅 pred_load+weather+calendar。
> actual 仅作离线验证(eval-only)。

幅度探针：combined 全特征，|r| 目标，2025 OOF 训。median(|r_train|)=1028 MW（"典型误差"阈值）。


## A. Confidence 模型（幅度探针 + isotonic 校准）

- 跨年 transfer R²=+0.182，risk_AUC=0.784（对照 P-Diag +0.182/0.760，一致）
- isotonic 校准：pred_|r| -> P(|r|>median)。点级 Brier=0.208（0=完美，1=最差，0.25=常数基线）


## B. 运行时四元组（日级，val 2026 模拟运行时）

Confidence_day = 96 点 mean(1−risk) = 期望可靠点比例。Warning = Confidence<0.5 OR Reason=Input anomaly。

| 日期 | Pred均值 MW | Pred峰值 MW | Confidence | Reason | Warning |
|---|---|---|---|---|---|
| 2026-03-01 | 67182 | 75062 | 0.50 | Unknown | ⚠️YES |
| 2026-03-02 | 70866 | 80680 | 0.43 | Input anomaly | ⚠️YES |
| 2026-03-03 | 65727 | 73715 | 0.47 | Likely renewable | ⚠️YES |
| 2026-03-04 | 69890 | 79628 | 0.47 | Unknown | ⚠️YES |
| 2026-03-05 | 73313 | 85058 | 0.55 | Input anomaly | ⚠️YES |
| 2026-03-27 | 61811 | 74143 | 0.62 | Unknown | no |
| 2026-03-26 | 62510 | 73992 | 0.62 | Unknown | no |
| 2026-03-31 | 62887 | 74565 | 0.63 | Unknown | no |
| 2026-03-25 | 60674 | 73340 | 0.64 | Unknown | no |
| 2026-03-24 | 63536 | 74528 | 0.65 | Unknown | no |

（共 57/107 日触发 Warning）


## C. 验证（eval-only）

### Reliability（Confidence 分位 -> 实际可靠点比例）

| Confidence 区间 | 日数 | 实际可靠比例 | 实际 mean|r| |
|---|---|---|---|
| [0.16, 0.39] | 22 | 0.48 | 1848 |
| [0.40, 0.47] | 22 | 0.52 | 1513 |
| [0.48, 0.53] | 21 | 0.59 | 1395 |
| [0.54, 0.58] | 21 | 0.60 | 1477 |
| [0.58, 0.65] | 21 | 0.70 | 972 |

良好校准=低 Confidence 区间实际可靠比例低、mean|r|高（单调）。

### Warning 子集 + 置信分位

- 全局 daily mean|r|=1446
- Warning 日 mean|r|=1675（ratio=1.16）
- 非 Warning 日 mean|r|=1185（ratio=0.82）
- 最低 20% Confidence 日 mean|r|=1827；最高 20% Confidence 日 mean|r|=959（分离度=1.91x）
- 日级 Spearman(Confidence, mean|r|)=-0.329（负=高置信低误差）
- 日级 Brier=0.282


## D. Reason（cause 标签）验证

| Reason | 日数 | mean|r| | /全局 |
|---|---|---|---|
| Demand shift | 8 | 1282 | 0.89 |
| Input anomaly | 8 | 2018 | 1.40 |
| Weather OOD | 16 | 1274 | 0.88 |
| Likely renewable | 8 | 1333 | 0.92 |
| Unknown | 67 | 1451 | 1.00 |

仅 Input anomaly 稳定有效（ratio>1，对照工作流 A/B）。其他 cause 为最佳猜测但跨年不稳。


## E. predict.py 集成方式

```
# 离线训练一次（train 时）：
#   1. 幅度探针 booster（2025 OOF |r| 目标）-> models/qa_mag_booster.txt
#   2. isotonic 校准器 -> models/qa_calib.pkl
#   3. cause 阈值（A1/OOD/renewable top-15%）-> models/qa_thresholds.pkl
# 运行时（predict.py，D+1）：
#   features = build_features(14d window)   # 同 v6，无 actual
#   pred = v6.predict(features)            # 96 点
#   pred_mag = qa_mag_booster.predict(features)
#   risk = qa_calib.predict(pred_mag)      # P(|r|>median)
#   confidence = 1 - risk                  # 96 点 -> 日级 mean
#   reason = cause_label(A1, OOD, calendar, renewable)  # 日级
#   warning = (confidence_day < 0.5) or (reason == 'Input anomaly')
#   output: pred(96) + confidence(96) + reason(day) + warning(day)
```


## F. 结论

- **Confidence 模型有效**：transfer R²=+0.182/AUC=0.784（复现 P-Diag），reliability 单调，日级 corr=-0.329，低/高置信分离度=1.91x。
- **Warning 可操作**：Warning 日 mean|r|=1675（全局 1446，ratio 1.16），57/107 日触发 -> 人工复核/fallback 触发。
- **能力定位**：Forecast QA = 可解释/可诊断系统能力，**非 MAE 杠杆**（符号通道仍关，Confidence 预测可靠性不预测方向）。Reason 中仅 Input anomaly 稳定有效。
- **价值**：预测从"67182 MW"单值 -> 附 Confidence+Reason+Warning，满足工业 Forecast QA 需求（DESIGN 终点里程碑的"可解释/可诊断"落地）。