# FDS — Forecast Diagnosis System（预测诊断体系）

对生产系统 **v6** 负荷预测结果进行**多角度纯诊断分析**的独立子系统。

## 核心约束（不可违反）

本子系统是**只读诊断**，与生产系统完全解耦：

- ❌ 不修改生产模型 / 不修改任何生产代码 / 不改变任何训练流程
- ❌ 不改变任何数据泄露约束 / 不增加任何新的人工特征
- ❌ 不为了降低 MAE 而进行任何实验
- ✅ 所有分析仅建立在 v6 已有预测结果之上

生产模型 `load_pred/model_bundle.pkl` 仅以**只读**方式加载用于生成验证集预测；生产代码无任何写入。`load_pred_v6/` 归档完好，可随时恢复。

## 目录结构

```
FDS/
├── README.md            # 本文件
├── report.md            # ★ 最终诊断报告（8 项结论 + 候选建议）
├── ds_prep.py           # 数据准备：加载 v6 只读，生成 val 预测 + 诊断列 + OOF
├── diag_lib.py          # 共享库：加载/指标/ACF/PACF/绘图/CJK 字体
├── a01_distribution.py  # (一) 误差分布
├── a02_temporal.py      # (二) 误差时间规律（时/周/月/季/节假日）
├── a03_load_bins.py     # (三) 负荷区间（峰值/谷值/U 型）
├── a04_weather.py       # (四) 天气条件
├── a05_heatmaps.py      # (五) 二维 Heatmap（温×时 / 晴×时）
├── a06_autocorr.py      # (六) 残差自相关（ACF/PACF）
├── a07_consecutive.py   # (七) 连续误差 run + 逐日 Bias
├── a08_ramp.py          # (八) 爬坡（方向/幅度）
├── a09_extreme.py       # (九) 极端天气场景
├── a10_decomp.py        # (十) 误差来源方差拆解
├── a11_feature.py       # (十一) 特征贡献（Gain/SHAP/置换/PDP/ICE）
├── a12_viz.py           # (十二) 拟合可视化（散点/时间序列/案例日）
├── a13_learnable.py     # (十三) 剩余可学习信息（R² 上限判定）
├── run_all.py           # 一键复现编排
└── output/
    ├── diag_val.csv     # 验证集诊断数据（10272×38）
    ├── diag_oof.csv     # 训练期 3 折 OOF
    ├── X_val.csv        # 验证集特征（10272×126）
    ├── ds_prep.log      # 数据准备日志
    ├── figures/         # 18 张图
    └── tables/          # 26 张数据表
```

## 运行方式

```bash
# 从项目根目录运行（诊断模块需 import load_pred）
python -m FDS.run_all                 # 一键复现：数据准备 -> 13 项分析 -> 报告提示

# 或单独运行某项分析（数据准备须先完成）
python -m FDS.ds_prep                 # 生成 output/diag_val.csv 等
python -m FDS.a01_distribution        # 单项诊断
...
python -m FDS.a13_learnable
```

## 数据流

1. `ds_prep.py` 以**只读**加载 `load_pred/model_bundle.pkl`，对验证窗（2026/03/01–06/15，10272 点）调用 `predict_load()` 得到 v6 预测；同时构建 weather/calendar/ramp/分解列；并从 `build_train_oof()` 产出训练期 3 折 OOF（用于 a13 小时结构 R² 对照）。
2. 各 `aNN_*.py` 经 `diag_lib.load_val()/load_oof()` 读取缓存 CSV，独立计算并写出 `output/figures/`、`output/tables/`。
3. `report.md` 汇总 13 项分析 -> 8 项最终结论 + 候选建议。

## 关键诊断结论（详见 report.md）

- val MAE 1445.62，无系统偏移（p=0.44），重尾（峰度 5.38）。
- **误差首要来源：外部预测特异性误差 83.7%**（不可学习）；随机 7.5%；可迁移结构 ~9% 已被 v6 捕获。
- **峰值低估 −2 GW 是模型收缩**（验证峰值在训练分布内，非外推），属模型能力而非数据限制。
- 雨天贡献 48.9% MAE；多云午 MAE 3854（threshold_corr 未充分校正）；高温过估 +1347。
- 残差 ~78.5% 不可学习；逐日 oracle 上限 1284 MW 不可达 -> **v6 已接近当前数据信息上限**。

## 与生产的关系

本系统**不向生产反馈任何变更**。report.md 中的候选建议（CS1 特征剪枝 / CS2 峰值收缩 / CS3 多云午校准）均带强制免责声明，须独立实验验证后方可考虑实施。
