# TCN 调优实验脚本工程计划

## 目标与约束
- **目标**：在 `load_pred_tcn/` 内搭建一套覆盖全部调优元素的实验脚本，支撑系统化超参搜索与拟合诊断（首要：用纯训练集 MAE 判过/欠拟合）。
- **约束（不变）**：
  - 所有新脚本/文件仅创建在 `load_pred_tcn/` 内，**不修改 `load_pred/` 及任何外部文件**。
  - 6 条泄露不变量逐条遵守：actual 仅作目标/评估；pred_load 滞后特征不变（lag_192 在）；气象去重三阶段一致；时间边界（TRAIN_END=2026-02-28，val 2026-03-01~06-15 eval-only）；共享 `features.build_features`；模块形式运行 `python -m load_pred_tcn.exp_xxx`。
  - `predict_load` 后处理（锚→median→λ收缩→hour_bias→drift_corr `+=`→threshold_corr→clip）逐行不变。

## 设计原则
1. **选择信号用 WF-CV，不用官方 val。** 超参选择基于 3 折 walk-forward（训练期内 `best_it_folds`）折内 MAE 均值+标准差；官方 val 仅作每配置最终读数，**绝不参与选择**（防 val 过拟合/泄露）。train MAE 仅用于过/欠拟合诊断，不用于选择。
2. **复用生产管线，不重写逻辑。** 实验脚本调用 `load_pred_tcn.train` 的 `build_dataset`/`usable_mask`/`train_ensemble`/`compute_hour_bias`/`_evaluate`，通过 cfg 覆盖传超参。标准化、样本权重、OOF 校正逻辑全部复用，保证实验与生产同构、无 train/serve skew。
3. **减速搜索：缩成员。** 全量 40 成员×3 折 OOF=160 次训练太慢。harness 支持 cfg 覆盖 `seeds`/`objectives`/`residual_modes`（如 5 成员快速模式）做广搜，top 配置再用全量 40 成员验证。
4. **不写生产 artifact。** 实验脚本不保存模型到 `models/`、不覆盖 `model_bundle.pkl`；仅打印对比表，可选写小结果 txt 到 `load_pred_tcn/exp_results/`。
5. **每脚本聚焦一个调优元素组**，统一输出"配置 → WF-CV MAE / 折CV / 官方val MAE / Bias / debiased / 午间MAE"对比表 + 判定语。

## 脚本清单（10 个，均在 `load_pred_tcn/`）

### 0. `exp_common.py`（共享 harness，基础）
所有实验脚本复用的原语：
- `build_cached()`：`build_dataset`+`MismatchModel`+`MosModel` 一次性构造（times/X/pred_load/actual/usable/anchor/feat_cols/val_m），缓存避免重复。
- `train_ens(cfg_override, usable, best_it=None, mos_model=None)`：合并 `cfg={**TRAIN_CONFIG, **override}`，调用生产 `train_ensemble`，返回 `EnsembleModel`。
- `ensemble_raw(model, X_sub, anchor_sub)`：无校正的原始集成预测（`anchor+λ(ens-anchor)`）。
- `wf_cv(cfg_override, best_it=None)`：3 折 walk-forward（复用 `best_it_folds`），每折 `train_ens(ftr)`→`predict(fva)`，返回 `{fold_mae[], mean, std, cv}`。
- `compute_oof_corr(cfg_override, best_it)`：复用 `compute_hour_bias` 返回 hour_bias/drift_corr/threshold_corr + oof_pred/oof_mask。
- `eval_val(...)` / `metrics_block(...)`：官方 val `_evaluate` 指标 + 统一打印 train_raw/OOF/val_raw/val_full 对比表。
- `_mae/_r2/_debiased/_midday` 工具；常量 `V6_VAL_MAE=1445.62`、`TCN_BASE_MAE=2027.47`（60ep 基线）。

### 1. `exp_fit_diag.py`【门控·首要】
用户点名的"纯训练集 MAE 判过/欠拟合"诊断。三类预测：**train_raw**（含已见，乐观下界）/ **OOF**（3 折无泄露）/ **val_raw** / **val_full**。输出：
- `train_raw − OOF` gap（>400 过拟合；<150 欠拟合；中间轻度）。
- `OOF vs val_raw`（|<200| 泛化一致=数据信号弱/结构性；OOF<<val 漂移；OOF>>val 训练期更难）。
- 分时段(00-06/06-11/11-14/15-18/18-24) train vs val raw MAE 表。
- 可选 LightGBM v6 同条件对比（若 lightgbm 可用）：判"模型拟合能力差" vs "泛化/数据差"。
- 标准化 sanity：打印 feat_std/target_std 分布，确认无尺度异常。
- 判定语决定后续脚本优先级。

### 2. `exp_curve.py`【门控·配合 1】
逐 epoch 学习曲线：训练一个小集成（5 成员），每 K epoch 在 (a)训练窗 (b)一折验证窗 评估 MW-MAE，打印 `train_mae`/`valfold_mae` vs epoch 曲线表。
- 直接看 val-fold MAE 第几 epoch 触底回升 = 过拟合 onset；客观定 `best_it_fixed` 最优点（替代当前"凭经验 60"）。
- 当前配置 + 一两个对照（如 dropout 0.1 vs 0.3）对比曲线。

### 3. `exp_capacity.py`
容量扫：`num_channels` ∈ {[32]×4, [64]×4, [128]×4} × 深度(层数/dilations/RF) ∈ {3层RF43, 4层RF91, 5层RF189, 6层RF379}，kernel_size 固定 7。WF-CV 选，val 读数。

### 4. `exp_regularize.py`
正则扫：`dropout` ∈ {0.1,0.2,0.3,0.4} × `weight_decay` ∈ {1e-5,1e-4,5e-4,1e-3}，其余固定（含当前 60ep）。WF-CV 选，val 读数。（Tier2 方向的系统化版）

### 5. `exp_train_dyn.py`
训练动力学扫：`epochs` ∈ {40,60,80,100} × `lr` ∈ {3e-4,1e-3,3e-3} × `scheduler` ∈ {none,cosine,step} × `batch_size` ∈ {32,64,128}，含 `grad_clip`。逐组单变量扫（不全组合爆炸）。WF-CV 选，val 读数。

### 6. `exp_window.py`
滑窗扫：`seq_len` ∈ {240,480,672,960}（2.5/5/7/10 天）× `stride` ∈ {48,96}。WF-CV 选。

### 7. `exp_ensemble.py`
集成配置扫：`shrinkage λ` ∈ {0.3,0.5,0.7,1.0} × `aggregation` ∈ {median,mean,trimmed} × 成员数 ∈ {10,20,40}（缩 seeds）× `trim_frac` ∈ {0.1,0.2,0.3}。
- 注：λ 是"其他东西"（v6=1.0），本脚本作诊断/探针，明确标注偏离忠实端口；若 λ<1 显著更优说明 TCN 弱于锚。`quantile_alphas`/`objectives` 集合亦可在此扫。

### 8. `exp_oof_ablation.py`
OOF 校正消融：{无校正 / +hour_bias / +drift_corr / +threshold_corr / 全校正} 五档，看每档对 val MAE 边际贡献；并扫 `hour_bias_slots` ∈ {24,48,96}。判断 OOF 校正对 TCN 是否仍正向（v6 上 +67MW；TCN 上待验）。

### 9. `exp_members.py`
成员分析：40 成员逐个 val MAE、direct vs residual / regression vs quantile 分组贡献、成员间预测相关矩阵（多样性）、leave-one-out（去某成员后集成 val MAE 变化）。判断拖后腿/冗余(相关>0.98)/可否剪枝缩成员。

## 执行顺序
1. **先跑 `exp_fit_diag` + `exp_curve`（门控）**，判定决定后续优先级：
   - **过拟合**（train_raw≪OOF，curve 的 val-fold 早回升）→ 先 `exp_regularize` + `exp_capacity`(减) + `exp_train_dyn`(减 epoch)。
   - **欠拟合**（train_raw≈OOF 都高）→ 先 `exp_capacity`(增) + `exp_train_dyn`(增 epoch/lr)。
   - **结构性**（OOF≈val 都高、train 能拟合、curve 无明显回升）→ TCN 在本特征集结构性落后（同 CatBoost 结论），停止单独调优，转异质集成或停手。
2. 门控判定后，按优先级跑 3-5 个聚焦脚本（单变量组扫）。
3. top 配置汇总对比，最后用全量 40 成员复跑确认。

## 不在范围内
- 不修改 `load_pred/` 任何文件、不改 `load_pred_tcn/` 的 `data_loader`/`features`/`predict`（verbatim）。
- 不修改生产 `config.TRAIN_CONFIG`（实验通过 cfg 覆盖临时变超参，`config.py` 保持 Tier2 当前值 60ep/dropout0.2/wd1e-4）。
- 不用官方 val 做超参选择；不写模型到 `models/`。

## 成本与运行
- 单脚本（缩 5 成员 + 3 折）约 3-8 min/4090；全量 40 成员单配置约 15-45 min。
- 全部门控+聚焦脚本一轮约 1-2 小时 GPU。
- 运行：`python -m load_pred_tcn.exp_fit_diag` 等，从项目根目录。
