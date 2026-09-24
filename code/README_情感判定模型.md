# 情感强度与状态判定模型 —— 方法说明与复现指南

> 本文档说明**方法**（怎么做的、为什么这么做、哪些结论不能下）。
> 本次运行的**具体数字**见 `data/model/README_情感判定模型_运行结果.md`（自动生成，每次运行刷新）。

---

## 1. 这个模块在做什么

问题1 产出了三模态对齐特征（`(100, 50, D)` 张量 + 有效掩码），但特征与监督标签从未配对过。
本模块把两者接起来，按题目要求判定每条样本的：

- **情感强度** —— 回归连续 `label`，强度取 `|预测|`
- **情感状态** —— Positive / Neutral / Negative

并按用户要求，产出可直接对照 `label-100.xlsx` 后两列（`label`、`annotation`）的测试样本预测表。

---

## 2. 三条决定方案形态的实测事实

### 2.1 `annotation` 是 `sign(label)` 的确定性函数，零例外

实测 100 条全部满足：`label>0 → Positive`(57)、`label=0 → Neutral`(25)、`label<0 → Negative`(18)。
因此「强度」与「状态」不是两个独立任务，一次回归天然自洽（不会出现「预测为正但强度为负」）。
代码中 `build_design_matrix` 对此做了**硬断言**，一旦不成立直接抛错。

### 2.2 同一视频的片段必须同折（否则泄漏）

100 条来自 **37 个视频**（14 个单片段、23 个多片段，最多 9 片段，中位 2）。
同视频片段共享说话人、话题与录制环境。若随机划分，模型可以靠「认出这个视频」而非「理解情感」得分。
本模块一律使用 `StratifiedGroupKFold`，以 `video_id` 为组。

### 2.3 文本有效槽只占 43%，且 `*_valid` 不能判断模态可用性

- **文本**：有效槽平均 43.16%（每条中位 20/50）。句子短，词级 RoBERTa 向量只占少数槽，其余是**精确零填充**。
  若对全部 50 槽求均值，768 维文本向量会被稀释过半。→ **必须只在 valid 槽上池化**。
- **`*_valid` 的语义边界**（容易误用）：`valid=True` 只表示「该槽被分配到了采样单元」，
  **不表示该单元是真观测**。反例：`-mJ2ud6oKI8_1` 的 `vision_valid` 为 50/50 全 True，
  但视觉特征能量恰为 0.0（MTCNN 一帧人脸都没检出）。插值/零填充同样被标 True。
  → 判断模态是否真的可用要用原始 `face_ratio`（见 `load_availability`），不能用 valid 掩码。

---

## 3. 特征装配口径

| 模态 | 池化 | 维度 | 依据 |
|---|---|---|---|
| 文本 | **仅均值** | 768 | 有效槽仅 43%，且句内词向量 std 是噪声。固定 α=1000 实测：改仅均值后 acc 0.590→0.620、r +0.359→+0.421（嵌套诚实协议下文本单模态 r=+0.422 / ρ=+0.496，acc 0.582——超参由内层 CV 决定，收缩更重） |
| 音频 | 均值 + std | 148 | 真时间序列（20 Hz），沿时间轴的波动含信息 |
| 视觉 | 均值 + std | 70 | 真时间序列（15 Hz），同上 |

- 一律**只在 `*_valid` 为真的槽上统计**；某样本该模态无有效槽则整块置零
  （语义即「该模态对此样本不可用」，与 `face_ratio=0` 一致）。
- 合计 **986 维**。`meta["slices"]` 记录各模态列区间，消融时按区间取子矩阵，保证口径不漂移。
- 6 条视觉退化样本（`face_ratio<0.5`）：其中 2 条特征全零、4 条为「插值借用」（特征非零）。
  **保留其原始特征不丢弃**，并在预测表中带 `face_ratio` / `vision_available` 两列供问题2 使用。
  保留理由：删样本会偷偷改变测试集分布。

---

## 4. 评价协议（三条铁律）

1. **按视频分组 5 折**，`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`。
   实测折大小 18/19/19/24/20，每折含三类，无退化折。
2. **超参必须嵌套选**：外层 5 折评估，超参在**每个训练折内**用内层 3 折选。
   绝不「全量选完超参再评估」——那会让测试折信息通过超参泄漏，分数系统性偏高。
3. **out-of-fold 预测必须按原始下标写回**（`oof[te] = pred`），不可按折顺序拼接。
   按折序拼接再与 `y` 比较会完全错位，相关系数假性掉到 ≈0（这是本项目实际踩过的坑）。

### 4.1 固定 held-out 测试集

取上述固定种子 5 折的**第 1 折**作 held-out：18 条（正10/中5/负3，多数类基线 0.556）。
选它是因为它是五折中类别最平衡的一折（偏差 0.056），且是先验取第一折、**不是按表现挑选**。

> 为什么不按 74/26 划分：扫描 60 个分组种子，最优测试集仍含 19正/4中/3负 = 73% 正向，
> 而全局只有 57%——在那个测试集上「无脑全猜 Positive」就有 73% 准确率，会制造假象。

---

## 5. 指标与基线（诚实性约束）

**必须并列报告该折/该测试集自己的基线**，不是全局基线：

- 分类：**多数类基线**（逐折算，实测 0.570）
- 回归：**训练折均值基线**（逐折算，实测 MAE 0.583）

**必须报告的诚实指标**：

| 指标 | 为什么必须报 |
|---|---|
| `recall_Neutral` | **最关键的一条**。回归模型的中性召回率为 **0.000**——它们从不预测 Neutral，准确率全部来自对 Positive 的过预测。不报这一条，「acc 0.60」会被严重误读 |
| `balanced_acc` | 57/25/18 不平衡下比 acc 诚实 |
| `macroF1` | 同上；「全猜 Positive」的 macroF1 仅 0.242 |
| `calib_slope` | 校准斜率。≪1 说明预测被强烈压向均值——即「能排序但不敢给大幅度值」 |
| `acc_std` / `macroF1_std` | 必须报标准差。若模型间差异落在标准差内，须写「无显著差异」 |

---

## 6. 运行

```bash
cd D:\23届建模\code

# 自检装配层（打印维度、组数、状态分布）
python emotion_features.py

# 全流程：对比 + 消融 + held-out + 出图 + 报告
python emotion_model.py

# 只跑对比与消融，不出图
python emotion_model.py --cv-only

# 只用文本单模态（对照用）
python emotion_model.py --modalities text
```

主要参数：`--modalities`（逗号分隔）、`--regimen`（回归模型，默认 ridge）、
`--classifier`（分类模型，默认 logreg）、`--seed`（默认 42）、`--out`（默认 `data/model`）。

**依赖**：仅 numpy / pandas / scipy / scikit-learn / matplotlib（均为本机已有，零安装、零网络）。

---

## 7. 产物

| 文件 | 内容 |
|---|---|
| `data/model/cv_comparison.csv` | 各模型 out-of-fold 指标（含基线、逐类召回） |
| `data/model/ablation.csv` | 模态消融（7 个组合） |
| `data/model/test_predictions.csv` | **held-out 18 条**逐样本预测，可直接对照 `label-100` 后两列 |
| `data/model/oof_predictions_all100.csv` | **全部 100 条** out-of-fold 预测（每条都由没见过其视频的折预测） |
| `data/model/fig_confusion.png` | 混淆矩阵（回归取符号 vs 分类器） |
| `data/model/fig_scatter.png` | 预测 vs 真值（含 y=x、r/ρ/MAE 与基线） |
| `data/model/fig_ablation.png` | 模态消融柱状图（r 与 macroF1 双面板） |
| `data/model/README_情感判定模型_运行结果.md` | 本次运行的完整结果与性能边界 |

预测表列：`sample_id, video_id, clip_id, true_label, pred_label, true_intensity, pred_intensity,
true_annotation, pred_annotation_from_regression, pred_annotation_from_classifier,
correct_from_regression, correct_from_classifier, face_ratio, vision_available`

其中 `true_label` / `true_annotation` 与 `label-100.xlsx` 逐行一致
（实测最大差 3.7e-07，纯属 CSV 六位小数舍入；`annotation` 完全一致）。

---

## 8. 明确不做的事

- 不使用附件2 的 4850 条官方样本（音频是 COVAREP、视觉是 FACET，与自提取特征不同源）
- 不微调 RoBERTa（N=100 下方差极大，且破坏「冻结特征」的可比性）
- 不引入 statsmodels / xgboost / lightgbm（未安装，零安装约束）
- 不做特征重要性归因（N=100 下不可靠）
- 不改动问题1 的任何文件与产物（本模块只读 `data/aligned/` 与 `data/unaligned_features/`）
