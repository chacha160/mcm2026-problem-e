# 2026 年研究生数学建模竞赛 E 题 —— 问题一

本仓库只做**问题一**：多模态特征提取与时序对齐。问题二/三的代码与产物已在一次
清理中出库，仓库不再包含它们（见 §9）。

> 本文件是入口：讲清输入是什么、仓库怎么组织、数据怎么流动、产出是什么、怎么跑。
> 方法细节在各模块自己的 docstring 与论文里，本文件只做索引，不复述数字——
> 数字都在自动生成的产物里，两处各写一份必然互相矛盾。

---

## 1. 输入：赛题给了什么

| 路径 | 是什么 | 是否入库 |
|---|---|---|
| `data/attachment1/` | 解压后的全部赛题数据，含附件1–4（约 5.6 GB） | 否，体量过大 |
| `data/attachment1/E题数据/E题数据/附件1-数据集原始多模态样本/` | 本问唯一使用的输入：100 条 MP4 与 `label-100.xlsx`（约 100 MB） | 否，同上 |
| `code/logs/题目.txt` | 赛题正文的纯文本提取版 | 是 |

样本身份是 `sample_key = video_id + "_$_" + clip_id`，例如 `-3g5yACwYnA$_$13`。
每条的 MP4、官方文本、以及二者的 SHA-256 在 `data/q1_v2/manifest.csv` 里一一对应。

**输入口径以附件1 为准，不使用附件2/3/4。** 那三份是问题二/三的官方特征，
与本问无关。

---

## 2. 输出：产出了什么

### 2.1 主产物 —— `data/q1_v2/`

v2 链路的产物，计入提交口径 **125 个文件 / 19,164,936 B**：

| 路径 | 内容 |
|---|---|
| `features/*.npz` | 100 个样本的三模态时序特征（计入口径的主要体积，18.7 MB） |
| `manifest.csv` | 100 条样本的身份、路径与校验和 |
| `results_100.csv` | 56 列逐样本汇总表（路由、掩码、覆盖区间、校验和） |
| `config_snapshot.json` | 本次运行的对齐侧与提取侧参数快照 |
| `metadata/` | `environment` / `model_assets` / `registry` / `route_policy` / `protected_baseline` |
| `reports/` | 各阶段小结、核验报告、体积台账、身份扫描 |
| `audit/` | `anomaly_ledger.csv`（异常只标注不删除）、`v2_route_100.csv`（逐样本路由） |
| `correspondence/` | 典型样本的逐词对应表（赛题四(二)2.(3) 那条要求） |
| `logs/`、`.cache/` | 按样本的运行日志与提取/对齐缓存，**不计入**提交口径 |

`logs/` 与 `.cache/` 的「不计入」是台账自己的口径（`q1v2_size_ledger.json` 的
`not_counted`），不是本文件的判断。

### 2.2 旧产物 —— `data/q1_delivery/`（v1 链路）

27 个文件 / 16,129,185 B，含 `features_q1.npz`（11.6 MB）。v2 的 `replaces`
字段声明的就是它。两份是同一问的先后实现，不要同时当作最终交付；v1 留着是因为
M2 基线钉住了它，动它会让 v2 核验硬失败。

### 2.3 论文与图

| 文件 | 说明 |
|---|---|
| `问题一论文正文.docx` | 定稿：17 张表、12 张图、137 个 OMML 公式 |
| `问题一论文.docx` | 单篇稿：10 张表、2 张图、53 个公式 |
| `问题一提交.docx` | 提交稿：13 张表、8 张图（有一处残留引用，见 §9） |
| `问题一论文正文_展开.md` | 注入表格后的展开稿，是 docx 的直接输入 |
| 根目录 `图*` | 18 个文件：14 张 300 dpi png，其中 4 张另给了同名矢量 pdf |

---

## 3. 代码结构

仓库里有**两条建模链**和**一条文档链**。两条建模链是同一问的先后实现，
不是流水线的前后两段。

| 链 | 入口 | 产物 | 状态 |
|---|---|---|---|
| **v2** | `code/run_q1v2_all.py` | `data/q1_v2/` | 当前链路，论文的表格与图都取自这里 |
| **v1** | `code/run_all.py` | `data/q1_delivery/` | 已被 v2 取代，保留是因为 M2 基线钉着它 |
| **文档** | `_build_q1_paper.py` / `_build_q1_body.py` | 根目录的 md 与 docx | 只读 `data/`，不重新提取 |

### 3.1 v2 六个阶段

`code/run_q1v2_all.py` 是纯编排器，自己不含建模逻辑，按序调用各模块。
`--stage {prepare,text,align,media,tables,verify,all}`，可单阶段或单样本重跑。

| 阶段 | 模块 | 产物 |
|---|---|---|
| `prepare` | `q1v2_assets`、`q1v2_route` | `metadata/*.json`、`config_snapshot.json` |
| `text` | `q1v2_text`、`q1v2_build` | `.cache/text/`、`logs/*_text.json` |
| `align` | `q1v2_align`、`q1v2_media`、`q1v2_build` | `.cache/align/`、`logs/*_align.json` |
| `media` | 逐样本子进程 | `features/*.npz`（20 MB 里的绝大部分） |
| `tables` | `q1v2_build` | `results_100.csv`、`manifest.csv`、`audit/` |
| `verify` | `q1v2_verify` | `reports/verify_report.{json,md}` |

**为什么媒体阶段单独起子进程**：PyAV 解码、MediaPipe(TF Lite)、openSMILE(C 扩展)
三个原生栈叠在一起，段错误会直接带走整个进程，try/except 拦不住。文本与对齐阶段
是纯 PyTorch / stable-ts，没有这个风险，起子进程反而每条白付数秒的模型加载时间，
所以改成批内逐样本 try/except。两种模式都可用 `--isolate` / `--no-isolate` 覆盖。

每个阶段逐样本即写即落，重跑默认跳过已有产物（`--overwrite` 反选），中断后
重跑只补缺失部分。媒体子进程先写 `.tmp` 再 `replace`，不会留下半截 NPZ 被当成成品。

### 3.2 v2 模块

| 模块 | 职责 |
|---|---|
| `q1v2_contract` | 冻结常量与通用 IO。刻意不导入 torch/mediapipe/opensmile/stable_whisper，使核验器与体积台账在没装重库的机器上也能跑 |
| `q1v2_assets` | 资产定位、预取、SHA-256 身份核验、环境冻结、受保护基线（M2） |
| `q1v2_text` | 两层文本单元规则（官方词 / 词单元）与 RoBERTa 词级池化 |
| `q1v2_align` | 官方文本强制对齐与「块→词」归并 |
| `q1v2_media` | 音频/视觉原生观测与词级聚合，含全部 HardStop 断言 |
| `q1v2_route` | 五类路由与三处声明。纯函数：只吃标量、只吐标量，便于核验器从成品 NPZ 复算 |
| `q1v2_build` | 单样本装配与输出表 |
| `q1v2_correspondence` | 逐词对应表导出 |
| `q1v2_verify` | 核验层 V1–V14 |
| `q1v2_package` | v2 的体积台账与身份扫描 |

### 3.3 v1 链路的模块

`code/run_all.py` 串起三步 `q1raw` → `q1` → `pack`，产物在则跳过。

- `run_unaligned_all.py`：三模态分别提取 → 统一对齐 → 时间轴可视化
  （`unaligned_text` / `unaligned_audio` / `unaligned_vision` / `word_align` /
  `align_multimodal` / `timeline_visualize`）
- `q1_delivery`、`q1_verify`、`q1_readme`、`q1_fusion`、`q1_typical`

### 3.4 辅助模块

`config.py`（全局参数）、`utils.py`（通用工具）、`check_data.py`（数据诊断）、
`face_probe.py`（「多人或远景」类别的人脸探测取证）、`package_check.py`
（提交体积核算与红线扫描）。

### 3.5 文档链

两级入口各自把三段串起来：

```
_gen_paper_tables_q1.py  →  _tables_q1/*.md（12 张表）
        ↓
_inject_paper_tables.py  →  问题一论文正文_展开.md
        ↓
_make_paper_docx.py      →  问题一论文正文.docx
```

`_build_q1_body.py` 与 `_build_q1_paper.py` 是这两个入口。旁支：`_make_q1_body_figures.py`
（图，只读 `data/q1_v2/`）、`_make_flowchart.py`（流程图）、`_make_report_docx.py`
（普通报告转 Word）、`_make_q1_submit.py`（整合提交包）、`_verify_docx.py`（结构核对）。

---

## 4. 数据怎么流动

```
附件1 的 100 条 MP4 + label-100.xlsx
        │
        ▼  ① 身份与质量检查
sample_key，MP4/文本/SHA-256 一一对应        → manifest.csv
        │
        ▼  ② 建立统一时间基准
音频与视频共用同一个 t0，时间取真实解码 PTS，不用 frame_index/fps 估算
        │
        ├──────────────┬──────────────┐
        ▼              ▼              ▼
   ③ 文本          ③ 语音          ③ 视觉
   RoBERTa 词级     openSMILE        MediaPipe
   向量             eGeMAPSv02       FaceLandmarker
        │              │              │
        └──────────────┴──────────────┘
                       │
                       ▼  ④ 强制对齐与路由
        stable-ts 把官方文本对齐到音频，得到逐词时间
        按文本—语音对应关系的可信度分五类路由，粒度落为 词级 / 片段级
                       │
                       ▼  ⑤ 词区间聚合
        落词内的观测取均值与总体标准差拼接；区间内无有效观测则判该模态无效
                       │
                       ▼  ⑥ 装配与核验
        features/*.npz + results_100.csv → 16 项核验 → verify_report
```

---

## 5. 怎么跑

### 5.1 环境

v2 需要独立 venv，且**路径必须是纯 ASCII**：openSMILE 的 Python 绑定在
`opensmile/core/lib.py` 里执行 `bytes(config_file, "ascii")`，包路径含非 ASCII
字符时 `Smile()` 构造会抛 `UnicodeEncodeError`。本机项目路径含中文，故 venv 放在
`<盘符>:\q1v2work\.venv`（该约束已编码进 `q1v2_contract._resolve_work_root()`）。

```
python -m venv --system-site-packages <ASCII_PATH>\.venv
<ASCII_PATH>\.venv\Scripts\python.exe -m pip install -r code/requirements-q1v2.txt
```

`--system-site-packages` 是为了继承既有的大依赖（torch / transformers / av 不必重装），
venv 内只新增 `opensmile==2.6.0` 与 `mediapipe==0.10.35`。实测版本：
Python 3.13.9 / torch 2.14.0+cpu / transformers 5.17.0 / stable-ts 2.19.1 / av 18.1.0。

### 5.2 跑 v2

```bash
python code/run_q1v2_all.py --stage prepare --allow-download   # 首次需下模型
python code/run_q1v2_all.py --stage all
```

在仓库根目录运行即可，脚本会自己把 `code/` 放进 `sys.path`。

其他开关：`--sample-key`（单样本重跑）、`--limit`、`--overwrite`、
`--isolate/--no-isolate`、`--timeout`、`--retries`、`--quick`、
`--correspondence-csv`。

**跑核验务必用上面那个 venv 的解释器。** 用系统 Python 会因缺 mediapipe/opensmile
而失败，而失败报告会写进交付物目录。

### 5.3 跑 v1

```bash
python code/run_all.py                      # 全量，产物在则跳过
python code/run_all.py --only q1            # 只跑交付/核验/文档层
```

### 5.4 重建论文

```bash
python _build_q1_body.py
python _verify_docx.py                      # 结构核对
```

---

## 6. 核验

`code/q1v2_verify.py` 共 **16 项**（V1–V14 与 M1、M2）。它只做四件事：内部自洽
（产物能不能复算出自己）、外部交叉（ffprobe 声明与解码覆盖）、结构硬锚点
（1926 / 1932 / 0 这些冻结数）、位精确锚点（两条数字静音样本的每个采样点）。
它**不与对照实现比数值**——对照实现的逐样本数组全空，比不了。

当前状态：执行 15 项、跳过 1 项（V12「路由对拍」默认关闭）、断言 20 686 条、
失败 0。`all_passed` 只对实际执行过的项成立，跳过的单独列出，不混进通过数。

轻量项（V1–V5、V9–V11、V13）只依赖 numpy 与成品产物，裸解释器就能跑；V6、V7、V14
需要外部工具或重依赖，`--quick` 下跳过。这正是 `q1v2_contract` 禁止在模块层导入
torch/mediapipe/opensmile 的原因。

其中两项值得单独说：

- **M2「受保护基线未变」**：用 `metadata/protected_baseline.json` 里 530 个文件的
  SHA-256 比对既有交付。基线里有两个文件（`data/q1_delivery/verify_report.{json,md}`）
  被 `q1v2_contract.VOLATILE_BY_DESIGN` 显式列为按设计可变，v1 核验器每跑一次就重写。
  **复核 M2 要调 `q1v2_assets.compare_protected_baseline()`，自己逐文件重算哈希会误报失败。**
- **V11「路由可复算」**：之所以能把 `q1v2_route` 做成纯函数，就是为了让核验器从成品
  NPZ 里把标量读回来、在原实现之外重放一遍逐字段比对。

`_verify_docx.py` 查的是另一件事：占位符残留、LaTeX 残留、公式编号与正文引用闭合、
图表引用是否有对应标题。它把断链计入结论——列出来却仍报「通过」是不行的。

---

## 7. 关键约定

- **时间基准**：一律取真实解码 PTS，音频与视频共用同一个 t0，区间半开。逐样本的
  文本时间基准是二选一的证据路由结果，落在 `word_time_basis` 字段。
- **词单元口径**：官方词与词单元是两层，核验围绕 1926 这个硬锚点复算。
- **对齐参数**：`remove_instant_words` 取 False，零时长词保留。
- **有效性命名分层**：`*_valid` 掩码、`valid_length`、`audio_speech_valid` 三者
  语义不同，不可互相替代。
- **异常只标注不删除**：`counts_as_deletion ≡ 0` 由核验项强制。

---

## 8. 诚实边界：这些结论不能下

1. **文本没有真实的逐词时间戳。** 全部 100 条的 `text_audio_correspondence` 都是
   `not_asserted`（98 条）或 `no_speech`（2 条），`content_assertion` 恒为 0。
   词时间来自本仓库自己跑的 stable-ts 强制对齐（`alignment_source` 全为
   `q1v2_own_stable_ts_run`），那是估计值，不是标注。
2. **`audio_speech_valid` 从不取 1。** 实测 98 条为 -1（未判）、2 条为 0（无语音），
   没有一条能断言「这段音频就是语音」。不要把它当模态可用性判断。
3. **98/100 的样本路由落在 `UNCERTAIN_REVIEW`。** 对齐粒度为词级的有 97 条、
   片段级 3 条；其中 `word_valid` 47 条、`word_partial` 50 条。这两个词差的是
   可靠性层次，逐词时间在 `word_partial` 上只作参考。
4. **24/100 的样本人脸特征无效**（`face_feature_valid = 0`，人脸检出帧 16 806 / 23 241）。
   这是实测结论，不是提取失败。

---

## 9. 已知问题与遗留

- **`问题一提交.docx` 有两处断链引用**：正文出现「第 4.6 节」与「表 4-3」，
  而全文只有三章、最后一表是 3-3。像是从更长的稿子裁剪后落下的。`_verify_docx.py`
  会报出来（`表引用无标题: ['4-3']`）。尚未修。
- **`data/q1_delivery/README_问题一交付与验证.md` 的 D1 条目**仍在引用已出库的
  `emotion_model.py` / `data/model/` / `data/_d1_check_model/` 作证据。该文件被
  `protected_baseline.json` 钉住且不在 volatile 名单，改它会让 M2 硬失败；
  生成器是 `code/q1_readme.py`，要改得连带重做基线快照。
- **`code/package_check.py` 仍把 `data/q2`、`data/q3` 列入体积台账**，并有
  `import q2q3_common`（已出库，try/except 内优雅降级）。台账合计
  42 331 339 B 与论文那句同源；单独核 v2 的体积请用
  `data/q1_v2/reports/q1v2_size_ledger.json`。
- **v1 链路的 `config.py` 有遗留常量**（`OUTPUT_DIR`、`TEMP_AUDIO_DIR`），
  指向已不写出的目录，保留只为诊断脚本能 import。

---

## 10. 仓库收录范围

入库：代码 + 文档 + 论文与图 + 真正的交付物。出库：赛题原始题包（约 5.6 GB）、
可重跑的中间产物、一次性核验的旧快照、由 `_make_q1_submit.py` 从 `data/` 与
`code/` 复制而来的 `问题一提交/`。判据与逐条规则见 `.gitignore` 第 5 节。

出库者**本地一律保留**，只是不进版本库。
