# 问题一交付物说明：三模态特征提取与时序对齐（v2 词级多粒度）

本目录是问题一的交付物，内容包括 100 条样本的逐词三模态特征文件、两份全量汇总表、
3 个典型样本的逐词对应表、异常台账，以及 16 项机器核验的报告。本文说明这批文件的
组织与用法：各文件存放的内容、特征文件的字段与读取方式、不填充约定下数组长度与
有效性的表示方法、核验覆盖到哪些量，以及由赛题附件 1 重跑出这批产物所需的步骤与
环境约束。使用特征文件之前，需要先读第 2.5 节与第 4 节。

---

## 0 交付概要

| 项目 | 值 |
|---|---|
| 样本数 | 100（与附件 1 一一对应，未增未删） |
| 词单元总数 | 1926（空白块 1932） |
| 三模态共同有效词 | 见 `results_100.csv` 的 `valid_length` 列 |
| 逐词特征维度 | 文本 768 + 语音 50 + 视觉 104 = 922 |
| 核验 | 16 项，执行 15 项（V12 为开发期回归，默认跳过）；断言 20 686 条，失败 0 条 |
| 交付体积 | 19 164 936 B（不含 `.cache/` 与 `logs/`，二者可重跑再生） |
| 样本与标签删除数 | **0**（异常只登记，`counts_as_deletion ≡ 0`） |

---

## 1 目录清单

```
data/q1_v2/
├── features/                     100 个 NPZ 逐词特征文件，本问的核心交付物
├── manifest.csv                  59 列全量汇总表（含执行期列：内存、耗时、错误）
├── results_100.csv               56 列全量汇总表（manifest 去掉执行期三列）
├── config_snapshot.json          本次运行的有效配置
├── README_问题一交付与验证.md      本文档
├── correspondence/
│   ├── _meta.json                选样判据、坐标口径、CSR 独立复算结果
│   └── *_correspondence.csv      3 个典型样本的逐词对应表
├── audit/
│   ├── anomaly_ledger.csv        异常台账（29 条，counts_as_deletion 恒为 0）
│   └── v2_route_100.csv          逐样本路由判定
├── metadata/
│   ├── registry.json             样本登记（五方集合相等的基准之一）
│   ├── route_policy.json         路由政策说明书（含两类恒不发出类别的原因）
│   ├── model_assets.json         模型资产身份与 SHA-256
│   ├── environment.json          运行环境快照
│   └── protected_baseline.json   既有交付物的基线快照（供 M2 核验）
├── reports/
│   ├── verify_report.json        16 项核验的机器可读报告
│   ├── verify_report.md          同一报告的可读版
│   ├── q1v2_size_ledger.json     体积台账
│   ├── q1v2_size_ledger.md       同一台账的可读版
│   ├── q1v2_identity_scan.json   身份红线扫描回执
│   ├── prepare_summary.json      准备阶段台账
│   └── stage_text/align/media/tables.json   四个阶段的执行台账
├── .cache/                       分模态原始观测缓存（可重建，不计提交体积）
└── logs/                         逐样本执行日志（可重建，不计提交体积）
```

`.cache/` 与 `logs/` 不含任何不可再得的特征或标注，二者由 `run_q1v2_all.py` 完整重跑
一次即可再生，因此不计入提交体积，这与 `reports/q1v2_size_ledger.json` 的口径一致。
目录树中除 `.cache/` 与 `logs/` 之外的 125 个文件计入提交，合计 19 164 936 B。

---

## 2 特征文件规范

### 2.1 格式与读取方式

每个样本一个 `.npz`，字段 85 个，读取统一用

```python
numpy.load(path, allow_pickle=False)
```

完成。文件内不含 pickle，读取过程不执行任何反序列化代码，这是提交物不含可执行内容
的前提。

### 2.2 字段分组

| 组 | 代表字段 | 说明 |
|---|---|---|
| 身份与来源 | `sample_key` `video_id` `clip_id` `official_text` `source_video_relpath` `source_video_sha256` | 与附件 1 的对应关系与来源指纹 |
| 配置指纹 | `schema_version` `config_sha256` `roberta_model_sha256` `mediapipe_model_sha256` `opensmile_config_sha256` | 复现时逐项比对，任一不符即说明环境已偏离 |
| 时间与覆盖 | `shared_t0_sec` `audio_presentation_coverage_sec` `video_presentation_coverage_sec` `av_common_coverage_sec` `original_effective_duration_sec` | 共享零点与各模态的呈现覆盖区间 |
| 词级身份 | `words` `word_char_start` `word_char_end` | 词单元及其在 `official_text` 中的字符区间 |
| 词级时间 | `word_start_sec` `word_end_sec` `word_time_valid` | 半开区间 `[start, end)` 与结构合法性掩码 |
| 四条掩码 | `text_word_valid` `word_audio_valid` `word_vision_valid`（另有帧级 `raw_video_face_feature_valid`） | 各模态在本词上是否有真实观测 |
| 词级特征 | `text_word_feat (L,768)` `word_audio_feat (L,50)` `word_vision_feat (L,104)` | `L` 为该样本自身的词单元数 |
| 原生观测 | `raw_audio_lld_values (K,25)` `raw_audio_lld_center_sec` `raw_video_blendshape_values (M,52)` `raw_video_pts_sec` | 未聚合的逐窗与逐帧观测，供独立复算 |
| 冻结名序 | `raw_audio_lld_feature_names (25,)` `raw_video_blendshape_names (52,)` | 逐维物理量的名字，顺序已冻结 |
| CSR 溯源 | `audio_word_feature_indptr/indices` `vision_word_feature_indptr/indices` | 词到源观测的归属，见 2.4 |
| 文本子词 | `roberta_token_ids` `roberta_token_offsets` `roberta_word_token_indptr/indices` | RoBERTa 子词与词单元的归属 |
| 对齐诊断 | `aligner_slot_count` `aligner_slot_to_word_index` `alignment_zero_duration_word_count` `alignment_mapping_fail_count` `word_without_slot_count` | 结构诊断量 |
| 结论字段 | `alignment_mode` `alignment_granularity` `text_av_time_mapping_status` `text_audio_correspondence` `content_assertion` `audio_speech_valid` `word_time_basis` | 见 2.5，使用前需先读该节 |
| 溯源 | `decode_provenance_json` | 解码路径与重采样的完整记录 |

### 2.3 维度与聚合口径

| 模态 | 原生观测 | 原生维度 | 词级聚合 | 词级维度 |
|---|---|---|---|---|
| 文本 text | RoBERTa 子词 | 768 | 该词所辖子词的均值 | 768 |
| 语音 audio | openSMILE eGeMAPSv02 LLD 窗 | 25 | `[mean, std(ddof=0)]` | 50 |
| 视觉 vision | MediaPipe FaceLandmarker blendshape 帧 | 52 | `[mean, std(ddof=0)]` | 104 |

标准差取总体口径（`ddof=0`），以免单观测词出现 $0/0$。

一个原生观测是否属于某个词，取决于它的时间标签是否落在该词的半开区间内：

```
center_sec ∈ [word_start_sec, word_end_sec)
```

本方案不做插值或重采样，参与聚合的只有落在区间内的观测本身，不把观测拉到统一网格
上取值。落点判定之外，另有三条不予补足的规则：其一，词区间非法或时长为零时，该模态
在本词判为无效，特征整行为零；其二，区间内没有任何有效观测时同样判为无效，不用邻近
观测顶替；其三，观测含非有限值时该词判为无效，不静默跳过该维。

### 2.4 CSR 溯源结构：怎么从词回到原始观测

`audio_word_feature_indptr`（长度 $L{+}1$）与 `audio_word_feature_indices` 构成标准的
压缩稀疏行结构：

```python
idx = audio_word_feature_indices[indptr[i]:indptr[i + 1]]
# idx 是第 i 个词所对应的 raw_audio_lld_values 的行号
```

视觉侧的 `vision_word_feature_indptr` 与 `vision_word_feature_indices` 同理，指向
`raw_video_blendshape_values` 的行号。每个有效词的特征都能回到具体的窗口号与帧号，
核验项 V4 正是用这套结构回捞源观测并重算均值与标准差。无效词
（`word_time_valid == 0`）的 CSR 区间为空，三路特征整行为零。

### 2.5 结论字段

「存在对齐区间」不等于「已确认文本内容与讲话对应」，两者性质不同，因此拆成独立字段
分别记录，不合并为同一个结论：

| 字段 | 取值 | 判定依据 | 本批分布 |
|---|---|---|---|
| `text_av_time_mapping_status` | `word_valid` / `word_partial` / `clip_only` / `unavailable` | 机器：官方词的时间区间够不够用 | `word_partial` 50、`word_valid` 47、`clip_only` 3 |
| `text_audio_correspondence` | `not_asserted` / `no_speech` | 人工听辨（唯一例外见下） | `not_asserted` 98、`no_speech` 2 |
| `content_assertion` | `uint8` | 需外部人工证据 | 恒为 0 |
| `audio_speech_valid` | `0` 确认数字静音 / `-1` 证据不足 | — | 从不取 1 |
| `alignment_mode` | 五类词汇表 | — | `UNCERTAIN_REVIEW` 98、`TRI_MODAL_WITH_AUDIO_CONTENT_INVALID` 2；`TRI_MODAL_WORD_VALID` 与 `AV_VALID_TEXT_UNALIGNED` 从不写出 |

`text_audio_correspondence` 只有 `no_speech` 一类由机器判定：int16 波形逐采样全零属于
编码事实，核验 V14 重新解码一遍予以确认。其余情况本套代码一律写 `not_asserted`。

下游使用 `content_assertion` 与 `text_audio_correspondence` 时，应按「尚无证据」对待，
不能按「已确认一致」对待。本套代码只给出机器可判的结论。

---

## 3 汇总表结构

两份表都是 100 行，一行一样本，区别在于执行期信息：`manifest.csv` 共 59 列，比
`results_100.csv` 多出 `process_rss_bytes`、`runtime_sec`、`error` 三列；后者共 56 列，
是 manifest 去掉这三列后的结果视图。

赛题要求的五项对应列：

| 赛题要求 | 对应列 |
|---|---|
| 样本编号 | `sample_key`（另有 `video_id` / `clip_id`） |
| 模态类型 | `modalities`（另有 `text_present` / `audio_present` / `video_present`） |
| 原始有效时长 | `original_effective_duration_sec`（另有 `ffprobe_format_duration_sec` 与各模态覆盖区间） |
| 特征维度 | 由 `schema_version` 冻结：文本 768 / 语音 50 / 视觉 104（逐词合计 922） |
| 对齐粒度 | `alignment_granularity`（`word` 97 / `clip` 3） |

其余列给出词单元数、LLD 窗数、视频帧数、人脸有效率、词并集覆盖率、三模态有效词数、
路由判定与来源指纹等，便于逐样本核对。

---

## 4 填充规则：本方案不做填充

赛题要求说明填充规则，本交付物的约定是不做填充，具体有三条。

其一，逐样本变长。词级数组的第一维 `L` 是该样本自己的词单元数，不同样本可以不同，
不做补齐。

其二，有效位置由掩码声明。四条 `*_valid` 掩码逐位置标记该模态在本词上是否有真实
观测。被标为 0 的位置，特征行是零值，其语义是零观测，并不代表某个被填充的真值。

其三，长度由显式列声明。`text_sequence_length`、`audio_observation_length`、
`video_frame_count`、`valid_length` 分别给出各模态长度与三模态共同有效词数。

与补齐到统一长度相比，这一表示在核验上更强。等长填充会让填充位置的零与真实的零观测
混在同一数组里，此后按位置做的统计无法区分两者；变长表示下填充位置根本不存在。代价
是下游必须按掩码处理不等长输入，这是问题二与问题三需要显式面对的建模问题。

`valid_length` 的定义：

```
valid_length = (text_word_valid & word_audio_valid & word_vision_valid).sum()
```

---

## 5 核验结论

16 项核验执行 15 项，共 20 686 条断言，失败 0 条。完整报告见 `reports/verify_report.json`
与 `reports/verify_report.md`。断言数较多的几项如下：

| 核验 | 断言数 | 核验内容 |
|---|---|---|
| V1 五方集合相等 | 300 | registry / features / manifest / results / logs 的键集合双向相等 |
| V2 NPZ schema 一致 | 8200 | 100 个 NPZ 的字段、形状、dtype、维度逐项一致 |
| V4 词级聚合可回捞 | 3006 | 用 CSR 回捞源观测重算 mean / std(ddof=0)，并按 `s≤c<e` 独立重选索引再比一次 |
| V6 外部 ffprobe 交叉 | 100 | 解码覆盖与容器声明相容（容差 0.15 s） |
| V7 权重真实性 | 14 | 只读探测：RoBERTa 逐字节 SHA、MediaPipe 枚举与逐帧 detect、openSMILE 逐维名序 |
| V10 异常只登记不删除 | 229 | 台账 29 行的 `counts_as_deletion` 全为 0 |
| V13 对齐结构硬断言 | 5787 | 词单元 1926、空白块 1932（文本的纯函数），`mapping_fail=0`、无槽词 = 0 |
| V14 数字静音位精确 | 14 | 对 2 条无语音样本重新解码一遍，确认 int16 逐采样为零 |
| M1 媒体全量硬锚点 | 3 | 视频帧 23 241、人脸检出帧 16 806、无检出人脸样本 24 |
| M2 既有交付未被破坏 | 530 | 受保护基线：新增 0 / 删除 0 / 实质改动 0 |
| V12 对照路由对拍 | 0 | 未提供对照文件，默认关闭（开发期回归项） |

### 5.1 软区间越界记录

V13 中有一条软区间（非断言项）越界，记录如下：含零时长词的样本数实测为 53，软区间为
`[30, 44]`。报告的归因是本套对齐固定 `remove_instant_words=False`（保留瞬时词），
说话人为空的片段必然产出一批零时长词；其中全部词都为零时长的样本共 3 条，即
`-mJ2ud6oKI8` 的第 1、2 段与 `-ri04Z7vwnc` 的第 1 段，报告把这 3 条无语音样本记作
这一部分的来源。本批零时长词合计 207 个，与结构有效的 1719 个词相加即为词单元总数
1926。

该软区间只用于记录差异，不参与通过或失败的判定。与它并列的对照值来自一次旧的对齐
trace，两者的数值不可互相引用。V13 的硬断言只落在客观量上，即文本的纯函数
（词单元 1926、空白块 1932）与两条零失败性质。对齐器产出的槽数（本批 1929）取决于
本次对齐运行，把它写成断言，等于用一次运行结果充当客观约束。

---

## 6 复现步骤

### 6.1 环境

依赖锁定见 `code/requirements-q1v2.txt`。venv 路径必须是纯 ASCII：`opensmile` 的
Python 绑定在初始化时执行 `bytes(config_file, "ascii")`，包路径含非 ASCII 字符会让
`Smile()` 构造抛 `UnicodeEncodeError`。该约束在 `code/q1v2_contract.py` 的
`_resolve_work_root()` 中实现（含非 ASCII 时退到 ASCII 兜底目录），实测环境见
`metadata/environment.json`。

### 6.2 命令

```
pip install -r code/requirements-q1v2.txt

# 全流程（已存在的产物默认跳过，支持断点续跑）
python code/run_q1v2_all.py --stage=all --isolate

# 只重建某个阶段 / 某个样本
python code/run_q1v2_all.py --stage=media --sample-key "-THoVjtIkeU$_$6"

# 导出典型样本逐词对应表（只读 NPZ，不重解码）
python code/q1v2_correspondence.py --count 3

# 16 项机器核验
python code/run_q1v2_all.py --stage=verify

# 提交体积与身份红线扫描
python code/package_check.py --sanitize
```

完整说明见项目根目录的 `运行指令.md` 中「问题一 v2」一节。

### 6.3 复现是否成功的首选判据

V14 的端到端复现锚点对解码路径、重采样与 openSMILE 特征窗划分的偏差最敏感：它要求
两条无语音样本的覆盖末端、LLD 窗数、词单元数与路由结论逐项相等，且重解码后 int16
逐采样为零。该核验通过，即可认为整条抽取链在等价环境中复现成功。

### 6.4 运行期网络

运行期不访问网络。RoBERTa 强制 `local_files_only=True`；MediaPipe 权重预取一次后离线
加载；whisper 检查点以本地路径传入，不用模型名。若传入模型名，缓存缺失时会静默联网
下载一个与冻结值不同的检查点，而落盘的版本号看起来仍然是对的。

---

## 7 数据使用与结论强度

(1) 数据来源。全部原始素材来自赛题附件 1（CMU-MOSEI 族数据集）。未引入任何外部数据、
预训练标签或第三方标注结论。所用预训练权重均为公开模型，身份由 SHA-256 锁定并在核验
中逐字节比对。

(2) 样本与标签完整性。100 个样本全部保留，未删除任何样本或标签。异常只登记、不剔除，
`counts_as_deletion` 恒为 0；异常阈值从不作为删除样本或标签的依据。

(3) 结论强度。本交付物不对「官方文本与视频中实际讲话内容是否对应」作任何断言。路由
设计中有两个类别恒不发出，其原因即在于证据等级不足，与措辞是否保守无关。

(4) 可核验性。每个有效词的特征都能通过 CSR 回到具体的原始窗口号与帧号；汇总表的
每一列都有明确的产生规则；核验报告的每一项都给出断言数与判据。
