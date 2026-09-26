# 未对齐三模态提取 + 统一时间轴对齐（问题1）

本目录新增的这套代码，把「特征提取」与「时序对齐」彻底拆成两步：

```
                          ┌─────────────────────────────────────────────┐
                          │  第一步：三个模态各自独立提取，互不干扰      │
                          └─────────────────────────────────────────────┘
  label-100.xlsx ──► unaligned_text.py    ──► (W, 768)  词级向量 + 词串
  *.mp4 音轨     ──► unaligned_audio.py   ──► (T, 74)   逐帧 @ 20 Hz + 实测 pts
  *.mp4 视频流   ──► unaligned_vision.py  ──► (T, 35)   逐帧 @ 15 Hz + 实测 pts
                          └─────────────────────────────────────────────┘
                                            │
                                            │
                          ┌─────────────────▼───────────────────────────┐
                          │  第二步：官方文本强制对齐到音轨              │
                          │  取逐词实测半开区间 [start, end)             │
                          │  按对齐质量给出逐样本证据路由档位            │
                          └─────────────────────────────────────────────┘
  word_align.py ─────────► word_level（全词对齐+无零时长词）/ clip_level
                          实测词时刻落盘，供对齐阶段按档位替换文本时间基准
                                            │
                          ┌─────────────────▼───────────────────────────┐
                          │  第三步：投影到同一根秒轴，离散成 50 槽      │
                          └─────────────────────────────────────────────┘
  align_multimodal.py ──► (50, 768) / (50, 74) / (50, 35) + valid_mask
                          + 槽↔时间可逆映射 + 逐槽来源单元（可溯源）
                                            │
  timeline_visualize.py ──► 共享时间轴图：视频帧 / 语音振幅 / 文本词 / 对齐槽 / 三类特征
```

这样拆的好处：三个模态的提取互不依赖，任一模态改进（换模型、换采样率、修 bug）
都不需要重跑其它两个；对齐规则也可以单独替换、单独消融。

---

## 一、新增文件清单

| 文件 | 作用 |
|---|---|
| `unaligned_common.py` | 公共工具层：ffmpeg 管道解码、真实 PTS 抽帧、未对齐文件读写、样本清单 |
| `unaligned_text.py` | 文本 → 词级 RoBERTa 向量 (W, 768)，**不做时间槽分配** |
| `unaligned_audio.py` | 语音 → 逐帧 74 维 @ 20 Hz + 实测 pts，**不做时间槽分配** |
| `unaligned_vision.py` | 视觉 → 逐帧 35 维人脸特征 @ 15 Hz + 实测 pts，**不做时间槽分配** |
| `word_align.py` | **词对齐模块**：官方文本 → stable-ts 强制对齐 → 逐词实测区间 + 证据路由档位 |
| `align_multimodal.py` | **对齐模块**：三模态归一到同一秒轴 → (50, D)，输出槽↔时间可逆映射 |
| `timeline_visualize.py` | 共享时间轴可视化 + 逐槽对应表 |
| `run_unaligned_all.py` | 一键跑通全流程 |

上一版的 `extractors/`、`main.py`、`alignment.py`、`report.py` **已从磁盘移除**，
本方案不依赖它们、也没有任何 `import` 指向它们。历史对照见下方「与上一版的差异」一节
（每个模块的文件头也各有一段同口径的说明）。

> **问题一的验证与交付**：`q1_verify.py`（13 项机器核验）、`q1_delivery.py`（四项交付物
> + 异常台账 + 体积核算 + 五类典型样本）、`face_probe.py`（「多人/远景」判据取证）、
> `q1_readme.py`（自动生成交付说明文档）是独立的一层，产物在 `data/q1_delivery/`。
> 逐条要求与实现见
> [`../data/q1_delivery/README_问题一交付与验证.md`](../data/q1_delivery/README_问题一交付与验证.md)
> ——该文档由 `python q1_readme.py` 自动生成，正文数字全部现读现算，
> 只此一份、不另存副本（避免两份文档各改各的而互相矛盾）。

---

## 二、对齐方案与对齐规则

### 2.1 一句话规则

> 以片段真实时长 `T` 为公共时间轴，把 `[0, T]` 等分为 `L = 50` 个槽（槽宽 `T/L`）；
> 每个未对齐单元按其**实测时间戳**归属到所在槽，同槽内特征取**均值**即该槽特征；
> 空槽按策略补齐并在 `valid_mask` 中标记为 `False`。

### 2.2 三条关键约定

1. **用「秒」而不是「数组下标」**
   三个模态采样率不同（音频 20 Hz、视觉 15 Hz、文本按词），片段时长也不同（2.26 ~ 29.29 s）。
   若按数组下标等分（既有 `utils.resample_feature_sequence` 的做法），
   会把 50 ms 的音频帧与 66.7 ms 的视觉帧当作等长，并抹平丢帧/VFR 造成的不均匀间隔。
   按秒归属则天然正确。

2. **三模态共用同一个 `T`**
   这是「共享时间轴」能成立的前提。`T` 取自 ffprobe 的视频流时长，
   在提取阶段就已写进各模态的 `meta.duration_used`，对齐阶段直接复用——
   因此**对齐这一步不需要原始视频，仅凭未对齐特征文件就能独立重跑**。

3. **空槽策略按模态的疏密特性分别设定**

   | 模态 | 策略 | 理由 |
   |---|---|---|
   | 文本 | `zero` + `valid_mask=False` | 一段 5 s 的话只有十几个词，天然填不满 50 槽。「没有词的槽就是没有文本内容」，补零才诚实；插值会把相邻词的语义抹到空白处，制造原文不存在的「词」 |
   | 语音 | `interpolate` | 20 Hz 远快于槽率（约 2.2 帧/槽），空槽只可能出现在音轨短于视频处 |
   | 视觉 | `interpolate` | 15 Hz 约 1.6 帧/槽，同上 |

### 2.3 槽 ↔ 时间的可逆映射（问题3 定位证据的前提）

槽是 `[0, T]` 上的等分区间，映射是闭式的、双向可逆：

```
槽号 → 时间： [k·T/L, (k+1)·T/L)
时间 → 槽号： min(L-1, floor(t·L/T))
```

每个样本的 `T` 随文件落盘，因此拿到任一槽号即可还原其真实时间区间，反之亦然。
这使问题3「模型关注哪段时间」的定位结果能直接投影回问题1 的时间轴。

比闭式公式更强的是**溯源链**：每槽同时记录「哪些未对齐单元落进了它」，
于是 `槽 → 单元下标 → 该单元的原文词 / 该帧的时间戳` 这条链是完整的，可一路追到原始证据。

实测往返自检（样本 `-3g5yACwYnA_13`，`T=5.51 s`，`L=50`）：

```
槽 0  -> [0.0000, 0.1102) 中心0.0551 -> 回投影槽 0   OK
槽 24 -> [2.6448, 2.7550) 中心2.6999 -> 回投影槽 24  OK
槽 49 -> [5.3998, 5.5100) 中心5.4549 -> 回投影槽 49  OK
```

溯源示例：

```
槽 1  [0.110~0.220s] <- 未对齐单元 [0] <- 词 'They'
槽 4  [0.441~0.551s] <- 未对齐单元 [1] <- 词 "'ve"
槽 7  [0.771~0.882s] <- 未对齐单元 [2] <- 词 'been'
```

### 2.4 文本为什么没有真实时间戳（必须说明的假设）

附件1 只提供整句转写文本，附件2 也不存储逐词时间戳。因此词级时刻只能来自**假设**，
本流水线把它显式标注出来，绝不伪装成实测值：

```
time_basis = "uniform_assumption"
词 i 的中心时刻 = (i + 0.5) · T / W
```

这是「在已知片段总时长下，词在时间上均匀分布」的功效最弱假设。
它使文本可以与音频/视觉落在同一根秒轴上用于可视化，
但必须与音频/视觉的 `time_basis = "measured_pts"`（解码器实测）区分开——
后者才是硬证据。每个文件的 `meta.time_basis` 字段如实记录这一点。

---

## 三、特征文件规范

### 3.1 未对齐文件：`data/unaligned_features/<modality>/<sample_id>.npz`

`sample_id` 形如 `-3g5yACwYnA_13`（`video_id_clip_id`）。
注意有 **4 个 `video_id` 自身就含下划线**（`-I_e4mIh0yE`、`-lzEya4AM_4`、`-tANM6ETl_M`、`-wMB_hJL-3o`），
这 4 个 video_id 下辖 **6 条样本**，
因此从 `sample_id` 反解必须按**最后一个**下划线切分（`sid.rsplit("_", 1)`），
不能用 `split("_", 1)`——写错会把 video_id 内部的下划线当成分隔符（波及 18/300 行）。
（「4」是 video_id 数、「6」是样本数，两者别混——本行曾误写成「6 个 video_id」。）

| 键 | 类型 | 说明 |
|---|---|---|
| `features` | `(T, D) float32` | 逐单元特征。文本 D=768，语音 D=74，视觉 D=35 |
| `pts` | `(T,) float32` | **每个单元的时间戳（秒）**，相对片段起点 |
| `modality` | str | `text` / `audio` / `vision` |
| `meta` | JSON 字符串 | 时间基准、时长、维度构成、投影指纹、代码指纹等 |
| `extra__*` | JSON 字符串 | 附属记录：`words`（词串）、`envelope`（振幅包络）、`face_flags` 等 |

时间基准约定（未对齐阶段最重要的字段）：

| 模态 | `time_basis` | pts 来源 |
|---|---|---|
| 文本 | `uniform_assumption` | 词在 `[0,T]` 均匀分布的假设值 |
| 语音 | `measured_pts` | `pts[k] = k·hop/sr`（hop=800@16k，等间隔 50 ms） |
| 视觉 | `measured_pts` | `cv2 CAP_PROP_POS_MSEC`，解码器逐帧实测，允许不均匀 |

### 3.2 对齐文件：`data/aligned/<sample_id>.npz`

| 键 | 类型 | 说明 |
|---|---|---|
| `text_features` / `audio_features` / `vision_features` | `(50, D) float32` | 对齐后的定长张量 |
| `text_valid` / `audio_valid` / `vision_valid` | `(50,) bool` | 该槽是否由真实单元直接支撑 |
| `slot_edges` | `(51,) float32` | 50 个槽的边界时刻，与 `duration` 一起构成可逆映射 |
| `duration` | float32 | 公共时间轴长度 `T` |
| `meta` | JSON 字符串 | 每模态的单元数、有效槽数、`slot_unit_index`（逐槽来源单元下标） |

### 3.3 汇总文件：`data/aligned/aligned_50.npz`（附件2 口径）

顶层按模态分字段，与附件2 / 附件4 的字段与维度组织一致：

```
text_features    (N, 50, 768) float32      text_valid    (N, 50) bool
audio_features   (N, 50,  74) float32      audio_valid   (N, 50) bool
vision_features  (N, 50,  35) float32      vision_valid  (N, 50) bool
text_lengths / audio_lengths / vision_lengths  (N,) int64
sample_id (N,)   duration (N,)   n_slots
```

维度核对（与附件2 完全一致）：文本 **768**、语音 **74**、视觉 **35**，序列位置 **50**。

### 3.4 汇总表：`data/aligned/align_summary.csv`

问题1 要求的那张表，每条样本每个模态一行：

| 字段 | 含义 |
|---|---|
| `sample_id` / `official_id` | 项目口径 ID / 附件2 口径 ID（`video$_$clip`） |
| `modality` | `text` / `audio` / `vision` |
| `clip_duration_sec` | 公共时间轴长度 `T`（= 片段真实时长） |
| `modality_span_sec` | 该模态实际覆盖到的时间（末单元时刻） |
| `raw_units` | 未对齐单元数（词数 / 帧数） |
| `feature_dim` | 特征维度 |
| `valid_slots` / `n_slots` | 有效槽数 / 总槽数 |
| `align_granularity_sec` | **对齐粒度 = 槽宽 = T/50** |
| `units_per_slot` | 平均每槽落入的原始单元数 |
| `time_basis` | `uniform_assumption` / `measured_pts` |

### 3.5 槽↔时间映射表：`data/aligned/slot_time_map.csv`

逐样本逐槽给出 `槽号 / 时间区间 / 中心时刻 / 三模态各有多少单元落在该槽 / 各模态该槽是否有效`。
问题3 定位到的槽号可以直接在这张表里读回真实时间区间。

---

## 四、运行说明

### 4.1 环境

- Python + `numpy` / `pandas` / `opencv-python` / `librosa` / `torch` / `torchvision` / `facenet-pytorch` / `transformers` / `matplotlib`
- **ffmpeg 与 ffprobe 必须在 PATH 中**（音频解码与时长探测都依赖它）
- 无 GPU 也能跑（CPU 推理），`--threads` 可指定 torch 线程数

### 4.2 一键全量（约 35 分钟，视觉占绝大部分）

```bash
cd D:\23届建模\code
python run_unaligned_all.py --threads 8
```

带典型样本出图：

```bash
python run_unaligned_all.py --threads 8 --timeline
```

### 4.2.1 视觉环节的逐样本看门狗（重要）

视觉提取依赖 torch 与 OpenCV 的原生线程，**实测出现过非确定性卡死**：
本批 100 条中曾有 1 条在批量运行里 8 核满载空转数小时不返回，
而把同一条样本单独重跑只需不到 1 秒（即与输入数据无关，属原生库的线程竞态）。
这类卡死发生在原生代码内部，Python 进程内的计时器无法中断它。

因此 `run_unaligned_all.py` 默认把视觉环节拆成**每样本一个子进程**，
由父进程按墙钟超时收割（`--sample-timeout`，默认 900 s）。效果：

- 任何一条卡死只影响它自己，其余样本照常完成；
- 超时的样本被明确记入日志（含可直接粘贴的重跑命令），不会拖着一整批任务无声停摆；
- 提取器默认跳过已存在的产物，所以**重跑本步就是断点续跑**。

代价是每样本多约 2 s 的解释器与模型加载。若确认环境稳定，可用
`--sample-timeout 0` 退回单进程整批运行（更快）。

### 4.3 分步运行（每步可独立重跑）

```bash
python unaligned_text.py                     # 文本（约 1.5 s/条）
python unaligned_audio.py                    # 语音（约 1.5 s/条）
python unaligned_vision.py --threads 8       # 视觉（约 135 ms/帧，全量约 27 分钟）
python word_align.py                         # 词对齐（实测约 0.8 s/条，全量约 1.5 分钟）
python align_multimodal.py                   # 对齐（< 1 s/条，读上一步的路由）
```

先小批量验证链路：

```bash
python run_unaligned_all.py --limit 3 --overwrite
```

### 4.4 可视化

```bash
# 画指定样本
python timeline_visualize.py -3g5yACwYnA_13 --correspondence

# 自动挑一条典型样本（文本非空、人脸率高、有声比例适中、时长接近中位数）
python timeline_visualize.py --pick typical --correspondence
```

> 注意：本批 MOSEI 的 `video_id` 以 `-` 开头（如 `-3g5yACwYnA`），直接当位置参数会被
> argparse 当成选项。脚本已内置 argv 规范化，也可以显式写成 `--id=-3g5yACwYnA_13`。

输出：
- `data/timeline/<sample_id>_timeline.png` —— 共享时间轴图
- `data/timeline/<sample_id>_correspondence.csv` —— 逐槽对应表

### 4.5 常用参数

| 参数 | 作用 |
|---|---|
| `--only <子串>` | 只处理 sample_id 含该子串的样本 |
| `--limit N` | 只处理前 N 条 |
| `--overwrite` | 覆盖已存在的产物（默认跳过，便于断点续跑） |
| `run_unaligned_all.py --sample-timeout 900` | 视觉环节每样本的墙钟超时（0 = 关闭看门狗） |
| `unaligned_audio.py --rate 20` | 语音帧率，默认 20 Hz（官方口径） |
| `unaligned_vision.py --fps 15 --dim 35` | 视觉帧率与维度 |
| `word_align.py --out-dir` | 词对齐记录输出目录，默认 `data/word_align/` |
| `word_align.py --overwrite` | 忽略缓存，全部重跑对齐（默认命中即复用） |
| `align_multimodal.py --word-align-dir` | 词对齐记录目录，默认 `data/word_align/` |
| `align_multimodal.py --no-word-align` | 关闭实测词时间（消融/对照用，文本一律均匀假设） |
| `run_unaligned_all.py --skip-word-align` | 跳过词对齐环节（无 stable-ts/whisper 环境时的降级路径） |
| `align_multimodal.py --slots 50` | 对齐槽数，默认 50 |
| `align_multimodal.py --method time_bin\|interp` | 对齐方法：按时间戳归属+均值 / 按时间插值 |
| `align_multimodal.py --duration video\|max_pts` | 公共时间轴取视频流时长 / 各模态末帧最大值 |

---

## 五、共享时间轴图说明了什么

`data/timeline/<sample_id>_timeline.png` 自上而下五段**共用同一根秒轴**：

| 段 | 内容 |
|---|---|
| 1 | **视频帧条**：在各自实测时刻贴出该时刻的真实视频帧，红竖线标出代表帧 |
| 2 | **语音振幅**：100 Hz RMS 包络曲线，底色标出 VAD 判定的有声区间 |
| 3 | **文本词**：每个词按自身时刻排布，分泳道避免重叠 |
| 4 | **对齐槽**：3×50 格，显示各模态哪些槽由真实单元直接支撑（深色）还是空槽（浅灰） |
| 5 | **三类对齐特征**：三模态 `(50, D)` 热力图，逐维度归一化，无效槽用浅灰区分 |

因此「某一秒」是一条竖线，穿过视频帧、振幅、词、槽与全部三类特征——
这就是「共享时间轴」的落地。图顶部标注了片段时长、槽数与**对齐粒度（槽宽）**。

---

## 六、绕过既有代码的三个地雷（实测确认）

既有 `utils.py` 的三处问题会静默污染特征，本流水线全部绕开：

1. **时长口径**：`utils.get_video_duration()` 用 `帧数 / fps` 估算，对本批 mp4 严重偏大。
   例：`-3g5yACwYnA/2.mp4` 容器声称 424 帧、30 fps → 14.13 s，实际只能解出 278 帧，真实时长 9.39 s。
   **后果**：整根时间轴被拉长 1.5 ~ 2.6 倍。
   本流水线改用 ffprobe 的**流时长**，并以逐帧 PTS 为准。

2. **音频临时文件冲突**：`utils.extract_audio_from_video()` 的临时 wav 仅以 `clip_id.wav` 命名，
   而本批 100 条样本只有 **19 个不同的 clip_id**（clip `"2"` 出现在 12 个不同视频中），
   "已存在则跳过" 会让不同视频的音频互相覆盖后被静默复用。
   **实测**：`-ri04Z7vwnc_0` 与 `-wny0OAz3g8_0` 的音频特征**逐字节相同**。
   本流水线用 ffmpeg 管道直接解码到内存，完全不落临时文件。

3. **抽帧方式**：`utils.safe_read_video_frames()` 会把整段视频所有帧读进内存
   （720p × 250 帧 × 3 通道 ≈ 0.7 GB/片段），且抽帧间隔按名义 fps 计算。
   本流水线改为**流式逐帧 + 真实 PTS 触发**，内存占用与视频长度无关，天然容忍丢帧/VFR。

---

## 七、关键数值（实测）

| 项 | 值 |
|---|---|
| 样本数 | 100 |
| 总时长 | 787.5 s（13.1 分钟） |
| 时长范围 | 2.26 ~ 29.29 s（中位 6.72 s） |
| 文本 | 768 维；词数 5 ~ 78（中位 20） |
| 语音 | 74 维 @ 20 Hz；45 ~ 583 帧（中位 132） |
| 视觉 | 35 维 @ 15 Hz |
| 对齐 | 50 槽；对齐粒度 = T/50（本批 45 ~ 586 ms） |

### 74 维语音特征的构成（逐段写入 `meta.dim_layout`）

| 区间 | 内容 | 维数 |
|---|---|---|
| 0–19 | MFCC 20 维 | 20 |
| 20–39 | 一阶差分 ΔMFCC | 20 |
| 40–59 | 二阶差分 ΔΔMFCC | 20 |
| 60 | log F0（pyin，清音帧取 `log(fmin)`） | 1 |
| 61 | 浊音概率（pyin voiced_prob） | 1 |
| 62 | log RMS 能量 | 1 |
| 63 | 过零率 ZCR | 1 |
| 64–67 | 谱质心 / 谱带宽 / 谱滚降 / 谱平坦度 | 4 |
| 68–72 | 谱对比度 5 维 | 5 |
| 73 | 有声/静音指示（VAD） | 1 |
| | **合计** | **74** |

> 官方 74 维来自 COVAREP（含 NAQ / QOQ / H1H2 等声门参数）。
> 本项目无 COVAREP，用上面这套「MFCC + 差分 + 韵律 + 频谱」的等价维度代理，
> **维度严格保持 74 与附件2 对齐**，构成逐段写进文件元数据以便复核。

### 视觉 35 维的来源

MTCNN(160) 裁人脸 → PIL Resize(224) → ImageNet 归一化 → ResNet-50 池化 2048 维
→ 固定随机投影（`W ∈ R^{2048×35}`，元素 `~ N(0, 1/√2048)`，seed=42）降到 35 维。

投影矩阵与上一版 `vision_extractor.py`（已移除）**逐元素完全相同**（md5 `992b29b991cd23f9…`），
因此「未对齐 → 对齐」的结果可与上一版「直接对齐」路径直接对照。
矩阵 md5 随每个文件落盘，任何一次运行都能验证用的是不是同一个投影。
缺失人脸的帧按相邻有效帧线性插值补齐，并逐帧记录 `face_flags` 供核对。

---

## 八、产物体积

| 目录 | 内容 |
|---|---|
| `data/unaligned_features/text/` | 100 × `.npz`（词级 768 维，压缩后很小） |
| `data/unaligned_features/audio/` | 100 × `.npz`（74 维 + 100 Hz 包络） |
| `data/unaligned_features/vision/` | 100 × `.npz`（35 维） |
| `data/aligned/` | 100 个单样本 `.npz` + `aligned_50.npz` + 两张映射/汇总 CSV |
| `data/timeline/` | 时间轴 PNG + 逐槽对应表 |

提交材料需 ≤ 50 MB，`data/timeline/` 的 PNG 与日志不属于算法代码，可按需裁剪。
