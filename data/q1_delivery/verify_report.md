# 问题一 机器核验报告

- 生成时间：2026-09-26 17:34:55
- 代码指纹：`7e4527279a76`
- 样本数：100    时间槽数：50
- **总体结论：全部通过**（13/13 项通过）

## 诚实性前置声明（读下面任何数字之前请先读这里）

1. **本数据集没有逐词人工时间戳真值。** 文本的时间轴逐样本二选一，来源由
   `alignment_mode` 字段标明：`word_level`（实测，stable-ts 强制对齐的区间中心，
   全词对齐且无零时长词）或 `clip_level`（均匀假设，词 i 中心时刻 = (i+0.5)·T/W）；
   音频/视觉的 `pts` 才是解码器实测值。因此本报告**只报告一致性检查与抽查结果**，
   **绝不声称对齐达到某个毫秒级平均误差**——那需要真值才能计算，这里没有。
   任何形如「平均误差 N 毫秒」的表述在本项目中都是无依据的。
   实测档位的准入证据是**官方词数是否被完整覆盖**这一可机检的弱证据，
   它不能证明逐词时刻正确，只能证明对齐器在该样本上放下了全部官方词。
2. **`*_valid=True` 只表示「该槽被分配了采样单元」，不等于该模态真实可用。**
   反例：`-mJ2ud6oKI8_1` 的 `vision_valid` 为 50/50 全 True，但其视觉特征能量恰为 0
   （全片未检出人脸）。判定模态可用性必须看 `face_ratio` / `voiced_ratio`。
3. **音频/视觉的空槽由插值填充，没有源帧。** 因此「每个聚合值都能溯源到源帧」
   这一条**无法 100% 满足**；本报告给出真实的可溯源率，并把这些槽标记为 `interp`，
   而不是假装全部可溯源。
4. **异常一律保留并标注，不删除样本。** 竞赛指南明确：置信度/质量阈值用于质量标注，
   不得作为删除样本的理由。本报告与交付台账中没有任何一条以质量为由剔除的样本。


## 检查结果总表

| 代号 | 检查项 | 结论 | 检查数 | 失败数 | 说明 |
|---|---|---|---|---|---|
| V1 | 覆盖完整性（双向集合相等） | 通过 | 100 | 0 | 六处样本集合互相相等（官方 label-100、三模态未对齐文件、单样本对齐文件、aligned_50.npz），均为 100 条，且无多余文件。 |
| V2 | 词区间单调且不越出视频范围 | 通过 | 900 | 0 | 文本 pts 严格递增且落在 (0,T] 内；100 条样本的字符区间能逐词精确回指规范化文本；100 条样本的词元索引严格递增；音频/视觉实测 pts 在 200/200 条样本上非递减。 |
| V3 | 序列长度与词索引一致 | 通过 | 1403 | 0 | 未对齐序列长度字段两两一致，三模态维度恒为 768/74/35，对齐张量形状 (100,50,D)，valid 掩码 valid.sum() == meta.valid_slots == (counts>0).sum()。 |
| V4 | 每个音视频聚合值都能找到来源帧 | 通过 | 12626 | 0 | 对每个有源单元的槽回捞源帧重算均值，最大相对偏差 5.949e-08（阈值 1e-06，余量 17×，出现在 -THoVjtIkeU_2/audio/槽37）；text 可溯源率 43.16%（插值槽 2842 个无源帧）；audio 可溯源率 98.82%（插值槽 59 个无源帧）；vision 可溯源率 98.54%（插值槽 73 个无源帧） |
| V5 | 槽几何与可逆映射 | 通过 | 300 | 0 | 100 条样本的槽边界严格递增、首尾为 0 与 T、等宽（最大槽宽浮动 3.55e-15 s），50 个槽的 time_to_slot 往返全部一致；越界单元被并入端点槽：末槽 7 个、首槽 0 个（其中真正越出 T 的 5 个）。 |
| V6 | 填充规则一致性 | 通过 | 401 | 0 | text 的 2842 个空槽全部恰为零向量；audio/vision 的 132 个空槽由插值填充（因此非零），三者 fill 口径与声明一致。 |
| V7 | 数值合规（无 NaN/Inf，dtype 正确） | 通过 | 1200 | 0 | 全部特征均为 float32 且无 NaN/Inf，掩码均为 bool；各模态 |x|max：文本 15.785、语音 7562.5、视觉 1.234（量级差 3 个数量级，故 V4 必须用相对容差）。 |
| V8 | 时长口径交叉核对（外部 ffprobe） | 通过 | 100 | 0 | 对 100 条样本重新执行 ffprobe，视频流时长与文件内 duration 的最大绝对偏差 3.30e-05 s（阈值 1e-3 s），耗时 5.4s。 |
| V9 | 预训练权重真实性（掩码补全，外部证据） | 通过 | 2 | 0 | 掩码补全 2/2 通过（top-5 命中语义正确词），词嵌入标准差 0.1315（随机初始化应≈1/√768≈0.036），hidden=768 vocab=50265；纯离线（local_files_only=True）加载成功；未预期键 0 个、缺失键 0 个。 |
| V10 | 跨样本重复检测 | 通过 | 300 | 0 | 三模态共 300 个特征数组按 md5 查重，未发现逐字节重复。 |
| V11 | 产物新鲜度（对齐晚于特征） | 通过 | 4 | 0 | 对齐产物最早写入时间 2026-09-26 15:33:48 晚于未对齐产物最新写入时间 2026-09-24 16:04:30，交付自洽（对齐确实建立在当前特征之上）。 |
| V12 | 异常普查（仅用于标注，不用于删除） | 通过 | 100 | 0 | A1_audio_digital_silence=2；A2_vad_zero_but_energy=1；A3_vision_face_all_fail=2；A4_vision_face_partial=4；A5_vision_energy_zero_but_valid=2；A6_audio_shorter_than_video=50；A7_audio_valid_slots_lt_50=46；A8_pts_beyond_axis=5；L |
| V13 | 实测词时间路由一致性 | 通过 | 336 | 0 | 7 条走实测路由、93 条保持均匀假设，逐条与落盘的时间基准字段严格对应；7 条实测样本均满足「全词对齐 + 零时长词为 0 + 区间不越界」三条准入条件，且其槽归属确由实测时刻推出。 各交付物计数：{'aligned_dir': 7, 'features_q1.npz': 7, 'summary_q1.csv': 7, 'alignment_q1.json': 7}。 |

## 逐项明细

### V1 覆盖完整性（双向集合相等） —— 通过

六处样本集合互相相等（官方 label-100、三模态未对齐文件、单样本对齐文件、aligned_50.npz），均为 100 条，且无多余文件。

> **限定**：本项只证明「一一对应」，不证明特征内容正确。

**V1b official_id 口径与官方一致** —— 通过：300 行的 official_id 全部等于 `video_id$_$clip_id` 口径。

> 历史缺陷：save_summary_csv 曾用 replace('_','$_$',1) 按**第一个**下划线切分，而 6 个 video_id 自身含下划线，波及 18 行；已改为按最后一个下划线切分（rsplit）。

<details><summary>证据</summary>

```json
{
 "counts": {
  "label-100（官方）": 100,
  "unaligned/text": 100,
  "unaligned/audio": 100,
  "unaligned/vision": 100,
  "aligned/单样本": 100,
  "aligned_50.npz": 100
 },
 "missing": {},
 "extra": {},
 "symmetric_difference": {}
}
```

</details>

### V2 词区间单调且不越出视频范围 —— 通过

文本 pts 严格递增且落在 (0,T] 内；100 条样本的字符区间能逐词精确回指规范化文本；100 条样本的词元索引严格递增；音频/视觉实测 pts 在 200/200 条样本上非递减。

> **限定**：本项检查的是**未对齐产物**里的文本 pts，它一律是均匀假设值（由 (i+0.5)·T/W **定义**而来），故由构造保证成立，只能证明实现与声明一致，**不作为词级时间正确的证据**。走实测路由的样本，其真正落进 50 槽的是实测词时刻——那一套时刻的合法性与一致性由 V13 单独核验。音频/视觉的 pts 为解码器实测值，该项才具实测意义。

<details><summary>证据</summary>

```json
{
 "text_pts": {
  "strict_increasing": 100,
  "in_range": 100
 },
 "char_spans": {
  "reconstruct_ok": 100,
  "monotone": 100
 },
 "token_index": {
  "strict_increasing": 100,
  "count_ok": 100
 },
 "av_pts": {
  "non_decreasing": 200,
  "samples": 200
 },
 "errors": []
}
```

</details>

### V3 序列长度与词索引一致 —— 通过

未对齐序列长度字段两两一致，三模态维度恒为 768/74/35，对齐张量形状 (100,50,D)，valid 掩码 valid.sum() == meta.valid_slots == (counts>0).sum()。

<details><summary>证据</summary>

```json
{
 "aggregate": {
  "n_samples": 100,
  "n_slots": 50,
  "text": {
   "shape": [
    100,
    50,
    768
   ],
   "dim": 768
  },
  "audio": {
   "shape": [
    100,
    50,
    74
   ],
   "dim": 74
  },
  "vision": {
   "shape": [
    100,
    50,
    35
   ],
   "dim": 35
  }
 },
 "errors": []
}
```

</details>

### V4 每个音视频聚合值都能找到来源帧 —— 通过

对每个有源单元的槽回捞源帧重算均值，最大相对偏差 5.949e-08（阈值 1e-06，余量 17×，出现在 -THoVjtIkeU_2/audio/槽37）；text 可溯源率 43.16%（插值槽 2842 个无源帧）；audio 可溯源率 98.82%（插值槽 59 个无源帧）；vision 可溯源率 98.54%（插值槽 73 个无源帧）

> **限定**：音频/视觉的空槽由相邻有效槽**插值**填充，没有源帧，因此不可能 100% 溯源——本项给出真实可溯源率，并在 alignment_q1.json 里把这些槽标记为 interp 而非冒充有源。

<details><summary>证据</summary>

```json
{
 "v4a_max_norm_rel_diff": 5.949114251251917e-08,
 "v4a_max_abs_diff": 0.000244140625,
 "v4a_max_abs_diff_note": "绝对偏差仅供记录；音频量级 ~5e3，绝对容差会误判，判据一律用相对偏差",
 "v4a_worst_location": "-THoVjtIkeU_2/audio/槽37",
 "v4a_rel_tol": 1e-06,
 "v4a_rel_diff_percentiles": {
  "p50": 1.883529118912394e-08,
  "p90": 4.06278514445188e-08,
  "p99": 5.400905696973862e-08,
  "max": 5.949114251251917e-08
 },
 "traceability": {
  "text": {
   "slots": 5000,
   "traceable": 2158,
   "interp": 2842
  },
  "audio": {
   "slots": 5000,
   "traceable": 4941,
   "interp": 59
  },
  "vision": {
   "slots": 5000,
   "traceable": 4927,
   "interp": 73
  }
 },
 "traceable_ratio": {
  "text": 0.4316,
  "audio": 0.9882,
  "vision": 0.9854
 },
 "errors": []
}
```

</details>

### V5 槽几何与可逆映射 —— 通过

100 条样本的槽边界严格递增、首尾为 0 与 T、等宽（最大槽宽浮动 3.55e-15 s），50 个槽的 time_to_slot 往返全部一致；越界单元被并入端点槽：末槽 7 个、首槽 0 个（其中真正越出 T 的 5 个）。

> **限定**：落在 [0,T] 之外的单元被 time_to_slot 按最近端截断并入端点槽（并排总比丢弃诚实）；本项**显式记账**而不是让它静默发生。

<details><summary>证据</summary>

```json
{
 "max_width_jitter_sec": 3.552713678800501e-15,
 "units_clipped_to_last_slot": 7,
 "units_clipped_to_first_slot": 0,
 "units_beyond_duration": 5
}
```

</details>

### V6 填充规则一致性 —— 通过

text 的 2842 个空槽全部恰为零向量；audio/vision 的 132 个空槽由插值填充（因此非零），三者 fill 口径与声明一致。

> **限定**：audio/vision 的空槽值是插值结果，**不具溯源资格**，已在 V4 的可溯源率中扣除；本条只断言掩码与计数自洽，不对插值值做零断言。

<details><summary>证据</summary>

```json
{
 "per_modality": {
  "text": {
   "slots": 5000,
   "empty": 2842,
   "empty_nonzero": 0
  },
  "audio": {
   "slots": 5000,
   "empty": 59,
   "empty_nonzero": 59
  },
  "vision": {
   "slots": 5000,
   "empty": 73,
   "empty_nonzero": 61
  }
 },
 "fills_seen": {
  "text": [
   "zero"
  ],
  "audio": [
   "interpolate"
  ],
  "vision": [
   "interpolate"
  ]
 },
 "expected_fill": {
  "text": "zero",
  "audio": "interpolate",
  "vision": "interpolate"
 },
 "errors": []
}
```

</details>

### V7 数值合规（无 NaN/Inf，dtype 正确） —— 通过

全部特征均为 float32 且无 NaN/Inf，掩码均为 bool；各模态 |x|max：文本 15.785、语音 7562.5、视觉 1.234（量级差 3 个数量级，故 V4 必须用相对容差）。

<details><summary>证据</summary>

```json
{
 "abs_max_per_modality": {
  "text": 15.784838676452637,
  "audio": 7562.5,
  "vision": 1.2338403463363647
 }
}
```

</details>

### V8 时长口径交叉核对（外部 ffprobe） —— 通过

对 100 条样本重新执行 ffprobe，视频流时长与文件内 duration 的最大绝对偏差 3.30e-05 s（阈值 1e-3 s），耗时 5.4s。

<details><summary>证据</summary>

```json
{
 "max_deviation_sec": 3.300000000017178e-05,
 "probed_samples": 100,
 "elapsed_sec": 5.42
}
```

</details>

### V9 预训练权重真实性（掩码补全，外部证据） —— 通过

掩码补全 2/2 通过（top-5 命中语义正确词），词嵌入标准差 0.1315（随机初始化应≈1/√768≈0.036），hidden=768 vocab=50265；纯离线（local_files_only=True）加载成功；未预期键 0 个、缺失键 0 个。

> **限定**：本项验证「权重确为预训练成果」，不验证「特征对情感任务有效」（后者属问题二）。

<details><summary>证据</summary>

```json
{
 "probes": [
  {
   "probe": "The capital of France is <mask>.",
   "top5": [
    "Paris",
    "Lyon",
    "Nice",
    "Nancy",
    "Napoleon"
   ],
   "expected": [
    "Paris",
    "paris"
   ],
   "passed": true
  },
  {
   "probe": "I love this movie, it was <mask>.",
   "top5": [
    "great",
    "amazing",
    "awesome",
    "fantastic",
    "good"
   ],
   "expected": [
    "amazing",
    "awesome",
    "excellent",
    "great",
    "wonderful"
   ],
   "passed": true
  }
 ],
 "word_embedding_std": 0.13147923350334167,
 "hidden_size": 768,
 "vocab_size": 50265,
 "loaded_local_files_only": true,
 "unexpected_keys": [],
 "missing_keys": [],
 "note": "本检查加载的是 RobertaForMaskedLM（含 lm_head），因此键集齐全；管线用的 AutoModel 会另报 pooler.* 缺失，那是正常的——管线只取 last_hidden_state，从不使用 pooler。"
}
```

</details>

### V10 跨样本重复检测 —— 通过

三模态共 300 个特征数组按 md5 查重，未发现逐字节重复。

> **限定**：重复**不必然**是缺陷；本项要求每一组都有合理解释，并把解释写进异常台账。

<details><summary>证据</summary>

```json
{
 "groups": []
}
```

</details>

### V11 产物新鲜度（对齐晚于特征） —— 通过

对齐产物最早写入时间 2026-09-26 15:33:48 晚于未对齐产物最新写入时间 2026-09-24 16:04:30，交付自洽（对齐确实建立在当前特征之上）。

<details><summary>证据</summary>

```json
{
 "unaligned_newest": "text/-yRb-Jum7EQ_6.npz",
 "aligned_oldest": "-3g5yACwYnA_13.npz",
 "unaligned_newest_per_modality": {
  "text": {
   "newest": "2026-09-24 16:04:30",
   "file": "-yRb-Jum7EQ_6.npz"
  },
  "audio": {
   "newest": "2026-09-24 05:28:03",
   "file": "-yRb-Jum7EQ_6.npz"
  },
  "vision": {
   "newest": "2026-09-24 10:53:44",
   "file": "-yRb-Jum7EQ_6.npz"
  }
 },
 "aligned_newest": "2026-09-26 15:33:51"
}
```

</details>

### V12 异常普查（仅用于标注，不用于删除） —— 通过

A1_audio_digital_silence=2；A2_vad_zero_but_energy=1；A3_vision_face_all_fail=2；A4_vision_face_partial=4；A5_vision_energy_zero_but_valid=2；A6_audio_shorter_than_video=50；A7_audio_valid_slots_lt_50=46；A8_pts_beyond_axis=5；LONG_PAUSE=6；VFR_frame_gap=28

> **限定**：本项**只分类不处置**。竞赛指南明确：置信度/质量阈值用于质量标注，**不得作为删除样本的理由**。所有命中的样本一律保留并标注。

<details><summary>证据</summary>

```json
{
 "counts": {
  "A1_audio_digital_silence": 2,
  "A2_vad_zero_but_energy": 1,
  "A3_vision_face_all_fail": 2,
  "A4_vision_face_partial": 4,
  "A5_vision_energy_zero_but_valid": 2,
  "A6_audio_shorter_than_video": 50,
  "A7_audio_valid_slots_lt_50": 46,
  "A8_pts_beyond_axis": 5,
  "VFR_frame_gap": 28,
  "LONG_PAUSE": 6,
  "EMPTY_TEXT": 0
 },
 "samples": {
  "A1_audio_digital_silence": [
   "-mJ2ud6oKI8_1",
   "-mJ2ud6oKI8_2"
  ],
  "A2_vad_zero_but_energy": [
   "-s9qJ7ATP7w_8"
  ],
  "A3_vision_face_all_fail": [
   "-mJ2ud6oKI8_1",
   "-ri04Z7vwnc_0"
  ],
  "A4_vision_face_partial": [
   "-HwX2H8Z4hY_9",
   "-NFrJFQijFE_1",
   "-NFrJFQijFE_2",
   "-ri04Z7vwnc_2"
  ],
  "A5_vision_energy_zero_but_valid": [
   "-mJ2ud6oKI8_1",
   "-ri04Z7vwnc_0"
  ],
  "A6_audio_shorter_than_video": [
   "-3g5yACwYnA_13",
   "-3g5yACwYnA_2",
   "-3g5yACwYnA_9",
   "-HwX2H8Z4hY_5",
   "-HwX2H8Z4hY_6",
   "-NFrJFQijFE_2",
   "-THoVjtIkeU_6",
   "-UuX1xuaiiE_6",
   "-aNfi7CP8vM_7",
   "-dxfTGcXJoc_1",
   "-dxfTGcXJoc_6",
   "-egA8-b7-3M_26",
   "-egA8-b7-3M_17",
   "-egA8-b7-3M_18",
   "-egA8-b7-3M_13",
   "-egA8-b7-3M_1",
   "-egA8-b7-3M_9",
   "-egA8-b7-3M_20",
   "-lzEya4AM_4_5",
   "-mJ2ud6oKI8_9",
   "-t217m2on-s_7",
   "-tANM6ETl_M_3",
   "-tPCytz4rww_10",
   "-tPCytz4rww_12",
   "-wny0OAz3g8_2",
   "-wny0OAz3g8_5",
   "-wny0OAz3g8_7",
   "-wny0OAz3g8_9",
   "-571d8cVauQ_5",
   "-I_e4mIh0yE_1",
   "-I_e4mIh0yE_3",
   "-UacrmKiTn4_10",
   "-uywlfIYOS8_4",
   "-9y-fZ3swSY_4",
   "-AUZQgSxyPQ_2",
   "-HeZS2-Prhc_2",
   "-MeTTeMJBNc_13",
   "-MeTTeMJBNc_7",
   "-RfYyzHpjk4_11",
   "-RfYyzHpjk4_2",
   "-ri04Z7vwnc_2",
   "-ri04Z7vwnc_5",
   "-s9qJ7ATP7w_1",
   "-s9qJ7ATP7w_5",
   "-s9qJ7ATP7w_4",
   "-s9qJ7ATP7w_7",
   "-s9qJ7ATP7w_6",
   "-s9qJ7ATP7w_8",
   "-yRb-Jum7EQ_1",
   "-yRb-Jum7EQ_5"
  ],
  "A7_audio_valid_slots_lt_50": [
   "-3g5yACwYnA_13",
   "-3g5yACwYnA_2",
   "-HwX2H8Z4hY_2",
   "-HwX2H8Z4hY_5",
   "-HwX2H8Z4hY_6",
   "-NFrJFQijFE_2",
   "-THoVjtIkeU_2",
   "-UuX1xuaiiE_0",
   "-UuX1xuaiiE_3",
   "-UuX1xuaiiE_6",
   "-aNfi7CP8vM_7",
   "-egA8-b7-3M_26",
   "-egA8-b7-3M_17",
   "-egA8-b7-3M_13",
   "-egA8-b7-3M_9",
   "-egA8-b7-3M_20",
   "-iRBcNs9oI8_3",
   "-iRBcNs9oI8_7",
   "-iRBcNs9oI8_6",
   "-mJ2ud6oKI8_6",
   "-tANM6ETl_M_3",
   "-tPCytz4rww_11",
   "-tPCytz4rww_10",
   "-wny0OAz3g8_0",
   "-wny0OAz3g8_2",
   "-wny0OAz3g8_5",
   "-wny0OAz3g8_7",
   "-wny0OAz3g8_9",
   "-571d8cVauQ_0",
   "-I_e4mIh0yE_1",
   "-I_e4mIh0yE_3",
   "-UacrmKiTn4_10",
   "-uywlfIYOS8_4",
   "-9y-fZ3swSY_4",
   "-MeTTeMJBNc_13",
   "-RfYyzHpjk4_11",
   "-RfYyzHpjk4_2",
   "-ri04Z7vwnc_0",
   "-ri04Z7vwnc_2",
   "-ri04Z7vwnc_5",
   "-s9qJ7ATP7w_1",
   "-s9qJ7ATP7w_5",
   "-s9qJ7ATP7w_4",
   "-s9qJ7ATP7w_7",
   "-s9qJ7ATP7w_6",
   "-s9qJ7ATP7w_8"
  ],
  "A8_pts_beyond_axis": [
   "-UuX1xuaiiE_0",
   "-571d8cVauQ_0",
   "-9y-fZ3swSY_0",
   "-UUCSKoHeMA_0",
   "-s9qJ7ATP7w_0"
  ],
  "VFR_frame_gap": [
   "-3g5yACwYnA_2",
   "-NFrJFQijFE_2",
   "-UuX1xuaiiE_6",
   "-aNfi7CP8vM_7",
   "-egA8-b7-3M_17",
   "-egA8-b7-3M_20",
   "-iRBcNs9oI8_3",
   "-iRBcNs9oI8_7",
   "-iRBcNs9oI8_6",
   "-iRBcNs9oI8_8",
   "-mJ2ud6oKI8_9",
   "-t217m2on-s_7",
   "-wny0OAz3g8_2",
   "-wny0OAz3g8_5",
   "-wny0OAz3g8_9",
   "-571d8cVauQ_0",
   "-571d8cVauQ_5",
   "-I_e4mIh0yE_1",
   "-I_e4mIh0yE_3",
   "-9y-fZ3swSY_4",
   "-MeTTeMJBNc_13",
   "-RfYyzHpjk4_11",
   "-ri04Z7vwnc_5",
   "-s9qJ7ATP7w_1",
   "-s9qJ7ATP7w_5",
   "-s9qJ7ATP7w_4",
   "-s9qJ7ATP7w_6",
   "-yRb-Jum7EQ_1"
  ],
  "LONG_PAUSE": [
   "-THoVjtIkeU_2",
   "-aqamKhZ1Ec_0",
   "-mJ2ud6oKI8_1",
   "-mJ2ud6oKI8_2",
   "-MeTTeMJBNc_13",
   "-s9qJ7ATP7w_8"
  ]
 }
}
```

</details>

### V13 实测词时间路由一致性 —— 通过

7 条走实测路由、93 条保持均匀假设，逐条与落盘的时间基准字段严格对应；7 条实测样本均满足「全词对齐 + 零时长词为 0 + 区间不越界」三条准入条件，且其槽归属确由实测时刻推出。 各交付物计数：{'aligned_dir': 7, 'features_q1.npz': 7, 'summary_q1.csv': 7, 'alignment_q1.json': 7}。

> **限定**：本项证明的是**路由实现与声明一致**，不是「实测词时刻与真值一致」。本数据集没有逐词人工时间戳真值，对齐质量只能用「官方词数是否被完整覆盖」这一可机检的弱证据来把关。

<details><summary>证据</summary>

```json
{
 "n_word_level": 7,
 "n_clip_level": 93,
 "n_illegal_mode": 0,
 "n_routing_missing": 0,
 "word_level_samples": [
  "-9y-fZ3swSY_4",
  "-9y-fZ3swSY_8",
  "-HwX2H8Z4hY_6",
  "-THoVjtIkeU_2",
  "-mJ2ud6oKI8_6",
  "-s9qJ7ATP7w_6",
  "-wny0OAz3g8_0"
 ],
 "counts_across_artifacts": {
  "aligned_dir": 7,
  "features_q1.npz": 7,
  "summary_q1.csv": 7,
  "alignment_q1.json": 7
 },
 "errors": []
}
```

</details>
