# 问题一交付物体积核算（50 MB 口径）

- 上限：50 MiB（= 50×1024² 字节；十进制 52.43 MB）
- 未压缩估算公式：`Σ_m (全部单元数) × D_m × 每元素字节数`
- 变长 float32 未压缩合计：**12.99 MB**（float16 为 6.50 MB）
- 若改存定长填充：文本 (100,50,768) 单独就要 **15.36 MB**，三模态合计 **17.54 MB**

## 实测交付物体积

| 交付物 | 大小 (MB) |
|---|---|
| anomaly_ledger.csv | 0.033 |
| features_q1.npz | 11.635 |
| alignment_q1.json | 0.759 |
| summary_q1.csv | 0.044 |
| extract_config.json | 0.010 |
| size_budget.csv | 0.001 |
| typical_samples.csv | 0.001 |
| figures/-a55Q6RWvTA_3_timeline.png | 0.841 |
| figures/-mJ2ud6oKI8_6_timeline.png | 0.654 |
| figures/-s9qJ7ATP7w_8_timeline.png | 0.648 |
| figures/-UuX1xuaiiE_1_timeline.png | 0.752 |
| figures/-ri04Z7vwnc_0_timeline.png | 0.325 |
| **合计** | **15.702** |

余量 34.30 MB（上限 50 MB）。

## float16 往返实验

| 模态 | 最大绝对偏差 | max-norm 相对偏差 | 逐元素最大相对误差（不具判别力） | |x|max |
|---|---|---|---|---|
| text | 3.9053e-03 | 2.4741e-04 | 0.0187 | 15.785 |
| audio | 2.0000e+00 | 2.6446e-04 | 0.0228 | 7562.500 |
| vision | 4.8816e-04 | 3.9564e-04 | 0.0213 | 1.234 |

**结论：不采用改用 float16。** 变长 float32 三模态合计仅 13.00 MB，距 50 MB 上限余量充足；改用 float16 只省 6.50 MB，却给量级最大的语音/谱对比度维引入最大 2.000 的绝对误差。**没有理由用精度换体积。**

> 关于「逐元素最大相对误差」：近零元素在 float16 下会下溢为 0，使该指标必然趋近 1，因此它**不能**用作判据；上表把它列出来是为了说明为什么不能只看这一列。判定一律使用 max-norm 相对偏差。

## 未随附的资源

- **原始视频**（约 4 GB）：不提交。交付只保留源视频的**相对位置**（见 `alignment_q1.json` 的 `source_video`）与**关键帧**（五类典型图的缩略帧条，逐帧对应时刻见对应表）。
- **RoBERTa 本体**（约 110M 参数，float32 ≈ 440 MB）：无法随附。改为在 `extract_config.json` 固定其**本地 revision 哈希、来源地址、各文件字节数与校验值**，使外部资源可被唯一定位。