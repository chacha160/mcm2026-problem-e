| 模态 | 词级特征形状 | 词级维度 | 原生观测 | 原生维度 | 提取工具与口径 |
|---|---|---|---|---|---|
| 文本 text | (L, 768) | 768 | RoBERTa 子词 | 768 | roberta-base 末层隐状态按词池化（子词归属见 roberta_word_token_*） |
| 语音 audio | (L, 50) | 50 | openSMILE LLD 窗 | 25 | eGeMAPSv02 LowLevelDescriptors，逐窗 25 维；词级取 [均值, 总体标准差(ddof=0)] |
| 视觉 vision | (L, 104) | 104 | MediaPipe blendshape 帧 | 52 | FaceLandmarker 逐帧 52 维 blendshape；词级取 [均值, 总体标准差(ddof=0)] |
