| 资产 | 文件 | SHA-256（前 16 位） | 说明 |
|---|---|---|---|
| RoBERTa 文本编码 | model.safetensors | 5bde1d28afb363d0… | 6 个文件逐个校验，任一不符即报错 |
| whisper 强制对齐 | base.en.pt | 25a8566e1d0c1e22… | base.en，与 stable-ts 配套 |
| MediaPipe 面部 | face_landmarker.task | 64184e229b263107… | 预取一次后离线加载（本次 downloaded_this_run=False） |
| openSMILE 配置 | eGeMAPSv02.conf | ef451953badced2e… | eGeMAPSv02 LowLevelDescriptors，25 维，顺序已冻结 |
