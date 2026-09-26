| 包 | 本机实测版本 | 问题一链路直接 import | 未被直接 import 时的角色 |
|---|---|---|---|
| av | 18.1.0 | 是 | — |
| librosa | 1.0.0 | 否 | 本问链路不调用；该包由整机环境快照记录 |
| mediapipe | 0.10.35 | 是 | — |
| numpy | 2.2.6 | 是 | — |
| openai-whisper | 20250625 | 否 | 运行期传递依赖：stable_whisper.load_model() 加载的即 whisper 检查点 |
| openpyxl | 3.1.5 | 否 | 运行期传递依赖：pandas.read_excel 读 label-100.xlsx 的引擎 |
| opensmile | 2.6.0 | 是 | — |
| pandas | 3.0.6 | 是 | — |
| stable-ts | 2.19.1 | 是 | — |
| torch | 2.14.0+cpu | 是 | — |
| torchaudio | 2.11.0+cpu | 否 | 本问链路未直接调用（随 torch 音频栈装入） |
| transformers | 5.17.0 | 是 | — |
