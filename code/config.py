"""
config.py
问题1：多模态情感特征提取与时序对齐 —— 全局配置参数

维度口径说明（重要）：
    本题附件2给出了标准特征文件的张量形状，附件4也要求「特征字段、张量维度和时序
    组织方式与附件2保持一致」。因此本流水线的输出维度严格对齐附件2：

        文本 text   : (50, 768)   —— RoBERTa 最后一层隐状态（词级平均后插值到50）
        语音 audio  : (50,  74)   —— MFCC/差分/韵律/频谱，时间分桶到50
        视觉 vision : (50,  35)   —— ResNet-50 池化特征经固定随机投影降到35维

    其中 50 为统一的最大时间槽数（对齐粒度），768/74/35 为各模态特征维度。
"""

import os

# ==================== 路径配置 ====================
# 附件1根目录（包含 label-100.xlsx 和若干 video_id 子文件夹）
# 目录结构：附件1根目录/
#   ├── label-100.xlsx              （工作表 label 中 text 列为英文转写文本）
#   ├── -3g5yACwYnA/13.mp4          （子文件夹名 = video_id，文件名 = clip_id.mp4）
#   └── ...
# 注意：路径含中文与空格，统一使用原始字符串，不做转义。
ATTACHMENT1_DIR = r"D:\23届建模\data\attachment1\exercise\附件1-数据集原始多模态样本\MOSEI数据集部分原始视频-100条"

# 标注表文件与工作表名（题目要求：读取 label 工作表中的 text 列内容）
LABEL_XLSX_NAME = "label-100.xlsx"
LABEL_SHEET_NAME = "label"

# ==================== 遗留常量（已无任何代码使用，勿据此判断路径有效性）====================
# 下面三条属于**已删除的上一版流水线**（extractors/ + main.py + alignment.py + report.py）：
#   · data/output_features/ 是空目录（0 文件），feature_summary.csv 从未由当前流水线写出；
#   · data/temp_audio/ 也不会再出现——现流水线用 ffmpeg 管道直接解码到内存，不落临时 wav
#     （原因见 unaligned_common.py 的说明：按 clip_id 命名的临时 wav 会跨视频互相覆盖）。
# 保留它们只是为了让 check_data.py 之类的诊断脚本仍可 import 而不报 AttributeError。
# **当前流水线的产物路径**以各提取器与 align_multimodal.py 的 out_dir 参数为准：
#   data/unaligned_features/{text,audio,vision}/<sample_id>.npz  未对齐特征
#   data/aligned/<sample_id>.npz 与 data/aligned/aligned_50.npz   对齐结果
#   data/q1_delivery/                                            问题一交付物
OUTPUT_DIR = r"D:\23届建模\data\output_features"      # 遗留，不再写出
TEMP_AUDIO_DIR = r"D:\23届建模\data\temp_audio"       # 遗留，不再写出
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "feature_summary.csv")   # 遗留，不再写出

# 日志目录（在用）
LOG_DIR = r"D:\23届建模\code\logs"

# ==================== 模型与工具配置 ====================
# 文本预训练模型（Hugging Face 模型名）
TEXT_MODEL_NAME = "roberta-base"  # 备选: "bert-base-uncased"
# 文本最大子词长度（RoBERTa 位置编码上限为 514）
TEXT_MAX_LENGTH = 512
# 文本时序对齐策略：
#   "word_interpolate"   —— 子词先按词求平均得到「词」向量，再线性插值到50步（本题要求，默认）
#   "token_interpolate"  —— 不做词级平均，直接对子词序列插值（消融对比用）
#   "sentence_pooling"   —— 全句均值向量复制50份（消融对比用）
TEXT_ALIGN_STRATEGY = "word_interpolate"

# 视觉人脸检测与特征提取模型
VISION_FACE_DETECTOR = "mtcnn"           # facenet-pytorch 内置
VISION_FEATURE_EXTRACTOR = "resnet50"    # torchvision 内置

# 音频特征提取方式："librosa" 或 "opensmile"
AUDIO_EXTRACTOR = "librosa"

# ==================== 特征维度配置 ====================
# 对齐后的统一时间步长（序列位置数）。与附件2 的序列长度口径一致；
# 本项目的实际产物是 data/aligned/aligned_50.npz（不是 .pkl，上一版流水线的
# aligned_50.pkl 已随 extractors/ 一起删除）。
ALIGN_SEQ_LEN = 50

# 各模态输出特征维度（与附件2完全一致）
TEXT_FEATURE_DIM = 768     # RoBERTa-base hidden size
AUDIO_FEATURE_DIM = 74     # 与CMU-MOSEI官方对齐（附件2语音维度）
VISION_FEATURE_DIM = 35    # 与CMU-MOSEI官方对齐（附件2视觉维度）

# 视觉原始特征维度（ResNet-50 全局平均池化后）
VISION_RAW_DIM = 2048
# 视觉降维方式："random_projection"（固定随机投影，单遍可复现）
VISION_REDUCTION = "random_projection"
# 随机投影矩阵的随机种子（固定后100条样本共用同一个投影矩阵，保证可复现）
VISION_PROJECTION_SEED = 42

# ==================== 音频提取参数 ====================
AUDIO_SAMPLE_RATE = 16000   # 重采样目标频率
AUDIO_HOP_LENGTH = 512      # 帧移（samples），@16kHz ≈ 32ms
AUDIO_WIN_LENGTH = 1024     # 窗长（samples），@16kHz = 64ms
AUDIO_N_FFT = 1024          # FFT点数
AUDIO_N_MFCC = 20           # MFCC维数

# ---------------- 语音活动检测（VAD）与「无人声片段」判定 ----------------
# 判定规则：一帧同时满足 (1) pyin 判为浊音（周期性）且 (2) 短时能量高于门限，
#           才计为「有声帧」；某时间槽内有声帧占比低于阈值即判为「无人声槽（静音槽）」。
AUDIO_VAD_ENABLED = True
# 能量门限 = max(绝对下限, 相对比例 × 该片段RMS的95分位数)，用于自适应不同录音音量
AUDIO_VAD_ABS_FLOOR = 1e-3
AUDIO_VAD_REL_RATIO = 0.05
# 槽内有声帧占比低于该值时，判为无人声（静音）槽
AUDIO_VAD_MIN_VOICED_RATIO = 0.02

# ==================== 视频提取参数 ====================
VIDEO_FPS = 25              # 抽帧帧率
VIDEO_FACE_SIZE = 160       # 人脸图像缩放尺寸

# ==================== 可复现性配置 ====================
RANDOM_SEED = 42
PIPELINE_VERSION = "1.1.0"  # 特征提取流水线版本号

# ==================== 运行配置 ====================
# 批处理时并行worker数（Windows建议设为0，即主进程串行，避免multiprocessing问题）
NUM_WORKERS = 0

# 是否覆盖已存在的特征文件
OVERWRITE_EXISTING = False
