"""
utils.py
通用工具函数：日志设置、文件检查、时间戳处理、安全读取视频等
"""

import os
import sys
import logging
import subprocess
import cv2
import numpy as np
from pathlib import Path

import config


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """配置日志记录器，同时输出到控制台和文件"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台输出
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件输出
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# 全局logger
LOGGER = setup_logger(
    "feature_extraction",
    os.path.join(config.LOG_DIR, "extraction.log")
)


def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)


def load_label_table(xlsx_path: str = None, sheet_name: str = None):
    """
    读取附件1的标注表 label-100.xlsx，返回按行组织的数据表。

    题目要求：附件1的标注信息存放在 label-100.xlsx 的 **label 工作表** 中，
    其中 `text` 列是该视频片段对应的英文转写文本，是文本模态特征提取的唯一输入；
    `video_id` + `clip_id` 共同确定一条视频样本，`label` / `annotation` 为情感标注。

    Args:
        xlsx_path:  标注表路径，默认取 config 中配置的附件1目录下的 label-100.xlsx。
        sheet_name: 工作表名，默认取 config.LABEL_SHEET_NAME（即 "label"）。

    Returns:
        pandas.DataFrame，列至少包含 video_id / clip_id / text，
        并按需携带 label / annotation。clip_id 统一转为字符串（与文件名对齐）。

    Raises:
        FileNotFoundError: 标注表不存在。
        ValueError:        标注表缺少必要列。
    """
    import pandas as pd  # 延迟导入：避免不涉及数据读取的脚本额外加载 pandas

    xlsx_path = xlsx_path or os.path.join(config.ATTACHMENT1_DIR, config.LABEL_XLSX_NAME)
    sheet_name = sheet_name or config.LABEL_SHEET_NAME

    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"找不到标注文件: {xlsx_path}")

    # 优先按指定的工作表名读取（题目明确要求读取 label 工作表）
    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        LOGGER.info(f"[数据] 已读取工作表 '{sheet_name}': {len(df)} 行")
    except ValueError:
        # 工作表名不存在时回退到第一张表，并在日志中说明，避免中断整个流程
        LOGGER.warning(f"[数据] 未找到工作表 '{sheet_name}'，回退为默认读取第一张工作表")
        df = pd.read_excel(xlsx_path)

    if len(df) == 0:
        # 表头被误当作数据/无表头的异常情况，按无表头模式重读
        LOGGER.warning("[数据] 检测到0行数据，尝试无表头模式读取...")
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)
        expected = ["video_id", "clip_id", "text", "label", "annotation"]
        if df.shape[1] == len(expected):
            df.columns = expected
        else:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]

    # 统一列名大小写与空白，避免下游取值失败
    df.columns = [str(c).strip() for c in df.columns]

    required = {"video_id", "clip_id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"标注表缺少必要列: {missing}，当前列名: {list(df.columns)}")

    # clip_id 统一为字符串（与 clip_id.mp4 的文件名做字符串匹配）
    df["clip_id"] = df["clip_id"].astype(str).str.strip()
    # text 列做空值填充与首尾空白清理（NaN → 空串，避免向量化时报错）
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    LOGGER.info(f"[数据] 加载标注表: {len(df)} 条样本, 列名: {list(df.columns)}")
    return df


def check_ffmpeg():
    """检查ffmpeg是否可用（兼容Windows/CMD/GitBash）"""
    for cmd in ["ffmpeg", "ffmpeg.exe"]:
        try:
            result = subprocess.run(
                [cmd, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )
            version_line = result.stdout.splitlines()[0]
            LOGGER.info(f"检测到 FFmpeg: {version_line}")
            return True
        except Exception:
            continue
    LOGGER.error("未检测到 FFmpeg，请检查安装与环境变量 PATH 是否包含 ffmpeg/bin")
    return False


def extract_audio_from_video(video_path: str, output_wav: str, sr: int = 16000):
    """
    使用 ffmpeg 从视频中提取单声道、指定采样率的 WAV 音频。
    若已存在且不要求覆盖，则直接返回路径。
    """
    if os.path.exists(output_wav) and not config.OVERWRITE_EXISTING:
        LOGGER.info(f"[跳过] 音频已存在: {output_wav}")
        return output_wav

    ensure_dir(os.path.dirname(output_wav))
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                # 禁用视频
        "-acodec", "pcm_s16le",  # 16bit PCM
        "-ac", "1",           # 单声道
        "-ar", str(sr),       # 采样率
        output_wav
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        LOGGER.info(f"[成功] 提取音频: {output_wav}")
        return output_wav
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"[失败] ffmpeg提取音频失败: {video_path}, stderr: {e.stderr.decode('utf-8', errors='ignore')[:200]}")
        return None


def get_video_duration(video_path: str) -> float:
    """使用OpenCV获取视频时长（秒），失败则返回0"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return round(frame_count / fps, 3)
    return 0.0


def get_video_fps(video_path: str) -> float:
    """获取视频帧率"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    return fps


def safe_read_video_frames(video_path: str, target_fps: int = None):
    """
    安全读取视频帧，返回 (frames: list[np.array], actual_fps: float, total_duration: float)
    若指定 target_fps，则按时间间隔均匀抽帧；否则读取全部帧。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        LOGGER.error(f"无法打开视频: {video_path}")
        return [], 0.0, 0.0

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / original_fps if original_fps > 0 else 0.0

    frames = []
    frame_idx = 0
    interval = max(1, int(round(original_fps / target_fps))) if target_fps and original_fps > 0 else 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if target_fps is None or (frame_idx % interval == 0):
            # OpenCV读取为BGR，后续模型可能需要RGB，这里先保持BGR，由extractor转换
            frames.append(frame)
        frame_idx += 1

    cap.release()
    effective_fps = target_fps if target_fps else original_fps
    LOGGER.debug(f"视频 {os.path.basename(video_path)}: 原始fps={original_fps}, 抽帧fps={effective_fps}, 共读取{len(frames)}帧")
    return frames, effective_fps, duration


def resample_feature_sequence(features: np.ndarray, target_len: int, mode="interpolate") -> np.ndarray:
    """
    将特征序列 (T, D) 重采样到目标长度 (target_len, D)。
    mode: "interpolate" 线性插值; "repeat" 最近邻重复
    """
    T, D = features.shape
    if T == target_len:
        return features

    if mode == "interpolate":
        # 对每个维度独立进行1D线性插值
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, target_len)
        new_features = np.zeros((target_len, D), dtype=features.dtype)
        for d in range(D):
            new_features[:, d] = np.interp(x_new, x_old, features[:, d])
        return new_features
    else:
        # 最近邻下采样/上采样
        indices = np.round(np.linspace(0, T - 1, target_len)).astype(int)
        return features[indices]
