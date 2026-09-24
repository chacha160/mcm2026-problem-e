# -*- coding: utf-8 -*-
"""
unaligned_common.py
三模态「未对齐」特征提取的公共工具层。

设计动机（本模块绕开既有 utils.py 的三个已知问题，均已在本批 100 条 mp4 上实测确认）：

  1) 时长口径：utils.get_video_duration() 用 `帧数 / fps` 估算，对本批 mp4 严重偏大。
     容器元数据中的帧数虚高（例：-3g5yACwYnA/2.mp4 声称 424 帧、30fps → 14.13s，
     实际只能解出 278 帧，真实时长 9.39s）。本模块改用 ffprobe 的「流时长」，
     并以逐帧 PTS 为准，避免把时间轴整体拉长 1.5~2.6 倍。

  2) 抽帧方式：utils.safe_read_video_frames() 会把整段视频所有帧读进内存
     （720p × 250 帧 × 3 通道 ≈ 0.7 GB/片段），且抽帧间隔按名义 fps 计算。
     本模块改为「流式逐帧 + 真实 PTS 触发」，内存占用与视频长度无关，
     且按 PTS 采样，天然容忍丢帧/VFR。

  3) 音频提取：utils.extract_audio_from_video() 的临时 wav 仅以 `clip_id.wav` 命名，
     而本批 100 条样本只有 19 个不同的 clip_id（clip "2" 出现在 12 个不同视频中），
     "已存在则跳过" 会导致不同视频的音频互相覆盖后被静默复用
     （实测 -ri04Z7vwnc_0 与 -wny0OAz3g8_0 的音频特征逐字节相同）。
     本模块用 ffmpeg 管道直接解码到内存，完全不落临时文件。

时间基准约定
------------
所有「未对齐」特征文件都必须携带自己的时间基准，这是后续对齐能否成立的前提：
    features : (T, D) float32   逐帧特征
    pts      : (T,)   float32   每帧对应的时间戳（秒），相对片段起点
    lengths  : int              有效帧数 T
音频的 pts 由 hop 恒定推出（等间隔）；视觉的 pts 由解码器实际给出，可能存在不均匀间隔。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# ---- 让 code/ 目录可被 import（与同目录其余模块的做法一致）----
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import config  # noqa: E402
from utils import LOGGER  # noqa: E402

# Windows 中文控制台默认 GBK，日志中的中文一旦无法编码就会抛 UnicodeEncodeError
# 并中断整个批处理。这里统一把标准流重配为 UTF-8，无法编码的字符替换而非报错。
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # 少数环境（如被重定向的管道）不允许重配，忽略即可
            pass

# ==================== 外部可执行文件定位 ====================


def _find_bin(name: str, extra_candidates: Optional[List[str]] = None) -> Optional[str]:
    """定位 ffmpeg / ffprobe。先查 PATH，再查显式候选路径（便于环境变量未生效时兜底）。"""
    found = shutil.which(name) or shutil.which(f"{name}.exe")
    if found:
        return found
    for cand in extra_candidates or []:
        if os.path.isfile(cand):
            return cand
    # Windows 上常见的 WinGet 安装位置
    local = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages")
    if os.path.isdir(local):
        for root, _dirs, files in os.walk(local):
            for f in files:
                if f.lower() in (f"{name}.exe", name):
                    return os.path.join(root, f)
    return None


FFMPEG = _find_bin("ffmpeg")
FFPROBE = _find_bin("ffprobe")


def ensure_ffmpeg() -> bool:
    """确认 ffmpeg / ffprobe 可用。音频解码与视频探测都依赖它。"""
    if FFMPEG and FFPROBE:
        return True
    LOGGER.error("[系统] 未找到 ffmpeg/ffprobe，无法解码音频或探测时长，请检查 PATH")
    return False


# ==================== 片段标识 ====================


def sample_id(video_id: str, clip_id) -> str:
    """
    项目内部使用的样本 ID，形如 `-3g5yACwYnA_13`。

    注意 `video_id` 本身可能含下划线（本批 100 条里有 6 个如此，如 `-I_e4mIh0yE`），
    因此**任何从 sample_id 反解的动作都必须按最后一个下划线切分**，不能用
    `split("_", 1)` 或 `replace("_", "$_$", 1)`——那会把 video_id 内部的第一个下划线
    当成分隔符。正确写法见 `official_id()` 的调用方，例如 `sid.rsplit("_", 1)`。
    """
    return f"{video_id}_{clip_id}"


def official_id(video_id: str, clip_id) -> str:
    """附件2 使用的样本 ID 口径（video_id$_$clip_id），用于与官方特征做匹配对标。"""
    return f"{video_id}$_${clip_id}"


def video_path_of(video_id: str, clip_id, root: Optional[str] = None) -> str:
    """由 video_id + clip_id 还原 mp4 路径：<附件1根目录>/<video_id>/<clip_id>.mp4"""
    root = root or config.ATTACHMENT1_DIR
    return os.path.join(root, str(video_id), f"{clip_id}.mp4")


# ==================== 时长与流信息（ffprobe 口径）====================


def probe_streams(video_path: str) -> Dict[str, object]:
    """
    用 ffprobe 读取音视频流信息。

    返回 dict：
        video_duration : float  视频流时长（秒），失败为 0.0
        audio_duration : float  音频流时长（秒），无音轨为 0.0
        has_audio      : bool
        video_codec / audio_codec / width / height / nominal_fps
        n_video_packets: int    视频包数（注意：容器 np_frames 字段可能虚高，此处以包数为准）

    注意：不要用 OpenCV 的 CAP_PROP_FRAME_COUNT / CAP_PROP_FPS 估算时长——
    本批 mp4 的该字段虚高 1.5~2.6 倍。
    """
    import json

    info: Dict[str, object] = {
        "video_duration": 0.0, "audio_duration": 0.0, "has_audio": False,
        "video_codec": None, "audio_codec": None, "width": None, "height": None,
        "nominal_fps": 0.0, "n_video_packets": 0,
    }
    if not (FFPROBE and os.path.isfile(video_path)):
        return info

    cmd = [FFPROBE, "-v", "quiet", "-print_format", "json",
           "-show_streams", "-select_streams", "", video_path]
    try:
        # 显式用 utf-8 解码：路径含中文，Windows 默认 GBK 会抛 UnicodeDecodeError
        proc = subprocess.run(cmd, capture_output=True, check=True,
                              encoding="utf-8", errors="replace")
        streams = json.loads(proc.stdout).get("streams", [])
    except Exception as exc:
        LOGGER.warning("[探测] ffprobe 失败 %s: %s", os.path.basename(video_path), exc)
        return info

    for s in streams:
        if s.get("codec_type") == "video" and info["video_codec"] is None:
            info["video_codec"] = s.get("codec_name")
            info["width"] = s.get("width")
            info["height"] = s.get("height")
            info["nominal_fps"] = _parse_fraction(s.get("avg_frame_rate") or s.get("r_frame_rate"))
            info["video_duration"] = float(s.get("duration") or 0.0)
            try:
                info["n_video_packets"] = int(s.get("nb_frames") or 0)
            except (TypeError, ValueError):
                info["n_video_packets"] = 0
        elif s.get("codec_type") == "audio" and info["audio_codec"] is None:
            info["audio_codec"] = s.get("codec_name")
            info["audio_duration"] = float(s.get("duration") or 0.0)
            info["has_audio"] = True
    return info


def _parse_fraction(text) -> float:
    """把 ffprobe 的 '30/1' 形式帧率转成 float。"""
    try:
        if isinstance(text, str) and "/" in text:
            num, den = text.split("/")
            return float(num) / float(den) if float(den) else 0.0
        return float(text)
    except (TypeError, ValueError):
        return 0.0


# ==================== 音频解码（管道，不落临时文件）====================


def decode_audio(video_path: str, sr: int = None) -> Optional[np.ndarray]:
    """
    用 ffmpeg 把视频音轨解码为单声道 float32 波形（不写临时文件）。

    返回 (n_samples,) 的 float32 数组；失败或无音轨返回 None。
    """
    sr = sr or config.AUDIO_SAMPLE_RATE
    if not (FFMPEG and os.path.isfile(video_path)):
        return None
    cmd = [FFMPEG, "-v", "error", "-i", video_path,
           "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"]
    try:
        proc = subprocess.run(cmd, capture_output=True)
    except Exception as exc:
        LOGGER.warning("[音频] ffmpeg 调用失败 %s: %s", os.path.basename(video_path), exc)
        return None
    if proc.returncode != 0 or not proc.stdout:
        LOGGER.warning("[音频] 解码失败或无音轨: %s", os.path.basename(video_path))
        return None
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


# ==================== 视频流式读取（真实 PTS）====================


def iter_video_frames(video_path: str, target_fps: Optional[float] = None,
                      max_seconds: Optional[float] = None) -> Iterator[Tuple[float, np.ndarray]]:
    """
    流式逐帧读取，产出 (pts_seconds, frame_bgr)。

    target_fps 为 None 时逐帧产出；否则按「真实 PTS 跨过下一个采样时刻」触发产出，
    采样时刻为 0, 1/fps, 2/fps, ...，产出的是该时刻上或之后的第一个真实帧，
    其 pts 为解码器给出的真实时间（因此不均匀间隔/丢帧都被如实保留）。

    内存占用与视频长度无关：始终只持有一帧。
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        LOGGER.error("[视觉] 无法打开视频: %s", video_path)
        return

    step = (1.0 / target_fps) if target_fps and target_fps > 0 else None
    next_target = 0.0
    n_yield = 0
    try:
        while True:
            if not cap.grab():
                break
            pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if pts is None or pts != pts:      # 部分后端可能返回 NaN
                pts = next_target if step else float(n_yield)
            if max_seconds is not None and pts > max_seconds:
                break
            if step is None or pts + 1e-9 >= next_target:
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                n_yield += 1
                yield float(pts), frame
                # 跳过落在同一采样时刻的所有后续帧，避免重复采样
                if step is not None:
                    while next_target <= pts + 1e-9:
                        next_target += step
    finally:
        cap.release()


def video_true_duration(video_path: str) -> float:
    """优先用 ffprobe 流时长；失败时退回逐帧 PTS 的末值。"""
    d = float(probe_streams(video_path).get("video_duration") or 0.0)
    if d > 0:
        return d
    last = 0.0
    for pts, _ in iter_video_frames(video_path):
        last = pts
    return last


# ==================== 未对齐特征文件的读写 ====================


def save_unaligned(path: str, features: np.ndarray, pts: np.ndarray,
                   modality: str, meta: Optional[Dict] = None,
                   extra: Optional[Dict[str, object]] = None) -> str:
    """
    保存单个样本的单模态「未对齐」特征。

    存储内容（.npz）：
        features (T, D) float32 , pts (T,) float32 , modality , meta(JSON 字符串)
        以及 extra 中每个键的 JSON 字符串（键名以 `extra__` 前缀存储）

    extra 用于承载「非等长」的模态附属信息，例如：
        文本 —— words（词串，供时间轴直接显示文字）/ char_spans / token_index（溯源）
        视觉 —— face_flags（逐帧「该帧是否检出人脸」，供 QC 与时间轴核对）
    注意：**并不存在 `face_boxes` 字段**。现有视觉提取器的 MTCNN 是 keep_all=False、
    不保留框坐标的，所以既没有框也没有框面积；需要框信息的场合必须另起一个
    keep_all=True 的探测器，见 face_probe.py。
    之所以一律走 JSON 而不是 ndarray：这些字段长度不一、含字符串与混合类型，
    且 npz 在 allow_pickle=False 下无法保存 object 数组。
    """
    import json
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    features = np.asarray(features, dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32).reshape(-1)
    if features.ndim != 2:
        raise ValueError(f"features 必须为 (T, D)，实际 {features.shape}")
    if features.shape[0] != pts.shape[0]:
        raise ValueError(f"features 帧数 {features.shape[0]} 与 pts 长度 {pts.shape[0]} 不一致")
    payload = {
        "features": features,
        "pts": pts,
        "modality": np.array(modality),
        "meta": np.array(json.dumps(meta or {}, ensure_ascii=False)),
    }
    for key, value in (extra or {}).items():
        payload[f"extra__{key}"] = np.array(json.dumps(value, ensure_ascii=False))
    np.savez_compressed(path, **payload)
    return path


def load_unaligned(path: str) -> Dict[str, object]:
    """读取未对齐特征文件，返回 dict(features, pts, modality, meta, extra)。"""
    import json
    with np.load(path, allow_pickle=False) as d:
        extra = {k[len("extra__"):]: json.loads(str(d[k]))
                 for k in d.files if k.startswith("extra__")}
        return {
            "features": d["features"],
            "pts": d["pts"],
            "modality": str(d["modality"]),
            "meta": json.loads(str(d["meta"])),
            "extra": extra,
        }


def module_fingerprint(obj: object) -> str:
    """把类/函数的源码算成 md5，写入元数据，用于证明「特征由哪份代码产出」。"""
    import hashlib
    import inspect
    try:
        src = inspect.getsource(obj)
    except (OSError, TypeError):
        src = repr(obj)
    return hashlib.md5(src.encode("utf-8")).hexdigest()[:12]


def unaligned_dir(modality: str, root: Optional[str] = None) -> str:
    """未对齐特征的输出目录：<项目>/data/unaligned_features/<modality>/"""
    # _CODE_DIR 是 <项目>/code，故项目根 = 其父目录（只剥一层）
    root = root or os.path.join(os.path.dirname(_CODE_DIR), "data", "unaligned_features")
    return os.path.join(root, modality)


# ==================== 样本清单（三个提取器共用）====================


def list_samples(only: Optional[str] = None, limit: Optional[int] = None,
                 label_xlsx: Optional[str] = None) -> List[Dict[str, object]]:
    """
    从附件1的 label-100.xlsx 读出全部样本清单，供三个提取器遍历。

    返回 list[dict]，每项含：
        video_id / clip_id / sample_id / official_id / text / video_path / label
    only  : 子串过滤（按 sample_id 匹配，便于只跑单条样本做调试）
    limit : 只取前 N 条
    """
    from utils import load_label_table

    xlsx = label_xlsx or os.path.join(config.ATTACHMENT1_DIR, config.LABEL_XLSX_NAME)
    df = load_label_table(xlsx, sheet_name=config.LABEL_SHEET_NAME)
    rows: List[Dict[str, object]] = []
    for _, r in df.iterrows():
        vid = str(r["video_id"]).strip()
        cid = str(r["clip_id"]).strip()
        sid = sample_id(vid, cid)
        if only and only not in sid:
            continue
        rows.append({
            "video_id": vid,
            "clip_id": cid,
            "sample_id": sid,
            "official_id": official_id(vid, cid),
            "text": str(r.get("text", "") or ""),
            "label": r.get("label", None),
            "video_path": video_path_of(vid, cid),
        })
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows
