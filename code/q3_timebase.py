# -*- coding: utf-8 -*-
"""
q3_timebase.py —— 问题三的「槽位 → 真实时间 → 原始素材」还原层

这是问题三最硬的一步，也是本项目相对早期实现的关键增量。
赛题第 28 行要求「关键证据需可对应至原始文本片段、语音时段或视觉关键帧」，
但第 73 行同时说明「附件2 保存的是处理后的数值特征和标签，不保存原始视频帧、
音频波形或逐词时间戳」——也就是说，官方特征本身**不含**时间戳，
槽号到秒的映射必须自己重建。

可用的官方素材
--------------
附件4 的每个样本都配套一个原始 mp4（`videos/<id>.mp4`，共 20 个），
这就是重建时间轴的全部依据：真实时长、真实语音活动、真实关键帧都从这里来。

重建方法（三级，逐级降级，每级都在输出里标注用了哪一级）
--------------------------------------------------------
第 1 级 R（真实音频）：
    用 ffmpeg 解码 16 kHz 单声道，10 ms 帧移做能量 VAD，得到**真实语音区间**；
    语音区间给出「哪里在说话」，这是从原始波形直接测出来的，不是假设。
第 2 级 S（音节加权）：
    在语音区间内，把 n_slots 个词元槽按**音节权重**分配时间。权重取每个词的
    元音组数（vowel-group），它比「每词等长」更接近真实语速分布，因为
    长单词占用时间确实更长。这一步是**近似**，在输出中显式标注。
第 3 级 U（均匀兜底）：
    若视频缺失或音频解码失败，退化为按总时长均匀分配，并标注 `uniform`。

为什么不用「按帧数/50」这种更简单的做法
----------------------------------------
那等于假设语速恒定且全程有声，会在句首停顿、句尾静音处把证据时间整体挪位，
而问题三要看的关键证据恰恰常出现在这些位置。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import q2q3_common as Q  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

SR = 16000          # 解码采样率
FRAME_MS = 25       # VAD 帧长
HOP_MS = 10         # VAD 帧移
VOWELS = set("aeiouyAEIOUY")


# ==================== 视频探测与解码 ====================

def probe_duration(video_path: str) -> Optional[float]:
    """用 ffprobe 取**容器真实时长**（不是「帧数 ÷ 名义 fps」——后者对丢帧/VFR 会严重偏大）。"""
    if not FFPROBE or not os.path.isfile(video_path):
        return None
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", video_path],
            capture_output=True, text=True, timeout=60)
        return float(r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else None
    except Exception:
        return None


def decode_audio(video_path: str, sr: int = SR) -> Optional[np.ndarray]:
    """ffmpeg 管道直接解成 float32 单声道，不落临时 wav。

    早期实现曾按 `clip_id.wav` 命名临时文件，而 clip_id 会在不同视频间重复，
    「已存在则跳过」导致音频互相覆盖后被静默复用。管道解码从结构上消除该风险。
    """
    if not FFMPEG or not os.path.isfile(video_path):
        return None
    try:
        r = subprocess.run(
            [FFMPEG, "-v", "error", "-i", video_path, "-vn", "-f", "f32le",
             "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sr), "-"],
            capture_output=True, timeout=300)
        if r.returncode != 0 or not r.stdout:
            return None
        return np.frombuffer(r.stdout, dtype="<f4").astype(np.float64)
    except Exception:
        return None


# ==================== 第 1 级：真实语音活动检测 ====================

def frame_energy_db(audio: np.ndarray, sr: int = SR,
                    frame_ms: int = FRAME_MS, hop_ms: int = HOP_MS
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """逐帧 RMS 能量（dB）及其中心时刻。返回 (db, times)。"""
    fl = max(1, int(sr * frame_ms / 1000))
    hp = max(1, int(sr * hop_ms / 1000))
    if len(audio) < fl:
        return np.array([]), np.array([])
    n = 1 + (len(audio) - fl) // hp
    idx = np.arange(n)[:, None] * hp + np.arange(fl)[None, :]
    frames = audio[idx]
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    times = (np.arange(n) * hp + fl / 2.0) / sr
    return db, times


def vad_intervals(audio: np.ndarray, sr: int = SR,
                  rel_db: float = 10.0, min_speech_sec: float = 0.12,
                  min_gap_sec: float = 0.10) -> List[Tuple[float, float]]:
    """自适应能量门限 VAD，返回若干**真实语音区间** [start, end)（秒）。

    门限 = 噪声底（能量 10 分位）+ rel_db。自适应是必要的：
    这批素材录音电平差异很大，固定绝对门限会把轻声样本整段判成静音。
    """
    db, t = frame_energy_db(audio, sr)
    if len(db) == 0:
        return []
    noise = np.percentile(db, 10)
    thr = max(noise + rel_db, np.percentile(db, 95) - 30.0)
    voiced = db > thr
    intervals: List[Tuple[float, float]] = []
    runs = Q.missing_runs(voiced)          # 复用同一套连续段归并逻辑
    for s, e in runs:
        start = float(t[s])
        end = float(t[min(e - 1, len(t) - 1)] + HOP_MS / 1000.0)
        if end - start >= min_speech_sec:
            intervals.append((start, end))
    # 合并间隔过短的相邻段
    merged: List[Tuple[float, float]] = []
    for s, e in intervals:
        if merged and s - merged[-1][1] < min_gap_sec:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


# ==================== 第 2 级：音节加权分配 ====================

def _syllables(word: str) -> int:
    """极简音节估计：元音组数，至少为 1。用于给词分配相对时长权重。"""
    w = "".join(ch for ch in word if ch.isalpha())
    if not w:
        return 1
    groups = 0
    prev_v = False
    for ch in w:
        is_v = ch in VOWELS
        if is_v and not prev_v:
            groups += 1
        prev_v = is_v
    return max(1, groups)


def word_time_plan(raw_text: str, n_slots: int, duration: float,
                   speech: Sequence[Tuple[float, float]]
                   ) -> Tuple[np.ndarray, str]:
    """把 n_slots 个槽位铺到时间轴上，返回每个槽的 (start, end) 与所用级别标签。

    级别标签：`speech+syllable`（真实 VAD + 音节加权）/ `speech+uniform`
    （有 VAD 但无文本）/ `uniform`（完全兜底）。
    """
    out = np.zeros((n_slots, 2), dtype=np.float64)
    if n_slots <= 0:
        return out, "empty"

    words = str(raw_text).split()
    if speech and len(words) > 0:
        speech_total = sum(e - s for s, e in speech)
        weights = np.array([_syllables(w) for w in words], dtype=np.float64)
        # 词 → 时间：按音节权重在「全部语音区间」内顺序展开
        cum = np.concatenate(([0.0], np.cumsum(weights / weights.sum()) * speech_total))
        word_times = np.zeros((len(words), 2), dtype=np.float64)
        for i in range(len(words)):
            word_times[i] = _offset_to_time(cum[i], cum[i + 1], speech, weights, i)
        # 槽 → 词：按索引比例映射（BPE 词元数 ≈ 词数，实测 n_slots 与词数相差 1~8）
        if len(words) == 1:
            map_idx = np.zeros(n_slots, dtype=np.int64)
        else:
            map_idx = np.round(np.linspace(0, len(words) - 1, n_slots)).astype(np.int64)
        out[:, 0] = word_times[map_idx, 0]
        out[:, 1] = word_times[map_idx, 1]
        return out, "speech+syllable"

    if speech:
        # 有语音区间但拿不到词：把槽位按语音总时长均分后铺回语音区间
        total = sum(e - s for s, e in speech)
        cum = np.linspace(0.0, total, n_slots + 1)
        for i in range(n_slots):
            s, e = _offset_to_time(cum[i], cum[i + 1], speech, None, i)
            out[i] = (s, e)
        return out, "speech+uniform"

    # 兜底：按真实（或缺失时退化为 1.0 秒）总时长均匀分配
    d = duration if duration and duration > 0 else max(1.0, n_slots * 0.3)
    edges = np.linspace(0.0, d, n_slots + 1)
    out[:, 0] = edges[:-1]
    out[:, 1] = edges[1:]
    return out, "uniform"


def _offset_to_time(a: float, b: float, speech: Sequence[Tuple[float, float]],
                    weights: Optional[np.ndarray], i: int) -> Tuple[float, float]:
    """把「语音内累计偏移」[a,b) 映射回真实时钟时间。"""
    def to_clock(x: float) -> float:
        acc = 0.0
        for s, e in speech:
            seg = e - s
            if x <= acc + seg:
                return s + (x - acc)
            acc += seg
        return speech[-1][1] if speech else x
    t0, t1 = to_clock(a), to_clock(b)
    if t1 <= t0:
        t1 = t0 + 1e-3
    return t0, t1


# ==================== 第 3 级：关键帧抽取（含隐私处理）====================

def detect_faces(img: np.ndarray) -> List[np.ndarray]:
    """返回人脸的 `(x0, y0, x1, y1)` 像素框列表；检测不可用时返回空列表。

    优先用 MTCNN（facenet-pytorch 自带本地权重，纯离线）——
    本机 OpenCV 5.0.0 **没有** `CascadeClassifier`，也没有
    `cv2.data.haarcascades` 下的级联文件，所以早期基于 Haar 的实现
    在任何机器状态下都只是「什么都不做」地返回原图。
    MTCNN 是本仓库问题一视觉特征提取所用的同一检测器（`config.VISION_FACE_DETECTOR`）。
    """
    try:
        from facenet_pytorch import MTCNN
    except Exception:
        return []
    try:
        img = np.asarray(img)
        if img.ndim != 3 or img.shape[0] < 8 or img.shape[1] < 8:
            return []
        rgb = img[:, :, ::-1].copy()          # cv2 给的是 BGR，MTCNN 要 RGB
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        det = MTCNN(keep_all=True, device="cpu", image_size=160, margin=0,
                    min_face_size=20, thresholds=[0.6, 0.7, 0.7],
                    factor=0.709, post_process=False)
        boxes, _ = det.detect(rgb)
        if boxes is None:
            return []
        return [np.asarray(b, dtype=float) for b in boxes]
    except Exception:
        return []


def _mosaic_box(out: np.ndarray, box) -> bool:
    """就地给一个框打马赛克；框无效时返回 False。"""
    import cv2
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    pad = int(0.12 * max(1, x1 - x0))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(out.shape[1], x1 + pad), min(out.shape[0], y1 + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    roi = out[y0:y1, x0:x1]
    if roi.size == 0:
        return False
    k = max(3, (min(roi.shape[:2]) // 5) | 1)
    out[y0:y1, x0:x1] = cv2.resize(
        cv2.resize(roi, (max(1, roi.shape[1] // k), max(1, roi.shape[0] // k)),
                   interpolation=cv2.INTER_AREA),
        (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    return True


def face_mosaic(img: np.ndarray) -> Tuple[np.ndarray, int]:
    """对检测到的人脸做马赛克。

    赛题第 56 行要求提交物不含身份信息。原视频由赛题方提供，但**我们产出的**
    图件若含清晰人脸，等于在提交物里引入了身份信息，故统一在此处处理。

    **返回 `(图, 实际打码的人脸数)`。** 这个计数是必须的：早期版本静默地
    在检测不可用时原样返回，而图注却写着「faces mosaicked」——
    于是提交物里出现了一句与事实不符的声明。调用方现在据实写图注。
    """
    img = np.asarray(img)
    if img.ndim != 3:
        return img, 0
    boxes = detect_faces(img)
    if not boxes:
        return img, 0
    out = img.copy()
    n = sum(1 for b in boxes if _mosaic_box(out, b))
    return out, n


def grab_frame(video_path: str, t: float, mosaic: bool = True,
               max_width: int = 640, return_faces: bool = False):
    """按绝对时刻抽一帧（OpenCV 定位 + 就近读取），可选人脸马赛克。

    `return_faces=True` 时返回 `(帧, 打码人脸数)`，便于调用方据实写图注。
    """
    try:
        import cv2
    except Exception:
        return None
    if not os.path.isfile(video_path):
        return None
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t * 1000.0))
            ok, frame = cap.read()
        if not ok:
            return None
        if max_width and frame.shape[1] > max_width:
            s = max_width / frame.shape[1]
            frame = cv2.resize(frame, (max_width, int(round(frame.shape[0] * s))))
        if not mosaic:
            return (frame, 0) if return_faces else frame
        out, n = face_mosaic(frame)
        return (out, n) if return_faces else out
    except Exception:
        return None if not return_faces else (None, 0)
    finally:
        if cap is not None:
            cap.release()


# ==================== 对外主入口 ====================

@dataclass
class Timebase:
    """一个附件4 样本的槽位时间基准。"""
    key: str
    n_slots: int                    # 词元槽数 = L - 2
    duration: float                 # 容器真实时长（秒）
    speech: List[Tuple[float, float]]
    slot_times: np.ndarray          # (n_slots, 2) 每个槽的 [start, end) 秒
    level: str                      # 时间轴重建所用级别
    speech_ratio: float             # 语音区间占全片时长比例

    def of_slot(self, j: int) -> Tuple[float, float]:
        """词元槽号 j（1 基，对应官方序列里的第 j 个词元位置）→ (start, end)。"""
        k = int(j) - 1
        if k < 0 or k >= self.n_slots:
            return (float("nan"), float("nan"))
        return (float(self.slot_times[k, 0]), float(self.slot_times[k, 1]))


def build_timebase(sample: "Q.Sample4") -> Timebase:
    """为单个附件4 样本重建槽位时间基准。"""
    L = sample.info.valid_len
    n_slots = max(0, L - 2)              # 去掉 [CLS] 与 [SEP]
    dur = probe_duration(sample.video_path) if sample.video_path else None
    audio = decode_audio(sample.video_path) if sample.video_path else None
    speech = vad_intervals(audio) if audio is not None and len(audio) else []
    if dur is None and audio is not None:
        dur = len(audio) / SR
    slot_times, level = word_time_plan(sample.raw_text, n_slots,
                                       float(dur or 0.0), speech)
    return Timebase(
        key=sample.key, n_slots=n_slots, duration=float(dur or 0.0),
        speech=speech, slot_times=slot_times, level=level,
        speech_ratio=(sum(e - s for s, e in speech) / dur) if dur else 0.0,
    )


def build_all_timebases(samples: Sequence["Q.Sample4"]) -> List[Timebase]:
    return [build_timebase(s) for s in samples]
