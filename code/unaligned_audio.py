# -*- coding: utf-8 -*-
"""
unaligned_audio.py —— 语音模态「未对齐」特征提取

职责边界：
    本文件只负责「音频 → 逐帧 74 维特征 (T, 74) + 每帧实测时间戳 pts」，
    **不做任何时间槽分配**。对齐见 align_multimodal.py。

与已删除的上一版提取器（原 extractors/audio_extractor.py，已随 extractors/ 目录一并移除）
的三点关键差异
--------------------------------------------------------------------------------
  1) 时间轴口径：用 ffmpeg 管道直接解码到内存（不落临时 wav）。上一版用
     `clip_id.wav` 命名临时文件，而本批 100 条样本只有 19 个不同 clip_id
     （clip "2" 出现在 12 个不同视频里），"已存在则跳过" 会让不同视频的音频互相
     覆盖后被静默复用——实测 -ri04Z7vwnc_0 与 -wny0OAz3g8_0 的音频特征逐字节相同。
  2) 帧率：按官方采样率 20 Hz（hop=800@16k）逐帧，而不是 config 里的 512（31.25 Hz）。
     「未对齐」阶段用官方帧率，是为了让后面的 50 槽对齐在时间上站得住：
     50 槽 × 50 ms = 2.5 s/槽，正好对应官方 20 Hz 口径下 50 帧并入 1 槽。
  3) 每帧带实测 pts（帧中心时刻 = k*hop/sr），使 (T,74) 与 (T,) 一一对应，
     对齐时可以按「秒」而不是按「数组下标」重采样。

74 维构成（合计 74，逐段写入 meta.dim_layout 落盘）
--------------------------------------------------
    0..19   MFCC 20 维（n_mfcc=20）
   20..39   一阶差分 ΔMFCC
   40..59   二阶差分 ΔΔMFCC
   60       log F0（pyin；清音帧取 log(fmin)）
   61       浊音概率（pyin voiced_prob）
   62       log RMS 能量
   63       过零率 ZCR
   64       谱质心
   65       谱带宽
   66       谱滚降（85%）
   67       谱平坦度
   68..72   谱对比度 5 维（n_bands=4 → librosa 返回 6 行，取后 5 行，见 _spectral_contrast5）
   73       有声/静音指示（VAD：浊音 且 能量高于自适应门限）

    注：官方 74 维来自 COVAREP（含 NAQ/QOQ/H1H2 等声门参数），本项目无 COVAREP，
    用上面这套「MFCC+差分+韵律+频谱」的等价维度代理，维度严格保持 74 与附件2 对齐。
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import config  # noqa: E402
import unaligned_common as U  # noqa: E402
from utils import LOGGER  # noqa: E402

MODALITY = "audio"

# 官方对齐文件的语音帧率（由附件2 的 500 帧上限与最长片段 25 s 反推：500/25 = 20 Hz）
OFFICIAL_RATE_HZ = 20.0
# 可视化用的振幅包络帧率（10 ms 一格）
ENVELOPE_RATE_HZ = 100.0

DIM_LAYOUT = {
    "mfcc": [0, 20],
    "delta_mfcc": [20, 40],
    "delta2_mfcc": [40, 60],
    "log_f0": [60, 61],
    "voiced_prob": [61, 62],
    "log_rms": [62, 63],
    "zcr": [63, 64],
    "spectral_centroid": [64, 65],
    "spectral_bandwidth": [65, 66],
    "spectral_rolloff": [66, 67],
    "spectral_flatness": [67, 68],
    "spectral_contrast": [68, 73],
    "vad_flag": [73, 74],
}


def _spectral_contrast5(y: np.ndarray, sr: int, n_fft: int, hop: int, win: int) -> np.ndarray:
    """
    取 5 维谱对比度。

    librosa 的 spectral_contrast(n_bands=B) 返回 B+1 行：第 0 行是 fmin(默认 200 Hz)
    以下那一段的对比度，其余 B 行才是真正的频带。因此要得到 5 维需传 n_bands=5，
    再丢掉第 0 行（实测 n_bands=5 → 6 行，n_bands=4 → 5 行）。
    """
    import librosa
    sc = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop,
                                           win_length=win, n_bands=5)
    return sc[1:6]


class AudioFrameExtractor:
    """逐帧 74 维音频特征提取器（不对齐）。"""

    def __init__(self, rate_hz: float = OFFICIAL_RATE_HZ, sr: Optional[int] = None,
                 n_mfcc: Optional[int] = None, n_fft: Optional[int] = None,
                 win_length: Optional[int] = None) -> None:
        self.sr = int(sr or config.AUDIO_SAMPLE_RATE)
        self.rate_hz = float(rate_hz)
        # hop 由目标帧率决定：20 Hz @16k → 800 样点 = 50 ms
        self.hop = int(round(self.sr / self.rate_hz))
        self.n_fft = int(n_fft or config.AUDIO_N_FFT)
        self.win_length = int(win_length or config.AUDIO_WIN_LENGTH)
        self.n_mfcc = int(n_mfcc or config.AUDIO_N_MFCC)
        self.feature_dim = 74
        LOGGER.info("[音频] 逐帧 %d 维，帧率 %.1f Hz（sr=%d, hop=%d=%.1f ms, win=%d, n_mfcc=%d）",
                    self.feature_dim, self.rate_hz, self.sr, self.hop,
                    self.hop / self.sr * 1000, self.win_length, self.n_mfcc)

    # ---------- 特征 ----------
    def extract_frames(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
        """
        波形 → (features (T,74) float32, pts (T,), meta)

        pts 为帧中心时刻：librosa 用 center=True，第 k 帧中心落在样点 k*hop，
        故 pts[k] = k * hop / sr。该定义让帧与时间严格线性对应，
        对齐时按秒换算即可，不必依赖数组下标。
        """
        import librosa

        if y is None or len(y) == 0:
            return np.zeros((0, self.feature_dim), np.float32), np.zeros((0,), np.float32), \
                   {"empty_audio": True, "num_frames": 0}

        y = np.asarray(y, dtype=np.float32)
        kw = dict(hop_length=self.hop, win_length=self.win_length, n_fft=self.n_fft)

        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc, **kw)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.win_length, hop_length=self.hop)

        S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win_length))
        # 注意：凡是显式传 S 的谱特征，都必须同时传 n_fft——librosa 默认 n_fft=2048，
        # 会拿它去校验 S 的行数（513 行对应 1024 点 FFT），不传就报 ParameterError。
        rms = librosa.feature.rms(S=S, frame_length=self.win_length)
        cen = librosa.feature.spectral_centroid(S=S, sr=self.sr, n_fft=self.n_fft)
        bw = librosa.feature.spectral_bandwidth(S=S, sr=self.sr, n_fft=self.n_fft)
        ro = librosa.feature.spectral_rolloff(S=S, sr=self.sr, n_fft=self.n_fft, roll_percent=0.85)
        # power=2.0 表示 S 是幅度谱、内部按 |S|² 取功率谱（避免手工加 1 破坏几何均值语义）
        fl = librosa.feature.spectral_flatness(S=S, power=2.0)
        sc = _spectral_contrast5(y, self.sr, self.n_fft, self.hop, self.win_length)

        # pyin 每帧给出 F0 与浊音概率；实测单段耗时 ~0.25 s（5.4 s 音频），全量可接受
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=65.0, fmax=400.0, sr=self.sr,
            frame_length=self.win_length, hop_length=self.hop)

        T = min(mfcc.shape[1], d1.shape[1], d2.shape[1], zcr.shape[1], rms.shape[1],
                cen.shape[1], bw.shape[1], ro.shape[1], fl.shape[1], sc.shape[1], len(f0))
        mfcc, d1, d2 = mfcc[:, :T], d1[:, :T], d2[:, :T]
        zcr, rms = zcr[:, :T], rms[:, :T]
        cen, bw, ro, fl, sc = cen[:, :T], bw[:, :T], ro[:, :T], fl[:, :T], sc[:, :T]
        f0, voiced_prob, voiced_flag = f0[:T], voiced_prob[:T], voiced_flag[:T]

        # ---- 韵律通道 ----
        voiced = np.isfinite(f0)
        log_f0 = np.where(voiced, np.log(np.maximum(f0, 1e-6)), np.log(65.0)).astype(np.float32)
        vprob = np.where(voiced, voiced_prob, 0.0).astype(np.float32)
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)[0].astype(np.float32)

        # ---- VAD：浊音 且 能量高于自适应门限 ----
        thr = max(config.AUDIO_VAD_ABS_FLOOR,
                  config.AUDIO_VAD_REL_RATIO * float(np.percentile(rms[0], 95)))
        vad = ((rms[0] > thr) & voiced).astype(np.float32)

        feats = np.concatenate([
            mfcc, d1, d2,
            log_f0[None, :], vprob[None, :], rms_db[None, :], zcr,
            cen, bw, ro, fl, sc,
            vad[None, :],
        ], axis=0).T.astype(np.float32)

        if feats.shape[1] != self.feature_dim:
            raise RuntimeError(f"音频维度核算错误：得到 {feats.shape[1]}，应为 {self.feature_dim}。"
                               f"通道长度 {[a.shape[0] for a in [mfcc, d1, d2, log_f0[None], vprob[None], rms_db[None], zcr, cen, bw, ro, fl, sc, vad[None]]]}")

        pts = (np.arange(T, dtype=np.float64) * self.hop / self.sr).astype(np.float32)
        meta = {
            "num_frames": int(T),
            "sample_rate": self.sr,
            "hop_length": self.hop,
            "win_length": self.win_length,
            "n_fft": self.n_fft,
            "n_mfcc": self.n_mfcc,
            "frame_rate_hz": round(self.sr / self.hop, 4),
            "frame_time_convention": "pts[k] = k * hop / sr（librosa center=True，帧中心时刻）",
            "duration_audio_sec": round(len(y) / self.sr, 4),
            "dim_layout": DIM_LAYOUT,
            "voiced_ratio": round(float(voiced.mean()), 4),
            "vad_ratio": round(float(vad.mean()), 4),
            "vad_threshold": round(float(thr), 6),
            "f0_range_hz": [65.0, 400.0],
        }
        return feats, pts, meta


def envelope(y: np.ndarray, sr: int, rate: float = ENVELOPE_RATE_HZ) -> Tuple[np.ndarray, np.ndarray]:
    """
    振幅包络（RMS），供时间轴可视化画「音频振幅」。

    为什么单独算：74 维特征里只有 log RMS 一个能量通道，画不出波形的起伏形状；
    而时间轴要展示的是「音频的振幅」这一物理量，用 10 ms 步长的 RMS 才看得出包络。
    随音频特征一并落盘，使可视化无需重新解码音频即可复现。
    """
    hop = max(1, int(round(sr / rate)))
    T = max(1, len(y) // hop)
    y = np.asarray(y, dtype=np.float32)[:T * hop].reshape(T, hop)
    env = np.sqrt((y ** 2).mean(axis=1)).astype(np.float32)
    times = (np.arange(T, dtype=np.float64) * hop / sr).astype(np.float32)
    return env, times


# ==================== 单样本抽取 ====================

def extract_one(extractor: AudioFrameExtractor, sample: Dict[str, object]
                ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object], Dict[str, object]]:
    """对一条样本抽音频特征。返回 (features, pts, meta, extra)。"""
    video_path = str(sample["video_path"])
    probe = U.probe_streams(video_path)
    y = U.decode_audio(video_path, sr=extractor.sr)

    if y is None or len(y) == 0:
        feats = np.zeros((0, extractor.feature_dim), np.float32)
        pts = np.zeros((0,), np.float32)
        meta = {"num_frames": 0, "empty_audio": True, "has_audio_stream": bool(probe.get("has_audio"))}
        return feats, pts, meta, {"envelope": [], "envelope_times": []}

    feats, pts, meta = extractor.extract_frames(y)
    env, env_t = envelope(y, extractor.sr)

    meta = {
        "modality": MODALITY,
        "feature_dim": extractor.feature_dim,
        "unit_level": "frame",
        "alignment_rule": "无（本文件不做时间槽分配，对齐见 align_multimodal.py）",
        "time_basis": "measured_pts",
        "time_basis_desc": "pts 由解码后按 hop 恒定推出（等间隔），与视频 PTS 同以秒为单位",
        "duration_used": round(U.video_true_duration(video_path), 4),
        "container_video_duration": round(float(probe.get("video_duration") or 0.0), 4),
        "container_audio_duration": round(float(probe.get("audio_duration") or 0.0), 4),
        "audio_codec": probe.get("audio_codec"),
        "pipeline_version": config.PIPELINE_VERSION,
        "code_fingerprint": U.module_fingerprint(AudioFrameExtractor.extract_frames),
        **meta,
    }
    extra = {
        "envelope": env.tolist(),
        "envelope_times": env_t.tolist(),
        "envelope_rate_hz": ENVELOPE_RATE_HZ,
    }
    return feats, pts, meta, extra


def run(samples: List[Dict[str, object]], out_dir: str, overwrite: bool = False,
        rate_hz: float = OFFICIAL_RATE_HZ) -> List[Dict[str, object]]:
    """批量抽取音频未对齐特征。返回汇总行列表。"""
    os.makedirs(out_dir, exist_ok=True)
    extractor = AudioFrameExtractor(rate_hz=rate_hz)

    rows: List[Dict[str, object]] = []
    n_ok = n_skip = n_fail = 0
    for i, s in enumerate(samples, 1):
        sid = str(s["sample_id"])
        out_path = os.path.join(out_dir, f"{sid}.npz")
        if os.path.exists(out_path) and not overwrite:
            n_skip += 1
            continue
        if not os.path.isfile(str(s["video_path"])):
            LOGGER.warning("[音频] (%d/%d) 找不到视频: %s", i, len(samples), s["video_path"])
            n_fail += 1
            continue
        try:
            feats, pts, meta, extra = extract_one(extractor, s)
            U.save_unaligned(out_path, feats, pts, MODALITY, meta, extra=extra)
            n_ok += 1
            LOGGER.info("[音频] (%d/%d) %s -> (T=%d, D=%d) 音频=%.2fs 浊音占比=%.2f",
                        i, len(samples), sid, feats.shape[0], feats.shape[1],
                        float(meta.get("duration_audio_sec") or 0.0),
                        float(meta.get("voiced_ratio") or 0.0))
            rows.append({
                "sample_id": sid, "official_id": s["official_id"], "modality": MODALITY,
                "n_units": int(feats.shape[0]), "feature_dim": int(feats.shape[1]),
                "duration_sec": round(float(meta.get("duration_audio_sec") or 0.0), 3),
                "time_basis": "measured_pts", "out_path": out_path, "status": "ok",
            })
        except Exception as exc:
            n_fail += 1
            LOGGER.error("[音频] (%d/%d) 失败 %s: %s: %s", i, len(samples), sid, type(exc).__name__, exc)

    LOGGER.info("[音频] 完成：成功 %d / 跳过 %d / 失败 %d（输出目录 %s）", n_ok, n_skip, n_fail, out_dir)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="语音模态未对齐特征提取（逐帧 74 维，不做时间对齐）")
    p.add_argument("--only", default=None, help="只处理 sample_id 含该子串的样本")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 条")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出")
    p.add_argument("--out", default=None, help="输出目录，默认 data/unaligned_features/audio")
    p.add_argument("--rate", type=float, default=OFFICIAL_RATE_HZ, help="帧率 Hz，默认 20（官方口径）")
    args = p.parse_args(argv)

    samples = U.list_samples(only=args.only, limit=args.limit)
    LOGGER.info("[音频] 待处理样本 %d 条", len(samples))
    out_dir = args.out or U.unaligned_dir(MODALITY)
    run(samples, out_dir, overwrite=args.overwrite, rate_hz=args.rate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
