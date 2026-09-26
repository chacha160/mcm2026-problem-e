# -*- coding: utf-8 -*-
"""q1v2_media.py —— 音频/视觉原生观测与词级聚合（问题一 v2）

本模块是对照实现的 `q1_feature_core.py`（L69-272, L331-387）的移植，
**含全部 HardStop 断言**。那些断言不是装饰——它们正是「不强行制造伪对齐」的执行体：

* 音频：要求每个重采样帧都有 presentation PTS，且重采样后**首尾相接**（残差 ≤1e-8）。
  出现空洞或重叠即 HardStop，而不是把两个不连续的时刻当成连续时间轴。
* 视觉：要求帧 PTS 严格递增；支撑区间由**相邻帧 PTS 的中点**构造，
  末帧支撑取**实际解码到的最后一帧末尾**（`decoded_end`），
  而**不是**容器声明的流时长——后者是元数据，不构成「这一帧覆盖到那时」的证据。
* 聚合：只认**特征代表时刻落在词的半开区间 `[start, end)` 内**的观测；
  一个都落不到就记 0 并标 `valid=0`，绝不外扩窗口去凑。

移植时保留了一处 对照实现的**命名陷阱**：`aggregate_by_center` 的形参 `starts/ends`
是**词区间**，而 `centers` 是**特征代表时刻**——与名字给人的直觉相反。
调用方务必对照 `q1v2_build.media_stage` 的实参顺序。

本模块 import 时不加载 mediapipe / opensmile（延迟到函数内），
使核验器仍可裸跑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from q1v2_contract import (
    FACE_MODEL_DIM,
    OPENMILE_LLD_DIM,
    TOL,
    HardStop,
    sha256,
)

#: MediaPipe 的 LLD 步长不是常量——实测值由 `opensmile_features` 从返回值量出。
#: 这里只放一个"是否已量过"的哨兵，防止有人误把它当配置常量去推算 center。
_SAMPLE_RATE = 16000


# --------------------------------------------------------------------------
# MediaPipe blendshape 名单
# --------------------------------------------------------------------------

_BLENDSHAPE_NAMES: Optional[List[str]] = None


def camel_name(enum_name: str) -> str:
    """`BROW_DOWN_LEFT` → `browDownLeft`；`NEUTRAL` → `_neutral`（特例）。

    那个下划线不是笔误：MediaPipe 的 `_neutral` 与其余 51 项命名法不同，
    照搬枚举名的转换结果才能与对照实现的 `blendshape_category_names_order` 逐项对上。
    """
    parts = enum_name.lower().split("_")
    if parts == ["neutral"]:
        return "_neutral"
    return parts[0] + "".join(part.title() for part in parts[1:])


def blendshape_names() -> List[str]:
    """52 项 blendshape 名字，按枚举顺序。顺序即身份——它决定 (K,52) 每列的含义。"""
    global _BLENDSHAPE_NAMES
    if _BLENDSHAPE_NAMES is None:
        from mediapipe.tasks.python.vision.face_landmarker import Blendshapes
        names = [camel_name(member.name) for member in Blendshapes]
        if len(names) != FACE_MODEL_DIM or len(set(names)) != FACE_MODEL_DIM:
            raise HardStop(
                f"MediaPipe 枚举未定义 {FACE_MODEL_DIM} 个互异的 blendshape 名称"
                f"（实得 {len(names)}）"
            )
        _BLENDSHAPE_NAMES = names
    return list(_BLENDSHAPE_NAMES)


def create_face_landmarker(model_path: Path):
    """每个样本**新建**一个 landmarker，用完必须 `close()`。

    A0 踩过复用实例的坑：MediaPipe 的 VIDEO 模式内部维护时间戳状态，
    跨样本复用会让下一个样本的毫秒时间戳被判定为回退而整批失败。
    另外 `num_faces=1` + `output_face_blendshapes=True` 缺一不可。
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


# --------------------------------------------------------------------------
# 共享时间原点
# --------------------------------------------------------------------------


def probe_presentation_starts(path: Path) -> Dict[str, Any]:
    """先探两条流的 presentation 起点，供编排层确定共享 `t0`。

    **为什么必须单独探一次**：`decode_audio_16k` 会断言
    `abs(音频起点 - shared_t0) <= 1/16000`，而 `shared_t0` 本该由音频起点定义——
    这是个先有鸡还是先有蛋的问题。先在不解码的前提下把两条流的起点读出来，
    确认二者一致后再定 `t0`，解码器里那条断言才有意义（它校验的是「解码实际
    拿到的首帧 PTS」与「容器声明的流起点」是否真的一致）。

    两者不一致的样本记 `EXTRACTION_OR_TIMELINE_ANOMALY`，不硬凑。
    """
    import av

    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            raise HardStop("no audio stream")
        if not container.streams.video:
            raise HardStop("no video stream")
        audio = container.streams.audio[0]
        video = container.streams.video[0]
        audio_start = (float(audio.start_time * audio.time_base)
                       if audio.start_time is not None else None)
        video_start = (float(video.start_time * video.time_base)
                       if video.start_time is not None else None)
    if audio_start is None:
        raise HardStop("audio stream start PTS is missing")
    if video_start is None:
        raise HardStop("video stream start PTS is missing")
    return {
        "audio_stream_start_sec": audio_start,
        "video_stream_start_sec": video_start,
        "abs_offset_sec": abs(audio_start - video_start),
        "aligned_within_tolerance": abs(audio_start - video_start) <= TOL,
    }


# --------------------------------------------------------------------------
# 音频
# --------------------------------------------------------------------------


def decode_audio_16k(path: Path, shared_t0: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """PyAV 解码音轨 → 16 kHz 单声道 float32。返回 `(波形, 溯源信息)`。

    用 PyAV 而非 ffmpeg CLI，是为了拿到**每个重采样帧的 presentation PTS**：
    CLI 管道只给一段裸字节，无从校验时间轴是否连续。这里逐帧核验首尾相接，
    残差超过 1e-8 s 即 HardStop——那种情况下「第 t 秒的音频」已不可信，
    再去谈词级聚合就是把错误藏进特征里。
    """
    import av

    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            raise HardStop("no audio stream")
        stream = container.streams.audio[0]
        raw_info = {
            "codec": stream.codec_context.name,
            "time_base": str(stream.time_base),
            "stream_start_time": stream.start_time,
            "stream_start_sec": (float(stream.start_time * stream.time_base)
                                 if stream.start_time is not None else None),
            "stream_duration_sec": (float(stream.duration * stream.time_base)
                                    if stream.duration is not None else None),
            "input_sample_rate": stream.codec_context.sample_rate,
            "input_channels": stream.codec_context.channels,
        }
        if raw_info["stream_start_sec"] is None:
            raise HardStop("audio stream start PTS is missing")

        raw_frames = 0
        raw_total = 0
        raw_pts: List[float] = []
        raw_gaps: List[float] = []
        previous_end: Optional[float] = None
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=_SAMPLE_RATE)
        output_chunks: List[np.ndarray] = []
        output_pts: List[float] = []
        output_info: List[Dict[str, Any]] = []

        def collect(converted):
            if converted is None:
                return
            frames = converted if isinstance(converted, list) else [converted]
            for out in frames:
                if out.pts is None or out.time_base is None:
                    raise HardStop("resampled audio frame is missing presentation PTS")
                pts_sec = float(out.pts * out.time_base)
                arr = np.asarray(out.to_ndarray(), dtype=np.float32).reshape(-1)
                if arr.size != out.samples:
                    raise HardStop("resampled mono frame shape does not match sample count")
                output_pts.append(pts_sec)
                output_chunks.append(arr)
                output_info.append({"pts_sec": pts_sec, "samples": int(out.samples),
                                    "rate": int(out.sample_rate)})

        for frame in container.decode(stream):
            raw_frames += 1
            raw_total += int(frame.samples)
            if frame.pts is None or frame.time_base is None:
                raise HardStop("decoded audio frame is missing presentation PTS")
            pts_sec = float(frame.pts * frame.time_base)
            raw_pts.append(pts_sec)
            if previous_end is not None:
                raw_gaps.append(pts_sec - previous_end)
            previous_end = pts_sec + frame.samples / frame.sample_rate
            collect(resampler.resample(frame))
        collect(resampler.resample(None))

    if not output_chunks:
        raise HardStop("audio resampling produced no samples")
    waveform = np.concatenate(output_chunks).astype(np.float32, copy=False)
    if not np.isfinite(waveform).all():
        raise HardStop("16 kHz audio contains nonfinite samples")

    start_sec = output_pts[0]
    end_sec = start_sec + len(waveform) / float(_SAMPLE_RATE)
    if abs(start_sec - shared_t0) > 1 / _SAMPLE_RATE:
        raise HardStop(
            f"audio presentation start {start_sec:.9f} differs from shared t0 {shared_t0:.9f}"
        )
    expected = start_sec
    max_residual = 0.0
    for frame_info in output_info:
        residual = frame_info["pts_sec"] - expected
        max_residual = max(max_residual, abs(residual))
        if abs(residual) > 1e-8:
            raise HardStop(
                f"resampled audio has a non-contiguous presentation gap/overlap "
                f"of {residual:.9f}s"
            )
        expected = frame_info["pts_sec"] + frame_info["samples"] / float(_SAMPLE_RATE)

    raw_max_abs_residual = max((abs(g) for g in raw_gaps), default=0.0)
    if raw_max_abs_residual > 1 / raw_info["input_sample_rate"] + 1e-6:
        raise HardStop(
            f"decoded source audio contains unexplained PTS discontinuity "
            f"({raw_max_abs_residual:.9f}s)"
        )

    info = {
        **raw_info,
        "decoded_input_audio_frames": raw_frames,
        "decoded_input_samples": raw_total,
        "decoded_input_duration_sec": raw_total / raw_info["input_sample_rate"],
        "decoded_input_first_pts_sec": raw_pts[0],
        "decoded_input_last_frame_end_sec": previous_end,
        "decoded_input_max_abs_pts_residual_sec": raw_max_abs_residual,
        "resampler": "PyAV AudioResampler(format=fltp, layout=mono, rate=16000)",
        "resampled_output_rate": _SAMPLE_RATE,
        "resampled_output_samples": int(waveform.size),
        "resampled_start_sec": start_sec,
        "resampled_end_sec": end_sec,
        "resampled_duration_sec": len(waveform) / float(_SAMPLE_RATE),
        "resampled_frame_count": len(output_info),
        "resampled_max_abs_pts_residual_sec": max_residual,
        "resampled_frame_pts": output_info,
        "audio_time_map": [{
            "sample_start": 0, "sample_end": int(waveform.size),
            "presentation_start_sec": start_sec, "presentation_end_sec": end_sec,
            "rate": _SAMPLE_RATE,
        }],
    }
    return waveform, info


def quantize_int16(audio: np.ndarray) -> np.ndarray:
    """重现对照实现的 `write_wav` 所用的量化：`rint(clip(x,-1,1) * 32767) → int16`。

    数字静音的判定必须建立在这个量化结果上，而不是浮点波形上：
    一段幅度 1e-6 的浮点噪声量化后就是全零，对照实现的判据也是这么定的。
    走同一条量化路径，两条静音样本的锚点才能位精确复现。
    """
    clipped = np.clip(audio, -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype("<i2", copy=False)


def digital_silence_evidence(audio: np.ndarray) -> Dict[str, Any]:
    """数字静音的位精确证据。**这是本套代码里唯一由机器直接证实的"内容"结论。**

    `confirmed_match` / `confirmed_mismatch` 都需要人工听辨，我们没有；
    但「量化后的 int16 逐采样全零」是波形层面的确定事实，不依赖任何主观判断，
    因此可以如实写 `audio_speech_valid=0` 与 `text_audio_correspondence=no_speech`。
    """
    quantized = quantize_int16(audio)
    nonzero = int(np.count_nonzero(quantized))
    return {
        "method": "int16 quantization (round(clip(x,-1,1)*32767)) of the 16 kHz resampled waveform",
        "n_samples": int(quantized.size),
        "n_nonzero_samples": nonzero,
        "all_zero": nonzero == 0,
        "max_abs_quantized": int(np.abs(quantized).max()) if quantized.size else 0,
        "float_max_abs": float(np.abs(audio).max()) if audio.size else 0.0,
    }


# --------------------------------------------------------------------------
# 视觉
# --------------------------------------------------------------------------


def decode_video_blendshapes(path: Path, t0: float, face_landmarker):
    """逐帧解码 → MediaPipe blendshape。返回 `(info, pts, support_start, support_end, vectors, valid)`。

    支撑区间的构造是本模块最容易被"想当然"写错的地方，三处必须照搬：

    1. `support_start[0] = pts[0]`，内部边界取**相邻帧 PTS 的中点**；
    2. `support_end[-1] = coverage_end = decoded_end`，即**实际解码到的最后一帧末尾**，
       不是容器声明的流时长——元数据不能当成覆盖证据；
    3. 未检出人脸的帧填 `zeros(52)` 且 `valid=False`。**填零不等于"中性表情"**，
       故必须同时带 `valid` 掩码；聚合时按掩码剔除，而不是把零当观测。
    """
    import av
    import mediapipe as mp

    with av.open(str(path), mode="r") as container:
        if not container.streams.video:
            raise HardStop("no video stream")
        stream = container.streams.video[0]
        info = {
            "codec": stream.codec_context.name,
            "time_base": str(stream.time_base),
            "stream_start_time": stream.start_time,
            "stream_start_sec": (float(stream.start_time * stream.time_base)
                                 if stream.start_time is not None else None),
            "stream_duration_sec": (float(stream.duration * stream.time_base)
                                    if stream.duration is not None else None),
            "average_rate_metadata_only": (str(stream.average_rate)
                                           if stream.average_rate else None),
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "decoder": "PyAV/FFmpeg decoded frame presentation PTS",
        }
        if info["stream_start_sec"] is None:
            raise HardStop("video stream start PTS is missing")

        names_order = blendshape_names()
        expected_names = set(names_order)
        pts: List[float] = []
        durations: List[Optional[float]] = []
        vectors: List[np.ndarray] = []
        valid: List[bool] = []
        per_frame_counts: List[int] = []
        last_ms: Optional[int] = None

        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise HardStop("decoded video frame is missing presentation PTS")
            frame_pts = float(frame.pts * frame.time_base)
            relative_ms = int(round((frame_pts - t0) * 1000.0))
            if last_ms is not None and relative_ms <= last_ms:
                raise HardStop(
                    f"MediaPipe VIDEO millisecond timestamps collide or regress "
                    f"at {frame_pts:.9f}s"
                )
            last_ms = relative_ms
            image_np = frame.to_ndarray(format="rgb24")
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
            result = face_landmarker.detect_for_video(image, relative_ms)

            pts.append(frame_pts)
            durations.append(float(frame.duration * frame.time_base)
                             if frame.duration else None)
            found = result.face_blendshapes or []
            per_frame_counts.append(len(found))
            if len(found) == 0:
                vectors.append(np.zeros(FACE_MODEL_DIM, dtype=np.float32))
                valid.append(False)
                continue
            if len(found) != 1:
                raise HardStop(
                    f"MediaPipe num_faces=1 yielded {len(found)} face blendshape sets"
                )
            categories = found[0]
            names = [c.category_name for c in categories]
            if len(names) != FACE_MODEL_DIM or len(set(names)) != FACE_MODEL_DIM:
                raise HardStop(
                    f"MediaPipe returned {len(names)} blendshape categories "
                    f"(expected {FACE_MODEL_DIM} unique)"
                )
            if set(names) != expected_names:
                raise HardStop(
                    "MediaPipe blendshape category names differ from its frozen "
                    "52-category schema"
                )
            by_name = {c.category_name: float(c.score) for c in categories}
            vector = np.asarray([by_name[n] for n in names_order], dtype=np.float32)
            if not np.isfinite(vector).all():
                raise HardStop("MediaPipe returned nonfinite blendshape values")
            vectors.append(vector)
            valid.append(True)

    if not pts:
        raise HardStop("video decode returned no presentation frames")
    if abs(pts[0] - info["stream_start_sec"]) > 1e-6:
        raise HardStop("first decoded video PTS differs from video stream presentation start")
    deltas = np.diff(np.asarray(pts, dtype=np.float64))
    if np.any(deltas <= 0):
        raise HardStop("decoded video presentation PTS is not strictly increasing")

    stream_end = (info["stream_start_sec"] + info["stream_duration_sec"]
                  if info["stream_duration_sec"] is not None else None)
    decoded_end = pts[-1] + (durations[-1] or (pts[-1] - pts[-2] if len(pts) > 1 else 0.0))
    # 流/影片时长是元数据，不构成「最后一帧覆盖到那时」的证据。
    # 保留为诊断字段，支撑区间只由最后一帧的实际时长（或最后一个 PTS 步长）推出。
    coverage_end = decoded_end
    if pts[0] > t0 + 1 / 90000 or pts[-1] > coverage_end + TOL:
        raise HardStop("video presentation coverage cannot be reconciled with decoded PTS")

    support_start = np.empty(len(pts), dtype=np.float64)
    support_end = np.empty(len(pts), dtype=np.float64)
    support_start[0] = pts[0]
    support_end[-1] = coverage_end
    for i in range(1, len(pts)):
        midpoint = (pts[i - 1] + pts[i]) / 2.0
        support_end[i - 1] = midpoint
        support_start[i] = midpoint

    median_delta = float(np.median(deltas)) if len(deltas) else None
    info.update({
        "frames_decoded": len(pts),
        "first_pts_sec": pts[0],
        "last_pts_sec": pts[-1],
        "median_pts_delta_sec": median_delta,
        "max_pts_delta_sec": float(np.max(deltas)) if len(deltas) else None,
        "pts_gap_count_over_1_5x_median": (int(np.sum(deltas > median_delta * 1.5))
                                           if len(deltas) else 0),
        "pts_gaps": [{"left_pts_sec": float(pts[i]), "right_pts_sec": float(pts[i + 1]),
                      "delta_sec": float(deltas[i])}
                     for i in range(len(deltas))
                     if deltas[i] > np.median(deltas) * 1.5],
        "decoded_last_frame_end_sec": decoded_end,
        "actual_video_coverage_start_sec": pts[0],
        "actual_video_coverage_end_sec": coverage_end,
        "stream_or_movie_end_minus_decoded_frame_end_sec": (
            stream_end - decoded_end if stream_end is not None else None),
        "num_faces_per_frame": per_frame_counts,
        "blendshape_category_names_order": names_order,
        "blendshape_order_basis": (
            "mediapipe.tasks.python.vision.face_landmarker.Blendshapes enum order"),
        "face_detected_frame_count": int(np.count_nonzero(valid)),
        "video_frame_count": len(pts),
    })
    return (info, np.asarray(pts, dtype=np.float64), support_start, support_end,
            np.asarray(vectors, dtype=np.float32), np.asarray(valid, dtype=np.uint8))


# --------------------------------------------------------------------------
# openSMILE
# --------------------------------------------------------------------------


def opensmile_features(audio: np.ndarray) -> Dict[str, Any]:
    """16 kHz 波形 → eGeMAPSv02 LLD，25 维/窗，含实测 start/end/center。

    **只读返回值，绝不自己按 20 ms 窗 / 10 ms 步推算 center**（R10）。
    openSMILE 的 `_series_to_frame` 会把首窗 start 夹到切片起点、末窗 end 夹到切片终点，
    所以第一窗的宽度与其余窗不同（本机实测：1 s 静音 → 96 窗，首窗 `[0, 0.02]`）。
    自算会把这一层"元数据式的时间标签"当成物理分析窗，进而在词的边界处错判归属。

    同理，eGeMAPSv02 内部含 60 ms 与 20 ms 两种窗、10 ms 步长，部分 LLD 还做了 3 帧平滑，
    因此这些 start/end 的正确称呼是**「LLD 输出时间标签及归词代表时刻」**，
    不是「物理分析窗」。
    """
    import pandas as pd
    from opensmile import FeatureLevel, FeatureSet, Smile

    smile = Smile(feature_set=FeatureSet.eGeMAPSv02,
                  feature_level=FeatureLevel.LowLevelDescriptors,
                  num_workers=1, multiprocessing=False, verbose=False)
    frame = smile.process_signal(audio, sampling_rate=_SAMPLE_RATE)

    if frame.shape[1] != OPENMILE_LLD_DIM or len(smile.feature_names) != OPENMILE_LLD_DIM:
        raise HardStop(
            f"openSMILE produced {frame.shape[1]} columns, "
            f"expected {OPENMILE_LLD_DIM} eGeMAPSv02 LLDs"
        )
    if not isinstance(frame.index, pd.MultiIndex) or list(frame.index.names) != ["start", "end"]:
        raise HardStop(
            f"openSMILE returned no interpretable start/end support intervals: "
            f"{frame.index.names}"
        )

    starts = np.asarray([x.total_seconds() if hasattr(x, "total_seconds") else float(x)
                         for x in frame.index.get_level_values("start")], dtype=np.float64)
    ends = np.asarray([x.total_seconds() if hasattr(x, "total_seconds") else float(x)
                       for x in frame.index.get_level_values("end")], dtype=np.float64)
    values = frame.to_numpy(dtype=np.float32, copy=True)
    if len(starts) != len(values) or np.any(ends <= starts) or np.any(np.diff(starts) < -TOL):
        raise HardStop("openSMILE start/end support intervals are malformed or nonmonotonic")

    # 实测帧长与帧移。写成"量出来的"而非"配置里的"，供核验 V5 与报告引用。
    frame_len = float(ends[1] - starts[1]) if len(starts) > 1 else float(ends[0] - starts[0])
    frame_step = float(starts[1] - starts[0]) if len(starts) > 1 else float("nan")

    return {
        "values": values,
        "starts": starts,
        "ends": ends,
        "centers": (starts + ends) / 2.0,
        "feature_names": list(smile.feature_names),
        "config_path": str(smile.config_path),
        "config_sha256": sha256(Path(smile.config_path)),
        "feature_set": "eGeMAPSv02",
        "feature_level": "LowLevelDescriptors",
        "frame_count": int(len(values)),
        "lld_frame_len_sec": frame_len,
        "lld_frame_step_sec": frame_step,
    }


# --------------------------------------------------------------------------
# 词级聚合
# --------------------------------------------------------------------------


def aggregate_by_center(words: Sequence[Any],
                        starts: Sequence[float],
                        ends: Sequence[float],
                        centers: np.ndarray,
                        values: np.ndarray,
                        valid_frames: Optional[np.ndarray] = None,
                        dim: int = OPENMILE_LLD_DIM,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """把逐帧观测按「代表时刻落在词区间内」聚合到词。

    **形参命名陷阱（照搬对照实现，勿改）**：
        `starts` / `ends`  —— **词**的半开区间 `[start, end)`，长度 = 词数 L
        `centers`          —— **特征**的代表时刻，长度 = 观测帧数 K
    名字看起来像反的，但调用点（`q1v2_build.media_stage`）就是这么传的。

    返回 `(means_stds, valid, indptr, indices, local_issues)`：
        means_stds  `(L, 2*dim)`  —— 前 `dim` 列是均值，后 `dim` 列是**总体标准差**
        valid       `(L,)`        —— 有观测且观测全有限
        indptr/indices            —— CSR，用于**从成品回捞源观测重算**
        local_issues              —— 逐词的问题清单

    三处"不凑合"：
      * 词区间非法（非有限 / 时长为零或负）→ 直接 `valid=0`，不去找最近的帧顶替；
      * 区间内一个合法观测都没有 → `valid=0`，不回退到相邻区间；
      * 观测里有非有限值 → 整词 `valid=0` 并记录，不"跳过坏的取平均"。
    """
    L = len(words)
    means_stds = np.zeros((L, dim * 2), dtype=np.float32)
    valid = np.zeros(L, dtype=np.uint8)
    indptr: List[int] = [0]
    all_indices: List[int] = []
    local_issues: List[Dict[str, Any]] = []
    mask = None if valid_frames is None else np.asarray(valid_frames, dtype=bool)

    for wi in range(L):
        a, b = starts[wi], ends[wi]
        if not np.isfinite(a) or not np.isfinite(b) or b <= a:
            indptr.append(len(all_indices))
            local_issues.append({"word_index": wi,
                                 "reason": "invalid or zero-duration word interval"})
            continue
        indexes = np.flatnonzero((centers >= a) & (centers < b))
        if mask is not None:
            indexes = indexes[mask[indexes]]
        all_indices.extend(int(i) for i in indexes)
        indptr.append(len(all_indices))
        if len(indexes) == 0:
            local_issues.append({
                "word_index": wi,
                "reason": "no valid feature center in half-open word interval",
            })
            continue
        selected = values[indexes]
        if selected.shape[1] != dim:
            raise HardStop(
                f"center assignment expected {dim} columns, got {selected.shape[1]}"
            )
        if not np.isfinite(selected).all():
            local_issues.append({
                "word_index": wi,
                "reason": "nonfinite raw feature in assigned support",
                "feature_indices": np.flatnonzero(~np.isfinite(selected)).tolist()[:20],
            })
            continue
        means_stds[wi, :dim] = selected.mean(axis=0, dtype=np.float64).astype(np.float32)
        means_stds[wi, dim:] = selected.std(axis=0, ddof=0, dtype=np.float64).astype(np.float32)
        valid[wi] = 1

    return (means_stds, valid, np.asarray(indptr, dtype=np.int32),
            np.asarray(all_indices, dtype=np.int32), local_issues)


__all__ = [
    "camel_name", "blendshape_names", "create_face_landmarker",
    "probe_presentation_starts", "decode_audio_16k", "quantize_int16",
    "digital_silence_evidence", "decode_video_blendshapes",
    "opensmile_features", "aggregate_by_center",
]
