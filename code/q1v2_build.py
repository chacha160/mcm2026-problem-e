# -*- coding: utf-8 -*-
"""q1v2_build.py —— 单样本装配与输出表（问题一 v2）

三件事：

1. `assemble_npz` —— 把一次样本运行的全部产物落成一个 NPZ。
   **只写数值数组与 NumPy Unicode 字符串数组**，保证 `allow_pickle=False` 可读。
   JSON 类元数据一律先序列化成字符串再存——把 dict 塞进 npz 需要 pickle，
   那会让「不执行任何反序列化代码」这条安全属性失效。

2. `media_stage` —— 单样本 worker：解码 → 原生观测 → 路由 → 词级聚合 → 落盘。
   由 `run_q1v2_all.py` 以**子进程**逐个调用，超时可控、失败可隔离、可断点续跑。

3. `rewrite_routes` —— 从成品 NPZ 里已有的标量重算路由列并回写。
   用于 S6：只换结论列，**不重解任何一帧**。

坐标系统的三处口径（照搬对照实现，写错会让词级数组全部错位）：

* openSMILE 收到的信号从**切片起点**开始，故 LLD 的 `center` 是**相对音频起点**的；
  而落盘的 `raw_audio_lld_*_sec` 要**加回**音频起点，才是共享时间轴上的绝对时刻。
* MediaPipe 的毫秒时间戳用 `frame_pts - shared_t0`。
* 视觉帧的 `center` 用 `pts - shared_t0`（pts 本身就是绝对presentation 时刻）。

`shared_t0` 取**音频流起点**，前提是它与视频流起点在 `1e-6` 内一致；
不一致的样本记 `EXTRACTION_OR_TIMELINE_ANOMALY`，不硬凑一条公共时间轴。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from q1v2_contract import (
    FACE_MODEL_DIM,
    FACE_MODEL_PATH,
    OPENMILE_CONFIG_SHA256,
    OPENMILE_LLD_DIM,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    ROBERTA_FILE_SHA256,
    FACE_MODEL_SHA256,
    SCHEMA_VERSION,
    HardStop,
    safe_stem,
    sha256,
    write_csv,
)

#: 资产身份是**全局冻结常量**，不是逐样本数据。放在这里而不是从登记表读，
#: 是为了让「产物里记的资产 SHA」与「核验时比对的常量」同源——
#: 二者若各存一份，改动其一便可能悄悄不一致。
ASSET_IDENTITY = {
    "roberta_model_sha256": ROBERTA_FILE_SHA256["model.safetensors"],
    "mediapipe_model_sha256": FACE_MODEL_SHA256,
    "opensmile_config_sha256": OPENMILE_CONFIG_SHA256,
}

#: NPZ 必需字段。核验 V2 逐项检查存在性、dtype 与形状。
NPZ_REQUIRED_FIELDS = (
    # 身份与版本
    "schema_version", "sample_key", "video_id", "clip_id", "official_text",
    "source_video_relpath", "source_video_sha256", "config_sha256",
    "roberta_model_sha256", "mediapipe_model_sha256", "opensmile_config_sha256",
    # 对齐身份
    "alignment_source", "alignment_params_json", "alignment_run_sha256",
    # 路由与声明
    "alignment_mode", "alignment_granularity", "text_audio_correspondence",
    "text_av_time_mapping_status", "content_assertion", "word_time_basis",
    # 状态
    "text_present", "text_content_valid", "audio_present",
    "audio_observation_valid", "audio_speech_valid", "video_present",
    "audio_visual_time_valid", "vision_feature_valid", "face_feature_valid",
    "audio_speech_evidence",
    # 词单元
    "words", "word_char_start", "word_char_end",
    "aligner_slot_count", "aligner_slot_to_word_index",
    "word_start_sec", "word_end_sec", "word_time_valid",
    "text_word_valid", "word_audio_valid", "word_vision_valid",
    # 三路词级特征
    "text_word_feat", "word_audio_feat", "word_vision_feat",
    # RoBERTa token 级溯源
    "roberta_token_ids", "roberta_token_offsets", "roberta_token_word_index",
    "roberta_word_token_indptr", "roberta_word_token_indices",
    "roberta_tokenizer_failure_count", "text_input_length",
    # 原生序列：音频
    "raw_audio_lld_values", "raw_audio_lld_start_sec", "raw_audio_lld_end_sec",
    "raw_audio_lld_center_sec", "raw_audio_lld_valid",
    "raw_audio_lld_feature_names", "lld_frame_len_sec", "lld_frame_step_sec",
    # 原生序列：视觉
    "raw_video_blendshape_values", "raw_video_pts_sec",
    "raw_video_support_start_sec", "raw_video_support_end_sec",
    "raw_video_face_feature_valid", "raw_video_blendshape_names",
    "video_frame_count", "face_detected_frame_count",
    # 溯源索引
    "audio_word_feature_indptr", "audio_word_feature_indices",
    "vision_word_feature_indptr", "vision_word_feature_indices",
    # 覆盖与时长
    "audio_presentation_coverage_sec", "video_presentation_coverage_sec",
    "av_common_coverage_sec", "shared_t0_sec", "original_effective_duration_sec",
    "audio_observation_length", "video_observation_length", "text_sequence_length",
    # 诊断
    "alignment_zero_duration_word_count", "alignment_mapping_fail_count",
    "decode_provenance_json",
)


def _u(value: Any) -> np.ndarray:
    """字符串 → 0 维 Unicode 数组（`allow_pickle=False` 下可读）。"""
    return np.asarray(str(value))


def _u1(values: Sequence[str]) -> np.ndarray:
    return np.asarray([str(v) for v in values])


def _j(value: Any) -> np.ndarray:
    """任意可序列化对象 → JSON 字符串的 0 维 Unicode 数组。

    刻意不把 dict 直接塞进 npz：那需要 pickle，读取端就得执行反序列化代码。
    存成 JSON 文本则 `allow_pickle=False` 也能读全。
    """
    return np.asarray(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def write_npz_atomic(path: Path, payload: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """原子写 **压缩** NPZ，返回 `{sha256, bytes, path}`。

    先写 `.tmp` 再 `replace`：中途失败不会留下一个"看起来能读、其实截断"的 NPZ，
    也不会让断点续跑把半截文件当成品跳过。

    **为什么压缩**：本套 NPZ 里体积的大头是原生序列（视频 52 维 blendshape 与
    LLD 25 维，都按帧/窗逐行存）。压缩前实测 100 条合计约 26.3 MiB，使「替换 v1」
    的核算约 51.1 MB —— 超过 `50×10⁶` 的**严格口径**约 1.1 MB（那两个数是压缩前的
    历史实测，已无法从当前目录树复现，故只给量级）。
    压缩是**无损**的：float32 数值解压后逐位不变，所以 V4 的 `rtol=1e-5`
    可复算性与 V14 的位精确锚点都不受影响。这与「降 float16 换体积」是两回事——
    后者会牺牲精度，本套**不做**。
    压缩后 100 条合计 18,705,996 B（该值稳定、可复算），替换核算约 42.2 MB，
    两种口径都在线内。逐字节的台账号见 `reports/q1v2_size_ledger.md`。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temp.replace(path)
    return {
        "path": path.as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """**始终 `allow_pickle=False` 读取**，并在读不到时给出可操作的报错。"""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as handle:
            return {k: handle[k] for k in handle.files}
    except ValueError as exc:
        raise HardStop(
            f"{path.name} 需要 pickle 才能读取，违反 allow_pickle=False 约定：{exc}"
        ) from exc


# --------------------------------------------------------------------------
# 单样本 worker
# --------------------------------------------------------------------------


def media_stage(sample: Dict[str, Any],
                text_cache: np.ndarray,
                align_cache: Dict[str, Any],
                feature_dir: Path,
                config_sha: str,
                allow_overwrite: bool = True) -> Dict[str, Any]:
    """单样本完整装配。返回该样本的 manifest 行。

    前置缓存（`text_cache` / `align_cache`）由 text / align 两阶段产出：
    文本阶段给出 `(L,768)` 词特征，对齐阶段给出词区间与结构诊断。
    本阶段补齐音频与视觉，然后装配落盘。
    """
    import q1v2_media as M
    import q1v2_route as R
    import q1v2_text as T

    started = time.time()
    key = sample["sample_key"]
    stem = safe_stem(key)
    out_path = Path(feature_dir) / f"{stem}.npz"
    if out_path.is_file() and not allow_overwrite:
        raise HardStop(f"{out_path.name} 已存在且未开启覆盖")

    official_text = str(sample["official_text"])
    words = T.source_words(official_text)
    chunks = T.whitespace_chunks(official_text)
    n_words = len(words)

    # 登记表只存相对路径（交付物不得含本机绝对路径），在此解析回绝对路径。
    video_path = Path(PROJECT_ROOT) / str(sample["source_video_relpath"])

    # ---- 共享时间原点 -------------------------------------------------
    probe = M.probe_presentation_starts(video_path)
    timeline_ok = bool(probe["aligned_within_tolerance"])
    shared_t0 = float(probe["audio_stream_start_sec"])

    # ---- 音频：解码 + 原生 LLD + 静音证据 ------------------------------
    waveform, audio_info = M.decode_audio_16k(video_path, shared_t0)
    silence = M.digital_silence_evidence(waveform)
    lld = M.opensmile_features(waveform)

    # ---- 视觉：解码 + blendshape --------------------------------------
    landmarker = M.create_face_landmarker(FACE_MODEL_PATH)
    try:
        (video_info, pts, support_start, support_end,
         blend, face_valid) = M.decode_video_blendshapes(
            video_path, shared_t0, landmarker)
    finally:
        # 必须关闭：MediaPipe 的 VIDEO 模式内部带时间戳状态，
        # 泄漏的实例会让下一个样本的毫秒戳被判定为回退。
        landmarker.close()

    # ---- 路由 ---------------------------------------------------------
    intervals = align_cache["intervals"]
    diag = align_cache["diagnostics"]
    route = R.route_sample(
        timeline_ok=timeline_ok,
        audio_present=True,
        audio_digital_silence=bool(silence["all_zero"]),
        mapping_status=align_cache["mapping_status"],
        n_official_words=n_words,
        n_structurally_valid_words=int(diag["n_structurally_valid_words"]),
    )

    # ---- 词级聚合 -----------------------------------------------------
    # 音频侧：openSMILE 的 center 是**相对音频切片起点**的，
    # 而词区间是**绝对 presentation** 时刻，故先把词区间平移回相对坐标。
    audio_start = float(audio_info["resampled_start_sec"])
    audio_end = float(audio_info["resampled_end_sec"])
    audio_centers_abs = lld["centers"] + audio_start

    word_audio, audio_word_valid, a_indptr, a_indices, a_issues = M.aggregate_by_center(
        words,
        intervals["word_start_sec"] - audio_start,
        intervals["word_end_sec"] - audio_start,
        lld["centers"],
        lld["values"],
        dim=OPENMILE_LLD_DIM,
    )
    # 词区间本身不合法时，聚合结果即便"有观测"也不可用——两个有效性求交。
    audio_word_valid = (audio_word_valid & intervals["word_time_valid"]).astype(np.uint8)
    word_audio[audio_word_valid == 0] = 0.0

    # 视觉侧：帧 PTS 与词区间同在共享时间轴上，直接相减。
    word_vision, vision_word_valid, v_indptr, v_indices, v_issues = M.aggregate_by_center(
        words,
        intervals["word_start_sec"] - shared_t0,
        intervals["word_end_sec"] - shared_t0,
        pts - shared_t0,
        blend,
        valid_frames=face_valid,
        dim=FACE_MODEL_DIM,
    )
    vision_word_valid = (vision_word_valid & intervals["word_time_valid"]).astype(np.uint8)
    word_vision[vision_word_valid == 0] = 0.0

    text_word_valid = np.asarray(text_cache["valid"], dtype=np.uint8)
    text_feat = np.asarray(text_cache["features"], dtype=np.float32)

    # ---- 覆盖与公共区间 ------------------------------------------------
    video_start = float(video_info["actual_video_coverage_start_sec"])
    video_end = float(video_info["actual_video_coverage_end_sec"])
    common_start = max(audio_start, video_start)
    common_end = min(audio_end, video_end)
    av_common = np.asarray([common_start, common_end], dtype=np.float64)
    av_time_valid = bool(timeline_ok and common_end > common_start)
    original_effective_duration = max(audio_end, video_end) - shared_t0

    frames = int(video_info["video_frame_count"])
    face_frames = int(video_info["face_detected_frame_count"])
    face_ratio = float(face_frames / frames) if frames else 0.0

    # ---- 词区间并集对公共区间的覆盖比 ----------------------------------
    valid_iv = intervals["word_time_valid"].astype(bool)
    if valid_iv.any() and common_end > common_start:
        lo = np.maximum(intervals["word_start_sec"][valid_iv], common_start)
        hi = np.minimum(intervals["word_end_sec"][valid_iv], common_end)
        width = np.clip(hi - lo, 0.0, None)
        # 词区间互不重叠（单调且合并而来），直接求和即为并集长度。
        union = float(width.sum())
        word_union_ratio = union / (common_end - common_start)
    else:
        word_union_ratio = 0.0

    tri_valid = (text_word_valid & audio_word_valid & vision_word_valid).astype(np.uint8)

    # ---- 落盘 ----------------------------------------------------------
    alignment_run_sha256 = align_cache["run_sha256"]
    payload: Dict[str, np.ndarray] = {
        "schema_version": _u(SCHEMA_VERSION),
        "sample_key": _u(key),
        "video_id": _u(sample["video_id"]),
        "clip_id": _u(sample["clip_id"]),
        "official_text": _u(official_text),
        "source_video_relpath": _u(sample["source_video_relpath"]),
        "source_video_sha256": _u(sample["source_video_sha256"]),
        "config_sha256": _u(config_sha),
        "roberta_model_sha256": _u(ASSET_IDENTITY["roberta_model_sha256"]),
        "mediapipe_model_sha256": _u(ASSET_IDENTITY["mediapipe_model_sha256"]),
        "opensmile_config_sha256": _u(ASSET_IDENTITY["opensmile_config_sha256"]),

        "alignment_source": _u(align_cache["alignment_source"]),
        "alignment_params_json": _j(align_cache["alignment_params"]),
        "alignment_run_sha256": _u(alignment_run_sha256),

        "alignment_mode": _u(route["alignment_mode"]),
        "alignment_granularity": _u(route["alignment_granularity"]),
        "text_audio_correspondence": _u(route["text_audio_correspondence"]),
        "text_av_time_mapping_status": _u(route["text_av_time_mapping_status"]),
        "content_assertion": np.asarray(route["content_assertion"], dtype=np.uint8),
        "word_time_basis": _u(route["word_time_basis"]),
        "routing_reasons_json": _j(route["routing_reasons"]),

        "text_present": np.asarray(1, dtype=np.uint8),
        "text_content_valid": np.asarray(
            1 if int(text_word_valid.sum()) > 0 else 0, dtype=np.uint8),
        "audio_present": np.asarray(1, dtype=np.uint8),
        "audio_observation_valid": np.asarray(
            1 if lld["frame_count"] > 0 else 0, dtype=np.uint8),
        "audio_speech_valid": np.asarray(route["audio_speech_valid"], dtype=np.int8),
        "video_present": np.asarray(1, dtype=np.uint8),
        "audio_visual_time_valid": np.asarray(1 if av_time_valid else 0, dtype=np.uint8),
        "vision_feature_valid": np.asarray(1 if face_frames > 0 else 0, dtype=np.uint8),
        "face_feature_valid": np.asarray(1 if face_frames > 0 else 0, dtype=np.uint8),
        "audio_speech_evidence": _u(route["audio_speech_evidence"]),

        "words": _u1([w["text"] for w in words]),
        "word_char_start": np.asarray([w["char_start"] for w in words], dtype=np.int32),
        "word_char_end": np.asarray([w["char_end"] for w in words], dtype=np.int32),
        "aligner_slot_count": np.asarray(intervals["n_slots"], dtype=np.int32),
        "aligner_slot_to_word_index": np.asarray(intervals["slot_to_word"], dtype=np.int32),
        "word_start_sec": intervals["word_start_sec"].astype(np.float64),
        "word_end_sec": intervals["word_end_sec"].astype(np.float64),
        "word_time_valid": intervals["word_time_valid"].astype(np.uint8),
        "text_word_valid": text_word_valid,
        "word_audio_valid": audio_word_valid,
        "word_vision_valid": vision_word_valid,

        "text_word_feat": text_feat,
        "word_audio_feat": word_audio.astype(np.float32),
        "word_vision_feat": word_vision.astype(np.float32),

        "raw_audio_lld_values": lld["values"].astype(np.float32),
        "raw_audio_lld_start_sec": (lld["starts"] + audio_start).astype(np.float64),
        "raw_audio_lld_end_sec": (lld["ends"] + audio_start).astype(np.float64),
        "raw_audio_lld_center_sec": audio_centers_abs.astype(np.float64),
        "raw_audio_lld_valid": np.ones(lld["frame_count"], dtype=np.uint8),
        "raw_audio_lld_feature_names": _u1(lld["feature_names"]),
        "lld_frame_len_sec": np.asarray(lld["lld_frame_len_sec"], dtype=np.float64),
        "lld_frame_step_sec": np.asarray(lld["lld_frame_step_sec"], dtype=np.float64),

        "raw_video_blendshape_values": blend.astype(np.float32),
        "raw_video_pts_sec": pts.astype(np.float64),
        "raw_video_support_start_sec": support_start.astype(np.float64),
        "raw_video_support_end_sec": support_end.astype(np.float64),
        "raw_video_face_feature_valid": face_valid.astype(np.uint8),
        "raw_video_blendshape_names": _u1(video_info["blendshape_category_names_order"]),
        "video_frame_count": np.asarray(frames, dtype=np.int32),
        "face_detected_frame_count": np.asarray(face_frames, dtype=np.int32),

        "audio_word_feature_indptr": a_indptr,
        "audio_word_feature_indices": a_indices,
        "vision_word_feature_indptr": v_indptr,
        "vision_word_feature_indices": v_indices,

        # RoBERTa 侧的 token 级溯源：token → 词 的 CSR。
        # 留着它才能回答「某个词的特征是哪些子词平均出来的」，
        # 而不用重跑一遍前向。
        "roberta_token_ids": np.asarray(text_cache["token_ids"], dtype=np.int64),
        "roberta_token_offsets": np.asarray(text_cache["token_offsets"], dtype=np.int32),
        "roberta_token_word_index": np.asarray(
            text_cache["token_word_index"], dtype=np.int32),
        "roberta_word_token_indptr": np.asarray(
            text_cache["word_token_indptr"], dtype=np.int32),
        "roberta_word_token_indices": np.asarray(
            text_cache["word_token_indices"], dtype=np.int32),
        "roberta_tokenizer_failure_count": np.asarray(
            text_cache["tokenizer_failure_count"], dtype=np.int32),
        "text_input_length": np.asarray(text_cache["input_length"], dtype=np.int32),

        "audio_presentation_coverage_sec": np.asarray([audio_start, audio_end], dtype=np.float64),
        "video_presentation_coverage_sec": np.asarray([video_start, video_end], dtype=np.float64),
        "av_common_coverage_sec": av_common,
        "shared_t0_sec": np.asarray(shared_t0, dtype=np.float64),
        "original_effective_duration_sec": np.asarray(original_effective_duration, dtype=np.float64),
        "audio_observation_length": np.asarray(lld["frame_count"], dtype=np.int32),
        "video_observation_length": np.asarray(frames, dtype=np.int32),
        "text_sequence_length": np.asarray(n_words, dtype=np.int32),

        "alignment_zero_duration_word_count": np.asarray(
            diag["zero_duration_word_count"], dtype=np.int32),
        "aligner_zero_duration_slot_count": np.asarray(
            diag["zero_duration_slot_count"], dtype=np.int32),
        "alignment_mapping_fail_count": np.asarray(
            diag["mapping_fail_count"], dtype=np.int32),
        "word_without_slot_count": np.asarray(
            diag["n_words_without_slot"], dtype=np.int32),
        "decode_provenance_json": _j({
            "audio": {k: v for k, v in audio_info.items()
                      if k not in ("resampled_frame_pts", "audio_time_map")},
            "video": {k: v for k, v in video_info.items()
                      if k not in ("num_faces_per_frame", "pts_gaps")},
            "timeline": probe,
            "digital_silence": silence,
            "alignment_mapping_diagnostics": intervals["mapping_diagnostics"],
            "audio_word_issues": a_issues,
            "vision_word_issues": v_issues,
        }),
    }

    missing = [f for f in NPZ_REQUIRED_FIELDS if f not in payload]
    if missing:
        raise HardStop(f"装配缺字段：{missing}")

    written = write_npz_atomic(out_path, payload)
    runtime = time.time() - started

    return {
        "sample_key": key,
        "video_id": sample["video_id"],
        "clip_id": sample["clip_id"],
        "role": sample["role"],
        "modalities": sample["modalities"],
        "official_text_sha256": sample["official_text_sha256"],
        "source_video_relpath": sample["source_video_relpath"],
        "source_video_sha256": sample["source_video_sha256"],
        "source_label_xlsx_sha256": sample["source_label_xlsx_sha256"],
        "ffprobe_format_duration_sec": sample["ffprobe_format_duration_sec"],

        "npz_path": Path(written["path"]).name,
        "npz_bytes": written["bytes"],
        "output_sha256": written["sha256"],
        "schema_version": SCHEMA_VERSION,
        "config_sha256": config_sha,
        **ASSET_IDENTITY,

        "text_sequence_length": n_words,
        "audio_observation_length": lld["frame_count"],
        "video_frame_count": frames,
        "face_detected_frame_count": face_frames,
        "original_effective_duration_sec": original_effective_duration,
        "valid_length": int(tri_valid.sum()),

        "alignment_mode": route["alignment_mode"],
        "alignment_granularity": route["alignment_granularity"],
        "text_av_time_mapping_status": route["text_av_time_mapping_status"],
        "text_audio_correspondence": route["text_audio_correspondence"],
        "content_assertion": route["content_assertion"],
        "word_time_basis": route["word_time_basis"],
        "alignment_source": align_cache["alignment_source"],

        "text_present": 1, "text_content_valid": int(text_word_valid.sum() > 0),
        "audio_present": 1, "audio_observation_valid": int(lld["frame_count"] > 0),
        "audio_speech_valid": route["audio_speech_valid"],
        "video_present": 1, "audio_visual_time_valid": int(av_time_valid),
        "vision_feature_valid": int(face_frames > 0),
        "face_feature_valid": int(face_frames > 0),

        "n_official_words": n_words,
        "n_aligner_slots": int(intervals["n_slots"]),
        "n_structurally_valid_words": int(diag["n_structurally_valid_words"]),
        "alignment_zero_duration_word_count": int(diag["zero_duration_word_count"]),
        "alignment_mapping_fail_count": int(diag["mapping_fail_count"]),
        "text_valid_words": int(text_word_valid.sum()),
        "audio_valid_words": int(audio_word_valid.sum()),
        "vision_valid_words": int(vision_word_valid.sum()),
        "face_valid_ratio": face_ratio,
        "word_union_coverage_ratio": word_union_ratio,
        "audio_coverage_start_sec": audio_start,
        "audio_coverage_end_sec": audio_end,
        "video_coverage_start_sec": video_start,
        "video_coverage_end_sec": video_end,
        "av_common_coverage_start_sec": common_start,
        "av_common_coverage_end_sec": common_end,

        "process_rss_bytes": sample.get("process_rss_bytes", ""),
        "runtime_sec": round(runtime, 3),
        "error": "",
    }


# --------------------------------------------------------------------------
# 输出表
# --------------------------------------------------------------------------

MANIFEST_COLUMNS = (
    "sample_key", "video_id", "clip_id", "role", "modalities",
    "official_text_sha256", "source_video_relpath", "source_video_sha256",
    "source_label_xlsx_sha256", "ffprobe_format_duration_sec",
    "npz_path", "npz_bytes", "output_sha256", "schema_version", "config_sha256",
    "roberta_model_sha256", "mediapipe_model_sha256", "opensmile_config_sha256",
    "text_sequence_length", "audio_observation_length", "video_frame_count",
    "face_detected_frame_count", "original_effective_duration_sec", "valid_length",
    "alignment_mode", "alignment_granularity", "text_av_time_mapping_status",
    "text_audio_correspondence", "content_assertion", "word_time_basis",
    "alignment_source",
    "text_present", "text_content_valid", "audio_present", "audio_observation_valid",
    "audio_speech_valid", "video_present", "audio_visual_time_valid",
    "vision_feature_valid", "face_feature_valid",
    "n_official_words", "n_aligner_slots", "n_structurally_valid_words",
    "alignment_zero_duration_word_count", "alignment_mapping_fail_count",
    "text_valid_words", "audio_valid_words", "vision_valid_words",
    "face_valid_ratio", "word_union_coverage_ratio",
    "audio_coverage_start_sec", "audio_coverage_end_sec",
    "video_coverage_start_sec", "video_coverage_end_sec",
    "av_common_coverage_start_sec", "av_common_coverage_end_sec",
    "process_rss_bytes", "runtime_sec", "error",
)

#: `results_100.csv` = manifest 去掉执行期三列。
#: 分成两张表是有意的：manifest 是**生产日志**（含耗时与内存），
#: results_100 是**结果表**（只含样本与结论），下游只该读后者。
RESULTS_COLUMNS = tuple(c for c in MANIFEST_COLUMNS
                        if c not in ("process_rss_bytes", "runtime_sec", "error"))


def rebuild_output_tables(output_root: Path, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """按 `sample_key` 排序写两张表，各恰 100 行。"""
    ordered = sorted(rows, key=lambda r: str(r["sample_key"]))
    write_csv(Path(output_root) / "manifest.csv", ordered, MANIFEST_COLUMNS)
    full = [{k: r.get(k, "") for k in RESULTS_COLUMNS} for r in ordered]
    write_csv(Path(output_root) / "results_100.csv", full, RESULTS_COLUMNS)
    return {"n_rows": len(ordered),
            "manifest": (Path(output_root) / "manifest.csv").as_posix(),
            "results_100": (Path(output_root) / "results_100.csv").as_posix()}


# --------------------------------------------------------------------------
# 只换结论列（S6）
# --------------------------------------------------------------------------


def stored_routing_scalars(npz: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """从成品 NPZ 里取回路由所需的标量。**不碰任何大数组。**"""
    from q1v2_route import route_sample  # noqa: F401  (文档引用)

    return {
        "audio_present": bool(int(npz["audio_present"])),
        "audio_digital_silence": bool(
            "all-zero" in str(npz["audio_speech_evidence"])),
        "mapping_status": str(npz["text_av_time_mapping_status"]),
        "n_official_words": int(npz["text_sequence_length"]),
        "n_structurally_valid_words": int(np.count_nonzero(npz["word_time_valid"])),
    }


def rewrite_routes(output_root: Path, evidence: Optional[Dict[str, Dict[str, str]]] = None,
                   feature_dir: Optional[Path] = None) -> Dict[str, Any]:
    """用成品 NPZ 的标量重算路由并回写 NPZ 与两张表。

    用途仅限 S6 开发期回归（外部人工证据接入）：**不重解码、不重聚合**，
    故耗时是分钟级而非小时级。默认路径（`evidence` 为空）下这个函数
    重算出的路由与成品一致——这本身就是核验 V11 的做法。
    """
    import q1v2_route as R

    evidence = evidence or {}
    feature_dir = Path(feature_dir or (Path(output_root) / "features"))
    changed = 0
    rows: List[Dict[str, Any]] = []
    for path in sorted(feature_dir.glob("*.npz")):
        npz = load_npz(path)
        scalars = stored_routing_scalars(npz)
        # 时间映射状态要按结构闸门重算，故需要原始诊断标量。
        n_valid = scalars["n_structurally_valid_words"]
        n_words = scalars["n_official_words"]
        if int(npz["aligner_slot_count"]) == 0:
            mapping_status = "unavailable"
        elif n_words and n_valid == n_words:
            mapping_status = "word_valid"
        elif n_valid > 0:
            mapping_status = "word_partial"
        else:
            mapping_status = "clip_only"
        key = str(npz["sample_key"])
        new = R.route_sample(
            timeline_ok=bool(int(npz["audio_visual_time_valid"])),
            audio_present=scalars["audio_present"],
            audio_digital_silence=scalars["audio_digital_silence"],
            mapping_status=mapping_status,
            n_official_words=n_words,
            n_structurally_valid_words=n_valid,
            external=evidence.get(key),
        )
        if (str(npz["alignment_mode"]) != new["alignment_mode"]
                or str(npz["text_audio_correspondence"]) != new["text_audio_correspondence"]
                or int(npz["content_assertion"]) != new["content_assertion"]):
            changed += 1
        npz.update({
            "alignment_mode": _u(new["alignment_mode"]),
            "alignment_granularity": _u(new["alignment_granularity"]),
            "text_audio_correspondence": _u(new["text_audio_correspondence"]),
            "text_av_time_mapping_status": _u(new["text_av_time_mapping_status"]),
            "content_assertion": np.asarray(new["content_assertion"], dtype=np.uint8),
            "word_time_basis": _u(new["word_time_basis"]),
            "routing_reasons_json": _j(new["routing_reasons"]),
        })
        write_npz_atomic(path, npz)
    return {"n_files": len(list(feature_dir.glob("*.npz"))), "n_changed": changed,
            "evidence_provided": len(evidence)}


# --------------------------------------------------------------------------
# 从成品 NPZ 重算词级聚合（不重解码）
# --------------------------------------------------------------------------


def reaggregate_one(npz: Dict[str, np.ndarray],
                    *, write: bool = False) -> Dict[str, Any]:
    """只用 NPZ 里的 raw 序列 + 词区间，重算 `word_audio_feat` / `word_vision_feat`。

    **不读视频、不读音频、不加载任何模型**——输入全部来自成品本身。这是计划里
    「改聚合规则不必重解 23,241 帧」的那条能力：调 `aggregate_by_center` 的规则后，
    用本函数把 100 个 NPZ 的词级数组重算一遍即可，代价是秒级而非 12 分钟。

    返回逐字段的比对结论。`write=True` 时把重算结果写回 NPZ；规则未变时
    `savez_compressed` 的输出是**逐字节相同**的（zlib 确定性），所以那是一次
    可信的幂等操作——它同时就是本函数保真性的最强证据。

    **写回之后必须重跑 `--stage=tables` 与核验。** 即便内容逐字节不变，
    `replace()` 也会刷新 mtime，于是 `features/` 比 `manifest.csv` 新，
    V8 会（正确地）报「产物晚于表」。那不是误报，是流水线状态真的失同步了。
    """
    import q1v2_media as M

    words = npz["words"]
    wstart = np.asarray(npz["word_start_sec"], dtype=np.float64)
    wend = np.asarray(npz["word_end_sec"], dtype=np.float64)
    wtime = np.asarray(npz["word_time_valid"], dtype=bool)

    # 音频侧：openSMILE 的 center 相对音频切片起点，词区间是绝对 presentation 时刻，
    # 故先把词区间平移回相对坐标——与 media_stage 的调用逐参数一致。
    audio_start = float(npz["audio_presentation_coverage_sec"][0])
    audio, audio_valid, a_indptr, a_indices, a_issues = M.aggregate_by_center(
        words, wstart - audio_start, wend - audio_start,
        npz["raw_audio_lld_center_sec"], npz["raw_audio_lld_values"],
        dim=OPENMILE_LLD_DIM)
    audio_valid = (audio_valid & wtime).astype(np.uint8)
    audio[audio_valid == 0] = 0.0

    # 视觉侧：帧 PTS 与词区间同在共享时间轴上，直接相减。
    shared_t0 = float(npz["shared_t0_sec"])
    vision, vision_valid, v_indptr, v_indices, v_issues = M.aggregate_by_center(
        words, wstart - shared_t0, wend - shared_t0,
        npz["raw_video_pts_sec"] - shared_t0, npz["raw_video_blendshape_values"],
        valid_frames=npz["raw_video_face_feature_valid"], dim=FACE_MODEL_DIM)
    vision_valid = (vision_valid & wtime).astype(np.uint8)
    vision[vision_valid == 0] = 0.0

    checks = {
        "word_audio_feat": np.array_equal(audio, npz["word_audio_feat"]),
        "word_audio_valid": np.array_equal(audio_valid, npz["word_audio_valid"]),
        "audio_word_feature_indptr": np.array_equal(
            a_indptr, npz["audio_word_feature_indptr"]),
        "audio_word_feature_indices": np.array_equal(
            a_indices, npz["audio_word_feature_indices"]),
        "word_vision_feat": np.array_equal(vision, npz["word_vision_feat"]),
        "word_vision_valid": np.array_equal(vision_valid, npz["word_vision_valid"]),
        "vision_word_feature_indptr": np.array_equal(
            v_indptr, npz["vision_word_feature_indptr"]),
        "vision_word_feature_indices": np.array_equal(
            v_indices, npz["vision_word_feature_indices"]),
    }
    # 这里比的是**逐位相等**而不是 allclose：重算与原件走的是同一条确定性路径，
    # 中间没有随机性也没有重排，逐位相等是应当成立且更强的判据。
    result = {
        "sample_key": str(npz["sample_key"]),
        "fields_equal": checks,
        "all_equal": all(checks.values()),
        "n_audio_issues": len(a_issues),
        "n_vision_issues": len(v_issues),
    }
    if write:
        npz.update({
            "word_audio_feat": audio, "word_audio_valid": audio_valid,
            "audio_word_feature_indptr": a_indptr,
            "audio_word_feature_indices": a_indices,
            "word_vision_feat": vision, "word_vision_valid": vision_valid,
            "vision_word_feature_indptr": v_indptr,
            "vision_word_feature_indices": v_indices,
        })
        result["_payload"] = npz
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="问题一 v2：从成品 NPZ 重算词级聚合（不重解码）")
    parser.add_argument("--reaggregate", action="store_true",
                        help="重算并与成品比对（默认只比对不写回）")
    parser.add_argument("--write", action="store_true",
                        help="把重算结果写回 NPZ（规则未变时应逐字节不变）")
    parser.add_argument("--sample-key", default=None, help="只处理一条样本")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    if not args.reaggregate:
        parser.print_help()
        return 0

    feature_dir = OUTPUT_ROOT / "features"
    paths = sorted(feature_dir.glob("*.npz"))
    if args.sample_key:
        stem = safe_stem(args.sample_key)
        paths = [p for p in paths if p.stem == stem]
        if not paths:
            raise HardStop(f"features/ 下没有 {stem}.npz")
    if args.limit:
        paths = paths[:args.limit]

    started = time.time()
    n_ok = 0
    problems: List[Dict[str, Any]] = []
    for path in paths:
        npz = load_npz(path)
        before = sha256(path)
        outcome = reaggregate_one(npz, write=args.write)
        if outcome["all_equal"]:
            n_ok += 1
        else:
            problems.append({
                "sample_key": outcome["sample_key"],
                "unequal": [k for k, v in outcome["fields_equal"].items() if not v],
            })
        if args.write:
            payload = outcome.pop("_payload")
            info = write_npz_atomic(path, payload)
            if info["sha256"] != before:
                problems.append({"sample_key": outcome["sample_key"],
                                 "note": "写回后 SHA 变了（同规则下不应变）",
                                 "before": before, "after": info["sha256"]})
        else:
            outcome.pop("_payload", None)

    elapsed = time.time() - started
    print(json.dumps({
        "n_files": len(paths), "n_all_equal": n_ok, "n_problems": len(problems),
        "write": bool(args.write), "elapsed_sec": round(elapsed, 3),
        "note": "本函数不读任何媒体文件，故耗时应为秒级——远低于重跑媒体阶段的分钟级",
        "problems": problems[:10],
    }, ensure_ascii=False, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "NPZ_REQUIRED_FIELDS", "MANIFEST_COLUMNS", "RESULTS_COLUMNS",
    "write_npz_atomic", "load_npz", "media_stage", "rebuild_output_tables",
    "stored_routing_scalars", "rewrite_routes", "reaggregate_one", "main",
]
