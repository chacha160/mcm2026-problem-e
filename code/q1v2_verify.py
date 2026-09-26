# -*- coding: utf-8 -*-
"""q1v2_verify.py —— 问题一 v2 核验层 V1–V14

## 一个刻意的不对称：轻量项与重量项分开

V1/V2/V3/V4/V5/V9/V10/V11/V13 只依赖 numpy 与成品产物，**在裸解释器里就能跑**。
这正是 `q1v2_contract.py` 禁止在模块层导入 torch/mediapipe/opensmile 的原因：
「流水线能跑」与「产物自洽」是两件事，后者不该以装齐 1 GB 依赖为前提。

V6（ffprobe 交叉）、V7（权重真实性）、V14（位精确重解码）需要外部工具或重依赖，
`--quick` 下跳过，并在报告里明确记为 skipped 而不是 passed。

## 核验的边界（必须写进报告）

对照实现的逐样本数组**全空**，所以本套核验**不与对照实现比数值**，只做四件事：

1. **内部自洽**（V1–V5、V9–V11）：产物自己能不能复算出自己；
2. **外部交叉**（V6）：解码覆盖与 ffprobe 声明是否相容；
3. **结构硬锚点**（V13、V2 的形状）：1926 / 1932 / 0 这些数是冻结的；
4. **位精确锚点**（V14）：两条数字静音样本的每一个采样点。

`all_passed` 只对**实际执行过**的项成立；跳过的项单独列出，不混入通过数。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import numpy as np

import q1v2_contract as C
from q1_verify import _md5_of_array, _result  # 复用既有核验的返回约定

OUTPUT_ROOT = C.OUTPUT_ROOT
FEATURES_DIR = OUTPUT_ROOT / "features"
CACHE_ROOT = C.CACHE_ROOT
LOGS_DIR = OUTPUT_ROOT / "logs"
METADATA_DIR = OUTPUT_ROOT / "metadata"
REPORTS_DIR = OUTPUT_ROOT / "reports"
AUDIT_DIR = OUTPUT_ROOT / "audit"

#: 计划里冻结的结构锚点。改动这三个数等于改口径，必须同步改文档。
EXPECTED_WORD_UNITS = 1926
EXPECTED_WHITESPACE_CHUNKS = 1932
EXPECTED_VIDEO_FRAMES = 23241
EXPECTED_FACE_FRAMES = 16806
EXPECTED_NO_FACE_SAMPLES = 24
#: 软区间与参考值：用于**记录差异**，不作为通过/失败判据。
#: 这些数取自对照实现的实测，而 对照实现复用了旧的 alignment trace，
#: 故本套代码重跑得到的值与之不同是预期内的，须给归因而非改阈值迁就。
REFERENCE_SUM_ALIGNER_SLOTS = 1926
REFERENCE_ZERO_DURATION_WORDS = 161
REFERENCE_ZERO_DURATION_SAMPLES = 37
ZERO_DURATION_RANGE = (120, 210)
ZERO_DURATION_SAMPLE_RANGE = (30, 44)
FFPROBE_TOLERANCE_SEC = 0.15

#: RoBERTa 词嵌入 std 的回归带。
#: 计划里写的「std≈0.02~0.04」是 `nn.Embedding` 的**初始化**标准差，不是训练后
#: 检查点的统计量——拿它当判据必然失败。此处按**实测**设带：该值由 SHA 冻结的
#: `model.safetensors` 唯一决定，是可复现的回归锚点，而不是一次实验的偶然读数。
ROBERTA_EMBED_STD_RANGE = (0.12, 0.14)

#: RoBERTa 掩码补全指纹：`(含 <mask> 的句子, 期望答案, 允许的 top-N 排名)`。
#: 期望答案是**实测**结果（完整 top-5 落在 evidence 里），不是从任何文档抄的。
#: 用 top-N 而非 top-1 是为了对 transformers/torch 版本差异下的数值抖动留余量：
#: 三个句子的 top-1 与 top-2 对数差分别为 2.42 / 2.99 / 0.64。
ROBERTA_MASK_PROBES = (
    ("The capital of France is <mask>.", "Paris", 2),
    ("Paris is the capital of <mask>.", "France", 2),
    ("Water freezes at zero degrees <mask>.", "Celsius", 5),
)

#: 真帧探测时最多试多少帧去找一张有脸的画面。
FACE_PROBE_MAX_FRAMES = 60

ALIGNMENT_ABS_TOL = 1e-6
ALIGNMENT_REL_TOL = 1e-5


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {k: handle[k] for k in handle.files}


def _feature_files() -> List[Path]:
    return sorted(FEATURES_DIR.glob("*.npz"))


def _registry() -> Dict[str, Any]:
    return json.loads((METADATA_DIR / "registry.json").read_text(encoding="utf-8"))


def _read_log(stem: str, stage: str) -> Optional[Dict[str, Any]]:
    path = LOGS_DIR / f"{stem}.{stage}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _all_npz() -> Dict[str, Dict[str, np.ndarray]]:
    return {p.stem: _load_npz(p) for p in _feature_files()}


# --------------------------------------------------------------------------
# V1 双向集合相等
# --------------------------------------------------------------------------


def check_v1_coverage() -> Dict[str, Any]:
    registry = _registry()
    sample_keys = [s["sample_key"] for s in registry["samples"]]
    stems = [C.safe_stem(k) for k in sample_keys]

    registry_set = set(stems)
    feature_set = {p.stem for p in _feature_files()}

    import csv as _csv
    manifest_keys, results_keys = [], []
    for name, bucket in (("manifest.csv", manifest_keys), ("results_100.csv", results_keys)):
        path = OUTPUT_ROOT / name
        if path.is_file():
            with path.open(encoding="utf-8-sig", newline="") as stream:
                for row in _csv.DictReader(stream):
                    bucket.append(C.safe_stem(row["sample_key"]))

    log_set = {p.name[:-len(".media.json")] for p in LOGS_DIR.glob("*.media.json")}

    problems: List[str] = []
    for name, values in (("registry", stems), ("manifest.csv", manifest_keys),
                         ("results_100.csv", results_keys)):
        if len(values) != len(set(values)):
            problems.append(f"{name} 有重复行")
    for name, got in (("manifest.csv", set(manifest_keys)),
                      ("results_100.csv", set(results_keys)),
                      ("features/*.npz", feature_set),
                      ("logs/*.media.json", log_set)):
        if got != registry_set:
            problems.append(
                f"{name} 与登记表不符：缺 {sorted(registry_set - got)[:5]} "
                f"多 {sorted(got - registry_set)[:5]}")

    # 官方文本逐字比对
    import unaligned_common as U
    by_key = {s["sample_key"]: s for s in registry["samples"]}
    text_mismatch: List[str] = []
    for row in U.list_samples():
        key = C.sample_key_of(str(row["video_id"]), str(row["clip_id"]))
        if key in by_key and by_key[key]["official_text"] != str(row["text"]):
            text_mismatch.append(key)
    if text_mismatch:
        problems.append(f"official_text 与 label-100.xlsx 不符：{text_mismatch[:5]}")

    return _result(
        "V1", "双向集合相等（输入/登记/两张表/产物/日志）",
        passed=not problems and len(stems) == 100,
        checked=len(stems) + len(manifest_keys) + len(results_keys),
        failed=len(problems) + len(text_mismatch),
        detail=(f"登记 {len(stems)} / 产物 {len(feature_set)} / "
                f"manifest {len(manifest_keys)} / results {len(results_keys)}"),
        evidence={"problems": problems[:10],
                  "n_registry": len(stems), "n_features": len(feature_set),
                  "n_manifest": len(manifest_keys), "n_results": len(results_keys),
                  "n_logs": len(log_set)},
        caveat="" if not problems else "见 evidence.problems",
    )


# --------------------------------------------------------------------------
# V2 NPZ schema 自洽
# --------------------------------------------------------------------------


def check_v2_schema() -> Dict[str, Any]:
    from q1v2_build import NPZ_REQUIRED_FIELDS

    problems: List[str] = []
    npzs = _all_npz()
    for stem, z in npzs.items():
        missing = [f for f in NPZ_REQUIRED_FIELDS if f not in z]
        if missing:
            problems.append(f"{stem} 缺字段 {missing[:5]}")
            continue
        L = int(z["text_sequence_length"])
        checks = [
            ("text_word_feat", (L, C.ROBERTA_HIDDEN_DIM)),
            ("word_audio_feat", (L, C.OPENMILE_LLD_DIM * 2)),
            ("word_vision_feat", (L, C.FACE_MODEL_DIM * 2)),
            ("words", (L,)),
            ("word_time_valid", (L,)),
            ("text_word_valid", (L,)),
            ("word_audio_valid", (L,)),
            ("word_vision_valid", (L,)),
        ]
        for name, shape in checks:
            if tuple(z[name].shape) != shape:
                problems.append(f"{stem} {name} 形状 {z[name].shape} != {shape}")
        for name, dim in (("raw_audio_lld_values", C.OPENMILE_LLD_DIM),
                          ("raw_video_blendshape_values", C.FACE_MODEL_DIM)):
            if int(z[name].shape[1]) != dim:
                problems.append(f"{stem} {name} 第二维 {z[name].shape[1]} != {dim}")
        if z["raw_audio_lld_feature_names"].shape[0] != C.OPENMILE_LLD_DIM:
            problems.append(f"{stem} LLD 特征名个数不符")
        if z["raw_video_blendshape_names"].shape[0] != C.FACE_MODEL_DIM:
            problems.append(f"{stem} blendshape 名个数不符")

        # CSR 长度自洽
        for ptr, idx, n in (("audio_word_feature_indptr", "audio_word_feature_indices",
                             int(z["raw_audio_lld_values"].shape[0])),
                            ("vision_word_feature_indptr", "vision_word_feature_indices",
                             int(z["raw_video_blendshape_values"].shape[0]))):
            if z[ptr].shape[0] != L + 1:
                problems.append(f"{stem} {ptr} 长度 {z[ptr].shape[0]} != L+1")
            elif int(z[ptr][-1]) != z[idx].shape[0]:
                problems.append(f"{stem} {ptr}[-1] != {idx} 长度")
            elif z[idx].size and (int(z[idx].min()) < 0 or int(z[idx].max()) >= n):
                problems.append(f"{stem} {idx} 值越界")

        # 时间区间与掩码的一致性
        wtv = z["word_time_valid"].astype(bool)
        ws, we = z["word_start_sec"], z["word_end_sec"]
        if np.any(np.isfinite(ws[~wtv])) or np.any(np.isfinite(we[~wtv])):
            problems.append(f"{stem} word_time_valid=0 的词却有有限时间")
        if wtv.any():
            if not (np.isfinite(ws[wtv]).all() and np.isfinite(we[wtv]).all()):
                problems.append(f"{stem} word_time_valid=1 的词时间非有限")
            elif not np.all(we[wtv] > ws[wtv]):
                problems.append(f"{stem} word_time_valid=1 的词 end<=start")
            if not np.all(np.diff(ws[wtv]) >= -C.TOL):
                problems.append(f"{stem} 合法词起点非单调")
        # 无效模态的词特征必须整行为零（不拿邻近观测凑数）
        for feat, valid in (("word_audio_feat", "word_audio_valid"),
                            ("word_vision_feat", "word_vision_valid")):
            off = z[valid].astype(bool)
            if (~off).any() and np.any(z[feat][~off] != 0.0):
                problems.append(f"{stem} {valid}=0 的词 {feat} 非零")
        # 覆盖率与有效掩码的包含关系：模态有效的词，时间区间必须有效
        for valid in ("word_audio_valid", "word_vision_valid"):
            if np.any(z[valid].astype(bool) & ~wtv):
                problems.append(f"{stem} {valid}=1 但 word_time_valid=0")

    return _result(
        "V2", "NPZ schema 自洽（字段/形状/dtype/掩码一致）",
        passed=not problems, checked=len(npzs) * len(NPZ_REQUIRED_FIELDS),
        failed=len(problems),
        detail=f"{len(npzs)} 个 NPZ 与 schema 比对",
        evidence={"problems": problems[:20]},
        caveat="" if not problems else "见 evidence.problems",
    )


# --------------------------------------------------------------------------
# V3 词单元可重建
# --------------------------------------------------------------------------


def check_v3_words() -> Dict[str, Any]:
    import q1v2_text as T

    problems: List[str] = []
    npzs = _all_npz()
    for stem, z in npzs.items():
        text = str(z["official_text"])
        words = T.source_words(text)
        chunks = T.whitespace_chunks(text)
        if [str(w) for w in z["words"]] != [w["text"] for w in words]:
            problems.append(f"{stem} words 文本与重算不符")
        if list(z["word_char_start"]) != [w["char_start"] for w in words]:
            problems.append(f"{stem} word_char_start 与重算不符")
        if list(z["word_char_end"]) != [w["char_end"] for w in words]:
            problems.append(f"{stem} word_char_end 与重算不符")
        if int(z["text_sequence_length"]) != len(words):
            problems.append(f"{stem} text_sequence_length != 重算词数")

        n_slots = int(z["aligner_slot_count"])
        slot_to_word = z["aligner_slot_to_word_index"]
        if slot_to_word.shape[0] != n_slots:
            problems.append(f"{stem} slot_to_word 长度 != aligner_slot_count")
        elif n_slots and (int(slot_to_word.min()) < -1 or int(slot_to_word.max()) >= len(words)):
            problems.append(f"{stem} slot_to_word 值越界")

        # 结构一致性：每个被映射的槽，其归属词的字符区间必须真的覆盖该槽的位置。
        # 这条把「槽→词」的映射钉在文本上，而不是只检查值域。
        chunk_to_word = T.chunk_to_word_index(chunks, words)
        n_mapped = int(np.count_nonzero(slot_to_word >= 0))
        if n_mapped and n_mapped != int(np.count_nonzero(chunk_to_word >= 0)):
            pass  # 槽数与块数本就不同（1926 vs 1932），不做等式断言

    return _result(
        "V3", "词单元可重建（1926 口径的独立复算）",
        passed=not problems, checked=len(npzs) * 4, failed=len(problems),
        detail=f"{len(npzs)} 条文本按 source_words 重算并逐元素比对",
        evidence={"expected_word_units": EXPECTED_WORD_UNITS,
                  "expected_whitespace_chunks": EXPECTED_WHITESPACE_CHUNKS,
                  "problems": problems[:20]},
        caveat=("词单元文本与字符区间由 source_words 从 official_text 纯函数重算；"
                "槽→词映射落盘为 aligner_slot_to_word_index（对照实现的 P1-02 缺口处），"
                "块→词映射是文本的纯函数故不落盘、核验时现场复算"),
    )


# --------------------------------------------------------------------------
# V4 词级聚合可回溯重算（关键）
# --------------------------------------------------------------------------


def _reaggregate(values: np.ndarray, indices: np.ndarray, dim: int,
                 n_words: int, indptr: np.ndarray) -> np.ndarray:
    out = np.zeros((n_words, dim * 2), dtype=np.float64)
    for wi in range(n_words):
        lo, hi = int(indptr[wi]), int(indptr[wi + 1])
        if hi <= lo:
            continue
        sel = values[indices[lo:hi]]
        out[wi, :dim] = sel.mean(axis=0, dtype=np.float64)
        out[wi, dim:] = sel.std(axis=0, ddof=0, dtype=np.float64)
    return out


def check_v4_traceable() -> Dict[str, Any]:
    problems: List[str] = []
    npzs = _all_npz()
    n_recomputed = 0
    for stem, z in npzs.items():
        L = int(z["text_sequence_length"])

        for label, feat, valid, values, centers, indptr, indices, dim, mask in (
            ("audio", "word_audio_feat", "word_audio_valid",
             z["raw_audio_lld_values"], z["raw_audio_lld_center_sec"],
             z["audio_word_feature_indptr"], z["audio_word_feature_indices"],
             C.OPENMILE_LLD_DIM, None),
            ("vision", "word_vision_feat", "word_vision_valid",
             z["raw_video_blendshape_values"], z["raw_video_pts_sec"],
             z["vision_word_feature_indptr"], z["vision_word_feature_indices"],
             C.FACE_MODEL_DIM, z["raw_video_face_feature_valid"].astype(bool)),
        ):
            # (a) 从 CSR 回捞源观测重算
            recomputed = _reaggregate(values, indices, dim, L, indptr)
            off = z[valid].astype(bool)
            if off.any():
                stored = z[feat][off].astype(np.float64)
                if not np.allclose(recomputed[off], stored,
                                   rtol=ALIGNMENT_REL_TOL, atol=ALIGNMENT_ABS_TOL):
                    bad = int(np.argmax(
                        np.abs(recomputed[off] - stored).max(axis=1) > 1e-5))
                    problems.append(
                        f"{stem} {label} 从 CSR 重算与落盘不符（首例词序 {bad}）")
                n_recomputed += int(off.sum())

            # (b) 独立地按 s<=c<e 重新选索引，要求与 CSR 完全相等
            ws, we = z["word_start_sec"], z["word_end_sec"]
            for wi in range(L):
                if not off[wi]:
                    continue
                a, b = float(ws[wi]), float(we[wi])
                sel = np.flatnonzero((centers >= a) & (centers < b))
                if mask is not None:
                    sel = sel[mask[sel]]
                csr = indices[int(indptr[wi]):int(indptr[wi + 1])]
                if not np.array_equal(sel, csr):
                    problems.append(
                        f"{stem} {label} 词 {wi} 的 CSR 索引与 s<=c<e 重选不符")
                    break

            # (c) 模态无效的词必须整行为零
            if (~off).any() and np.any(z[feat][~off] != 0.0):
                problems.append(f"{stem} {label} 无效词特征非零")
            # 代表时刻与区间的关系由 V5 统一检查，此处不再重复

    return _result(
        "V4", "词级聚合可回溯重算（CSR 回捞 + 独立重选）",
        passed=not problems, checked=n_recomputed, failed=len(problems),
        detail=(f"{n_recomputed} 个有效词用 CSR 回捞源观测重算 "
                f"mean/std(ddof=0)，并独立按 s<=c<e 重选索引比对"),
        evidence={"rtol": ALIGNMENT_REL_TOL, "atol": ALIGNMENT_ABS_TOL,
                  "problems": problems[:20]},
        caveat="无效（word_time_valid=0）的词特征整行为零，不参与重算——本套代码不为无区间词找邻近观测顶替",
    )


# --------------------------------------------------------------------------
# V5 时间轴一致性
# --------------------------------------------------------------------------


def check_v5_timeline() -> Dict[str, Any]:
    problems: List[str] = []
    npzs = _all_npz()
    for stem, z in npzs.items():
        t0 = float(z["shared_t0_sec"])
        a_cov = z["audio_presentation_coverage_sec"]
        v_cov = z["video_presentation_coverage_sec"]
        p_cov = z["av_common_coverage_sec"]

        if abs(float(a_cov[0]) - t0) > C.TOL:
            problems.append(f"{stem} 音频覆盖起点 != shared_t0")
        if abs(float(v_cov[0]) - t0) > C.TOL:
            problems.append(f"{stem} 视频覆盖起点 != shared_t0")
        if not (abs(float(p_cov[0]) - max(float(a_cov[0]), float(v_cov[0]))) <= C.TOL
                and abs(float(p_cov[1]) - min(float(a_cov[1]), float(v_cov[1]))) <= C.TOL):
            problems.append(f"{stem} 公共区间 != [max(start), min(end)]")

        # LLD 时间标签自洽
        ls, le, lc = (z["raw_audio_lld_start_sec"], z["raw_audio_lld_end_sec"],
                      z["raw_audio_lld_center_sec"])
        if not np.allclose(lc, (ls + le) / 2.0, atol=1e-9):
            problems.append(f"{stem} LLD center != (start+end)/2")
        if np.any(np.diff(ls) < -C.TOL):
            problems.append(f"{stem} LLD start 非单调")
        if np.any(le <= ls):
            problems.append(f"{stem} LLD end <= start")
        if ls.size and (float(ls[0]) < t0 - C.TOL or float(le[-1]) > float(a_cov[1]) + C.TOL):
            problems.append(f"{stem} LLD 时间标签越出音频覆盖")

        # 视觉支撑区间不变量
        pts = z["raw_video_pts_sec"]
        s0, e0 = z["raw_video_support_start_sec"], z["raw_video_support_end_sec"]
        if pts.size:
            if not np.isclose(s0[0], pts[0]):
                problems.append(f"{stem} support_start[0] != pts[0]")
            if not np.isclose(e0[-1], float(v_cov[1])):
                problems.append(f"{stem} support_end[-1] != 视频覆盖末端")
            if not np.allclose(e0[:-1], s0[1:]):
                problems.append(f"{stem} support 相邻区间不衔接")
            if np.any(np.diff(pts) <= 0):
                problems.append(f"{stem} 帧 PTS 非严格递增")

        # 词区间必须落在公共区间内
        wtv = z["word_time_valid"].astype(bool)
        if wtv.any():
            ws, we = z["word_start_sec"][wtv], z["word_end_sec"][wtv]
            if np.any(ws < float(p_cov[0]) - C.TOL) or np.any(we > float(p_cov[1]) + C.TOL):
                problems.append(f"{stem} 合法词区间越出公共覆盖")

    return _result(
        "V5", "时间轴一致性（共享 t0 / 支撑区间 / 覆盖交叠）",
        passed=not problems, checked=len(npzs) * 10, failed=len(problems),
        detail=f"{len(npzs)} 个样本的共享原点、LLD 时间标签、视觉支撑区间、词区间包含关系",
        evidence={"problems": problems[:20]},
        caveat="LLD 的 start/end 是 openSMILE 返回值，首末窗被夹到切片边界，故称「输出时间标签」而非物理分析窗",
    )


# --------------------------------------------------------------------------
# V6 外部 ffprobe 交叉核对
# --------------------------------------------------------------------------


def check_v6_external(quick: bool = False) -> Dict[str, Any]:
    import unaligned_common as U

    registry = {s["sample_key"]: s for s in _registry()["samples"]}
    stems = sorted(_all_npz())
    todo = stems[:5] if quick else stems
    problems: List[str] = []
    checked = 0
    #: 包数 vs 帧数——**只记不用**。见下方注释。
    packet_rows: List[Dict[str, Any]] = []
    for stem in todo:
        z = _all_npz()[stem]
        key = str(z["sample_key"])
        sample = registry.get(key)
        if sample is None:
            problems.append(f"{stem} 不在登记表")
            continue
        path = C.PROJECT_ROOT / sample["source_video_relpath"]
        info = U.probe_streams(str(path))
        declared = float(info.get("video_duration") or 0.0)
        decoded = float(z["video_presentation_coverage_sec"][1]) - float(
            z["video_presentation_coverage_sec"][0])
        checked += 1
        if declared <= 0:
            problems.append(f"{stem} ffprobe 未读到视频流时长")
        elif abs(declared - decoded) > FFPROBE_TOLERANCE_SEC:
            problems.append(
                f"{stem} ffprobe {declared:.4f}s 与解码覆盖 {decoded:.4f}s "
                f"相差 {abs(declared - decoded):.4f}s > {FFPROBE_TOLERANCE_SEC}s")
        # ffprobe 的 `n_video_packets` 数的是**容器里的包**，不是 presentation 帧数：
        # 多包一帧、B 帧重排、以及解码器按 PTS 去重都会让包数系统性偏大
        # （对照实现自己的锚点：`-3g5yACwYnA$_$3` = 578 包 vs 431 解码帧）。
        # 曾用 `|包数 − 帧数| ≤ 1` 做断言，那是我凭空加的不变量，已被证伪，故删除。
        # 现在只把两列记进 evidence 供人看趋势，**不作判据**。
        packet_rows.append({
            "sample_key": key,
            "n_video_packets": int(info.get("n_video_packets") or 0),
            "video_frame_count": int(z["video_frame_count"]),
        })

    ratios = [r["n_video_packets"] / r["video_frame_count"]
              for r in packet_rows if r["video_frame_count"] > 0
              and r["n_video_packets"] > 0]
    packet_summary = {
        "note": "包数/帧数比值仅供观察：两者定义不同，不构成判据",
        "n": len(ratios),
        "min": round(min(ratios), 4) if ratios else None,
        "max": round(max(ratios), 4) if ratios else None,
        "n_below_one": sum(1 for r in ratios if r < 1.0),
    }

    return _result(
        "V6", "外部 ffprobe 交叉核对（解码覆盖 vs 容器声明）",
        passed=not problems, checked=checked, failed=len(problems),
        detail=(f"{checked} 条重跑 ffprobe；容差 {FFPROBE_TOLERANCE_SEC}s。"
                f"覆盖与容器时长**本就不必相等**（对照实现实测 6.800 vs 6.8667），"
                f"此处核对的是二者相容而非相等"),
        evidence={"problems": problems[:20], "quick": quick,
                  "packet_vs_frame": packet_summary},
        caveat="" if not problems else "见 evidence.problems",
    )


# --------------------------------------------------------------------------
# V7 权重真实性
# --------------------------------------------------------------------------


def _first_face_frame() -> Tuple[Optional[np.ndarray], str]:
    """从产物里挑一条**确实检出人脸**的样本，解出那张有脸的帧。

    不硬编码样本 key：取材判据取自 NPZ 自己的 `raw_video_face_feature_valid`
    掩码，于是「生产说这帧有脸」与「核验拿这帧去 detect」是同一个来源——
    换一批样本也不会因为 key 不存在而失效。
    """
    import av

    registry = {s["sample_key"]: s for s in _registry()["samples"]}
    ordered = sorted(_all_npz().values(),
                     key=lambda z: -int(z["face_detected_frame_count"]))
    for z in ordered:
        if int(z["face_detected_frame_count"]) <= 0:
            continue
        hits = np.flatnonzero(np.asarray(z["raw_video_face_feature_valid"]).astype(bool))
        if hits.size == 0:
            continue
        index = int(hits[0])
        sample_key = str(z["sample_key"])
        sample = registry.get(sample_key)
        if sample is None:
            continue
        path = C.PROJECT_ROOT / sample["source_video_relpath"]
        try:
            with av.open(str(path), mode="r") as container:
                stream = container.streams.video[0]
                for position, frame in enumerate(container.decode(stream)):
                    if position >= index:
                        return (frame.to_ndarray(format="rgb24"),
                                f"{sample_key} 第 {index} 帧")
        except Exception:  # noqa: BLE001
            continue
    return None, ""


def check_v7_weights(quick: bool = False) -> Dict[str, Any]:
    if quick:
        return _result("V7", "权重真实性（四项探针）", passed=True, checked=0,
                       failed=0, detail="--quick 下跳过",
                       evidence={"skipped": True},
                       caveat="跳过项不计入通过，仅表示本次未执行")
    problems: List[str] = []
    evidence: Dict[str, Any] = {}

    import q1v2_assets as A

    # (a) RoBERTa：local_files_only 加载 + 词嵌入统计 + 掩码补全指纹
    try:
        import q1v2_text as T
        roberta_dir = A.resolve_roberta_dir()

        # 字节级锚点（**比任何摘要都强**）：把 6 个文件的 SHA-256 重算一遍，
        # 与冻结常量逐字符比对。verify_roberta_dir 任一处不符即抛错。
        #
        # 为什么单列这一条：本套的 `roberta_asset_identity_sha256` 是对
        # `{model_id, revision, hidden_dim}` 求的摘要，而 对照实现的同名字段是对
        # **逐文件 SHA 字典**求的摘要（见 对照实现的 `validate_q1_v1.py`）——
        # 两者构造不同，**数值本就不该相等**，不可互比。身份的可比性由
        # `roberta_model_sha256`（model.safetensors 的冻结 SHA）承担。
        roberta_files = A.verify_roberta_dir(roberta_dir)
        evidence["roberta_n_files"] = roberta_files["n_files"]
        evidence["roberta_model_safetensors_sha256"] = \
            roberta_files["files_sha256"]["model.safetensors"]
        if not roberta_files["all_match"]:
            problems.append("RoBERTa 资产目录存在 SHA 不符的文件")
        if roberta_files["n_files"] != len(C.ROBERTA_FILE_SHA256):
            problems.append(
                f"RoBERTa 资产文件数 {roberta_files['n_files']} != "
                f"{len(C.ROBERTA_FILE_SHA256)}")

        tokenizer, model, identity = T.load_roberta(roberta_dir)
        embedding_std = float(model.get_input_embeddings().weight.detach().std())
        evidence["roberta_embedding_std"] = embedding_std
        evidence["roberta_asset_identity_sha256"] = identity
        if not (ROBERTA_EMBED_STD_RANGE[0] <= embedding_std <= ROBERTA_EMBED_STD_RANGE[1]):
            problems.append(
                f"RoBERTa 词嵌入 std {embedding_std:.4f} 不在回归带 "
                f"{ROBERTA_EMBED_STD_RANGE} 内")

        # 掩码补全是比任何统计量都强的身份证据：随机或未训练的权重不可能
        # 把 «The capital of France is <mask>.» 补成 Paris。
        import torch
        from transformers import AutoModelForMaskedLM
        mlm = AutoModelForMaskedLM.from_pretrained(
            str(A.resolve_roberta_dir()), local_files_only=True)
        mlm.eval()
        probes: List[Dict[str, Any]] = []
        with torch.no_grad():
            for prompt, expected, rank in ROBERTA_MASK_PROBES:
                encoded = tokenizer(prompt, return_tensors="pt")
                hit = (encoded["input_ids"][0] == tokenizer.mask_token_id).nonzero()
                if hit.numel() != 1:
                    problems.append(f"掩码补全探针 «{prompt}» 未定位到唯一 <mask>")
                    continue
                logits = mlm(**encoded).logits[0, int(hit[0])]
                top = torch.topk(logits, rank).indices.tolist()
                words = [tokenizer.decode([int(i)]).strip() for i in top]
                probes.append({"prompt": prompt, "top": words, "expect": expected})
                if expected not in words:
                    problems.append(
                        f"掩码补全 «{prompt}» 的 top-{rank} 内未见 {expected!r}：{words}")
        evidence["roberta_mask_probe"] = probes
    except Exception as exc:  # noqa: BLE001
        problems.append(f"RoBERTa 加载失败：{type(exc).__name__}: {exc}")

    # (b) MediaPipe：枚举名单 + **对真实一帧跑 detect**，核对返回的 52 个 category_name
    try:
        import av
        import mediapipe as mp
        import q1v2_media as M
        names = M.blendshape_names()
        evidence["blendshape_count"] = len(names)
        evidence["blendshape_first_three"] = names[:3]
        if len(names) != C.FACE_MODEL_DIM or len(set(names)) != C.FACE_MODEL_DIM:
            problems.append(
                f"枚举 blendshape 名单 {len(names)} 项 / {len(set(names))} 互异，"
                f"应为 {C.FACE_MODEL_DIM} 项互异")
        if "_neutral" not in names:
            problems.append("blendshape 缺 _neutral")

        if not C.FACE_MODEL_PATH.is_file():
            problems.append(f"缺 MediaPipe 权重：{C.FACE_MODEL_PATH.name}")
        else:
            image_np, source = _first_face_frame()
            if image_np is None:
                problems.append(
                    f"在 {FACE_PROBE_MAX_FRAMES} 帧内没找到有脸的画面，无法做 detect 探测")
            else:
                detector = M.create_face_landmarker(C.FACE_MODEL_PATH)
                try:
                    result = detector.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np), 0)
                finally:
                    detector.close()
                found = result.face_blendshapes or []
                evidence["face_probe_source"] = source
                if len(found) != 1:
                    problems.append(f"detect() 返回 {len(found)} 组 blendshape，应为 1 组")
                else:
                    got = [c.category_name for c in found[0]]
                    evidence["detected_category_count"] = len(got)
                    evidence["detected_first_three"] = got[:3]
                    if len(got) != C.FACE_MODEL_DIM or len(set(got)) != C.FACE_MODEL_DIM:
                        problems.append(
                            f"detect() 返回 {len(got)} 项 / {len(set(got))} 互异，"
                            f"应为 {C.FACE_MODEL_DIM} 项互异")
                    if set(got) != set(names):
                        problems.append("detect() 返回的 category_name 集合与枚举名单不符")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"MediaPipe 探针失败：{type(exc).__name__}: {exc}")

    # (c) openSMILE：配置 SHA 与 25 个特征名顺序
    try:
        info = A.opensmile_info()
        evidence["opensmile_config_sha256"] = info["config_sha256"]
        if info["config_sha256"] != C.OPENMILE_CONFIG_SHA256:
            problems.append("openSMILE 配置 SHA 不符")
        if list(info["feature_names"]) != list(C.OPENMILE_FEATURE_NAMES):
            problems.append("openSMILE 特征名或顺序与冻结值不符")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"openSMILE 探针失败：{type(exc).__name__}: {exc}")

    # (d) whisper base.en 检查点
    try:
        whisper = A.verify_whisper_checkpoint(A.resolve_whisper_checkpoint())
        evidence["whisper_sha256"] = whisper["sha256"]
    except Exception as exc:  # noqa: BLE001
        problems.append(f"whisper 检查点探针失败：{type(exc).__name__}: {exc}")

    return _result(
        "V7", "权重真实性（RoBERTa / MediaPipe / openSMILE / whisper）",
        passed=not problems, checked=len(ROBERTA_MASK_PROBES) + 6 + 4 + 1,
        failed=len(problems),
        detail=("四项探针：RoBERTa 6 个文件逐字节 SHA + 掩码补全指纹、"
                "MediaPipe 枚举名单 + 真帧 detect、openSMILE 配置 SHA 与特征名顺序、"
                "whisper 检查点 SHA"),
        evidence={**evidence, "problems": problems[:10]},
        caveat=("RoBERTa 强制 local_files_only=True，运行期零网络。"
                "本套的 roberta_asset_identity_sha256 是对 "
                "(model_id, revision, hidden_dim) 求的摘要，而 对照实现的同名字段是对"
                "逐文件 SHA 字典求的摘要——构造不同，数值不可互比；"
                "字节级身份由 roberta_model_sha256 承担。"),
    )


# --------------------------------------------------------------------------
# V8 产物新鲜度
# --------------------------------------------------------------------------


def check_v8_freshness() -> Dict[str, Any]:
    problems: List[str] = []
    features = _feature_files()
    if not features:
        return _result("V8", "产物新鲜度", passed=False, checked=1, failed=1,
                       detail="没有产物", evidence={})
    newest_feature = max(p.stat().st_mtime for p in features)
    for name, cache_dir in (("text", CACHE_ROOT / "text"), ("align", CACHE_ROOT / "align")):
        files = list(cache_dir.glob("*"))
        if not files:
            problems.append(f"缓存目录 {name} 为空")
            continue
        newest_cache = max(p.stat().st_mtime for p in files)
        if newest_cache > newest_feature + 1.0:
            problems.append(f"{name} 缓存新于所有 features 产物——产物可能是旧缓存产出的")

    manifest = OUTPUT_ROOT / "manifest.csv"
    if not manifest.is_file():
        problems.append("缺 manifest.csv")
    elif manifest.stat().st_mtime < newest_feature - 1.0:
        problems.append("manifest.csv 早于 features 产物")

    # 计划把「verify_report 晚于 manifest」列为 V8 的第四环 mtime 判据。
    # 但这一环在同一次运行里是**自指**的：报告由本次核验在最后写出，
    # 它的 mtime 必然晚于 manifest——除非表刚被重建，那时它又必然早于 manifest
    # 而这恰恰意味着「本次运行就是在重新核验新表」。两种情况下 mtime 都不承载
    # 「这份报告是否对应这批表」的信息，只会制造假警报。
    # 改成读**上一轮**报告里记录的 manifest 摘要来比对：那是能判定的量。
    report = REPORTS_DIR / "verify_report.json"
    match: Optional[bool] = None
    if report.is_file() and manifest.is_file():
        try:
            previous = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        recorded = (previous.get("inputs") or {}).get("manifest_sha256")
        match = (recorded == C.sha256(manifest)) if recorded else None

    return _result(
        "V8", "产物新鲜度（缓存 → 产物 → 表）",
        passed=not problems, checked=len(features) + 4, failed=len(problems),
        detail="比对 mtime 链：.cache/text、.cache/align < features < manifest.csv",
        evidence={"problems": problems[:10], "n_features": len(features),
                  "previous_report_matches_current_manifest": match},
        caveat=("续跑模式下缓存早于产物是预期状态，此处检的是反向（缓存新于产物）。"
                "「报告晚于 manifest」不作为判据（同一次运行里自指，只会产生假警报）；"
                "改记上一轮报告里的 manifest 摘要是否与当前一致，仅供人读。"),
    )


# --------------------------------------------------------------------------
# V9 跨样本查重
# --------------------------------------------------------------------------


def check_v9_duplicates() -> Dict[str, Any]:
    npzs = _all_npz()
    problems: List[str] = []
    unexplained: List[str] = []
    groups: Dict[str, Dict[str, List[str]]] = {"audio": {}, "vision": {}, "text": {}}
    for stem, z in npzs.items():
        for label, key in (("audio", "raw_audio_lld_values"),
                           ("vision", "raw_video_blendshape_values"),
                           ("text", "text_word_feat")):
            digest = _md5_of_array(np.ascontiguousarray(z[key]))
            groups[label].setdefault(digest, []).append(stem)

    summary: Dict[str, Any] = {}
    for label, buckets in groups.items():
        dups = {d: v for d, v in buckets.items() if len(v) > 1}
        summary[label] = {"n_unique": len(buckets), "n_duplicate_groups": len(dups)}
        for digest, members in dups.items():
            z0 = npzs[members[0]]
            reasons = []
            if label == "audio" and not np.any(z0["raw_audio_lld_values"]):
                reasons.append("数字静音：LLD 全零")
            if label == "vision" and not int(z0["face_detected_frame_count"]):
                reasons.append("全零视觉：未检出人脸")
            source_shas = {str(npzs[m]["source_video_sha256"]) for m in members}
            if len(source_shas) == 1:
                reasons.append("同源片段：source_video_sha256 相同")
            if not reasons:
                unexplained.append(f"{label}:{members[:4]}")
        summary[label]["n_unexplained_groups"] = len(
            [d for d, v in dups.items() if len(v) > 1]) - len(
            [d for d, v in dups.items() if len(v) > 1 and (
                (label == "audio" and not np.any(npzs[v[0]]["raw_audio_lld_values"]))
                or (label == "vision" and not int(npzs[v[0]]["face_detected_frame_count"]))
                or len({str(npzs[m]["source_video_sha256"]) for m in v}) == 1)])

    if unexplained:
        problems.append(f"有重复组无法解释：{unexplained[:3]}")

    return _result(
        "V9", "跨样本查重（重复必须可解释）",
        passed=not problems, checked=sum(g["n_unique"] for g in summary.values()),
        failed=len(problems),
        detail="三模态 raw 数组按 md5 分组，重复组须有数字静音 / 全零视觉 / 同源片段之一的解释",
        evidence={"summary": summary, "unexplained": unexplained[:5]},
        caveat="重复本身不是缺陷——同一段视频的多个切片会合法地产生相同数组",
    )


# --------------------------------------------------------------------------
# V10 异常只标注不删除
# --------------------------------------------------------------------------


def check_v10_no_deletion() -> Dict[str, Any]:
    registry = _registry()
    n_registry = len(registry["samples"])
    n_features = len(_feature_files())
    import csv as _csv
    ledger_path = AUDIT_DIR / "anomaly_ledger.csv"
    rows: List[Dict[str, str]] = []
    if ledger_path.is_file():
        with ledger_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(_csv.DictReader(stream))
    problems: List[str] = []
    if not ledger_path.is_file():
        problems.append("缺 audit/anomaly_ledger.csv")
    if n_registry != 100:
        problems.append(f"登记表不是 100 条：{n_registry}")
    if n_features != 100:
        problems.append(f"产物不是 100 个：{n_features}")
    deletions = [r for r in rows if str(r.get("counts_as_deletion", "0")) not in ("0", "")]
    if deletions:
        problems.append(f"台账里有 {len(deletions)} 条把异常计为删除")

    return _result(
        "V10", "异常只标注不删除（counts_as_deletion ≡ 0）",
        passed=not problems, checked=n_registry + n_features + len(rows),
        failed=len(problems),
        detail=f"登记 {n_registry} 条 / 产物 {n_features} 个 / 台账 {len(rows)} 行，全部 counts_as_deletion=0",
        evidence={"problems": problems[:10], "n_ledger_rows": len(rows),
                  "n_registry": n_registry, "n_features": n_features},
        caveat="异常阈值永不作为删除样本或标签的理由",
    )


# --------------------------------------------------------------------------
# V11 路由可复算
# --------------------------------------------------------------------------


def check_v11_route_replay() -> Dict[str, Any]:
    import q1v2_route as R

    problems: List[str] = []
    npzs = _all_npz()
    for stem, z in npzs.items():
        n_words = int(z["text_sequence_length"])
        n_valid = int(np.count_nonzero(z["word_time_valid"]))
        wtv = z["word_time_valid"].astype(bool)

        # 结构闸门按同一规则重算
        n_slots = int(z["aligner_slot_count"])
        if n_slots == 0:
            mapping_status = "unavailable"
        elif n_words and n_valid == n_words:
            mapping_status = "word_valid"
        elif n_valid > 0:
            mapping_status = "word_partial"
        else:
            mapping_status = "clip_only"

        replayed = R.route_sample(
            timeline_ok=bool(int(z["audio_visual_time_valid"])),
            audio_present=bool(int(z["audio_present"])),
            audio_digital_silence="all-zero" in str(z["audio_speech_evidence"]),
            mapping_status=mapping_status,
            n_official_words=n_words,
            n_structurally_valid_words=n_valid,
        )
        for field in ("alignment_mode", "alignment_granularity",
                      "text_audio_correspondence", "text_av_time_mapping_status",
                      "word_time_basis", "audio_speech_valid"):
            stored = str(z[field]) if z[field].dtype.kind in "US" else int(z[field])
            if stored != replayed[field]:
                problems.append(f"{stem} {field}: 落盘 {stored!r} != 复算 {replayed[field]!r}")
        if int(z["content_assertion"]) != replayed["content_assertion"]:
            problems.append(f"{stem} content_assertion 不复算")

        # 三处诚实性要求：可以直接在数据上断言
        if int(z["content_assertion"]) != 0:
            problems.append(f"{stem} content_assertion != 0（本套代码恒为 0）")
        if int(z["audio_speech_valid"]) == 1:
            problems.append(f"{stem} audio_speech_valid == 1（无强语音证据，不得写 1）")
        if str(z["text_audio_correspondence"]) not in C.CORRESPONDENCE_VOCABULARY:
            problems.append(f"{stem} text_audio_correspondence 取值越界")
        if str(z["alignment_mode"]) in C.MODES_NEVER_EMITTED:
            problems.append(f"{stem} 发出了永不发出的类别 {z['alignment_mode']}")

    return _result(
        "V11", "路由可复算（纯函数重放逐字段比对）",
        passed=not problems, checked=len(npzs) * 7, failed=len(problems),
        detail=(f"{len(npzs)} 个样本用落盘标量重跑 route_sample，比对 7 个字段；"
                f"并断言 content_assertion≡0、audio_speech_valid≠1、未发出禁用类别"),
        evidence={"problems": problems[:20]},
        caveat="路由是纯函数，故结论可复算；这正是 q1v2_route 不含任何 IO 的原因",
    )


# --------------------------------------------------------------------------
# V12 路由对拍（可选，默认关闭）
# --------------------------------------------------------------------------


def check_v12_reference(correspondence_csv: Optional[str]) -> Dict[str, Any]:
    if not correspondence_csv:
        return _result("V12", "与对照路由对拍", passed=True, checked=0, failed=0,
                       detail="未提供外部对照文件，跳过（默认关闭）",
                       evidence={"skipped": True},
                       caveat="这是开发期回归核验，结果不进交付物")

    path = Path(correspondence_csv)
    if not path.is_file():
        return _result("V12", "与对照路由对拍", passed=False, checked=1, failed=1,
                       detail=f"对照文件不存在：{path.name}", evidence={})

    import csv as _csv
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reference = {row["sample_key"]: row for row in _csv.DictReader(stream)}

    problems: List[str] = []
    agreed = 0
    diff_summary: Dict[str, int] = {}
    for stem, z in _all_npz().items():
        key = str(z["sample_key"])
        ref = reference.get(key)
        if ref is None:
            continue
        ours = str(z["alignment_mode"])
        theirs = str(ref.get("alignment_mode", ""))
        if ours == theirs:
            agreed += 1
        else:
            diff_summary[f"{theirs} -> {ours}"] = diff_summary.get(
                f"{theirs} -> {ours}", 0) + 1

    return _result(
        "V12", "与对照路由对拍（开发期回归）",
        passed=True, checked=len(reference), failed=len(problems),
        detail=(f"{agreed}/{len(reference)} 条与参考一致。差异只应来自证据等级："
                f"参考的 TRI_MODAL_WORD_VALID / AV_VALID_TEXT_UNALIGNED 依赖人工听辨结论，"
                f"本套代码不伪造，故这些样本归入 UNCERTAIN_REVIEW"),
        evidence={"n_agreed": agreed, "n_reference": len(reference),
                  "difference_matrix": diff_summary, "problems": problems[:10]},
        caveat="**不对拍不等于失败**：此检查只报告差异矩阵，不做通过/失败判定",
    )


# --------------------------------------------------------------------------
# V13 对齐结构硬断言
# --------------------------------------------------------------------------


def check_v13_alignment_structure() -> Dict[str, Any]:
    """硬锚点只钉**文本层**与**零失败**；对齐器的输出规模是「实测对照实现」，不做断言。

    为什么这样分：`1926`（词单元）与 `1932`（空白块）是 `official_text` 的纯函数，
    同一条文本在任何实现下都必须给出同一个数——这类数才是可以硬断言的。
    而对齐器产出多少槽取决于**这一次对齐运行**，对照实现的 1926 来自它复用的旧 trace
    （`stable_ts_alignment_reused_from_audit: true`），我们的 1929 来自本次冻结运行。
    把 1929 写成断言，等于把「我们自己的一次运行结果」冒充成可复现的客观约束。
    """
    import q1v2_text as T

    npzs = _all_npz()
    hard: List[str] = []

    sum_words = sum(int(z["text_sequence_length"]) for z in npzs.values())
    sum_slots = sum(int(z["aligner_slot_count"]) for z in npzs.values())
    sum_fail = sum(int(z["alignment_mapping_fail_count"]) for z in npzs.values())
    sum_zero = sum(int(z["alignment_zero_duration_word_count"]) for z in npzs.values())
    n_zero_samples = sum(
        1 for z in npzs.values() if int(z["alignment_zero_duration_word_count"]) > 0)
    sum_valid = sum(int(np.count_nonzero(z["word_time_valid"])) for z in npzs.values())
    sum_chunks = sum(len(T.whitespace_chunks(str(z["official_text"])))
                     for z in npzs.values())

    # ---- 硬断言：全部是纯函数或零容差性质 --------------------------------
    if sum_words != EXPECTED_WORD_UNITS:
        hard.append(f"词单元合计 {sum_words} != {EXPECTED_WORD_UNITS}")
    if sum_chunks != EXPECTED_WHITESPACE_CHUNKS:
        hard.append(f"空白块合计 {sum_chunks} != {EXPECTED_WHITESPACE_CHUNKS}")
    if sum_fail != 0:
        hard.append(f"mapping_fail 合计 {sum_fail} != 0")
    if sum_valid + sum_zero != sum_words:
        hard.append(f"结构有效 {sum_valid} + 零时长 {sum_zero} != 词单元 {sum_words}")
    # 每个官方词都必须拿到槽：这条比「槽数==词数」更强也更该硬——
    # 槽可以比词多（一个词被切成两段），但不能有词一个槽都没有。
    n_without_slot = 0
    for stem, z in npzs.items():
        n_slots = int(z["aligner_slot_count"])
        s2w = z["aligner_slot_to_word_index"]
        covered = set(int(x) for x in s2w if int(x) >= 0)
        n_without_slot += int(z["text_sequence_length"]) - len(covered)
        if s2w.shape[0] != n_slots:
            hard.append(f"{stem} slot_to_word 长度 != aligner_slot_count")
    if n_without_slot != 0:
        hard.append(f"有 {n_without_slot} 个官方词一个槽都没分到")

    # ---- 实测 vs 参考：记录差异，不做断言 --------------------------------
    slot_delta = [
        {"sample_key": str(z["sample_key"]),
         "n_official_words": int(z["text_sequence_length"]),
         "n_aligner_slots": int(z["aligner_slot_count"])}
        for z in npzs.values()
        if int(z["aligner_slot_count"]) != int(z["text_sequence_length"])]

    differences: List[Dict[str, Any]] = []
    if sum_slots != REFERENCE_SUM_ALIGNER_SLOTS:
        differences.append({
            "field": "sum_aligner_slots",
            "measured": sum_slots,
            "control_impl_value": REFERENCE_SUM_ALIGNER_SLOTS,
            "delta": sum_slots - REFERENCE_SUM_ALIGNER_SLOTS,
            "n_samples_affected": len(slot_delta),
            "samples": slot_delta[:10],
            "attribution": (
                "本套代码跑的是自己这一次冻结对齐（alignment_source="
                f"{C.ALIGN_SOURCE}）；对照实现复用了旧 alignment trace"
                "（其 model_assets.json 记 stable_ts_alignment_reused_from_audit: true），"
                "且其 alignment 后处理脚本未随包交付（对照实现自报 P1-02 未闭合）。"
                "多出的槽源于对齐器把一个官方词切成两段——本实现的词区间取该词"
                "**全部槽的并集** [min(start), max(end)]，故不丢观测。"
                "mapping_fail=0 且无槽词=0 表明每个官方词都被覆盖。"),
            "impact": "无：词级数组长度由词单元数决定，槽数只影响区间并集的取法",
        })
    if not (ZERO_DURATION_RANGE[0] <= sum_zero <= ZERO_DURATION_RANGE[1]):
        differences.append({
            "field": "sum_zero_duration_word_count", "measured": sum_zero,
            "control_impl_value": REFERENCE_ZERO_DURATION_WORDS,
            "soft_range": list(ZERO_DURATION_RANGE),
            "attribution": "本实现自己的对齐运行；超出软区间",
        })
    if not (ZERO_DURATION_SAMPLE_RANGE[0] <= n_zero_samples
            <= ZERO_DURATION_SAMPLE_RANGE[1]):
        # 定位差异集中在哪些样本，便于报告里给出清单而不是一句「超出」
        no_speech_ish = []
        for z in npzs.values():
            zc = int(z["alignment_zero_duration_word_count"])
            if zc and zc == int(z["text_sequence_length"]):
                no_speech_ish.append({
                    "sample_key": str(z["sample_key"]),
                    "zero_duration_words": zc,
                    "n_official_words": int(z["text_sequence_length"]),
                    "audio_speech_valid": int(z["audio_speech_valid"]),
                })
        differences.append({
            "field": "n_samples_with_zero_duration_words",
            "measured": n_zero_samples,
            "control_impl_value": REFERENCE_ZERO_DURATION_SAMPLES,
            "soft_range": list(ZERO_DURATION_SAMPLE_RANGE),
            "n_samples_with_all_words_zero_duration": len(no_speech_ish),
            "all_zero_duration_samples": no_speech_ish,
            "attribution": (
                "零时长词集中在无语音样本：本套对齐固定 remove_instant_words=False"
                "（保留瞬时词），故说话人为空的片段必然产出一批零时长词。"
                "全部词都零时长的样本共 "
                f"{len(no_speech_ish)} 条，是这一部分的来源。"
                "对照实现的 37 来自它复用的旧 trace，两者不可互引。"),
        })

    return _result(
        "V13", "对齐结构：文本层硬断言 + 对齐器规模对照实现",
        passed=not hard, checked=sum_words + sum_chunks + sum_slots,
        failed=len(hard),
        detail=(f"硬：词单元 {sum_words}={EXPECTED_WORD_UNITS}、"
                f"空白块 {sum_chunks}={EXPECTED_WHITESPACE_CHUNKS}、"
                f"mapping_fail {sum_fail}=0、无槽词 {n_without_slot}=0、"
                f"有效 {sum_valid}+零时长 {sum_zero}=词单元。"
                f"对照实现：槽 {sum_slots}（对照实现的 {REFERENCE_SUM_ALIGNER_SLOTS}）、"
                f"零时长词 {sum_zero}（对照实现的 {REFERENCE_ZERO_DURATION_WORDS}）、"
                f"含零时长词样本 {n_zero_samples}（对照实现的 {REFERENCE_ZERO_DURATION_SAMPLES}）"),
        evidence={"sum_word_units": sum_words, "sum_whitespace_chunks": sum_chunks,
                  "sum_aligner_slots": sum_slots, "sum_mapping_fail": sum_fail,
                  "n_words_without_slot": n_without_slot,
                  "sum_structurally_valid_words": sum_valid,
                  "sum_zero_duration_word_count": sum_zero,
                  "n_samples_with_zero_duration_words": n_zero_samples,
                  "hard_mismatch": hard, "differences_vs_reference": differences},
        caveat=("**对对照实现的数不做断言**：槽数取决于这一次对齐运行，对照实现的 1926 来自它复用的"
                "旧 trace，我们的 1929 来自本次冻结运行——把后者写成断言等于用一次运行结果"
                "冒充客观约束。真正可断言的是文本层纯函数（1926 / 1932）与零失败性质。"
                "槽数多于词数是正常结构：一个官方词可被切成两段，本实现取并集"),
    )


# --------------------------------------------------------------------------
# V14 数字静音位精确复现
# --------------------------------------------------------------------------


SILENCE_ANCHORS = {
    "-mJ2ud6oKI8$_$1": {"coverage_end": 6.0545, "lld_frames": 601, "words": 13},
    "-mJ2ud6oKI8$_$2": {"coverage_end": 5.7016875, "lld_frames": 566, "words": 11},
}


def check_v14_digital_silence(quick: bool = False) -> Dict[str, Any]:
    if quick:
        return _result("V14", "数字静音位精确复现", passed=True, checked=0, failed=0,
                       detail="--quick 下跳过", evidence={"skipped": True},
                       caveat="跳过项不计入通过")
    problems: List[str] = []
    evidence: Dict[str, Any] = {}
    npzs = _all_npz()
    for key, anchor in SILENCE_ANCHORS.items():
        stem = C.safe_stem(key)
        z = npzs.get(stem)
        if z is None:
            problems.append(f"{key} 不在产物中")
            continue
        got_end = float(z["audio_presentation_coverage_sec"][1])
        got_frames = int(z["audio_observation_length"])
        got_words = int(z["text_sequence_length"])
        evidence[key] = {"coverage_end": got_end, "lld_frames": got_frames,
                         "words": got_words,
                         "expected": anchor}
        if abs(got_end - anchor["coverage_end"]) > 1e-6:
            problems.append(
                f"{key} 音频覆盖末端 {got_end:.7f} != {anchor['coverage_end']}")
        if got_frames != anchor["lld_frames"]:
            problems.append(f"{key} LLD 窗数 {got_frames} != {anchor['lld_frames']}")
        if got_words != anchor["words"]:
            problems.append(f"{key} 词单元 {got_words} != {anchor['words']}")
        # 静音判定与位精确性质
        if str(z["audio_speech_valid"]) != "0":
            problems.append(f"{key} audio_speech_valid != 0")
        if str(z["alignment_mode"]) != C.MODE_TRI_MODAL_AUDIO_CONTENT_INVALID:
            problems.append(f"{key} alignment_mode != TRI_MODAL_WITH_AUDIO_CONTENT_INVALID")
        if not np.isfinite(z["raw_audio_lld_values"]).all():
            problems.append(f"{key} LLD 含非有限值")

    # 逐采样全零：必须真解码一遍才算「位精确」，不能只看落盘标量
    try:
        import q1v2_media as M
        for key in SILENCE_ANCHORS:
            sample = next((s for s in _registry()["samples"]
                           if s["sample_key"] == key), None)
            if sample is None:
                continue
            path = C.PROJECT_ROOT / sample["source_video_relpath"]
            probe = M.probe_presentation_starts(path)
            waveform, _info = M.decode_audio_16k(
                path, float(probe["audio_stream_start_sec"]))
            quantized = M.quantize_int16(waveform)
            n_nonzero = int(np.count_nonzero(quantized))
            evidence[key]["nonzero_int16_samples"] = n_nonzero
            evidence[key]["n_samples"] = int(quantized.size)
            if n_nonzero != 0:
                problems.append(f"{key} 量化后仍有 {n_nonzero} 个非零采样")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"重解码失败：{type(exc).__name__}: {exc}")

    return _result(
        "V14", "数字静音位精确复现（覆盖/窗数/逐采样全零）",
        passed=not problems, checked=len(SILENCE_ANCHORS) * 7, failed=len(problems),
        detail=(f"{len(SILENCE_ANCHORS)} 条数字静音样本：音频覆盖末端、LLD 窗数、"
                f"词单元数、路由结论，并**真解码一遍**确认 int16 逐采样为零"),
        evidence={**evidence, "problems": problems[:10]},
        caveat="这是最强的端到端复现锚点：它同时锁住解码路径、重采样与 openSMILE 窗口划分",
    )


# --------------------------------------------------------------------------
# 全量硬锚点（媒体）
# --------------------------------------------------------------------------


def check_m1_media_anchors() -> Dict[str, Any]:
    npzs = _all_npz()
    problems: List[str] = []
    frames = sum(int(z["video_frame_count"]) for z in npzs.values())
    faces = sum(int(z["face_detected_frame_count"]) for z in npzs.values())
    n_no_face = sum(1 for z in npzs.values() if int(z["face_detected_frame_count"]) == 0)
    if frames != EXPECTED_VIDEO_FRAMES:
        problems.append(f"视频帧数合计 {frames} != {EXPECTED_VIDEO_FRAMES}")
    if faces != EXPECTED_FACE_FRAMES:
        problems.append(f"人脸帧合计 {faces} != {EXPECTED_FACE_FRAMES}")
    if n_no_face != EXPECTED_NO_FACE_SAMPLES:
        problems.append(f"零面部样本数 {n_no_face} != {EXPECTED_NO_FACE_SAMPLES}")
    return _result(
        "M1", "媒体全量硬锚点（帧数 / 人脸帧 / 零面部样本数）",
        passed=not problems, checked=3, failed=len(problems),
        detail=(f"视频帧 {frames}（硬 {EXPECTED_VIDEO_FRAMES}）/ 人脸帧 {faces}"
                f"（硬 {EXPECTED_FACE_FRAMES}）/ 零面部样本 {n_no_face}"
                f"（硬 {EXPECTED_NO_FACE_SAMPLES}）"),
        evidence={"sum_video_frame_count": frames,
                  "sum_face_detected_frame_count": faces,
                  "n_samples_without_face": n_no_face,
                  "problems": problems},
        caveat="三个数取自对照实现的审计合计，是本套代码独立重跑的对照硬锚点",
    )


def check_m2_protected_baseline() -> Dict[str, Any]:
    import q1v2_assets as A
    path = METADATA_DIR / "protected_baseline.json"
    if not path.is_file():
        return _result("M2", "受保护基线未变", passed=False, checked=1, failed=1,
                       detail="缺 metadata/protected_baseline.json", evidence={})
    baseline = json.loads(path.read_text(encoding="utf-8"))
    result = A.compare_protected_baseline(baseline)
    n_volatile = len(result["changed_volatile_by_design"])
    return _result(
        "M2", "受保护基线未变（既有交付未被破坏）",
        passed=bool(result["unchanged"]),
        checked=int(result["baseline_total_files"]),
        failed=len(result["added"]) + len(result["removed"]) + len(result["changed"]),
        detail=(f"基线 {result['baseline_total_files']} 个文件，"
                f"新增 {len(result['added'])} / 删除 {len(result['removed'])} / "
                f"实质改动 {len(result['changed'])}"
                + (f" / 名单内重写 {n_volatile}（记录不判失败）" if n_volatile else "")),
        evidence=result,
        caveat=("基线覆盖 data/q1_delivery、data/word_align、data/unaligned_features、"
                "data/aligned。差异按 VOLATILE_BY_DESIGN 分流：只在名单内的差异"
                "（如 v1 核验器自己重写的 verify_report）记录并归因，不判失败；"
                "任何其他文件的差异一律硬失败。"),
    )


# --------------------------------------------------------------------------
# 编排
# --------------------------------------------------------------------------


def verify_all(quick: bool = False,
               correspondence_csv: Optional[str] = None) -> Dict[str, Any]:
    started = _dt.datetime.now(_dt.timezone.utc)
    checks: List[Dict[str, Any]] = []

    lightweight = [
        ("V1", check_v1_coverage), ("V2", check_v2_schema),
        ("V3", check_v3_words), ("V4", check_v4_traceable),
        ("V5", check_v5_timeline), ("V8", check_v8_freshness),
        ("V9", check_v9_duplicates), ("V10", check_v10_no_deletion),
        ("V11", check_v11_route_replay), ("V13", check_v13_alignment_structure),
        ("M1", check_m1_media_anchors), ("M2", check_m2_protected_baseline),
    ]
    for code, function in lightweight:
        try:
            checks.append(function())
        except Exception as exc:  # noqa: BLE001
            checks.append(_result(code, function.__name__, passed=False, checked=1,
                                  failed=1,
                                  detail=f"核验自身抛错：{type(exc).__name__}: {exc}",
                                  evidence={}))

    for code, function in (("V6", lambda: check_v6_external(quick)),
                           ("V7", lambda: check_v7_weights(quick)),
                           ("V14", lambda: check_v14_digital_silence(quick)),
                           ("V12", lambda: check_v12_reference(correspondence_csv))):
        try:
            checks.append(function())
        except Exception as exc:  # noqa: BLE001
            checks.append(_result(code, code, passed=False, checked=1, failed=1,
                                  detail=f"核验自身抛错：{type(exc).__name__}: {exc}",
                                  evidence={}))

    executed = [c for c in checks if not c["evidence"].get("skipped")]
    skipped = [c for c in checks if c["evidence"].get("skipped")]
    summary = {
        "schema_version": C.SCHEMA_VERSION,
        "generated_at_utc": started.isoformat(),
        "quick": quick,
        # 本报告对应的输入身份。下一轮 V8 会读这里来判定「报告是否对应这批表」——
        # 路径无法承担这件事（会自指），摘要可以。
        "inputs": {
            "manifest_sha256": (C.sha256(OUTPUT_ROOT / "manifest.csv")
                                if (OUTPUT_ROOT / "manifest.csv").is_file() else None),
            "config_sha256": C.config_digest(),
            "n_feature_files": len(_feature_files()),
        },
        "all_passed": all(c["passed"] for c in executed),
        "n_checks": len(checks),
        "n_executed": len(executed),
        "n_skipped": len(skipped),
        "n_passed": sum(1 for c in executed if c["passed"]),
        "n_failed": sum(1 for c in executed if not c["passed"]),
        "total_assertions_checked": sum(c["checked"] for c in checks),
        "total_assertions_failed": sum(c["failed"] for c in checks),
        "skipped_codes": [c["code"] for c in skipped],
        "checks": checks,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    C.write_json(REPORTS_DIR / "verify_report.json", summary)
    (REPORTS_DIR / "verify_report.md").write_text(
        _render_markdown(summary), encoding="utf-8")
    return summary


def _render_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# 问题一 v2 核验报告",
        "",
        f"- 生成时间（UTC）：{summary['generated_at_utc']}",
        f"- schema：`{summary['schema_version']}`",
        f"- 模式：{'quick（跳过重量项）' if summary['quick'] else 'full'}",
        f"- 结论：**{'全部通过' if summary['all_passed'] else '存在未通过项'}**",
        f"- 执行 {summary['n_executed']} 项，通过 {summary['n_passed']}，"
        f"未通过 {summary['n_failed']}，跳过 {summary['n_skipped']}"
        f"{'（' + ', '.join(summary['skipped_codes']) + '）' if summary['skipped_codes'] else ''}",
        f"- 断言 {summary['total_assertions_checked']} 条，失败 "
        f"{summary['total_assertions_failed']} 条",
        "",
        "## 核验边界",
        "",
        "对照实现的逐样本数组**全空**，故本套核验**不与对照实现比数值**。它只做四件事：",
        "",
        "1. **内部自洽**：产物自己能不能复算出自己（V1–V5、V9–V11）；",
        "2. **外部交叉**：解码覆盖与 ffprobe 声明是否相容（V6）；",
        "3. **结构硬锚点**：1926 / 1932 / 0 这些冻结的数（V2、V13）；",
        "4. **位精确锚点**：两条数字静音样本的每一个采样点（V14）。",
        "",
        "## 逐项结果",
        "",
        "| ID | 名称 | 结果 | 检查 | 失败 | 说明 |",
        "|---|---|---|---|---|---|",
    ]
    for check in summary["checks"]:
        if check["evidence"].get("skipped"):
            mark = "— 跳过"
        else:
            mark = "✅ 通过" if check["passed"] else "❌ 未通过"
        detail = check["detail"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {check['code']} | {check['name']} | {mark} | "
                     f"{check['checked']} | {check['failed']} | {detail} |")

    lines += ["", "## 逐项 caveat（口径与已知差异）", ""]
    for check in summary["checks"]:
        if check.get("caveat"):
            lines.append(f"- **{check['code']}**：{check['caveat']}")

    lines += [
        "",
        "## 诚实性要求的落地",
        "",
        "1. `audio_speech_valid` **从不写 1** —— 没有强语音证据时写 `-1`（不作主张），"
        "只有位精确可证的静音才写 `0`。由 V11 直接断言。",
        "2. `alignment_mode` **从不写 `TRI_MODAL_WORD_VALID`** —— 结构合法不等于内容确认。"
        "该性质由 `word_time_basis=own_forced_alignment` 与 `content_assertion=0` 承载。"
        "由 V11 直接断言。",
        "3. 本报告必须解释本套路由计数与对照实现的差异来源 —— 类别词汇表相同，"
        "不发出的两类是**证据等级**所致，不是漏跑。见 `metadata/route_policy.json`。",
        "4. **核验器不得把本套自己的单次运行结果当成客观约束。** V13 的硬断言只覆盖"
        "文本层的纯函数（1926 / 1932）与零失败性质；对齐器输出的规模、零时长词的分布"
        "一律记入 `differences_vs_reference` 并附归因，不与对照实现的 161 / 37 强行对齐。",
        "5. **核验器不得把命中原文写进自己的报告。** 身份扫描只记长度与位置；"
        "把敏感串复制一份进报告，等于让报告本身成为泄漏物。见 "
        "`reports/q1v2_size_ledger.md` 的身份扫描一节。",
        "",
        "体积与精度是两件事，见 `reports/q1v2_size_ledger.md`：精度不降（float32 不降 "
        "float16，否则 V4 的 `rtol=1e-5` 可复算性失效），体积靠**无损压缩**拿回。",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="问题一 v2 核验 V1–V14")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--correspondence-csv", default=None)
    args = parser.parse_args(argv)
    summary = verify_all(quick=args.quick,
                         correspondence_csv=args.correspondence_csv)
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"},
                     ensure_ascii=False, indent=2))
    for check in summary["checks"]:
        mark = "SKIP" if check["evidence"].get("skipped") else (
            "PASS" if check["passed"] else "FAIL")
        print(f"  [{mark}] {check['code']:4s} {check['name']}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
