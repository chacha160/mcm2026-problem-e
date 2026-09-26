# -*- coding: utf-8 -*-
"""q1v2_correspondence.py —— 逐词对应表导出（问题一 v2）

赛题四(二)2.(3) 要求「选取至少 1 个典型样本，展示其文本片段、对应语音时段、
视频帧段与三类特征的对应关系」。本模块把这条要求落成**可复现的产物**，
而不是手工摘抄一张表。

三条设计约束：

1. **只读成品 NPZ，不碰视频、不加载任何模型。** 输入全部来自 `features/*.npz`
   里已落盘的 raw 序列与 CSR 索引，因此耗时是秒级。这与 `q1v2_build.reaggregate_one`
   是同一条思路——凡是能由产物推出的东西，就不重解一遍。
2. **模块级不导入 torch / mediapipe / opensmile / av / stable_whisper。**
   与 `q1v2_contract` 同一条规矩：核验与导出要能在裸解释器里跑。
3. **值要么来自 NPZ，要么现场由 NPZ 复算。** 不引入任何外部常量到数值列里，
   唯一例外是三个特征维度——它们取自 `q1v2_contract` 的冻结常量，并被
   `q1v2_verify` 的 V2 逐样本断言过。

坐标口径（写错会让整张表错位，故在此写死）：

* 词区间 `word_start_sec` / `word_end_sec` 在**共享 presentation 时间轴**上。
* `raw_video_pts_sec` 同在共享轴上，可直接与词区间比较。
* `raw_audio_lld_center_sec` 由 openSMILE 按**音频切片**给出（相对切片起点）；
  `media_stage` 与 `reaggregate_one` 都用 `词区间 − audio_presentation_coverage_sec[0]`
  去与它比较。**本批 100 条样本的 `shared_t0_sec` 恒为 0，而
  `audio_presentation_coverage_sec[0] == shared_t0`（V5 断言），
  故该偏移在本批上是空操作，两种口径数值重合。** 本模块照 `reaggregate_one`
  的写法带上偏移，使口径在一开始就正确；同时在 `_meta.json` 里如实记下
  「本批 shared_t0 ≡ 0，该偏移未被数据区分」——不假装它被验证过。
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from q1v2_contract import (
    FACE_MODEL_DIM,
    OPENMILE_LLD_DIM,
    OUTPUT_ROOT,
    ROBERTA_HIDDEN_DIM,
    SCHEMA_VERSION,
    HardStop,
    safe_stem,
    sha256,
    write_csv,
    write_json,
)

#: 逐词对应表的列序。顺序即阅读顺序：先身份与文本，再时间，再掩码，
#: 再源观测，最后是特征维度与备注。
CORRESPONDENCE_COLUMNS = (
    "word_index", "word", "char_start", "char_end", "char_span_original",
    "word_start_sec", "word_end_sec", "word_time_valid",
    # 掩码列名与 NPZ 字段**逐字一致**（`word_audio_valid` / `word_vision_valid`，
    # 不是 `audio_word_valid`）。CSV 是 NPZ 的人读视图，同一个概念若在这里换个
    # 语序，读者就得自己建立一张对照表——而这正是最容易看错的地方。
    "text_word_valid", "word_audio_valid", "word_vision_valid",
    "n_audio_windows", "audio_window_range", "audio_center_span_sec",
    "n_face_frames", "face_frame_range", "face_pts_span_sec",
    "n_frames_in_span", "frame_range_in_span",
    "text_feat_dim", "audio_feat_dim", "vision_feat_dim", "note",
)

#: 自动选样判据（全部可复算，写进 `_meta.json`）。
SELECT_MIN_WORDS = 10
SELECT_MAX_WORDS = 30
SELECT_MIN_FACE_RATIO = 0.90
SELECT_MIN_TRI_VALID_RATIO = 0.80


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------


def _f(value: Any, digits: int = 6) -> str:
    """浮点写成定长小数字符串。NaN 写成空串——表里留白比写 `nan` 更易读。"""
    v = float(value)
    if not np.isfinite(v):
        return ""
    return f"{v:.{digits}f}"


def _span(values: np.ndarray, digits: int = 6) -> str:
    """把一串时刻写成 `起–止`；空串表示没有观测。"""
    if len(values) == 0:
        return ""
    return f"{_f(values[0], digits)}–{_f(values[-1], digits)}"


def _index_range(indexes: np.ndarray) -> str:
    if len(indexes) == 0:
        return ""
    return f"{int(indexes[0])}–{int(indexes[-1])}" if len(indexes) > 1 else f"{int(indexes[0])}"


def _csr_slice(npz: Dict[str, np.ndarray], prefix: str, word_index: int) -> np.ndarray:
    """取第 `word_index` 个词的 CSR 源观测下标。

    下标指向 `raw_audio_lld_values` / `raw_video_blendshape_values` 的**第 0 维**。
    """
    indptr = np.asarray(npz[f"{prefix}_word_feature_indptr"])
    indices = np.asarray(npz[f"{prefix}_word_feature_indices"])
    return indices[indptr[word_index]:indptr[word_index + 1]]


# --------------------------------------------------------------------------
# 逐词装配
# --------------------------------------------------------------------------


def build_rows(npz: Dict[str, np.ndarray]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """把一条样本的 NPZ 展开成逐词对应行。"""
    words = [str(w) for w in npz["words"]]
    official_text = str(npz["official_text"])
    char_start = np.asarray(npz["word_char_start"])
    char_end = np.asarray(npz["word_char_end"])
    wstart = np.asarray(npz["word_start_sec"], dtype=np.float64)
    wend = np.asarray(npz["word_end_sec"], dtype=np.float64)
    wtime = np.asarray(npz["word_time_valid"], dtype=np.uint8)
    tvalid = np.asarray(npz["text_word_valid"], dtype=np.uint8)
    avalid = np.asarray(npz["word_audio_valid"], dtype=np.uint8)
    vvalid = np.asarray(npz["word_vision_valid"], dtype=np.uint8)

    lld_center = np.asarray(npz["raw_audio_lld_center_sec"], dtype=np.float64)
    pts = np.asarray(npz["raw_video_pts_sec"], dtype=np.float64)
    face_valid = np.asarray(npz["raw_video_face_feature_valid"], dtype=bool)
    audio_offset = float(npz["audio_presentation_coverage_sec"][0])
    shared_t0 = float(npz["shared_t0_sec"])

    text_feat = npz["text_word_feat"]
    audio_feat = npz["word_audio_feat"]
    vision_feat = npz["word_vision_feat"]

    rows: List[Dict[str, Any]] = []
    zero_rows_with_valid_mask: List[int] = []
    valid_rows_all_zero: List[int] = []

    for i, word in enumerate(words):
        cs, ce = int(char_start[i]), int(char_end[i])
        ws, we = float(wstart[i]), float(wend[i])
        span_ok = bool(np.isfinite(ws) and np.isfinite(we) and we > ws)

        a_idx = _csr_slice(npz, "audio", i)
        v_idx = _csr_slice(npz, "vision", i)
        # 时段内的**全部**帧（含未检出人脸的帧）。与 v_idx（只含参与聚合的
        # 人脸有效帧）分开列，正是 `word_vision_valid` 语义的来源。
        in_span = (np.flatnonzero((pts >= ws) & (pts < we))
                   if span_ok else np.asarray([], dtype=int))

        notes: List[str] = []
        if not span_ok:
            notes.append("无合法区间，不凑观测")
        if span_ok and len(a_idx) == 0:
            notes.append("区间内无 LLD 窗")
        if span_ok and len(v_idx) == 0:
            notes.append("区间内无人脸有效帧" if len(in_span) else "区间内无视频帧")

        # 两条自检：掩码为 0 的词，其特征行必须整行为零；掩码为 1 的词必须非全零。
        tr = np.asarray(text_feat[i], dtype=np.float64)
        ar = np.asarray(audio_feat[i], dtype=np.float64)
        vr = np.asarray(vision_feat[i], dtype=np.float64)
        if tvalid[i] == 0 and tr.any():
            zero_rows_with_valid_mask.append(i)
        if avalid[i] == 0 and ar.any():
            zero_rows_with_valid_mask.append(i)
        if vvalid[i] == 0 and vr.any():
            zero_rows_with_valid_mask.append(i)
        if (tvalid[i] == 1 and not tr.any()) or (avalid[i] == 1 and not ar.any()) \
                or (vvalid[i] == 1 and not vr.any()):
            valid_rows_all_zero.append(i)

        rows.append({
            "word_index": i,
            "word": word,
            "char_start": cs,
            "char_end": ce,
            "char_span_original": official_text[cs:ce],
            "word_start_sec": _f(ws),
            "word_end_sec": _f(we),
            "word_time_valid": int(wtime[i]),
            "text_word_valid": int(tvalid[i]),
            "word_audio_valid": int(avalid[i]),
            "word_vision_valid": int(vvalid[i]),
            "n_audio_windows": len(a_idx),
            "audio_window_range": _index_range(a_idx),
            "audio_center_span_sec": _span(lld_center[a_idx]) if len(a_idx) else "",
            "n_face_frames": len(v_idx),
            "face_frame_range": _index_range(v_idx),
            "face_pts_span_sec": _span(pts[v_idx]) if len(v_idx) else "",
            "n_frames_in_span": len(in_span),
            "frame_range_in_span": _index_range(in_span),
            "text_feat_dim": int(text_feat.shape[1]),
            "audio_feat_dim": int(audio_feat.shape[1]),
            "vision_feat_dim": int(vision_feat.shape[1]),
            "note": "；".join(notes),
        })

    info = {
        "sample_key": str(npz["sample_key"]),
        "n_words": len(words),
        "n_words_valid_time": int(wtime.sum()),
        "n_words_valid_text": int(tvalid.sum()),
        "n_words_valid_audio": int(avalid.sum()),
        "n_words_valid_vision": int(vvalid.sum()),
        "valid_length": int((tvalid.astype(bool) & avalid.astype(bool)
                             & vvalid.astype(bool)).sum()),
        "face_valid_ratio": float(face_valid.mean()) if len(face_valid) else 0.0,
        "rows_with_mismatched_zero": zero_rows_with_valid_mask,
        "rows_flagged_valid_but_all_zero": valid_rows_all_zero,
        "audio_offset_sec": audio_offset,
        "shared_t0_sec": shared_t0,
    }
    return rows, info


# --------------------------------------------------------------------------
# 独立复算（与 V4 同源判据）
# --------------------------------------------------------------------------


def verify_rows(npz: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """按 `s ≤ center < e` 独立重选源观测，要求与落盘 CSR **逐元素相等**。

    这是 V4 的判据在导出侧的复述——导出表若与 NPZ 不一致，首先会在这一步暴露。
    """
    words = npz["words"]
    wstart = np.asarray(npz["word_start_sec"], dtype=np.float64)
    wend = np.asarray(npz["word_end_sec"], dtype=np.float64)
    wtime = np.asarray(npz["word_time_valid"], dtype=bool)

    audio_offset = float(npz["audio_presentation_coverage_sec"][0])
    shared_t0 = float(npz["shared_t0_sec"])
    a_center = np.asarray(npz["raw_audio_lld_center_sec"], dtype=np.float64)
    v_pts = np.asarray(npz["raw_video_pts_sec"], dtype=np.float64)
    v_face = np.asarray(npz["raw_video_face_feature_valid"], dtype=bool)

    a_ok = v_ok = 0
    a_bad: List[int] = []
    v_bad: List[int] = []
    for i in range(len(words)):
        a = float(wstart[i]) - audio_offset
        b = float(wend[i]) - audio_offset
        if wtime[i] and np.isfinite(a) and np.isfinite(b) and b > a:
            sel = np.flatnonzero((a_center >= a) & (a_center < b))
            if np.array_equal(sel, _csr_slice(npz, "audio", i)):
                a_ok += 1
            else:
                a_bad.append(i)
        a2 = float(wstart[i]) - shared_t0
        b2 = float(wend[i]) - shared_t0
        if wtime[i] and np.isfinite(a2) and np.isfinite(b2) and b2 > a2:
            sel = np.flatnonzero((v_pts >= a2) & (v_pts < b2) & v_face)
            if np.array_equal(sel, _csr_slice(npz, "vision", i)):
                v_ok += 1
            else:
                v_bad.append(i)

    return {
        "audio_csr_reproduced": a_ok,
        "audio_csr_mismatch": a_bad,
        "vision_csr_reproduced": v_ok,
        "vision_csr_mismatch": v_bad,
        "all_reproduced": not a_bad and not v_bad,
    }


# --------------------------------------------------------------------------
# 选样
# --------------------------------------------------------------------------


def _score(rows: List[Dict[str, Any]], info: Dict[str, Any]) -> int:
    """可复算的选样得分：**同时**含 ≥2 个语音窗与 ≥2 个人脸帧的词有几个。

    这个词数直接决定对应表能不能展示「多源观测聚合成一个词向量」这件事——
    若每词都只落 1 个窗、1 帧，表就看不出聚合过程。
    """
    return sum(1 for r in rows
               if r["n_audio_windows"] >= 2 and r["n_face_frames"] >= 2)


def select_typical(feature_dir: Path, count: int = 1) -> List[Tuple[str, Dict[str, Any]]]:
    """按冻结判据挑典型样本。判据全部写进 `_meta.json`，可复算。"""
    candidates: List[Tuple[int, str, Dict[str, Any]]] = []
    rejected: Dict[str, int] = {}
    for path in sorted(feature_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as handle:
            npz = {k: handle[k] for k in handle.files}
        n_words = int(npz["text_sequence_length"])
        reasons = []
        if not (SELECT_MIN_WORDS <= n_words <= SELECT_MAX_WORDS):
            reasons.append("词数不在区间内")
        face_ratio = float(np.asarray(npz["raw_video_face_feature_valid"]).mean())
        if face_ratio < SELECT_MIN_FACE_RATIO:
            reasons.append("人脸有效率不足")
        tri = (np.asarray(npz["text_word_valid"]).astype(bool)
               & np.asarray(npz["word_audio_valid"]).astype(bool)
               & np.asarray(npz["word_vision_valid"]).astype(bool))
        if n_words == 0 or tri.sum() < SELECT_MIN_TRI_VALID_RATIO * n_words:
            reasons.append("三模态共同有效词占比不足")
        if reasons:
            for r in reasons:
                rejected[r] = rejected.get(r, 0) + 1
            continue
        rows, info = build_rows(npz)
        if _score(rows, info) == 0:
            rejected["无『多源观测』词"] = rejected.get("无『多源观测』词", 0) + 1
            continue
        candidates.append((_score(rows, info), str(npz["sample_key"]), info))

    # 得分降序；同分按 sample_key 升序——保证选样是确定性的。
    candidates.sort(key=lambda t: (-t[0], t[1]))
    picked = [(key, info) for _, key, info in candidates[:count]]
    if not picked:
        raise HardStop(f"没有样本满足典型样本判据。拒绝原因统计：{rejected}")
    return picked


# --------------------------------------------------------------------------
# 总入口
# --------------------------------------------------------------------------


def export(sample_keys: Optional[Sequence[str]] = None, *,
           count: int = 1,
           feature_dir: Optional[Path] = None,
           out_dir: Optional[Path] = None,
           write_meta: bool = True) -> Dict[str, Any]:
    feature_dir = Path(feature_dir or (OUTPUT_ROOT / "features"))
    out_dir = Path(out_dir or (OUTPUT_ROOT / "correspondence"))
    out_dir.mkdir(parents=True, exist_ok=True)

    selection: Dict[str, Any] = {
        "mode": "explicit" if sample_keys else "auto",
        "criteria": {
            "词数区间": [SELECT_MIN_WORDS, SELECT_MAX_WORDS],
            "人脸有效率下限": SELECT_MIN_FACE_RATIO,
            "三模态共同有效词占比下限": SELECT_MIN_TRI_VALID_RATIO,
            "得分定义": "同时含 ≥2 个 LLD 窗与 ≥2 个人脸帧的词数（降序）；同分按 sample_key 升序",
        },
    }

    if sample_keys:
        targets = [(k, {}) for k in sample_keys]
    else:
        targets = select_typical(feature_dir, count=count)
        selection["selected"] = [k for k, _ in targets]

    written: List[Dict[str, Any]] = []
    for key, _info in targets:
        path = feature_dir / f"{safe_stem(key)}.npz"
        if not path.is_file():
            raise HardStop(f"features/ 下没有 {path.name}")
        with np.load(path, allow_pickle=False) as handle:
            npz = {k: handle[k] for k in handle.files}
        rows, info = build_rows(npz)
        check = verify_rows(npz)
        out_csv = out_dir / f"{safe_stem(key)}_correspondence.csv"
        write_csv(out_csv, rows, CORRESPONDENCE_COLUMNS)
        written.append({
            "sample_key": info["sample_key"],
            "csv": out_csv.name,
            "source_npz": path.name,
            "source_npz_sha256": sha256(path),
            "n_rows": len(rows),
            "score": _score(rows, info),
            **{k: v for k, v in info.items() if k != "sample_key"},
            "csr_check": check,
        })

    result = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "tool": {
            "python": platform.python_version(),
            "module": "code/q1v2_correspondence.py",
            "note": "只读成品 NPZ，不读媒体、不加载模型，故耗时为秒级",
        },
        "selection": selection,
        "coordinate_note": (
            "词区间与 raw_video_pts_sec 同在共享 presentation 时间轴上，可直接比较；"
            "raw_audio_lld_center_sec 由 openSMILE 按音频切片给出，本模块照 "
            "reaggregate_one 的写法用『词区间 − audio_presentation_coverage_sec[0]』与它比较。"
            "**本批 100 条样本的 shared_t0_sec 恒为 0，而 "
            "audio_presentation_coverage_sec[0] == shared_t0（V5 断言），故该偏移在本批上"
            "是空操作，两种口径数值重合——它未被本批数据区分，不声称已被验证。**"
        ),
        "column_note": (
            "n_face_frames / face_frame_range 只含参与 word_vision_feat 聚合的"
            "**人脸有效帧**；n_frames_in_span / frame_range_in_span 是该时段内的"
            "**全部帧**（含未检出人脸的帧）。两列并列，便于核对掩码语义。"
        ),
        "files": written,
    }
    if write_meta:
        # 清掉上一轮留下、本轮未入选的对应表。
        #
        # 这个目录里的 `*_correspondence.csv` 全部由本模块写出（文件名带本模块的
        # 固定后缀），故「不在本次 `written` 里」就等于「上一轮选样留下的孤儿」。
        # 不清的话，换个 `--count` 跑一次就会留下**列名与本次不同**的旧文件：
        # `_meta.json` 只列本轮结果，目录里却多出两份没人认领的表——读者无从判断
        # 哪份有效，而两份的表头还可能不一致。交付物里不能有这种状态。
        keep = {w["csv"] for w in written}
        removed = sorted(p.name for p in out_dir.glob("*_correspondence.csv")
                         if p.name not in keep)
        for name in removed:
            (out_dir / name).unlink()
        result["stale_removed"] = removed
        write_json(out_dir / "_meta.json", result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="问题一 v2：从成品 NPZ 导出逐词对应表（不读媒体）")
    parser.add_argument("--sample-key", action="append", default=None,
                        help="指定样本（可重复）；省略则按冻结判据自动选样")
    # 默认 3 而非 1：论文与交付物里列的是 3 个典型样本，而文档给出的复现命令是
    # **不带参数**的 `python code/q1v2_correspondence.py`。默认值与交付结果不一致，
    # 就等于「照文档跑一遍得到的是另一样东西」——这种不一致最难在事后发现。
    parser.add_argument("--count", type=int, default=3,
                        help="自动选样时取前 N 个（按得分降序），默认 3")
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    started = time.time()
    result = export(args.sample_key, count=args.count,
                    feature_dir=Path(args.feature_dir) if args.feature_dir else None,
                    out_dir=Path(args.out_dir) if args.out_dir else None)
    elapsed = time.time() - started
    print(json.dumps({
        "elapsed_sec": round(elapsed, 3),
        "selection": result["selection"]["mode"],
        "files": [{k: f[k] for k in ("sample_key", "csv", "n_rows", "score")}
                  for f in result["files"]],
        "csr_all_reproduced": all(f["csr_check"]["all_reproduced"]
                                  for f in result["files"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CORRESPONDENCE_COLUMNS", "build_rows", "verify_rows",
    "select_typical", "export", "main",
]
