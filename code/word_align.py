# -*- coding: utf-8 -*-
"""
word_align.py
问题一：stable-ts 强制对齐，产出实测词时间并做证据路由

本模块只做一件事：把 label-100.xlsx 的官方文本强制对齐到音频，得到每个词的
实测半开区间 [start, end)，并按对齐质量给出该样本的证据路由档位。

路由档位（严格档，经方案决议确定）：
    word_level —— 官方每一个词都对齐成功，且无零时长词、区间未越出公共时间轴。
                  只有这一档才允许用它替换文本模态的均匀假设时间戳。
    clip_level —— 其余全部样本，保持均匀假设 t_j=(j+0.5)*T/W，并如实标注。

为什么必须路由：100 条里有相当比例的样本，官方文本与实际音频内容对不上，
强制对齐器只能放下它真听到的词（实测 100 条合计仅对齐出 796/2191 = 36.3%）。
此时若强行采用对齐结果，等于用「对齐器的输出」冒充「官方文本的时间」——
词序与词数都已不再对应。故只在对齐完整时才采信。

与参考方案（阶段2.1.7_A0_Pilot）的关系：
    · 同版本依赖：stable-ts 2.19.1 + openai-whisper 20250625
    · 同口径：半开区间 [start, end)，不使用 frame_index/fps 估时间
    · 差异：参考方案在词级对齐之上再叠加「内容对应确认」，最终只有 5/100 可用；
      本模块只用对齐质量这一项可机检的证据，故通过者为 7 条（实测）。
      两者不冲突——本模块的档位更宽，判据更弱，但完全可复算。

依赖：stable-ts / openai-whisper。音频解码复用 unaligned_common.decode_audio
（ffmpeg 管道，不落临时文件），与提取阶段保持同一条解码路径。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import config  # noqa: E402
import unaligned_common as U  # noqa: E402
from utils import LOGGER  # noqa: E402

# 证据路由档位。严格档：只有 fully-aligned 才升为 word_level。
MODE_WORD_LEVEL = "word_level"
MODE_CLIP_LEVEL = "clip_level"

# 默认缓存目录：data/word_align/
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(_CODE_DIR), "data", "word_align")

# 对齐可接受的时间容差（秒）。对齐器输出的边界都落在 [0, T] 内，
# 这里留 1e-6 只是浮点比较余量，不是「允许越界」。
_RANGE_EPS = 1e-6


def words_of(sample_id: str) -> List[str]:
    """
    取该样本文本模态的权威词表——必须与未对齐特征的行一一对应。

    不能对 label 原文做 split()：``unaligned_text.py`` 的分词会把 ``They've``
    拆成 ``They`` + ``'ve``，两者词数不同（实测 -3g5yACwYnA_13 为 17 vs 15）。
    用错词表会让对齐出的时间戳与特征行错位——且错位是静默的，故此处以
    未对齐 npz 里的 ``extra__words`` 为唯一来源，并对行数做一次校验。
    """
    path = os.path.join(U.unaligned_dir("text"), f"{sample_id}.npz")
    if not os.path.isfile(path):
        return []
    d = U.load_unaligned(path)
    words = list(d.get("extra", {}).get("words") or [])
    n_rows = int(d["features"].shape[0]) if getattr(d["features"], "ndim", 0) == 2 else 0
    if words and n_rows and len(words) != n_rows:
        LOGGER.warning("[词对齐] %s 词表 %d 与特征行 %d 不一致，放弃使用",
                       sample_id, len(words), n_rows)
        return []
    return words


def align_text(model, audio: np.ndarray, words: Sequence[str],
               duration: float) -> Dict[str, object]:
    """
    对单条样本做强制对齐，返回质量记录（不含特征）。

    words 必须是与文本特征行一一对应的词表（见 words_of）。

    半开区间口径：第 j 个词占据 [start_j, end_j)，槽归属用区间中心
    t_j = (start_j + end_j) / 2（与 align_multimodal.time_to_slot 的入参口径一致）。
    """
    rec: Dict[str, object] = {
        "n_official_words": 0,
        "n_aligned_words": 0,
        "zero_duration_words": 0,
        "coverage": 0.0,
        "monotonic": None,
        "in_range": None,
        "first_t": None,
        "last_t": None,
        "span": None,
        "word_start_sec": [],
        "word_end_sec": [],
        "word_ts_sec": [],
        "error": None,
    }
    words = [w for w in (words or []) if w]
    rec["n_official_words"] = len(words)
    if not words:
        rec["error"] = "词表为空（未对齐文本特征缺失，或词表与特征行数不符）"
        return rec
    if audio is None or audio.size == 0:
        rec["error"] = "音频解码为空（无音轨或解码失败）"
        return rec

    res = model.align(audio, " ".join(words), language="en")
    seg = res.segments[0] if res.segments else None
    spans = [(float(w.start), float(w.end))
             for w in (list(getattr(seg, "words", []) or []) if seg else [])]
    if not spans:
        rec["error"] = "对齐器未返回任何词区间"
        return rec

    rec["n_aligned_words"] = len(spans)
    rec["coverage"] = round(len(spans) / len(words), 4)
    rec["zero_duration_words"] = int(sum(1 for a, b in spans if b - a <= 0.0))
    rec["monotonic"] = bool(all(b >= a for a, b in spans))
    rec["in_range"] = bool(spans[0][0] >= -_RANGE_EPS and spans[-1][1] <= float(duration) + _RANGE_EPS)
    rec["word_start_sec"] = [round(a, 6) for a, _ in spans]
    rec["word_end_sec"] = [round(b, 6) for _, b in spans]
    # 槽归属用区间中心；先 clip 到 [0, T) 以免中心恰好落在 T 上时越界
    centers = [min(max((a + b) / 2.0, 0.0), float(duration) * (1 - 1e-12)) for a, b in spans]
    rec["word_ts_sec"] = [round(c, 6) for c in centers]
    rec["first_t"] = round(spans[0][0], 6)
    rec["last_t"] = round(spans[-1][1], 6)
    rec["span"] = round(spans[-1][1] - spans[0][0], 6)
    return rec


def decide_mode(rec: Dict[str, object]) -> str:
    """
    证据路由（严格档）。

    升为 word_level 的三个条件同时成立，缺一不可：
        1. 官方每一个词都对齐到（n_aligned == n_official）
        2. 无零时长词（零时长词会让两个词共享同一时刻，槽归属无法唯一）
        3. 全部区间落在公共时间轴内（否则要动用 clip 兜底，时间已非实测）
    """
    if rec.get("error"):
        return MODE_CLIP_LEVEL
    n_off = int(rec.get("n_official_words") or 0)
    n_aln = int(rec.get("n_aligned_words") or 0)
    if n_off <= 0 or n_aln != n_off:
        return MODE_CLIP_LEVEL
    if int(rec.get("zero_duration_words") or 0) > 0:
        return MODE_CLIP_LEVEL
    if not rec.get("monotonic") or not rec.get("in_range"):
        return MODE_CLIP_LEVEL
    return MODE_WORD_LEVEL


def alignment_path(sample_id: str, out_dir: Optional[str] = None) -> str:
    return os.path.join(out_dir or DEFAULT_OUT_DIR, f"{sample_id}.json")


def load_word_align(sample_id: str, out_dir: Optional[str] = None) -> Optional[Dict[str, object]]:
    """读一条样本的词对齐记录；不存在返回 None。"""
    p = alignment_path(sample_id, out_dir)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("[词对齐] 记录损坏，按无记录处理 %s: %s", p, exc)
        return None


def measured_text_pts(sample_id: str, n_words: int,
                      out_dir: Optional[str] = None) -> Optional[np.ndarray]:
    """
    取该样本的实测词时间戳（区间中心），供对齐阶段替换均匀假设时间。

    仅当路由档位为 word_level 且词数与该样本的文本单元数一致时返回数组，
    否则返回 None（调用方保持原时间戳）。这里再校验一次词数，是因为
    「未对齐特征」与「词对齐记录」可能来自不同批次的重跑，词数不符说明不同源。
    """
    rec = load_word_align(sample_id, out_dir)
    if not rec or rec.get("mode") != MODE_WORD_LEVEL:
        return None
    ts = rec.get("word_ts_sec") or []
    if len(ts) != int(n_words):
        LOGGER.warning("[词对齐] %s 词数不符：记录 %d vs 特征 %d，回退均匀假设",
                       sample_id, len(ts), n_words)
        return None
    return np.asarray(ts, dtype=np.float64)


def run(samples: Sequence[Dict[str, object]], out_dir: Optional[str] = None,
        overwrite: bool = False) -> List[Dict[str, object]]:
    """对样本清单跑词对齐并落盘。已存在且未指定 overwrite 时直接复用缓存。"""
    out_dir = out_dir or DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    todo = []
    records: List[Dict[str, object]] = []
    for s in samples:
        sid = str(s["sample_id"])
        p = alignment_path(sid, out_dir)
        if os.path.isfile(p) and not overwrite:
            with open(p, "r", encoding="utf-8") as f:
                records.append(json.load(f))
            continue
        todo.append(s)

    if not todo:
        LOGGER.info("[词对齐] 全部命中缓存，%d 条", len(records))
        return records

    import stable_whisper  # 延迟导入：未装可让本模块仍可被 import

    t0 = time.time()
    model = stable_whisper.load_model("base.en")
    LOGGER.info("[词对齐] 模型加载 %.1fs，待处理 %d 条", time.time() - t0, len(todo))

    for i, s in enumerate(todo, 1):
        sid = str(s["sample_id"])
        ta = time.time()
        audio = U.decode_audio(str(s["video_path"]))
        # 公共时间轴 T 取自提取阶段写入的 ffprobe 视频流时长，与对齐阶段同源
        duration = U.video_true_duration(str(s["video_path"]))
        words = words_of(sid)
        rec = align_text(model, audio if audio is not None else np.zeros(0, np.float32),
                         words, duration)
        rec.update({
            "sample_id": sid,
            "official_id": s.get("official_id"),
            "duration_sec": round(float(duration), 6),
            "text_original": str(s.get("text") or ""),
            "word_source": f"data/unaligned_features/text/{sid}.npz::extra__words",
            "aligner": "stable-ts",
            "aligner_version": _pkg_version("stable-ts"),
            "whisper_version": _pkg_version("openai-whisper"),
            "whisper_model": "base.en",
            "interval_convention": "half-open [start,end)",
            "slot_query": "interval center",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        rec["mode"] = decide_mode(rec)
        with open(alignment_path(sid, out_dir), "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        records.append(rec)
        LOGGER.info("[词对齐] %3d/%3d %-22s 官方 %3d 对齐 %3d 零时长 %2d → %s (%.1fs)",
                    i, len(todo), sid, rec["n_official_words"], rec["n_aligned_words"],
                    rec["zero_duration_words"], rec["mode"], time.time() - ta)

    return records


def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:  # noqa: BLE001
        return "unknown"


def summarize(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """汇总统计，供交付物与核验引用。"""
    n = len(records)
    wl = [r for r in records if r.get("mode") == MODE_WORD_LEVEL]
    tot_off = sum(int(r.get("n_official_words") or 0) for r in records)
    tot_aln = sum(int(r.get("n_aligned_words") or 0) for r in records)
    return {
        "n_samples": n,
        "n_word_level": len(wl),
        "n_clip_level": n - len(wl),
        "total_official_words": tot_off,
        "total_aligned_words": tot_aln,
        "word_recovery_ratio": round(tot_aln / tot_off, 4) if tot_off else 0.0,
        "n_samples_with_zero_duration_words": sum(
            1 for r in records if int(r.get("zero_duration_words") or 0) > 0),
        "word_level_samples": sorted(str(r["sample_id"]) for r in wl),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="问题一：stable-ts 强制对齐与证据路由")
    p.add_argument("--out-dir", default=None, help="词对齐记录输出目录")
    p.add_argument("--only", default=None, help="只处理 sample_id 含该子串的样本")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 条")
    p.add_argument("--overwrite", action="store_true", help="忽略缓存，全部重跑")
    a = p.parse_args(argv)

    samples = U.list_samples(only=a.only, limit=a.limit)
    LOGGER.info("[词对齐] 样本 %d 条", len(samples))
    records = run(samples, out_dir=a.out_dir, overwrite=a.overwrite)
    s = summarize(records)
    LOGGER.info("[词对齐] 完成：word_level %d / clip_level %d；词恢复率 %.1f%%",
                s["n_word_level"], s["n_clip_level"], 100 * s["word_recovery_ratio"])
    print(json.dumps(s, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
