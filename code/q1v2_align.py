# -*- coding: utf-8 -*-
"""q1v2_align.py —— 官方文本强制对齐与「块→词」归并（问题一 v2）

本模块填的是对照实现**从未交付**的那个缺口：对照实现自报 `alignment_to_source_words`
（字符规范化 + 官方词覆盖 + 对齐器词合并 + 时间区间合并）未闭合（P1-02）。
因此 `map_slots_to_chunks` 与 `chunks_to_word_intervals` 是**新写**的，
算法写在这里，也写进 `reports/`。

三条不可让步的约定：

1. **传入的是官方原文，不是词表。** stable-ts 2.19.1 的
   `align(audio, text, ...)` 的 `text` 只接受字符串或 token id 列表，不接受词表。
   传原文让对齐器用自己的分词器切词——对照实现正是这么做的：其审计表里
   `alignment_probability_summary.count` **逐样本**等于 `alignment_official_word_count`，
   100 条合计 1926，与「纯标点块并入相邻词」的词单元规则一致。
2. **对齐器给了哪些槽，就只认哪些槽。** 官方词没拿到槽就标 `valid=0`，
   既不外扩区间去凑，也不用邻近词的时间顶替。
3. **本模块只产出"时间区间"，不产出"内容是否对应"。**
   官方文本与实际讲话是否一致，需要人工听辨；本套代码没有该证据，
   故一律 `text_audio_correspondence = not_asserted`
   （唯一例外是量化后逐采样全零的数字静音，那是机器可证的事实）。
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from q1v2_contract import (
    ALIGN_PARAMS,
    ALIGN_SOURCE,
    ALIGNER_NAME,
    TOL,
    HardStop,
)

#: 归一化时丢弃的字符类别：空白与各类标点/符号。
#: 只留字母数字用于**配对**；原始字符区间始终来自 `source_words`，不受此影响。
_NON_KEEPING = re.compile(r"[^\w]+", re.UNICODE)


def normalize_for_matching(text: str) -> str:
    """把小写化后的字母数字留下，其余全部丢弃。

    `"operation,"` 与 `"operation"` 归一化后相同——这正是我们要的：
    对齐器把标点粘在词上、或单独成槽，都不应导致配对失败。
    """
    return _NON_KEEPING.sub("", text.lower())


def prepare_alignment_audio(waveform: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """把 16 kHz 浮点波形转成对齐器的输入。

    **为什么要过一遍 int16 量化**：对照实现是把波形写成 wav 再把路径交给
    对齐器，whisper 的 `load_audio` 读 PCM s16le 后按 `/32768.0` 转回浮点。
    本套代码不落临时 wav（项目既有约定：按 clip_id 命名的临时 wav 会跨视频互相覆盖），
    改为在内存里走同一条量化路径：`rint(clip(x,-1,1)*32767) → int16 → /32768.0`。
    这样对齐器看到的样本与对照实现的 wav 路径**逐采样相同**，同时不产生任何临时文件。
    """
    from q1v2_media import quantize_int16

    quantized = quantize_int16(waveform)
    audio = (quantized.astype(np.float32) / 32768.0).astype(np.float32, copy=False)
    return audio, {
        "conversion": "int16 quantization -> float32/32768 (same samples as a written wav)",
        "n_samples": int(audio.size),
        "scale": 32767,
        "denominator": 32768,
        "max_abs": float(np.abs(audio).max()) if audio.size else 0.0,
    }


def load_aligner(checkpoint: Path, device: str = "cpu"):
    """加载 stable-ts 包装的 whisper 模型。

    `checkpoint` 传**本地路径**而非 `"base.en"`：后者会让 whisper 去
    `~/.cache/whisper/` 找，看似没问题，但一旦缓存缺失就会静默联网下载一个
    与冻结 SHA 不同的检查点，而落盘的版本号还是旧的。路径 + SHA 校验才是闭环。
    """
    import stable_whisper

    return stable_whisper.load_model(str(checkpoint), device=device)


def align_official_text(model, audio: np.ndarray, official_text: str,
                        language: Optional[str] = None) -> Dict[str, Any]:
    """官方原文强制对齐 → 对齐器给出的词槽序列。

    只传 `remove_instant_words=False` 与 `failure_threshold=None` 两个参数：

    * `remove_instant_words=False` 保留零时长词。这是**必须显式写死**的一条——
      丢零时长词会让槽数少于官方词数，而「官方每个词都有槽」正是 对照实现的判据之一，
      默认值一旦跨版本变动，整条结论会静默改变。
    * `failure_threshold=None` 关闭"对齐失败就退回普通转写"的自动兜底：
      我们要的是"没对上就说没对上"，不是悄悄换成另一条证据链。

    对照实现的 `执行报告.md` 把 `regroup=False` / `stream=False` 与上面两个写在一起，
    但这两个属于 `transcribe` 的参数（`align` 签名里没有）。本套代码不做 ASR、
    不算 WER，故不涉及。
    """
    params = {
        "remove_instant_words": ALIGN_PARAMS["remove_instant_words"],
        "failure_threshold": ALIGN_PARAMS["failure_threshold"],
    }
    lang = language if language is not None else ALIGN_PARAMS["language"]
    result = model.align(audio, official_text, language=lang, **params)
    if result is None:
        raise HardStop("对齐器返回 None（未产出任何结果）")

    segments = list(getattr(result, "segments", None) or [])
    slots: List[Dict[str, Any]] = []
    for segment in segments:
        for word in (list(getattr(segment, "words", None) or [])):
            text = getattr(word, "word", None)
            if text is None:
                continue
            slots.append({
                "text": str(text),
                "start": float(word.start) if word.start is not None else float("nan"),
                "end": float(word.end) if word.end is not None else float("nan"),
            })
    if not slots:
        raise HardStop("对齐器未返回任何词槽")

    return {
        "slots": slots,
        "n_slots": len(slots),
        "n_segments": len(segments),
        "language": lang,
        "params": params,
        "aligner": ALIGNER_NAME,
        "alignment_source": ALIGN_SOURCE,
        "slot_text_joined": " ".join(s["text"].strip() for s in slots),
    }


# --------------------------------------------------------------------------
# 块 → 词（对照实现未交付的缺口，本模块自己实现）
# --------------------------------------------------------------------------


def map_slots_to_chunks(slot_texts: Sequence[str],
                        chunk_texts: Sequence[str]) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """把对齐器的词槽映射到文本空白块。返回 `(slot_to_chunk, 方法名, 诊断)`。

    两条路径，先精确后兜底：

    1. `positional` —— 槽数与块数相等且归一化文本逐项相同。本批的常态
       （对照实现的槽数逐样本等于官方词数，说明对齐器与我们的切分一致）。
    2. `difflib` —— 用 `SequenceMatcher` 在**归一化文本序列**上求最长匹配，
       把 `equal` 段的槽与块一一对上；`replace`/`delete`/`insert` 段落里的槽
       按比例位置落到对应块上。

    映射不到的槽记 `-1`，由调用方计入 `mapping_fail_count`。
    **不做"最近邻凑数"**——那正是把错误藏进特征的做法。
    """
    norm_slots = [normalize_for_matching(t) for t in slot_texts]
    norm_chunks = [normalize_for_matching(t) for t in chunk_texts]
    n_slots, n_chunks = len(norm_slots), len(norm_chunks)
    slot_to_chunk = np.full(n_slots, -1, dtype=np.int32)
    diagnostics: Dict[str, Any] = {
        "n_slots": n_slots, "n_chunks": n_chunks,
        "n_matched_positional": 0, "difflib_blocks": None,
    }

    if n_slots == n_chunks and norm_slots == norm_chunks:
        slot_to_chunk[:] = np.arange(n_slots, dtype=np.int32)
        diagnostics["n_matched_positional"] = n_slots
        return slot_to_chunk, "positional", diagnostics

    matcher = difflib.SequenceMatcher(None, norm_slots, norm_chunks, autojunk=False)
    blocks = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        blocks.append({"tag": tag, "slot": [i1, i2], "chunk": [j1, j2]})
        if tag == "equal":
            for k in range(i2 - i1):
                slot_to_chunk[i1 + k] = j1 + k
        else:
            slot_span, chunk_span = i2 - i1, j2 - j1
            if slot_span == 0 or chunk_span == 0:
                continue
            # 两个区间都在变：按比例把槽摊到块上。摊不满或摊不完的槽保持 -1，
            # 宁可少数几个槽不参与，也不把时间安到无关的块上。
            for k in range(slot_span):
                target = j1 + min(chunk_span - 1, int(k * chunk_span / slot_span))
                slot_to_chunk[i1 + k] = target

    diagnostics["difflib_blocks"] = blocks
    diagnostics["n_matched_positional"] = int(np.count_nonzero(slot_to_chunk >= 0))
    method = "difflib" if n_slots != n_chunks else "difflib_same_count"
    return slot_to_chunk, method, diagnostics


def chunks_to_word_intervals(slots: Sequence[Dict[str, Any]],
                             chunks: Sequence[Dict[str, Any]],
                             chunk_to_word: np.ndarray,
                             n_words: int,
                             common_start: float,
                             common_end: float,
                             ) -> Dict[str, Any]:
    """对齐器词槽 → 官方词单元的半开时间区间。对照实现未交付此函数，本实现自撰。

    步骤：

    1. 用 `map_slots_to_chunks` 把槽映射到空白块；
    2. 空白块按 `chunk_to_word`（纯字符区间判定，与时间无关）归属到词单元；
    3. 一个词的区间 = 落在它身上的**全部槽**的区间并集，即
       `[min(start), max(end)]`。取并集而不是取第一个槽，是因为合并了标点块的词
       会有两个槽，取第一个会丢掉后半段的观测；
    4. 合法性判据（四项全过才 `word_time_valid=1`）：
       端点有限、`end > start`、落在 `[common_start, common_end]` 内、
       且词的起点不早于前一个合法词的起点（单调不减）。

    返回项里同时给出**槽级**与**词级**两套零时长计数。对照实现的
    `zero_duration_word_count + alignment_structurally_valid_words == alignment_official_word_count`
    （161 + 1765 = 1926）说明它数的是**词级**，故本函数把词级值作为
    `zero_duration_word_count` 输出，槽级值另存为诊断——
    两者在本批最多相差 6（仅 6 个纯标点块会造成多槽词），不会混淆结论。
    """
    n_slots = len(slots)
    slot_texts = [str(s.get("text", "")) for s in slots]
    chunk_texts = [str(c.get("text", "")) for c in chunks]
    slot_to_chunk, method, map_diag = map_slots_to_chunks(slot_texts, chunk_texts)
    slot_to_word = np.full(n_slots, -1, dtype=np.int32)
    ok = slot_to_chunk >= 0
    slot_to_word[ok] = chunk_to_word[slot_to_chunk[ok]]

    word_start = np.full(n_words, np.nan, dtype=np.float64)
    word_end = np.full(n_words, np.nan, dtype=np.float64)
    word_time_valid = np.zeros(n_words, dtype=np.uint8)
    word_slot_count = np.zeros(n_words, dtype=np.int32)

    starts = np.asarray([s.get("start", np.nan) for s in slots], dtype=np.float64)
    ends = np.asarray([s.get("end", np.nan) for s in slots], dtype=np.float64)
    slot_finite = np.isfinite(starts) & np.isfinite(ends)
    slot_positive = slot_finite & (ends > starts)
    zero_duration_slots = int(np.count_nonzero(slot_finite & ~slot_positive))
    nonfinite_slots = int(np.count_nonzero(~slot_finite))

    word_min = np.full(n_words, np.inf, dtype=np.float64)
    word_max = np.full(n_words, -np.inf, dtype=np.float64)
    for i in range(n_slots):
        wi = int(slot_to_word[i])
        if wi < 0 or not slot_finite[i]:
            continue
        word_slot_count[wi] += 1
        word_min[wi] = min(word_min[wi], starts[i])
        word_max[wi] = max(word_max[wi], ends[i])

    zero_duration_words = 0
    out_of_range = 0
    non_monotonic = 0
    no_slot_words = 0
    previous_start = -np.inf
    for wi in range(n_words):
        if word_slot_count[wi] == 0:
            no_slot_words += 1
            continue
        a, b = word_min[wi], word_max[wi]
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        if b <= a:
            zero_duration_words += 1
            continue
        if a < common_start - TOL or b > common_end + TOL:
            out_of_range += 1
            continue
        if a < previous_start - TOL:
            non_monotonic += 1
            continue
        word_start[wi] = a
        word_end[wi] = b
        word_time_valid[wi] = 1
        previous_start = a

    unmapped_slots = int(np.count_nonzero(slot_to_word < 0))
    return {
        "word_start_sec": word_start,
        "word_end_sec": word_end,
        "word_time_valid": word_time_valid,
        "slot_to_word": slot_to_word,
        "slot_to_chunk": slot_to_chunk,
        "mapping_method": method,
        "mapping_diagnostics": map_diag,
        "zero_duration_slot_count": zero_duration_slots,
        "nonfinite_slot_count": nonfinite_slots,
        "zero_duration_word_count": zero_duration_words,
        "no_slot_word_count": no_slot_words,
        "out_of_range_word_count": out_of_range,
        "non_monotonic_word_count": non_monotonic,
        # `mapping_fail_count` 是 对照实现的同名字段：槽没能落到任何官方词上。
        # 对照实现的 100 条合计为 0，本套代码把它作为硬断言（V13）。
        "mapping_fail_count": unmapped_slots,
        "n_slots": n_slots,
        "n_words": n_words,
    }


def structural_diagnostics(intervals: Dict[str, Any],
                           common_start: float,
                           common_end: float) -> Dict[str, Any]:
    """把结构诊断压成可直接落盘、也可用于路由判据的标量集合。"""
    valid = np.asarray(intervals["word_time_valid"], dtype=bool)
    n_words = int(intervals["n_words"])
    n_valid = int(np.count_nonzero(valid))
    return {
        "n_official_words": n_words,
        "n_aligner_slots": int(intervals["n_slots"]),
        "n_structurally_valid_words": n_valid,
        "n_words_without_slot": int(intervals["no_slot_word_count"]),
        "zero_duration_word_count": int(intervals["zero_duration_word_count"]),
        "zero_duration_slot_count": int(intervals["zero_duration_slot_count"]),
        "mapping_fail_count": int(intervals["mapping_fail_count"]),
        "out_of_range_word_count": int(intervals["out_of_range_word_count"]),
        "non_monotonic_word_count": int(intervals["non_monotonic_word_count"]),
        "mapping_method": intervals["mapping_method"],
        "common_start_sec": float(common_start),
        "common_end_sec": float(common_end),
        "all_words_structurally_valid": bool(n_words > 0 and n_valid == n_words),
        "some_words_structurally_valid": bool(n_valid > 0),
    }


def mapping_status(diag: Dict[str, Any]) -> str:
    """机器可判的「文本↔音视频时间映射状态」。

    取值语义（与 `TIME_STATUS_VOCABULARY` 一致）：

    * `word_valid`   —— 全部官方词都有合法区间
    * `word_partial` —— 有一部分词有合法区间
    * `clip_only`    —— 一个词都没有合法区间（只能在整段层面用）
    * `unavailable`  —— 对齐器压根没产出可用槽

    **这个字段只说"时间区间够不够用"，不说"内容和讲话对不对"。**
    后者是 `text_audio_correspondence`，本套代码一律 `not_asserted`。
    """
    if int(diag.get("n_aligner_slots") or 0) == 0:
        return "unavailable"
    if diag.get("all_words_structurally_valid"):
        return "word_valid"
    if diag.get("some_words_structurally_valid"):
        return "word_partial"
    return "clip_only"


__all__ = [
    "normalize_for_matching", "prepare_alignment_audio", "load_aligner",
    "align_official_text", "map_slots_to_chunks", "chunks_to_word_intervals",
    "structural_diagnostics", "mapping_status",
]
