# -*- coding: utf-8 -*-
"""
unaligned_text.py —— 文本模态「未对齐」特征提取

职责边界（刻意如此）：
    本文件只负责「文本 → 词级 RoBERTa 向量 (W, 768)」，**不做任何时间槽分配**。
    对齐（把 (W,768) 映射到 (50,768)）由独立的 align_multimodal.py 负责。

为什么文本没有真实时间戳
------------------------
    附件1只提供每个片段的整句转写文本，附件2也不存储逐词时间戳
    （题目说明中已明确：官方对齐文件仅保留切片后的特征，不含对齐前的时间轴）。
    因此词级时刻只能来自「假设」，本模块把它显式标注出来，绝不伪装成实测值：

        time_basis = "uniform_assumption"
        词 i 的中心时刻 = (i + 0.5) * duration / W

    这是「在已知片段总时长下，假设词在时间上均匀分布」的功效最弱假设；
    它使文本可以与音频/视觉落在同一秒轴上用于可视化，
    但必须与「音频/视觉由解码器实测 PTS 得到」区分开——后者才是硬证据。

输出（每样本一个 .npz，见 unaligned_common.save_unaligned）
    features (W, 768) float32   词级向量 = 该词全部子词隐状态的均值
    pts      (W,)     float32   词中心时刻（均匀假设，见上）
    extra.words                 词串列表，供时间轴直接显示文字
    extra.char_spans            每词在「规范化文本」中的字符区间 [start, end)
    extra.char_spans_original   同一区间换算到 label-100 原文的下标
    extra.token_index           每词占用的 input_ids 下标列表

后三个字段是问题1 交付要求的「原词、字符范围、词元索引」。它们在**提取时**记录而不是
事后由词串反推，因此与 features 出自同一趟编码，不存在二者不同源的风险；坐标口径写在
meta.char_span_basis 里，避免下游把规范化文本的区间误当原文区间。
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

MODALITY = "text"


# 规范化时会被改写的字符 → 替换串。抽成常量是为了让「带溯源映射」的实现
# 与下面 preprocess_text 共用同一张表，避免两处口径漂移。
_CHAR_REPLACEMENTS = {
    "…": "...", "’": "'", "‘": "'", "“": '"', "”": '"', "—": " - ", "–": " - ",
}


def preprocess_text(text: str) -> str:
    """
    轻量规范化：统一空白，unicode 引号/省略号/破折号归一为 ASCII。

    刻意不做停用词过滤与去标点——情感线索（否定词、程度副词、感叹号）分布在全部词上，
    过度清洗会破坏与音频/视觉模态的时序对应关系。
    """
    import re
    if not isinstance(text, str):
        return ""
    t = text
    for src, dst in _CHAR_REPLACEMENTS.items():
        t = t.replace(src, dst)
    return re.sub(r"\s+", " ", t).strip()


def preprocess_text_with_map(text: str) -> Tuple[str, List[int]]:
    """
    与 preprocess_text 逐字符等价，但额外给出「规范化后每个字符来自原文的哪个下标」。

    为什么需要它：问题1 的交付要求 alignment_q1.json 记录每个词的**字符范围**。
    词元（subword）的字符区间是相对于**规范化后**的文本的（fast tokenizer 的
    offset_mapping 就是这么给的），而评审要对照的是**原文**。两者之间隔着一层
    规范化，若不显式记录映射，就只能靠猜测对齐——那正是这一问要避免的。

    返回 (processed, src_index)：
        processed[j]     规范化后第 j 个字符
        src_index[j]     它在原文中的下标；替换产生的字符（如 "—" → " - " 里的空格）
                         归因到被替换的那个原字符上

    实现末尾用断言把 processed 与 preprocess_text(text) 对齐，一旦两者出现分歧
    立即报错而不是静默产出错误的字符区间。
    """
    if not isinstance(text, str):
        return "", []

    # 第一步：字符替换，逐字符记录来源
    buf_chars: List[str] = []
    buf_src: List[int] = []
    for i, ch in enumerate(text):
        rep = _CHAR_REPLACEMENTS.get(ch)
        if rep is None:
            buf_chars.append(ch)
            buf_src.append(i)
        else:
            for rc in rep:
                buf_chars.append(rc)
                buf_src.append(i)      # 替换串的每个字符都归因到原字符 i

    # 第二步：空白折叠（等价 re.sub(r"\s+", " ", ...)）
    col_chars: List[str] = []
    col_src: List[int] = []
    prev_space = False
    for ch, s in zip(buf_chars, buf_src):
        if ch.isspace():
            if prev_space:
                continue               # 连续空白只保留第一个
            col_chars.append(" ")
            col_src.append(s)
            prev_space = True
        else:
            col_chars.append(ch)
            col_src.append(s)
            prev_space = False

    # 第三步：strip（等价 str.strip()）
    lo, hi = 0, len(col_chars)
    while lo < hi and col_chars[lo] == " ":
        lo += 1
    while hi > lo and col_chars[hi - 1] == " ":
        hi -= 1

    processed = "".join(col_chars[lo:hi])
    src_index = col_src[lo:hi]

    ref = preprocess_text(text)
    if processed != ref:
        raise AssertionError(
            "preprocess_text_with_map 与 preprocess_text 口径不一致，"
            f"将产出错误的字符区间：地图版 {processed[:60]!r} vs 基准版 {ref[:60]!r}")
    return processed, src_index


class UnalignedTextExtractor:
    """
    RoBERTa 词级编码器（不对齐）。

    encode_words(text) -> (word_units (W,D), words List[str], meta)
    """

    def __init__(self, model_name: Optional[str] = None,
                 device: Optional[str] = None,
                 max_length: Optional[int] = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.model_name = model_name or config.TEXT_MODEL_NAME
        self.max_length = int(max_length or config.TEXT_MAX_LENGTH)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        self.feature_dim = int(self.model.config.hidden_size)
        self._special_ids = set(int(i) for i in (self.tokenizer.all_special_ids or []))
        LOGGER.info("[文本] 模型就绪 %s（hidden=%d, device=%s, max_length=%d）",
                    self.model_name, self.feature_dim, self.device, self.max_length)

    @property
    def torch(self):
        return self._torch

    def encode_words(self, text: str) -> Tuple[np.ndarray, List[str], List[Tuple[int, int]],
                                               List[Tuple[int, int]], List[List[int]],
                                               Dict[str, object]]:
        """
        文本 → 词级向量序列（**不分配时间槽**）。

        RoBERTa 用 GPT-2 式 BPE：一个词可能被切成多个子词，tokenizer.word_ids()
        给出每个子词所属的词序号（特殊符号为 None）。词向量取该词全部子词隐状态的均值，
        与 config.TEXT_ALIGN_STRATEGY="word_interpolate" 的口径保持一致，
        这样「未对齐 → 对齐」与既有的「直接对齐」两条路径可比。

        除特征外，还产出**逐词溯源**三件套（问题1 交付要求的「原词、字符范围、词元索引」）：
            char_spans          每词在**规范化文本**中的字符区间 (start, end)
            char_spans_original 同一个区间换算到**原文**的下标
            token_index         每词占用的 input_ids 下标列表

        返回 (word_units (W,D) float32, words List[str], char_spans, char_spans_original,
              token_index, meta)
        """
        processed, src_index = preprocess_text_with_map(text)
        if not processed:
            return (np.zeros((0, self.feature_dim), np.float32), [], [], [], [],
                    {"num_words": 0, "num_subwords": 0, "truncated": False,
                     "text_processed": "", "empty_text": True})

        # return_offsets_mapping 只有 fast tokenizer 支持，给出每个词元在 processed 中的
        # 字符区间。注意 offset_mapping 不能喂给模型，故下面 forward 显式只传两参数。
        enc = self.tokenizer(processed, return_tensors="pt", truncation=True,
                             max_length=self.max_length, add_special_tokens=True,
                             return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0].tolist()
        ids = enc["input_ids"][0].tolist()
        with self.torch.no_grad():
            hidden = self.model(input_ids=enc["input_ids"].to(self.device),
                                attention_mask=enc["attention_mask"].to(self.device)
                                ).last_hidden_state[0].cpu().numpy().astype(np.float32)

        word_ids = enc.word_ids(0)
        wid = np.array([-1 if w is None else int(w) for w in word_ids], dtype=np.int64)
        num_words = int(wid.max()) + 1 if (wid >= 0).any() else 0

        if num_words > 0:
            acc = np.zeros((num_words, self.feature_dim), dtype=np.float64)
            np.add.at(acc, wid[wid >= 0], hidden[wid[wid >= 0]])
            cnt = np.bincount(wid[wid >= 0], minlength=num_words)
            word_units = (acc / np.maximum(cnt, 1)[:, None]).astype(np.float32)
        else:
            word_units = np.zeros((0, self.feature_dim), np.float32)

        # 还原每个词的字符串：把该词的子词 id 交给 tokenizer.decode，
        # 由它处理字节级 BPE 的 Ġ 前缀与转义，比手工剥前缀可靠。
        # 同一趟循环里顺带收集溯源三件套，保证三者与 words 严格同序同长。
        words: List[str] = []
        char_spans: List[Tuple[int, int]] = []
        char_spans_original: List[Tuple[int, int]] = []
        token_index: List[List[int]] = []
        id_list = np.array(ids, dtype=np.int64)
        for w in range(num_words):
            pos = np.flatnonzero(wid == w)
            sub_ids = id_list[pos]
            words.append(self.tokenizer.decode(sub_ids.tolist(),
                                               skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False).strip())
            token_index.append([int(p) for p in pos])
            # 词的字符区间 = 首个子词起点到末个子词终点（BPE 前缀 Ġ 已由 offset 计入）
            start, end = int(offsets[pos[0]][0]), int(offsets[pos[-1]][1])
            char_spans.append((start, end))
            # 换算回原文：src_index 单调不减，故取首字符与末字符的来源下标即可
            if src_index:
                o_start = int(src_index[min(start, len(src_index) - 1)])
                o_end = int(src_index[min(max(end - 1, 0), len(src_index) - 1)]) + 1
            else:
                o_start = o_end = 0
            char_spans_original.append((o_start, o_end))

        meta = {
            "num_words": num_words,
            "num_subwords": int(len(ids)),
            "num_content_tokens": int(len([i for i, t in enumerate(ids) if t not in self._special_ids])),
            "truncated": bool(len(ids) >= self.max_length),
            "text_processed": processed,
            "empty_text": False,
            "avg_subwords_per_word": round(float(len(ids) / num_words), 3) if num_words else 0.0,
        }
        return word_units, words, char_spans, char_spans_original, token_index, meta


# ==================== 单样本抽取 ====================

# 无真实词时间戳时的均匀假设说明，随每个文件落盘，避免后续误用
TIME_BASIS = "uniform_assumption"
TIME_BASIS_DESC = (
    "文本不含逐词时间戳，pts 为均匀分布假设：词 i 中心时刻 = (i+0.5) * duration / W。"
    "该时间轴仅用于与音频/视觉共轴可视化，不是实测值。"
)


def extract_one(extractor: UnalignedTextExtractor, sample: Dict[str, object],
                duration: float) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, object],
                                          Dict[str, object]]:
    """
    对一条样本抽文本特征。

    duration: 该片段的有效时长（秒），由 ffprobe 给出，用于构造均匀假设时间轴。
    返回 (features (W,D), pts (W,), words, provenance, meta)

    provenance 承载逐词溯源三件套，随 extra 一并落盘，供问题1 交付的
    alignment_q1.json 直接读取（原词、字符范围、词元索引）。
    """
    original_text = str(sample.get("text", ""))
    word_units, words, char_spans, char_spans_original, token_index, enc_meta = \
        extractor.encode_words(original_text)
    W = word_units.shape[0]
    if W > 0 and duration > 0:
        pts = ((np.arange(W, dtype=np.float64) + 0.5) * duration / W).astype(np.float32)
    else:
        pts = np.zeros((W,), np.float32)

    meta: Dict[str, object] = {
        "modality": MODALITY,
        "model": extractor.model_name,
        "feature_dim": extractor.feature_dim,
        "unit_level": "word",
        "pooling_rule": "词向量 = 该词全部子词最后一层隐状态的均值",
        "alignment_rule": "无（本文件不做时间槽分配，对齐见 align_multimodal.py）",
        "time_basis": TIME_BASIS,
        "time_basis_desc": TIME_BASIS_DESC,
        "duration_used": round(float(duration), 4),
        "num_units": int(W),
        "pipeline_version": config.PIPELINE_VERSION,
        "code_fingerprint": U.module_fingerprint(UnalignedTextExtractor.encode_words),
        # 溯源字段的坐标口径必须写明，否则下游会误把 processed 区间当原文区间
        "char_span_basis": {
            "char_spans": "词在 meta.text_processed 中的字符区间 [start, end)",
            "char_spans_original": "同一区间换算到 label-100 原文的下标（半开区间）",
            "token_index": "该词占用的 input_ids 下标列表（含 RoBERTa 前导空格 BPE 前缀）",
        },
        "text_original": original_text,
        **enc_meta,
    }
    provenance = {
        "char_spans": [list(s) for s in char_spans],
        "char_spans_original": [list(s) for s in char_spans_original],
        "token_index": [list(t) for t in token_index],
    }
    return word_units, pts, words, provenance, meta


def run(samples: List[Dict[str, object]], out_dir: str, overwrite: bool = False,
        max_length: Optional[int] = None, device: Optional[str] = None) -> List[Dict[str, object]]:
    """批量抽取文本未对齐特征，逐样本落盘。返回汇总行列表。"""
    os.makedirs(out_dir, exist_ok=True)
    extractor = UnalignedTextExtractor(max_length=max_length, device=device)

    rows: List[Dict[str, object]] = []
    n_ok = n_skip = n_fail = 0
    for i, s in enumerate(samples, 1):
        sid = str(s["sample_id"])
        out_path = os.path.join(out_dir, f"{sid}.npz")
        if os.path.exists(out_path) and not overwrite:
            n_skip += 1
            LOGGER.info("[文本] (%d/%d) 已存在，跳过 %s", i, len(samples), sid)
            continue

        video_path = str(s["video_path"])
        if not os.path.isfile(video_path):
            LOGGER.warning("[文本] (%d/%d) 找不到视频，跳过 %s", i, len(samples), video_path)
            n_fail += 1
            continue

        try:
            duration = U.video_true_duration(video_path)
            feats, pts, words, prov, meta = extract_one(extractor, s, duration)
            U.save_unaligned(out_path, feats, pts, MODALITY, meta,
                             extra={"words": words, **prov})
            n_ok += 1
            LOGGER.info("[文本] (%d/%d) %s -> (W=%d, D=%d) 时长=%.2fs",
                        i, len(samples), sid, feats.shape[0], feats.shape[1], duration)
            rows.append({
                "sample_id": sid,
                "official_id": s["official_id"],
                "modality": MODALITY,
                "n_units": int(feats.shape[0]),
                "feature_dim": int(feats.shape[1]),
                "duration_sec": round(duration, 3),
                "time_basis": TIME_BASIS,
                "out_path": out_path,
                "status": "ok",
            })
        except Exception as exc:  # 单条失败不阻断整批
            n_fail += 1
            LOGGER.error("[文本] (%d/%d) 抽取失败 %s: %s: %s",
                         i, len(samples), sid, type(exc).__name__, exc)

    LOGGER.info("[文本] 完成：成功 %d / 跳过 %d / 失败 %d（输出目录 %s）",
                n_ok, n_skip, n_fail, out_dir)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="文本模态未对齐特征提取（RoBERTa 词级，不做时间对齐）")
    p.add_argument("--only", default=None, help="只处理 sample_id 含该子串的样本（调试用）")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 条")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出")
    p.add_argument("--out", default=None, help="输出目录，默认 data/unaligned_features/text")
    p.add_argument("--max-length", type=int, default=None, help="最大子词长度，默认取 config.TEXT_MAX_LENGTH")
    args = p.parse_args(argv)

    samples = U.list_samples(only=args.only, limit=args.limit)
    LOGGER.info("[文本] 待处理样本 %d 条", len(samples))
    out_dir = args.out or U.unaligned_dir(MODALITY)
    run(samples, out_dir, overwrite=args.overwrite, max_length=args.max_length)
    return 0


if __name__ == "__main__":
    sys.exit(main())
