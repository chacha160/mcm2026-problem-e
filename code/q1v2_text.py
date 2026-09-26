# -*- coding: utf-8 -*-
"""
q1v2_text.py —— 词单元规则与 RoBERTa 词级池化（问题一 v2）

两层文本单元，必须先分清，后面的核验全建立在这上面：

    whitespace_chunks(text)  ——  `\\S+` 的全部空白块。本批 100 条合计 **1932** 个。
    source_words(text)       ——  纯标点块并入相邻词后的**词单元**。合计 **1926** 个。

两者相差 6 个：某些官方文本里存在空格分隔的独立标点（如 `" ... word . "`），
它不构成新词，而是把前一个词的字符区间向右扩张。**词单元数 1926 是硬事实**，
已用对照实现的 `audit_100.csv` 独立复算：`sum(alignment_official_word_count) == 1926`。

本模块的两条规则**逐字符照搬**对照实现的 `q1_feature_core.py`（L24-66, L275-328），
不做任何"改进"——口径一变，1926 就复现不出来。

本模块 import 时不加载 torch / transformers（延迟到 `load_roberta` 内），
以便 `q1v2_verify.py` 在裸解释器里复算词单元规则（核验 V3）。
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from q1v2_contract import (
    ROBERTA_FILE_SHA256,
    ROBERTA_HIDDEN_DIM,
    ROBERTA_MAX_POSITIONS_FALLBACK,
    ROBERTA_MODEL_ID,
    ROBERTA_REVISION,
    HardStop,
    sha256,
)

#: 本批 100 条的期望值。写成常量供核验器引用，避免在核验器里重复硬编码。
#: 1932 是 `\S+` 空白块数（纯文本性质，与本套代码自身一致）；
#: 1926 是词单元数，已用对照实现的 audit 独立复算：`sum(alignment_official_word_count)==1926`。
EXPECTED_WHITESPACE_CHUNKS = 1932
EXPECTED_WORD_UNITS = 1926


# --------------------------------------------------------------------------
# 词单元规则（逐字符照搬对照实现的 core）
# --------------------------------------------------------------------------


def has_letter_or_digit(text: str) -> bool:
    """块里是否含字母或数字。

    两重判据缺一不可：`str.isalnum()` 覆盖拉丁与多数文字，
    `unicodedata.category[0] in {"L","N"}` 兜住 `isalnum()` 漏掉的字类
    （例如带组合附加符的字符）。只留任一条都会在个别样本上改变 1926 这个数。
    """
    return any(ch.isalnum() or unicodedata.category(ch)[0] in {"L", "N"} for ch in text)


def whitespace_chunks(text: str) -> List[Dict[str, Any]]:
    """全部 `\\S+` 空白块，保留原文字符区间。本批合计 1932。"""
    return [
        {"char_start": m.start(), "char_end": m.end(), "text": m.group(0)}
        for m in re.finditer(r"\S+", text, flags=re.UNICODE)
    ]


def source_words(text: str) -> List[Dict[str, Any]]:
    """词单元：含字母/数字的空白块，纯标点块并入相邻词。

    - 纯标点块出现在已有词之后 → 把 `words[-1]` 的 `char_end` 向右扩张到它末尾；
    - 纯标点块出现在第一个词之前 → 记入 `pending_prefix`，由第一个真词吸收；
    - 结尾只剩标点且尚无任何词 → 抛错（整条文本没有可用词）。

    这条规则给出 1926 这个数，与对照实现的 audit 逐样本吻合，**不要改动**。
    """
    words: List[Dict[str, Any]] = []
    pending_prefix: Optional[int] = None
    for match in re.finditer(r"\S+", text, flags=re.UNICODE):
        chunk = match.group(0)
        if has_letter_or_digit(chunk):
            start = pending_prefix if pending_prefix is not None else match.start()
            words.append({
                "char_start": start,
                "char_end": match.end(),
                "text": text[start:match.end()],
            })
            pending_prefix = None
        elif words:
            words[-1]["char_end"] = match.end()
            words[-1]["text"] = text[words[-1]["char_start"]:match.end()]
        else:
            pending_prefix = match.start()
    if pending_prefix is not None:
        raise HardStop("官方文本以纯标点开头且其后无任何词")
    if not words:
        raise HardStop("官方文本中没有任何可用词单元")
    return words


def chunk_to_word_index(chunks: Sequence[Dict[str, Any]],
                        words: Sequence[Dict[str, Any]]) -> np.ndarray:
    """每个空白块归属到哪个词单元。长度 1932 → 值域 `[0, 1926)`。

    由于 `source_words` 把标点并进相邻词，**每个空白块的字符区间都恰好被
    一个词单元的区间包含**，所以这是个定义良好的多对一映射：
    含字母的块自成映射，纯标点块映射到它并入的那个词。

    只按字符区间判定，**不看时间**——这正是 V3 能在裸解释器里独立复算的原因。
    """
    out = np.empty(len(chunks), dtype=np.int32)
    wi = 0
    last = len(words) - 1
    for ci, chunk in enumerate(chunks):
        start = chunk["char_start"]
        while wi < last and words[wi]["char_end"] <= start:
            wi += 1
        out[ci] = wi
    return out


# --------------------------------------------------------------------------
# RoBERTa 词级池化
# --------------------------------------------------------------------------


def asset_identity() -> Dict[str, Any]:
    """把模型身份压成一个可直接比对的 dict（写进 NPZ 与 environment.json）。"""
    return {
        "model_id": ROBERTA_MODEL_ID,
        "revision": ROBERTA_REVISION,
        "hidden_dim": ROBERTA_HIDDEN_DIM,
    }


def load_roberta(model_dir: Optional[Path] = None,
                 local_files_only: bool = True) -> Tuple[Any, Any, str]:
    """加载 RoBERTa 分词器与模型，返回 `(tokenizer, model, asset_identity_sha256)`。

    **强制 `local_files_only=True`**：赛题要求运行期零网络。若本地缓存缺失，
    宁可在这里失败，也不能悄悄联网下载一个 revision 不同的权重——
    那会让 768 维文本特征与冻结身份脱钩，而落盘的 revision 字符串却还是旧的。
    """
    import torch  # 延迟导入：核验器不需要 torch
    from transformers import AutoModel, AutoTokenizer

    source = str(model_dir) if model_dir else ROBERTA_MODEL_ID
    tokenizer = AutoTokenizer.from_pretrained(
        source, revision=None if model_dir else ROBERTA_REVISION,
        local_files_only=local_files_only, add_prefix_space=False,
    )
    model = AutoModel.from_pretrained(
        source, revision=None if model_dir else ROBERTA_REVISION,
        local_files_only=local_files_only,
        # 词级池化只用 last_hidden_state，pooler 从不参与计算。而该检查点里
        # 没有 pooler 权重，默认构造会**随机初始化**它——那会让进程里存在一组
        # 随机参数，虽然是死代码，却让"这个模型加载后是确定的"这句话不再显然。
        # 干脆不构造它：加载后不含任何随机初始化的参数。
        add_pooling_layer=False,
    )
    model.eval()
    identity_digest = json.dumps(asset_identity(), sort_keys=True)
    import hashlib
    return tokenizer, model, hashlib.sha256(identity_digest.encode("utf-8")).hexdigest()


def extract_text_features(text: str,
                          words: Sequence[Dict[str, Any]],
                          tokenizer: Any,
                          model: Any,
                          device: str = "cpu") -> Dict[str, Any]:
    """官方文本 → `(L, 768)` 词级特征，L = 词单元数。

    池化方式是**字符区间内的子词平均**：每个子词 token 按 `offset_mapping` 落到
    唯一一个词单元上，该词的特征即落在它身上的全部子词隐状态的均值。

    三个必须保住的细节：

    1. `truncation=False` 且**超长即报错**。静默截断会让尾部的词一个 token 都
       分不到，`text_word_valid` 却仍是 1——数据看起来正常，语义已经错了。
    2. `add_special_tokens=True` 但 `<s>/</s>` 的 offset 是 `(0,0)`，落不到任何词上，
       会被下面的"0 匹配 + 非空白"分支单独处理。
    3. 匹配不到任何词的**纯空白 token 直接跳过**（RoBERTa 在词首会切出
       `Ġ` 之类的空白片段），不算分词失败；非空白的落空才算失败并记入
       `tokenizer_failures`。这条区分决定了 `tokenizer_failures` 在 100 条上是否为 0。
    """
    import torch

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=False,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"][0].tolist()
    token_count = int(input_ids.shape[1])

    limit = int(getattr(model.config, "max_position_embeddings",
                        ROBERTA_MAX_POSITIONS_FALLBACK))
    if token_count > limit:
        raise HardStop(
            f"官方文本分词后 {token_count} 个 token，超过位置编码上限 {limit}；"
            "不静默截断，按样本级错误处理"
        )

    with torch.no_grad():
        hidden = model(input_ids=input_ids.to(device)).last_hidden_state[0]
    hidden = hidden.float().cpu().numpy()

    spans = [(int(w["char_start"]), int(w["char_end"])) for w in words]
    word_index_of_token = np.full(token_count, -1, dtype=np.int32)
    token_failures: List[Dict[str, Any]] = []
    # words 与 chunks 一样是有序不重叠的，用双指针把 O(T*L) 降到 O(T+L)
    wi = 0
    last = len(spans) - 1
    for ti, (a, b) in enumerate(offsets):
        a, b = int(a), int(b)
        while wi < last and spans[wi][1] <= a:
            wi += 1
        if a == b:
            # 零宽 offset：`<s>` / `</s>` 等特殊 token
            continue
        if spans[wi][0] <= a and b <= spans[wi][1]:
            word_index_of_token[ti] = wi
        else:
            raw = tokenizer.convert_ids_to_tokens(int(input_ids[0, ti]))
            if str(raw).strip() == "":
                continue  # 纯空白片段，不归属任何词是正常的
            token_failures.append({"token_index": ti, "token": str(raw),
                                   "char_start": a, "char_end": b})

    features = np.zeros((len(words), ROBERTA_HIDDEN_DIM), dtype=np.float32)
    valid = np.zeros(len(words), dtype=np.uint8)
    token_counts = np.zeros(len(words), dtype=np.int32)
    word_token_indices: List[int] = []
    word_token_indptr: List[int] = [0]
    for wi_ in range(len(words)):
        picked = np.flatnonzero(word_index_of_token == wi_)
        word_token_indices.extend(int(i) for i in picked)
        word_token_indptr.append(len(word_token_indices))
        if len(picked) == 0:
            continue
        features[wi_] = hidden[picked].mean(axis=0).astype(np.float32)
        valid[wi_] = 1
        token_counts[wi_] = len(picked)

    return {
        "features": features,
        "valid": valid,
        "token_counts": token_counts,
        "token_ids": input_ids[0].numpy().astype(np.int64),
        "token_offsets": np.asarray(offsets, dtype=np.int32).reshape(-1, 2),
        "token_word_index": word_index_of_token,
        "word_token_indptr": np.asarray(word_token_indptr, dtype=np.int32),
        "word_token_indices": np.asarray(word_token_indices, dtype=np.int32),
        "tokenizer_failures": token_failures,
        "input_length": token_count,
        "hidden_dim": ROBERTA_HIDDEN_DIM,
    }


def model_file_shas(model_dir: Path) -> Dict[str, str]:
    """逐个文件算 SHA-256，用于核验 V7(a)：权重文件确实是冻结的那一份。"""
    out: Dict[str, str] = {}
    for name in sorted(ROBERTA_MODEL_SHAS):
        p = Path(model_dir) / name
        out[name] = sha256(p) if p.is_file() else "missing"
    return out


#: 与 contract 里的 ROBERTA_FILE_SHA256 同源，单独取一份名字避免循环导入措辞混乱。
from q1v2_contract import ROBERTA_FILE_SHA256 as ROBERTA_MODEL_SHAS  # noqa: E402


__all__ = [
    "HardStop", "EXPECTED_WHITESPACE_CHUNKS", "EXPECTED_WORD_UNITS",
    "has_letter_or_digit", "whitespace_chunks", "source_words",
    "chunk_to_word_index", "asset_identity", "load_roberta",
    "extract_text_features", "model_file_shas",
]
