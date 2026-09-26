# -*- coding: utf-8 -*-
"""q1v2_route.py —— 五类路由与三处声明（问题一 v2）

本模块是**纯函数**：只吃标量、只吐标量与字符串，不碰文件、不加载模型。
这样做的唯一目的是让核验器（V11）能从成品 NPZ 里把标量读回来、
独立重跑一遍路由，逐字段比对——路由结论因此是**可复算的**，不是一次性的判断。

## 与对照实现的关键分歧：把一个合取判据拆成两问

对照实现的 `audit_rules.md` 第 4 条是一个**合取**：

> 人工确认文本与讲话基本对应，**且** 冻结 stable-ts 强制对齐后全部官方词都有合法区间

两个合取项回答的是**不同问题**，证据等级也不同：

| 合取项 | 问题 | 本套代码能否机器判定 |
|---|---|---|
| 全词区间合法 | 官方词的**时间**够不够用？ | **能** —— 对齐器输出可直接检验 |
| 人工确认内容对应 | 官方文本的**内容**对不对得上讲话？ | **不能** —— 需人工听辨 |

对照实现把它们捆进同一个 `alignment_mode`，于是 `TRI_MODAL_WORD_VALID` 隐含了内容确认。
本套代码**拆开**：

* `alignment_mode` —— 只由机器可证的判据决定，因此**永不发出**那两类需要人工证据的取值；
* `text_av_time_mapping_status` —— 纯结构闸门，机器可判；
* `text_audio_correspondence` —— 内容确认，一律 `not_asserted`，
  **唯一例外**是量化后逐采样全零的数字静音（位精确的机器事实，填 `no_speech`）；
* `content_assertion` —— `1` 仅当存在外部人工证据（本套代码**恒为 0**）；
* `word_time_basis` —— 区间来源，`own_forced_alignment` 当且仅当本样本有合法词区间。

**这不比对照实现宽松，反而更保守也更有据**：对照实现的 `word_valid` 隐含内容确认，
我们的结构合法样本照样产出词级特征，但用 `content_assertion=0` 明说
「这些区间来自冻结对齐器，我们不声称官方文本与讲话内容对应」。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from q1v2_contract import (
    CONTENT_ASSERTION_EXTERNAL,
    CONTENT_ASSERTION_NONE,
    CORRESPONDENCE_VOCABULARY,
    MODE_AV_VALID_TEXT_UNALIGNED,
    MODE_EXTRACTION_OR_TIMELINE_ANOMALY,
    MODE_TRI_MODAL_AUDIO_CONTENT_INVALID,
    MODE_TRI_MODAL_WORD_VALID,
    MODE_UNCERTAIN_REVIEW,
    TIME_STATUS_VOCABULARY,
    WORD_TIME_BASIS_NONE,
    WORD_TIME_BASIS_OWN,
    HardStop,
)

#: 外部证据（人工听辨）的合法取值。本套代码默认不收，仅 S6 开发期回归用。
EXTERNAL_ASSERTIONS = ("confirmed_match", "confirmed_mismatch")


def load_correspondence_evidence(path: Optional[Any] = None) -> Dict[str, Dict[str, str]]:
    """读外部人工听辨证据（可选，默认不用）。

    返回 `{sample_key: {"text_audio_correspondence": ..., "source": ...}}`。

    **默认路径不调用本函数。** 对照实现的人工证据文件
    （`human_review.json`、两份 `听辨报告.md`）在编写本套代码的机器上并不存在，
    而 对照实现的机器筛查也**无法反推**它们：18 条 `HIGH_DISAGREEMENT` 里只有 5 条被判
    `confirmed_mismatch`，而 `MODERATE` 里反而有 1 条 `confirmed_match`（WER 0.40）。
    任何 WER 阈值都复原不出对照实现的 5/88/2/5。
    故本函数只在显式传入路径时工作，用于 S6 的一次性回归对拍，其结果**不进交付物**。
    """
    if path is None:
        return {}
    import json
    from pathlib import Path

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, str]] = {}
    if isinstance(payload, dict) and "sample_key" in payload:
        payload = [payload]
    if isinstance(payload, dict):
        rows = [{"sample_key": k, **(v if isinstance(v, dict) else {"value": v})}
                for k, v in payload.items()]
    else:
        rows = list(payload)
    for row in rows:
        key = str(row.get("sample_key") or row.get("official_id") or "").strip()
        value = str(row.get("text_audio_correspondence")
                    or row.get("correspondence") or "").strip()
        if not key or value not in EXTERNAL_ASSERTIONS:
            continue
        out[key] = {
            "text_audio_correspondence": value,
            "source": str(row.get("human_evidence_source") or row.get("source") or path),
        }
    return out


def route_sample(timeline_ok: bool,
                 audio_present: bool,
                 audio_digital_silence: bool,
                 mapping_status: str,
                 n_official_words: int,
                 n_structurally_valid_words: int,
                 external: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """纯函数路由。输入全是标量，输出全是标量与短字符串。

    判定顺序（先证伪、后兜底）：

    1. `timeline_ok == False` → `EXTRACTION_OR_TIMELINE_ANOMALY`
       （身份/解码/共享 A/V 时间轴不可靠，后续一切时间语义都不成立）
    2. 音轨存在且**量化后逐采样全零** → `TRI_MODAL_WITH_AUDIO_CONTENT_INVALID`
       （时间轴可信，但音频内容被机器证伪）
    3. 其余全部 → `UNCERTAIN_REVIEW`（机器可判的部分到此为止）

    第 2 条是 `no_speech` 的唯一来源，也是 `audio_speech_valid=0` 的唯一依据。
    **本函数从不返回 `audio_speech_valid=1`**：那需要强语音证据，我们没有。

    `external` 非空时（仅 S6 开发期回归）才可能升为
    `TRI_MODAL_WORD_VALID` / `AV_VALID_TEXT_UNALIGNED`，并置 `content_assertion=1`，
    同时把证据来源记进 `reasons`。默认路径下这两类**永不发出**。
    """
    if mapping_status not in TIME_STATUS_VOCABULARY:
        raise HardStop(f"未知的 text_av_time_mapping_status：{mapping_status}")

    reasons: List[str] = []
    external = external or {}
    external_value = external.get("text_audio_correspondence")
    external_source = external.get("source")

    # 区间来源：有合法词区间就是自己跑出来的强制对齐，否则没有词时间。
    word_time_basis = (WORD_TIME_BASIS_OWN if n_structurally_valid_words > 0
                       else WORD_TIME_BASIS_NONE)

    # 粒度描述的是**实际产出**，与内容结论无关：有合法词区间就出词级数组。
    granularity = "word" if n_structurally_valid_words > 0 else "clip"

    # ---- 内容对应关系：默认不主张 -------------------------------------
    if external_value in EXTERNAL_ASSERTIONS:
        correspondence = external_value
        content_assertion = CONTENT_ASSERTION_EXTERNAL
        reasons.append(f"外部人工证据：{external_source}")
    elif audio_present and audio_digital_silence:
        correspondence = "no_speech"
        content_assertion = CONTENT_ASSERTION_NONE
        reasons.append("量化后 int16 逐采样全零（位精确机器事实）")
    else:
        correspondence = "not_asserted"
        content_assertion = CONTENT_ASSERTION_NONE
        reasons.append("无人工听辨证据，内容对应关系不作主张")

    # ---- 五类模式 ------------------------------------------------------
    if not timeline_ok:
        mode = MODE_EXTRACTION_OR_TIMELINE_ANOMALY
        reasons.append("身份/解码/共享 A/V 时间轴不可靠")
    elif external_value == "confirmed_match" and n_structurally_valid_words == n_official_words \
            and n_official_words > 0:
        mode = MODE_TRI_MODAL_WORD_VALID
        reasons.append("外部确认内容对应 且 全部官方词区间合法")
    elif external_value == "confirmed_mismatch":
        mode = MODE_AV_VALID_TEXT_UNALIGNED
        reasons.append("外部确认官方文字与实际讲话明显不对应")
    elif audio_present and audio_digital_silence:
        mode = MODE_TRI_MODAL_AUDIO_CONTENT_INVALID
    else:
        mode = MODE_UNCERTAIN_REVIEW
        reasons.append("其余情况：机器可判据到此为止，按 对照实现的原规则归入此类")

    # ---- 时间映射状态 --------------------------------------------------
    time_status = mapping_status
    if external_value == "confirmed_mismatch":
        # 与对照实现一致：内容既然已被外部证据证伪，文本的"时间映射"也就无从谈起。
        # 注意这是**内容证据驱动的赋值**，不是结构闸门的结论，故单独注明。
        time_status = "unavailable"
        reasons.append("时间映射状态置 unavailable 由外部内容证据驱动，非结构闸门结论")

    # ---- 音频语音状态：三态，且从不写 1 ---------------------------------
    if not audio_present:
        audio_speech_valid = 0
        audio_speech_evidence = "no audio stream in container"
    elif audio_digital_silence:
        audio_speech_valid = 0
        audio_speech_evidence = "int16-quantized waveform is all-zero at every sample"
    else:
        audio_speech_valid = -1
        audio_speech_evidence = (
            "no machine-provable speech evidence collected; writing -1 (not asserted) "
            "rather than 1 (speech confirmed)"
        )

    return {
        "alignment_mode": mode,
        "alignment_granularity": granularity,
        "text_audio_correspondence": correspondence,
        "text_av_time_mapping_status": time_status,
        "content_assertion": int(content_assertion),
        "word_time_basis": word_time_basis,
        "audio_speech_valid": int(audio_speech_valid),
        "audio_speech_evidence": audio_speech_evidence,
        "n_official_words": int(n_official_words),
        "n_structurally_valid_words": int(n_structurally_valid_words),
        "routing_reasons": reasons,
        "external_evidence_source": external_source if external_value else None,
    }


def route_policy_document() -> Dict[str, Any]:
    """路由政策说明书，落 `metadata/route_policy.json`。

    **必须写明两类"永不发出"的原因**，否则读者会把「证据等级不足」误读成「漏跑」。
    """
    from q1v2_contract import MODES_NEVER_EMITTED, MODE_VOCABULARY

    return {
        "vocabulary": list(MODE_VOCABULARY),
        "emitted_by_this_pipeline": [
            MODE_EXTRACTION_OR_TIMELINE_ANOMALY,
            MODE_TRI_MODAL_AUDIO_CONTENT_INVALID,
            MODE_UNCERTAIN_REVIEW,
        ],
        "never_emitted": MODES_NEVER_EMITTED,
        "expected_distribution": {
            MODE_EXTRACTION_OR_TIMELINE_ANOMALY: 0,
            MODE_TRI_MODAL_AUDIO_CONTENT_INVALID: 2,
            MODE_UNCERTAIN_REVIEW: 98,
            MODE_TRI_MODAL_WORD_VALID: 0,
            MODE_AV_VALID_TEXT_UNALIGNED: 0,
        },
        "distribution_note": (
            "本套代码的期望分布是 TRI_MODAL_WORD_VALID 0 / AV_VALID_TEXT_UNALIGNED 0 / "
            "其余三类非零。那两类恒为 0 **不是漏跑**，而是证据等级不达标："
            "它们都要求「内容是否对应」的人工听辨结论，本套代码不伪造该结论，"
            "把这类样本保守地归入 UNCERTAIN_REVIEW。因此本表的计数只描述本套代码，"
            "不能当作类别占比的经验估计。"
        ),
        "separate_fields": {
            "text_av_time_mapping_status": (
                "机器可判的结构闸门：官方词的时间区间够不够用。"
                "取值 word_valid / word_partial / clip_only / unavailable。"
            ),
            "text_audio_correspondence": (
                "内容确认：官方文本与实际讲话是否对应。"
                "本套代码一律 not_asserted，唯一例外是位精确可证的数字静音（no_speech）。"
            ),
            "content_assertion": (
                "uint8。1 仅当存在外部人工证据，本套代码恒为 0。"
                "它的作用是让下游一眼看出「对齐区间存在」不等于「内容已确认」。"
            ),
            "word_time_basis": (
                "词区间来源。own_forced_alignment 表示本样本至少有一个词的区间"
                "来自本次冻结 stable-ts 运行；none 表示没有合法词区间。"
                "它只声明区间来源，不声明文本内容与讲话对应。"
            ),
        },
    }


__all__ = [
    "EXTERNAL_ASSERTIONS", "load_correspondence_evidence", "route_sample",
    "route_policy_document",
]
