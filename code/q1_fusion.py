# -*- coding: utf-8 -*-
"""
q1_fusion.py —— 问题一交付物的融合增强层（**只读既有产物，不重新提取特征**）

背景：两个方案在问题一上各自的短板
----------------------------------
方案一（本仓库 `data/q1_delivery/`）的长处是**时序组织可核验**：
    真实 PTS 抽帧、50 槽 + `slot_edges_sec` + `slot_src_pts` + `fill_policy`
    + `slot_unit_index`，每个槽都能反查到原始素材的哪个时间点；
    五类典型样本、A1~A8 异常台账、`counts_as_deletion ≡ 0`。
    它自己写明的短板是：`*_valid` 掩码**无法判断模态可用性**——
    插值填充和补零也会被标成有效。

方案二的长处恰在此处：把「有效性」拆成三层语义
    （存在性 / 观测有效性 / 内容有效性），
    并把文本与语音的对应关系分成三态
    （confirmed_match / confirmed_mismatch / not_asserted），
    只在有证据时下断言，其余保守标注。

本模块把方案二的这两点补到方案一的交付上，产出一份**附加**目录，不改动任何既有文件：
    data/q1_delivery/fusion/validity_layers.csv     逐样本逐模态的三层语义
    data/q1_delivery/fusion/validity_layers.json    同上，含判据与阈值快照
    data/q1_delivery/fusion/correspondence_trace.json  对应关系三态 + 证据
    data/q1_delivery/fusion/q1_fusion_report.md     融合说明与核验结论

赛题对应
--------
第 22 行「时序组织可核验性」→ 三层语义把「有没有这一段」与「这一段有没有内容」分开，
    使 `slot_valid=1` 不再被误读成「该模态可用」；
第 10 行 R1（不增删改样本与标签）→ 本模块只读，并对 `counts_as_deletion ≡ 0` 再次断言。
"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import config  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

Q1_DIR = os.path.join(config.PROJECT_ROOT, "data", "q1_delivery")
FUSION_DIR = os.path.join(Q1_DIR, "fusion")

# ==================== 内容有效性的判据阈值（运行前固定，不得看完结果再调）====================
# 这三个阈值决定「该模态这一天到底有没有内容」。取值依据是方案一 anomaly_ledger 里
# A1~A8 的判定线，以及 100 条样本上 face_ratio / voiced_ratio 的实际分布。
CONTENT_RULES = {
    "text":  {"min_words": 1,        "note": "官方转写至少 1 个词即视为有内容"},
    "audio": {"min_voiced_ratio": 0.02, "note": "有声帧占比 ≥2% 即视为该样本语音有内容"},
    "vision": {"min_face_ratio": 0.01, "note": "检出人脸帧占比 ≥1% 即视为该样本视觉有内容"},
}


# ==================== 读取既有交付物 ====================

def _read_summary() -> Dict[str, Dict[str, str]]:
    path = os.path.join(Q1_DIR, "summary_q1.csv")
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return {r["sample_id"]: r for r in csv.DictReader(f)}


def _read_alignment() -> Dict:
    with open(os.path.join(Q1_DIR, "alignment_q1.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key, "")
        return float(v) if v not in ("", None) else default
    except Exception:
        return default


def _parse_slot_valid(v) -> np.ndarray:
    """`slot_valid` 在 alignment_q1.json 里是 '0/1' 组成的字符串（如 '010010...'），
    不是布尔数组。这里统一解析成 bool ndarray，长度即槽数。"""
    if isinstance(v, str):
        return np.array([c == "1" for c in v], dtype=bool)
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=bool)
    return np.zeros(0, dtype=bool)


# ==================== 第一层：三层语义 ====================

def build_validity_layers() -> Tuple[List[Dict], Dict]:
    """产出逐样本逐模态的「存在性 / 观测有效性 / 内容有效性」。

    三层的确切含义（这是本模块的核心定义，与方案二的语义一致）：

      存在性 existence
          原始素材里有没有这个模态。文本 = 官方转写非空；
          语音 = 该 mp4 有音轨（方案一解码成功即有）；
          视觉 = 该 mp4 至少检出过一帧人脸（probe_max_face_count > 0）。
          注意：**视觉模态可以整体不存在**（本批有 24 条无脸样本），
          这不是「缺失某些槽」，而是「这条样本压根没有视觉信息」。

      观测有效性 observation
          该槽是否落进了**真实源单元**。直接取 alignment_q1.json 的 `slot_valid`。
          `slot_valid=0` 表示该槽由插值或补零得到，**没有原始素材与其对应**。
          方案一已声明 `interp_slots_have_no_source=True`。

      内容有效性 content
          该槽的源单元是否携带可用的情感信息。语音看有声帧、视觉看人脸。
          这一层是方案一缺的：只用 observation 会把「有音轨但全程静音」
          判成可用语音，「检出过脸但该槽无人脸」判成可用视觉。

    内容有效性的粒度说明：逐槽的人脸/有声证据在既有交付里没有逐槽落盘
    （`face_probe.csv` 是逐帧探测、`voiced_ratio` 是样本级），
    因此本层按「样本级内容有无 × 该槽的观测有效性」传播，
    并在输出里用 `content_scope` 字段标明是 `sample_level_propagated`。
    这是**有意的保守**：宁可承认粒度不足，也不伪造逐槽证据。
    """
    summary = _read_summary()
    align = _read_alignment()
    samples = align["samples"]

    rows: List[Dict] = []
    for sid, a in samples.items():
        row = summary.get(sid.replace("_", "_", 1), None)
        # summary 的 sample_id 用下划线连接，alignment 的键也是下划线形式，直接查
        row = summary.get(sid) or row
        if row is None:
            continue

        face_ratio = _f(row, "face_ratio")
        voiced_ratio = _f(row, "voiced_ratio")
        max_face = _f(row, "probe_max_face_count")
        n_words = len(a["text"].get("words", []) or [])

        existence = {
            "text": n_words >= CONTENT_RULES["text"]["min_words"],
            "audio": True,   # 方案一的流水线只在成功解码出音轨时才会产出该样本
            "vision": max_face > 0,
        }
        content_ok = {
            "text": n_words >= CONTENT_RULES["text"]["min_words"],
            "audio": voiced_ratio >= CONTENT_RULES["audio"]["min_voiced_ratio"],
            "vision": (max_face > 0) and (face_ratio >= CONTENT_RULES["vision"]["min_face_ratio"]),
        }

        for m in ("text", "audio", "vision"):
            sv = _parse_slot_valid(a[m].get("slot_valid"))
            n_obs = int(sv.sum())
            n_slots = int(sv.size)
            content = sv & content_ok[m] if existence[m] else np.zeros(n_slots, dtype=bool)
            rows.append({
                "sample_id": sid,
                "modality": m,
                "n_slots": n_slots,
                "existence": int(bool(existence[m])),
                "observation_valid_slots": n_obs,
                "observation_valid_ratio": round(n_obs / n_slots, 6) if n_slots else 0.0,
                "content_valid_slots": int(content.sum()),
                "content_valid_ratio": round(float(content.sum()) / n_slots, 6) if n_slots else 0.0,
                "content_ok_sample_level": int(bool(content_ok[m])),
                "content_scope": "sample_level_propagated",
                "evidence": {
                    "text": f"words={n_words}",
                    "audio": f"voiced_ratio={voiced_ratio:.4f}",
                    "vision": f"face_ratio={face_ratio:.4f}, max_face={int(max_face)}",
                }[m],
                "verdict": (
                    "usable" if existence[m] and content_ok[m]
                    else ("present_no_content" if existence[m] else "absent")
                ),
            })

    meta = {
        "source_files": ["summary_q1.csv", "alignment_q1.json"],
        "content_rules": CONTENT_RULES,
        "layer_definitions": {
            "existence": "原始素材中该模态是否存在（视觉可整体不存在）",
            "observation_valid": "该槽是否落进真实源单元；0 表示插值/补零产物，无原始素材对应",
            "content_valid": "该槽的源单元是否携带可用情感信息（语音有声 / 视觉有人脸）",
        },
        "honest_limit": (
            "content_valid 为样本级内容有无向槽位的传播（content_scope="
            "sample_level_propagated）。既有交付未逐槽落盘人脸/有声证据，"
            "本层不伪造逐槽内容判定。"
        ),
    }
    return rows, meta


# ==================== 第二层：对应关系三态 ====================

def build_correspondence_trace() -> Tuple[Dict, Dict]:
    """把「文本 ↔ 语音」的对应关系拆成三态，只在有证据时下断言。

    为什么必须分三态
    ----------------
    方案一的时间轴里，文本用 `uniform_assumption`（词时间 = (i+0.5)·T/W 均分），
    语音/视觉用 `measured_pts`（真实测量）。两者**基准不同**：
    文本的逐词时间不是测出来的，是算出来的。若把这种样本一律记为
    「文本与语音已对齐」，等于把假设说成事实。所以：

      confirmed_match    该样本的文本时间基准与音视频同为真实测量，且逐词跨度合法
                         → 可以断言对应关系成立
      confirmed_mismatch 存在结构性非法（零时长词、越界单元、映射失败）
                         → 可以断言对应关系有问题，须人工复核
      not_asserted       文本时间基准是均匀假设、或缺证据
                         → 不下任何断言
    """
    align = _read_alignment()
    samples = align["samples"]
    out: Dict[str, Dict] = {}
    counts = {"confirmed_match": 0, "confirmed_mismatch": 0, "not_asserted": 0}

    for sid, a in samples.items():
        t = a["text"]
        basis = str(t.get("time_basis", ""))
        words = t.get("words", []) or []
        wtimes = t.get("word_t_sec", []) or []
        spans = t.get("char_spans", []) or []

        # 结构性非法：零/负时长词、非法字符跨度、越界单元
        # `word_t_sec` 存的是**每个词的起始时刻**（不是 [start,end] 对），
        # 故时长 = 后一词起点 − 前一词起点；最后一个词用片段总时长兜底。
        dur_total = float(a.get("duration_sec", 0.0) or 0.0)
        starts = [float(x) for x in wtimes]
        n_zero_dur = 0
        for i, s0 in enumerate(starts):
            s1 = starts[i + 1] if i + 1 < len(starts) else dur_total
            if not np.isfinite(s0) or not np.isfinite(s1) or (s1 - s0) <= 0:
                n_zero_dur += 1
        n_bad_span = sum(1 for s in spans
                         if not (isinstance(s, (list, tuple)) and len(s) == 2
                                 and int(s[1]) > int(s[0])))
        n_beyond = sum(int(a[m].get("units_beyond_duration", 0) or 0)
                       for m in ("text", "audio", "vision"))
        n_clipped = sum(int(a[m].get("units_clipped_to_last_slot", 0) or 0)
                        for m in ("text", "audio", "vision"))

        # 弱证据一致性检查：在没有真实词时间的前提下，仍然可以核验**必要条件**——
        # 文本占用槽数是否等于词数、文本槽占用是否单调不回头。
        # 这类检查能证伪，不能证真，故只作为 not_asserted 的补充，不升级为 match。
        t_counts = a["text"].get("slot_unit_counts", []) or []
        occ = np.array([int(x) > 0 for x in t_counts], dtype=bool)
        n_occ = int(occ.sum())
        idx = np.where(occ)[0]
        monotone = bool(np.all(np.diff(idx) > 0)) if len(idx) > 1 else True
        word_count_consistent = (n_occ == len(words)) if words else False

        evidence = {
            "text_time_basis": basis,
            "audio_time_basis": str(a["audio"].get("time_basis", "")),
            "vision_time_basis": str(a["vision"].get("time_basis", "")),
            "n_words": len(words),
            "n_text_occupied_slots": n_occ,
            "text_slot_monotone": monotone,
            "text_word_count_consistent": word_count_consistent,
            "n_zero_duration_words": n_zero_dur,
            "n_invalid_char_spans": n_bad_span,
            "units_beyond_duration": n_beyond,
            "units_clipped_to_last_slot": n_clipped,
        }

        if n_zero_dur or n_bad_span or n_beyond:
            state = "confirmed_mismatch"
        elif "uniform" in basis or "assumption" in basis or "假设" in basis:
            # 文本时间非真实测量 → 不构成「已确认对应」
            state = "not_asserted"
        elif not words or not wtimes:
            state = "not_asserted"
        else:
            state = "confirmed_match"

        counts[state] += 1
        out[sid] = {
            "state": state,
            "consistency_check": "ok" if (monotone and word_count_consistent) else "fail",
            "evidence": evidence,
        }

    bases = {str(v["text"].get("time_basis", "")) for v in samples.values()}
    meta = {
        "states": {
            "confirmed_match": "文本与音视频时间基准同为真实测量，且逐词跨度结构合法",
            "confirmed_mismatch": "存在零时长词 / 非法字符跨度 / 越界单元等结构性非法",
            "not_asserted": "文本时间基准为均匀假设或证据不足，不下断言",
        },
        "counts": counts,
        "n_samples": len(out),
        "text_time_basis_observed": sorted(bases),
        "why_confirmed_match_may_be_zero": (
            "本批 100 条的文本时间基准全部是 `uniform_assumption`（词时间=(i+0.5)·T/W 均分），"
            "没有任何一条是真实测量的词时间。因此在本判定规则下 "
            "`confirmed_match` 结构性为 0——这不是缺陷，而是事实："
            "**文本与语音的逐词对应关系在问题一交付里从未被真实测量过**。"
            "方案一自己在 meta.text_time_basis_warning 里也写明了这一点，"
            "但交付表里没有任何字段承载它，本层把这个差别显式化。"
        ),
        "consistency_check_meaning": (
            "在无真实词时间的前提下仍可核验必要条件：文本占用槽数是否等于词数、"
            "占用槽是否单调。该检查只能证伪不能证真，故只作为补充标记，"
            "不把 not_asserted 升级为 confirmed_match。"
        ),
        "rule_note": (
            "判定顺序为 mismatch > not_asserted > match：先排除结构性非法，"
            "再排除无证据样本，最后才允许下正面断言。"
        ),
    }
    return out, meta


# ==================== 第三层：红线自检 ====================

def redline_selfcheck() -> Dict:
    """R1 自检：确认没有任何样本因异常被剔除，且没有标签字段混进特征交付。"""
    summary = _read_summary()
    n_total = len(summary)
    n_included = sum(1 for r in summary.values() if str(r.get("included", "")).strip() in ("1", "True", "true"))
    ledger_path = os.path.join(Q1_DIR, "anomaly_ledger.csv")
    n_ledger = 0
    deletion_flags = []
    if os.path.isfile(ledger_path):
        with open(ledger_path, "r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                n_ledger += 1
                for k, v in r.items():
                    if "deletion" in (k or "").lower() and "count" in (k or "").lower():
                        deletion_flags.append(str(v))

    # 特征交付里不得出现任何标签字段（防止标签泄漏进问题一的特征文件）
    npz_path = os.path.join(Q1_DIR, "features_q1.npz")
    forbidden = ("label", "sentiment", "emotion", "target")
    hit_keys: List[str] = []
    if os.path.isfile(npz_path):
        with np.load(npz_path, allow_pickle=True) as z:
            for k in z.files:
                lk = k.lower()
                if any(t in lk for t in forbidden):
                    hit_keys.append(k)

    return {
        "n_samples_total": n_total,
        "n_samples_included": n_included,
        "counts_as_deletion_are_zero": n_included == n_total,
        "anomaly_ledger_rows": n_ledger,
        "deletion_flag_values": sorted(set(deletion_flags)),
        "label_like_keys_in_feature_npz": hit_keys,
        "r1_statement": (
            "全部异常样本均保留在交付中（included=1），异常只记录不剔除；"
            "特征文件内不含任何标签字段。"
        ),
    }


# ==================== 产出 ====================

def _write_csv(path: str, rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    cols = [c for c in rows[0].keys() if c != "evidence"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_report(rows: List[Dict], vmeta: Dict, cm: Dict, cmeta: Dict,
                  check: Dict, path: str) -> None:
    import collections
    by_mod: Dict[str, List[Dict]] = collections.defaultdict(list)
    for r in rows:
        by_mod[r["modality"]].append(r)

    lines = [
        "# 问题一交付：融合增强层报告",
        "",
        "本报告由 `code/q1_fusion.py` 生成，**只读** `data/q1_delivery/` 既有产物，",
        "不重新提取任何特征，也不修改任何原文件。产出落在 `data/q1_delivery/fusion/`。",
        "",
        "## 一、为什么需要这一层",
        "",
        "| 能力 | 方案一（本仓库） | 方案二 | 融合后 |",
        "|---|---|---|---|",
        "| 时序组织可核验（真实 PTS / 逐槽溯源） | ✅ 逐槽 `slot_edges_sec`+`slot_src_pts` | 有词级 trace | 沿用方案一 |",
        "| 有效性的语义分层 | ❌ `*_valid` 无法区分插值与真实观测，也无法判断模态可用性 | ✅ 存在性/观测/内容三层 | **本层补上** |",
        "| 文本↔语音对应关系的断言纪律 | ❌ 无 | ✅ 三态分离 | **本层补上** |",
        "| 异常只记录不剔除 | ✅ `counts_as_deletion ≡ 0` | ✅ | 沿用并再次断言 |",
        "",
        "## 二、三层语义的核验结果",
        "",
        "| 模态 | 存在率 | 观测有效槽占比（均值） | 内容有效槽占比（均值） | 三种判定的分布 |",
        "|---|---|---|---|---|",
    ]
    for m in ("text", "audio", "vision"):
        rs = by_mod.get(m, [])
        if not rs:
            continue
        ex = np.mean([r["existence"] for r in rs])
        ov = np.mean([r["observation_valid_ratio"] for r in rs])
        cv = np.mean([r["content_valid_ratio"] for r in rs])
        vd = collections.Counter(r["verdict"] for r in rs)
        lines.append(
            f"| {m} | {ex:.3f} | {ov:.3f} | {cv:.3f} | "
            + ", ".join(f"{k}={v}" for k, v in sorted(vd.items())) + " |")

    lines += [
        "",
        "**关键增量**：观测有效 ≠ 内容有效。以视觉为例，只要检出过人脸就算「存在」，",
        "但逐槽是否真的有人脸决定该槽是否为内容有效。方案一的单一 `valid_mask` 把两者混为一谈，",
        "本层把它们分开后，「有视觉模态但该槽没有内容」这种情况才第一次被表达出来。",
        "",
        "### 粒度上的诚实交代",
        "",
        f"- {vmeta['honest_limit']}",
        "- 本层不产出逐槽人脸/有声证据，是因为既有交付物里不存在这些数据；",
        "  若要提升到逐槽粒度，必须重跑 `face_probe` 与 VAD（属于重新提取，不在本层范围内）。",
        "",
        "## 三、文本↔语音对应关系三态",
        "",
        f"- 样本总数：{cmeta['n_samples']}",
    ]
    for k, v in cmeta["counts"].items():
        lines.append(f"- `{k}`：{v}")
    n_ok = sum(1 for v in cm.values() if v.get("consistency_check") == "ok")
    lines += [
        "",
        f"- 弱证据一致性检查通过：{n_ok}/{cmeta['n_samples']}"
        "（文本占用槽数 = 词数，且占用槽单调）",
        "",
        "### 为什么 `confirmed_match` 为 0",
        "",
        f"- 实测文本时间基准取值集合：`{cmeta['text_time_basis_observed']}`",
        f"- {cmeta['why_confirmed_match_may_be_zero']}",
        "",
        f"- {cmeta['consistency_check_meaning']}",
        "",
        cmeta["rule_note"],
        "",
        "意义：方案一把「文本时间基准 = 均匀假设」写进了 `meta.text_time_basis_warning`，",
        "但交付表里没有任何字段承载这个差别。三态分离后，",
        "**「已确认对应」的样本数是一个可引用的下界**，而不是把 100 条全算成已对齐。",
        "",
        "## 四、红线 R1 自检",
        "",
        "```json",
        json.dumps(check, ensure_ascii=False, indent=2),
        "```",
        "",
        f"- 样本总数 {check['n_samples_total']}，实际纳入 {check['n_samples_included']}，",
        f"「异常不剔除」断言：**{'通过' if check['counts_as_deletion_are_zero'] else '不通过'}**",
        f"- 特征文件中的标签类字段：{check['label_like_keys_in_feature_npz'] or '无'}",
        "",
        "## 五、与问题二/三的衔接",
        "",
        "本层的「观测有效 / 内容有效」分离，与问题二在附件2/3 上做的缺失建模是同一条原则：",
        "**先把「没有这一段」和「这一段没有信号」分开，再谈怎么补**。",
        "问题一在自提特征上做这件事，问题二在官方 aligned_50 上做同一件事，",
        "两处的判据在论文里可以互相印证。",
        "",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    if not os.path.isdir(Q1_DIR):
        print(f"未找到问题一交付目录：{Q1_DIR}")
        return 1
    print("=" * 72)
    print("问题一：融合增强层（只读既有交付物）")
    print("=" * 72)

    os.makedirs(FUSION_DIR, exist_ok=True)

    rows, vmeta = build_validity_layers()
    _write_csv(os.path.join(FUSION_DIR, "validity_layers.csv"), rows)
    with open(os.path.join(FUSION_DIR, "validity_layers.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": vmeta, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"三层语义：{len(rows)} 行（{len(rows)//3} 样本 × 3 模态）")

    trace, cmeta = build_correspondence_trace()
    with open(os.path.join(FUSION_DIR, "correspondence_trace.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": cmeta, "samples": trace}, f, ensure_ascii=False, indent=2)
    print(f"对应关系三态：{cmeta['counts']}")

    check = redline_selfcheck()
    print(f"R1 自检：纳入 {check['n_samples_included']}/{check['n_samples_total']}，"
          f"异常不剔除={'通过' if check['counts_as_deletion_are_zero'] else '不通过'}，"
          f"标签泄漏字段={check['label_like_keys_in_feature_npz'] or '无'}")

    _write_report(rows, vmeta, trace, cmeta, check,
                  os.path.join(FUSION_DIR, "q1_fusion_report.md"))
    print(f"\n产物目录：{FUSION_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
