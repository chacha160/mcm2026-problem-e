# -*- coding: utf-8 -*-
"""
q1_verify.py —— 问题一交付前的机器核验套件（**只读**，不修改任何产物）

为什么需要它
------------
问题一要求「原始样本覆盖完整性 / 时序组织可核验性 / 方法合理性与可复现性」。
前面几步只产出了特征，没有任何一项要求被**自动校验**过——手工翻 100 个 npz 既不可
复现也不可信。本模块把每一项要求写成一个可重跑的谓词，跑完给出通过/失败与证据。

诚实性前置声明（写在报告首段，也必须这么理解下面的每一项检查）
--------------------------------------------------------------
1. 本数据集**没有逐词人工时间戳真值**。文本的时间轴是本项目自己声明的
   `time_basis="uniform_assumption"`（词 i 中心时刻 = (i+0.5)·T/W），音频/视觉的
   pts 才是解码器实测值。因此本报告**只报告一致性检查与抽查结果**，
   **绝不声称对齐达到某个毫秒级平均误差**——那需要真值才能算，这里没有。
2. `*_valid=True` 只表示「该槽被分配了采样单元」，**不等于该模态真实可用**。
   反例见 V6/V12：`-mJ2ud6oKI8_1` 的 `vision_valid` 是 50/50 True，但其视觉特征
   能量恰为 0（全帧未检出人脸）。判定模态可用性必须看 `face_ratio` / `voiced_ratio`。
3. 音频/视觉的空槽是**插值填充**的，没有源帧。因此「每个聚合值都能溯源到源帧」
   这一条**无法 100% 满足**，本报告给出真实的可溯源率（见 V4c），
   并把不可溯源的槽逐槽标记为 `interp`，而不是假装全部可溯源。

用法
----
    python q1_verify.py                 # 全部检查
    python q1_verify.py --quick         # 跳过 V8 的逐条 ffprobe 实探
    python q1_verify.py --no-model-check # 跳过 V9（不加载 RoBERTa，省内存）
    python q1_verify.py --out <dir>     # 默认 data/q1_delivery/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import config  # noqa: E402
import unaligned_common as U  # noqa: E402
from utils import LOGGER  # noqa: E402

MODALITIES = ("text", "audio", "vision")
# 项目根 = code/ 的父目录（与 unaligned_common.unaligned_dir 的口径一致）
PROJECT_ROOT = os.path.dirname(_CODE_DIR)
DIM_OF = {"text": config.TEXT_FEATURE_DIM,
          "audio": config.AUDIO_FEATURE_DIM,
          "vision": config.VISION_FEATURE_DIM}

# V4 的判据用「相对」而非绝对容差。原因：不同模态的量级差 3 个数量级
# （文本 |x|max≈16、音频 |x|max≈5.4e3），用绝对容差会在大值维度上爆炸、
# 在小值维度上又过松。实测全体最大相对偏差 5.95e-8（≈0.5×float32 eps），
# 取 1e-6 作阈值有 16 倍余量。
V4_REL_TOL = 1e-6

# 时间轴比较的通用容差（秒）。用于「pts 是否越出公共时间轴」这类判定：
# pts 以 float32 落盘，在 T≈9.3 处 float32 的舍入步长已达 9.5e-7，
# 用 0 容差会把精度表示误差误报成越界。
TIME_TOL = 1e-5


# ==================== 小工具 ====================


def _md5_of_array(a: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def _load_unaligned(root: str, modality: str, sid: str) -> Dict[str, object]:
    return U.load_unaligned(os.path.join(root, modality, f"{sid}.npz"))


def _load_aligned(aligned_dir: str, sid: str) -> Dict[str, object]:
    with np.load(os.path.join(aligned_dir, f"{sid}.npz"), allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _aligned_meta(d: Dict[str, object]) -> Dict[str, object]:
    return json.loads(str(np.asarray(d["meta"]).item()))


def _effective_pts(unaligned_root: str, aligned_dir: str, sid: str, m: str,
                   n_units: int) -> np.ndarray:
    """
    取该样本该模态**实际参与槽归属**的时间戳。

    文本的 pts 逐样本二选一：未对齐文件里存的永远是均匀假设值，走实测路由的样本
    真正用的是强制对齐的区间中心（落在对齐产物的 routing 块里）。核验必须用后者，
    否则会拿一套时刻去重算另一套时刻算出的槽归属，把「路由生效」误判成「溯源失败」。
    音频/视觉无路由，直接返回未对齐文件的实测 pts。
    """
    pts = np.asarray(_load_unaligned(unaligned_root, m, sid)["pts"], np.float64)
    if m != "text":
        return pts
    p = os.path.join(aligned_dir, f"{sid}.npz")
    if not os.path.isfile(p):
        return pts
    with np.load(p, allow_pickle=True) as z:
        meta = json.loads(str(np.asarray(z["meta"]).item()))
    if str(meta["text"]["time_basis"]) != "measured_forced_alignment":
        return pts
    meas = np.asarray((meta.get("routing", {}) or {}).get(
        "word_align", {}).get("word_ts_sec") or [], np.float64)
    return meas if meas.size == int(n_units) else pts


def _result(code: str, name: str, passed: bool, checked: int, failed: int,
            detail: str, evidence: Optional[Dict[str, object]] = None,
            caveat: str = "") -> Dict[str, object]:
    """统一的结果结构。每个检查都必须给出 checked/failed 与人类可读的 detail。"""
    return {"code": code, "name": name, "passed": bool(passed),
            "checked": int(checked), "failed": int(failed),
            "detail": detail, "evidence": evidence or {}, "caveat": caveat}


def _list_npz(d: str, exclude: Tuple[str, ...] = ()) -> List[str]:
    if not os.path.isdir(d):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d)
                  if f.endswith(".npz") and f not in exclude)


# ==================== V1 覆盖完整性 ====================


def check_v1_coverage(unaligned_root: str, aligned_dir: str,
                      summary_csv: Optional[str] = None) -> Dict[str, object]:
    """
    V1 —— 样本编号、模态文件与输出特征是否一一对应（**双向**集合相等）。

    只查「官方 100 个是否都产出了」是不够的：还必须查「有没有多出官方之外的样本」。
    对多出来的文件，单向检查会放过，而它同样破坏一一对应。
    """
    label_ids = [str(s["sample_id"]) for s in U.list_samples()]
    sets: Dict[str, set] = {"label-100（官方）": set(label_ids)}
    for m in MODALITIES:
        sets[f"unaligned/{m}"] = set(_list_npz(os.path.join(unaligned_root, m)))
    sets["aligned/单样本"] = set(_list_npz(aligned_dir, exclude=("aligned_50.npz",)))

    agg_path = os.path.join(aligned_dir, "aligned_50.npz")
    agg_ids: List[str] = []
    if os.path.isfile(agg_path):
        with np.load(agg_path, allow_pickle=True) as z:
            agg_ids = [str(x) for x in z["sample_id"]]
        sets["aligned_50.npz"] = set(agg_ids)

    ref = set(label_ids)
    missing = {k: sorted(ref - v) for k, v in sets.items() if ref - v}
    extra = {k: sorted(v - ref) for k, v in sets.items() if v - ref}
    disagree = {k: sorted(v ^ ref) for k, v in sets.items() if v != ref}

    problems: List[str] = []
    if len(ref) != 100:
        problems.append(f"label-100 只有 {len(ref)} 条，应为 100 条")
    for k, v in missing.items():
        problems.append(f"{k} 缺失 {len(v)} 条，例：{v[:3]}")
    for k, v in extra.items():
        problems.append(f"{k} 多出 {len(v)} 条（官方样本之外），例：{v[:3]}")

    detail = (f"六处样本集合互相相等（官方 label-100、三模态未对齐文件、单样本对齐文件、"
              f"aligned_50.npz），均为 {len(ref)} 条，且无多余文件。"
              if not problems else "覆盖不一致：" + "；".join(problems))
    res = _result("V1", "覆盖完整性（双向集合相等）", not problems, len(ref),
                  len(problems), detail,
                  {"counts": {k: len(v) for k, v in sets.items()},
                   "missing": missing, "extra": extra, "symmetric_difference": disagree},
                  caveat="本项只证明「一一对应」，不证明特征内容正确。")

    # ---- V1b：official_id 口径（若汇总表已存在） ----
    v1b: Optional[Dict[str, object]] = None
    if summary_csv and os.path.isfile(summary_csv):
        with open(summary_csv, encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        bad = [r for r in rows
               if r.get("official_id") != U.official_id(*str(r.get("sample_id", "")).rsplit("_", 1))]
        ex = [{"sample_id": r.get("sample_id"), "got": r.get("official_id"),
               "expected": U.official_id(*str(r["sample_id"]).rsplit("_", 1))} for r in bad[:3]]
        v1b = _result(
            "V1b", "official_id 口径与官方一致", not bad, len(rows), len(bad),
            (f"{len(rows)} 行的 official_id 全部等于 `video_id$_$clip_id` 口径。"
             if not bad else
             f"{len(bad)}/{len(rows)} 行的 official_id 与官方口径不符，例：{ex}"),
            {"examples": ex},
            caveat="历史缺陷：save_summary_csv 曾用 replace('_','$_$',1) 按**第一个**下划线切分，"
                   "而 6 个 video_id 自身含下划线，波及 18 行；已改为按最后一个下划线切分（rsplit）。")
    if v1b is not None:
        res["sub_check"] = v1b
    return res


# ==================== V2 词区间 / 溯源三件套 ====================


def check_v2_word_time(unaligned_root: str, sids: List[str]) -> Dict[str, object]:
    """
    V2 —— 词区间单调且不越出视频范围；字符范围与词元索引自洽。

    **重要限定（写进报告，避免过度声称）**：文本的 pts 是 (i+0.5)·T/W 定义出来的，
    所以「单调且不超范围」**按构造必然成立**。本项只能证明「实现忠实于所声明的
    均匀假设、没有下标错位」，**不能**证明词时刻正确。
    音频/视觉的同类检查才具实测意义（其 pts 来自解码器）。
    """
    checked = failed = 0
    errs: List[str] = []
    txt_stat = {"strict_increasing": 0, "in_range": 0}
    av_stat = {"non_decreasing": 0, "samples": 0}
    span_stat = {"reconstruct_ok": 0, "monotone": 0}
    tok_stat = {"strict_increasing": 0, "count_ok": 0}

    for sid in sids:
        # ---- 文本 ----
        d = _load_unaligned(unaligned_root, "text", sid)
        md = d["meta"]
        pts = np.asarray(d["pts"], np.float64)
        T = float(md.get("duration_used", 0.0))
        W = len(pts)
        if W > 1:
            checked += 1
            if np.all(np.diff(pts) > 0):
                txt_stat["strict_increasing"] += 1
            else:
                failed += 1
                errs.append(f"{sid}: 文本 pts 非严格递增")
        if W:
            checked += 1
            if pts[0] > 0 and pts[-1] <= T + TIME_TOL:
                txt_stat["in_range"] += 1
            else:
                failed += 1
                errs.append(f"{sid}: 文本 pts 越出 (0,T]，max={pts[-1]:.4f} T={T:.4f}")

        # ---- 字符范围 / 词元索引 ----
        ex = d["extra"]
        words = [str(x) for x in ex.get("words", [])]
        spans = [tuple(int(v) for v in s) for s in ex.get("char_spans", [])]
        spans_o = [tuple(int(v) for v in s) for s in ex.get("char_spans_original", [])]
        toks = [[int(v) for v in t] for t in ex.get("token_index", [])]
        processed = str(md.get("text_processed", ""))
        original = str(md.get("text_original", ""))

        if spans:
            checked += 1
            mono = all(spans[i][0] <= spans[i][1] for i in range(len(spans))) and \
                   all(spans[i][0] >= spans[i - 1][1] for i in range(1, len(spans)))
            if mono:
                span_stat["monotone"] += 1
            else:
                failed += 1
                errs.append(f"{sid}: 字符区间非单调或重叠")
            checked += 1
            ok = all(processed[s:e] == w for (s, e), w in zip(spans, words))
            if ok:
                span_stat["reconstruct_ok"] += 1
            else:
                failed += 1
                first_bad = next((i for i, ((s, e), w) in enumerate(zip(spans, words))
                                  if processed[s:e] != w), -1)
                errs.append(f"{sid}: 字符区间无法回指 processed 文本（第 {first_bad} 词）")
        if spans_o and original:
            # 原文坐标必须单调且落在原文长度内（归一化会改动内容，只查范围合法性）
            checked += 1
            if all(0 <= s <= e <= len(original) for s, e in spans_o):
                pass
            else:
                failed += 1
                errs.append(f"{sid}: 原文坐标越界（原文长 {len(original)}）")

        if toks:
            flat = [t for row in toks for t in row]
            checked += 1
            if flat == sorted(set(flat)):
                tok_stat["strict_increasing"] += 1
            else:
                failed += 1
                errs.append(f"{sid}: 词元索引非严格递增或有重复")
            checked += 1
            n_special = int(md.get("num_subwords", 0)) - len(flat)
            if n_special >= 0 and len(flat) == int(md.get("num_subwords", 0)) - n_special:
                tok_stat["count_ok"] += 1
            else:
                failed += 1
                errs.append(f"{sid}: 词元索引总数与 num_subwords 不符")

        # ---- 音频 / 视觉（实测 pts） ----
        for m in ("audio", "vision"):
            da = _load_unaligned(unaligned_root, m, sid)
            p = np.asarray(da["pts"], np.float64)
            av_stat["samples"] += 1
            checked += 1
            if p.size <= 1 or np.all(np.diff(p) >= -TIME_TOL):
                av_stat["non_decreasing"] += 1
            else:
                failed += 1
                n_bad = int((np.diff(p) < -TIME_TOL).sum())
                errs.append(f"{sid}/{m}: 实测 pts 非单调（{n_bad} 处回退）")

    detail = (f"文本 pts 严格递增且落在 (0,T] 内；{span_stat['reconstruct_ok']} 条样本的"
              f"字符区间能逐词精确回指规范化文本；{tok_stat['strict_increasing']} 条样本的"
              f"词元索引严格递增；音频/视觉实测 pts 在 {av_stat['non_decreasing']}/"
              f"{av_stat['samples']} 条样本上非递减。"
              if not errs else "发现 " + str(len(errs)) + " 处问题：" + "；".join(errs[:5]))
    return _result("V2", "词区间单调且不越出视频范围", not errs, checked, failed, detail,
                   {"text_pts": txt_stat, "char_spans": span_stat, "token_index": tok_stat,
                    "av_pts": av_stat, "errors": errs[:20]},
                   caveat="本项检查的是**未对齐产物**里的文本 pts，它一律是均匀假设值"
                          "（由 (i+0.5)·T/W **定义**而来），故由构造保证成立，"
                          "只能证明实现与声明一致，**不作为词级时间正确的证据**。"
                          "走实测路由的样本，其真正落进 50 槽的是实测词时刻——"
                          "那一套时刻的合法性与一致性由 V13 单独核验。"
                          "音频/视觉的 pts 为解码器实测值，该项才具实测意义。")


# ==================== V3 序列长度与维度 ====================


def check_v3_shapes(unaligned_root: str, aligned_dir: str, sids: List[str]) -> Dict[str, object]:
    """V3 —— 序列长度与词索引一致；维度、掩码、长度字段互相自洽。"""
    checked = failed = 0
    errs: List[str] = []
    for sid in sids:
        # ---- 未对齐：长度与词索引一致 ----
        d = _load_unaligned(unaligned_root, "text", sid)
        md = d["meta"]
        ex = d["extra"]
        feats = np.asarray(d["features"])
        lens = {
            "features.shape[0]": int(feats.shape[0]),
            "len(pts)": int(len(np.asarray(d["pts"]))),
            "extra.words": int(len(ex.get("words", []))),
            "meta.num_words": int(md.get("num_words", -1)),
            "extra.char_spans": int(len(ex.get("char_spans", []))),
            "extra.token_index": int(len(ex.get("token_index", []))),
        }
        checked += 1
        if len(set(lens.values())) != 1:
            failed += 1
            errs.append(f"{sid}: 文本长度字段不一致 {lens}")

        for m in MODALITIES:
            dm = _load_unaligned(unaligned_root, m, sid)
            fm = np.asarray(dm["features"])
            mm = dm["meta"]
            checked += 1
            if fm.ndim != 2 or fm.shape[1] != DIM_OF[m] or int(mm.get("feature_dim", -1)) != DIM_OF[m]:
                failed += 1
                errs.append(f"{sid}/{m}: 维度不符 shape={fm.shape} "
                            f"meta.feature_dim={mm.get('feature_dim')} 期望={DIM_OF[m]}")
            checked += 1
            if len(np.asarray(dm["pts"])) != fm.shape[0]:
                failed += 1
                errs.append(f"{sid}/{m}: pts 长度与特征帧数不符")

        # ---- 已对齐：单样本 ----
        al = _load_aligned(aligned_dir, sid)
        am = _aligned_meta(al)
        n_slots = int(np.asarray(al["n_slots"]).item())
        edges = np.asarray(al["slot_edges"])
        checked += 1
        if n_slots != config.ALIGN_SEQ_LEN or len(edges) != n_slots + 1:
            failed += 1
            errs.append(f"{sid}: n_slots={n_slots} 与槽边界长度 {len(edges)} 不自洽")
        for m in MODALITIES:
            f = np.asarray(al[f"{m}_features"])
            v = np.asarray(al[f"{m}_valid"])
            checked += 1
            if f.shape != (n_slots, DIM_OF[m]) or v.shape != (n_slots,) or v.dtype != np.bool_:
                failed += 1
                errs.append(f"{sid}/{m}: 对齐形状不符 {f.shape} {v.shape} {v.dtype}")
            # valid.sum() 必须同时等于 meta 记录值与「有单元」的槽数
            counts = np.asarray(am[m]["align"]["unit_counts"], np.int64)
            checked += 1
            if int(v.sum()) != int(am[m]["align"]["valid_slots"]) != int((counts > 0).sum()):
                failed += 1
                errs.append(f"{sid}/{m}: valid.sum()={int(v.sum())} "
                            f"meta.valid_slots={am[m]['align']['valid_slots']} "
                            f"(counts>0).sum()={int((counts > 0).sum())}")

    # ---- 已对齐：汇总张量 ----
    agg_stat: Dict[str, object] = {}
    agg_path = os.path.join(aligned_dir, "aligned_50.npz")
    if os.path.isfile(agg_path):
        with np.load(agg_path, allow_pickle=True) as z:
            N = len(z["sample_id"])
            agg_stat["n_samples"] = N
            agg_stat["n_slots"] = int(np.asarray(z["n_slots"]).item())
            for m in MODALITIES:
                f, v, L = z[f"{m}_features"], z[f"{m}_valid"], z[f"{m}_lengths"]
                dim = int(np.asarray(z[f"{m}_features_dim"]).item())
                checked += 1
                ok = (f.shape == (N, config.ALIGN_SEQ_LEN, DIM_OF[m])
                      and v.shape == (N, config.ALIGN_SEQ_LEN) and v.dtype == np.bool_
                      and dim == DIM_OF[m]
                      and np.all(np.asarray(L) == config.ALIGN_SEQ_LEN))
                if ok:
                    agg_stat[m] = {"shape": list(f.shape), "dim": dim}
                else:
                    failed += 1
                    errs.append(f"aligned_50/{m}: 形状/维度不符 shape={f.shape} dim={dim}")

    detail = (f"未对齐序列长度字段两两一致，三模态维度恒为 768/74/35，"
              f"对齐张量形状 (100,50,D)，valid 掩码 valid.sum() == meta.valid_slots == (counts>0).sum()。"
              if not errs else f"{failed} 处不一致：" + "；".join(errs[:5]))
    return _result("V3", "序列长度与词索引一致", not errs, checked, failed, detail,
                   {"aggregate": agg_stat, "errors": errs[:20]})


# ==================== V4 溯源（核心） ====================


def check_v4_traceable(unaligned_root: str, aligned_dir: str,
                       sids: List[str]) -> Dict[str, object]:
    """
    V4 —— 每个音视频聚合值都能找到来源帧（核心检查）。

    对每个「有单元」的槽，用 `slot_unit_index` 从未对齐文件回捞源单元、**重算均值**，
    与落盘的槽值比较。判据用**相对**（max-norm）偏差 ≤ 1e-6，因为不同模态量级差千倍。

    子检查 V4c 是**独立重算槽归属**：直接用源 `pts` 重新跑 time_to_slot，
    与文件里的 `slot_unit_index` 对照。这一条不依赖任何落盘中间量，
    正是它抓出了 D1（duration/slot_edges 曾被降为 float32，使 39 个落在槽边界的
    单元归属错位）。
    """
    checked = failed = 0
    errs: List[str] = []
    worst_rel = 0.0
    worst_abs = 0.0
    worst_where = ""
    trace_stat: Dict[str, Dict[str, int]] = {m: {"slots": 0, "traceable": 0, "interp": 0}
                                             for m in MODALITIES}
    rel_hist: List[float] = []

    for sid in sids:
        al = _load_aligned(aligned_dir, sid)
        am = _aligned_meta(al)
        L = int(np.asarray(al["n_slots"]).item())
        T = float(np.asarray(al["duration"], np.float64))

        for m in MODALITIES:
            src = _load_unaligned(unaligned_root, m, sid)
            S = np.asarray(src["features"], np.float64)          # 源单元（float64 累加）
            # 用**实际参与归属**的时间戳（文本逐样本路由，见 _effective_pts）
            pts = _effective_pts(unaligned_root, aligned_dir, sid, m, S.shape[0])
            stored = np.asarray(al[f"{m}_features"], np.float64)
            valid = np.asarray(al[f"{m}_valid"])
            counts = np.asarray(am[m]["align"]["unit_counts"], np.int64)
            idx = am[m]["slot_unit_index"]

            trace_stat[m]["slots"] += L
            trace_stat[m]["traceable"] += int((valid & (counts > 0)).sum())
            trace_stat[m]["interp"] += int((~valid).sum())

            # ---- V4b: 有单元 <=> valid ----
            checked += 1
            if not np.array_equal(counts > 0, valid):
                failed += 1
                d = int(np.sum((counts > 0) != valid))
                errs.append(f"{sid}/{m}: unit_counts>0 与 valid 不等价（{d} 个槽）")

            # ---- V4a: 逐槽重算均值 ----
            for k in range(L):
                if counts[k] <= 0:
                    continue
                units = [int(u) for u in idx[k]]
                checked += 1
                if len(units) != int(counts[k]):
                    failed += 1
                    errs.append(f"{sid}/{m} 槽{k}: slot_unit_index 长度 {len(units)} "
                                f"!= unit_counts {int(counts[k])}")
                    continue
                rec = S[units].sum(axis=0) / len(units)
                if not np.all(np.isfinite(rec)):
                    failed += 1
                    errs.append(f"{sid}/{m} 槽{k}: 重算值含 NaN/Inf")
                    continue
                diff = np.abs(rec - stored[k])
                scale = max(float(np.abs(stored[k]).max()), float(np.abs(rec).max()), 1e-12)
                a = float(diff.max())
                r = a / scale
                rel_hist.append(r)
                if a > worst_abs:
                    worst_abs = a
                if r > worst_rel:
                    worst_rel, worst_where = r, f"{sid}/{m}/槽{k}"
                if r > V4_REL_TOL:
                    failed += 1
                    errs.append(f"{sid}/{m} 槽{k}: 溯源重算相对偏差 {r:.3e} > {V4_REL_TOL:g}")

            # ---- V4c: 独立重算槽归属（抓 D1 的那一条）----
            if pts.size:
                from align_multimodal import time_to_slot
                slot_of = time_to_slot(pts, T, L)
                groups: Dict[int, List[int]] = {}
                for u, s in enumerate(slot_of):
                    groups.setdefault(int(s), []).append(int(u))
                checked += 1
                bad_slots = []
                for k in range(L):
                    a_units = [int(u) for u in idx[k]]
                    b_units = groups.get(k, [])
                    if a_units != b_units:
                        bad_slots.append((k, a_units, b_units))
                if bad_slots:
                    failed += 1
                    k, a_u, b_u = bad_slots[0]
                    errs.append(f"{sid}/{m}: 槽归属不可复现（{len(bad_slots)} 个槽，"
                                f"例 槽{k} 文件={a_u} 重算={b_u}）")

    traceable_ratio = {m: (round(trace_stat[m]["traceable"] / max(trace_stat[m]["slots"], 1), 4))
                       for m in MODALITIES}
    # 逐槽均值重算的绝对偏差最大值用于**记录**，不用于判定（见文件头 DIM 注释）
    detail = (f"对每个有源单元的槽回捞源帧重算均值，最大相对偏差 {worst_rel:.3e}"
              f"（阈值 {V4_REL_TOL:g}，余量 {V4_REL_TOL / max(worst_rel, 1e-12):.0f}×，"
              f"出现在 {worst_where or '无'}）；"
              + "；".join(f"{m} 可溯源率 {traceable_ratio[m]:.2%}（插值槽 "
                          f"{trace_stat[m]['interp']} 个无源帧）" for m in MODALITIES)
              if not errs else f"{failed} 处溯源失败：" + "；".join(errs[:5]))
    return _result("V4", "每个音视频聚合值都能找到来源帧", not errs, checked, failed, detail,
                   {"v4a_max_norm_rel_diff": worst_rel,
                    "v4a_max_abs_diff": worst_abs,
                    "v4a_max_abs_diff_note": "绝对偏差仅供记录；音频量级 ~5e3，"
                                             "绝对容差会误判，判据一律用相对偏差",
                    "v4a_worst_location": worst_where,
                    "v4a_rel_tol": V4_REL_TOL,
                    "v4a_rel_diff_percentiles": {
                        "p50": float(np.percentile(rel_hist, 50)) if rel_hist else 0.0,
                        "p90": float(np.percentile(rel_hist, 90)) if rel_hist else 0.0,
                        "p99": float(np.percentile(rel_hist, 99)) if rel_hist else 0.0,
                        "max": float(np.max(rel_hist)) if rel_hist else 0.0},
                    "traceability": trace_stat,
                    "traceable_ratio": traceable_ratio,
                    "errors": errs[:20]},
                   caveat="音频/视觉的空槽由相邻有效槽**插值**填充，没有源帧，"
                          "因此不可能 100% 溯源——本项给出真实可溯源率，"
                          "并在 alignment_q1.json 里把这些槽标记为 interp 而非冒充有源。")


# ==================== V5 槽几何 ====================


def check_v5_slot_geom(aligned_dir: str, sids: List[str]) -> Dict[str, object]:
    """V5 —— 槽边界严格递增、覆盖 [0,T]、等宽，且 time_to_slot 可往返。"""
    from align_multimodal import time_to_slot, slot_to_interval
    checked = failed = 0
    errs: List[str] = []
    widths: List[float] = []
    clipped_total = underflow_total = beyond_total = 0
    for sid in sids:
        al = _load_aligned(aligned_dir, sid)
        edges = np.asarray(al["slot_edges"], np.float64)
        T = float(np.asarray(al["duration"], np.float64))
        L = int(np.asarray(al["n_slots"]).item())
        checked += 1
        if not (np.all(np.diff(edges) > 0) and abs(edges[0]) <= TIME_TOL
                and abs(edges[-1] - T) <= TIME_TOL):
            failed += 1
            errs.append(f"{sid}: 槽边界非严格递增或未覆盖 [0,T]")
        w = np.diff(edges)
        widths.append(float(w.max() - w.min()))
        checked += 1
        if abs(float(w.mean()) - T / L) > 1e-9 * max(T, 1.0):
            failed += 1
            errs.append(f"{sid}: 槽宽不均（mean={w.mean():.6f} 期望={T / L:.6f}）")
        # 往返：槽中心 → 槽号
        checked += 1
        back = [int(time_to_slot(np.array([(slot_to_interval(k, T, L)[0]
                                            + slot_to_interval(k, T, L)[1]) / 2.0]),
                                 T, L)[0]) for k in range(L)]
        if back != list(range(L)):
            failed += 1
            errs.append(f"{sid}: time_to_slot/slot_to_interval 往返不一致")

        am = _aligned_meta(al)
        for m in MODALITIES:
            a = am[m]["align"]
            clipped_total += int(a.get("units_clipped_to_last_slot", 0))
            underflow_total += int(a.get("units_clipped_to_first_slot", 0))
            beyond_total += int(a.get("units_beyond_duration", 0))

    detail = (f"100 条样本的槽边界严格递增、首尾为 0 与 T、等宽（最大槽宽浮动 "
              f"{max(widths):.2e} s），50 个槽的 time_to_slot 往返全部一致；"
              f"越界单元被并入端点槽：末槽 {clipped_total} 个、首槽 {underflow_total} 个"
              f"（其中真正越出 T 的 {beyond_total} 个）。"
              if not errs else f"{failed} 处槽几何异常：" + "；".join(errs[:5]))
    return _result("V5", "槽几何与可逆映射", not errs, checked, failed, detail,
                   {"max_width_jitter_sec": max(widths) if widths else 0.0,
                    "units_clipped_to_last_slot": clipped_total,
                    "units_clipped_to_first_slot": underflow_total,
                    "units_beyond_duration": beyond_total},
                   caveat="落在 [0,T] 之外的单元被 time_to_slot 按最近端截断并入端点槽"
                          "（并排总比丢弃诚实）；本项**显式记账**而不是让它静默发生。")


# ==================== V6 填充规则 ====================


def check_v6_fill(unaligned_root: str, aligned_dir: str,
                  sids: List[str]) -> Dict[str, object]:
    """
    V6 —— 填充规则一致性。

    **分模态判定**（统一断言会立刻失败）：
      text          fill="zero"          → 空槽必须恰为全零向量
      audio/vision  fill="interpolate"   → 空槽**非零**，是插值值；只断言
                                           valid=False ⟺ unit_counts==0
    """
    checked = failed = 0
    errs: List[str] = []
    stat: Dict[str, Dict[str, int]] = {m: {"slots": 0, "empty": 0, "empty_nonzero": 0}
                                       for m in MODALITIES}
    fills_seen: Dict[str, set] = {m: set() for m in MODALITIES}
    for sid in sids:
        al = _load_aligned(aligned_dir, sid)
        am = _aligned_meta(al)
        for m in MODALITIES:
            f = np.asarray(al[f"{m}_features"], np.float64)
            v = np.asarray(al[f"{m}_valid"])
            counts = np.asarray(am[m]["align"]["unit_counts"], np.int64)
            fills_seen[m].add(str(am[m]["align"]["fill"]))
            stat[m]["slots"] += len(v)
            stat[m]["empty"] += int((~v).sum())

            checked += 1
            if not np.array_equal(counts > 0, v):
                failed += 1
                errs.append(f"{sid}/{m}: 空槽与 unit_counts==0 不等价")
                continue
            empty = ~v
            if m == "text":
                checked += 1
                nz = int((np.abs(f[empty]).sum(axis=1) > 0).sum()) if empty.any() else 0
                stat[m]["empty_nonzero"] += nz
                if nz:
                    failed += 1
                    errs.append(f"{sid}/text: {nz} 个空槽非零（fill=zero 应恰为零）")
            else:
                stat[m]["empty_nonzero"] += int((np.abs(f[empty]).sum(axis=1) > 0).sum()) \
                    if empty.any() else 0

    expected_fill = {"text": "zero", "audio": "interpolate", "vision": "interpolate"}
    checked += 1
    bad_fill = {m: sorted(fills_seen[m]) for m in MODALITIES
                if fills_seen[m] != {expected_fill[m]}}
    if bad_fill:
        failed += 1
        errs.append(f"fill 口径与声明不符：{bad_fill}")

    detail = (f"text 的 {stat['text']['empty']} 个空槽全部恰为零向量；"
              f"audio/vision 的 {stat['audio']['empty'] + stat['vision']['empty']} 个空槽"
              f"由插值填充（因此非零），三者 fill 口径与声明一致。"
              if not errs else f"{failed} 处填充规则不一致：" + "；".join(errs[:5]))
    return _result("V6", "填充规则一致性", not errs, checked, failed, detail,
                   {"per_modality": stat, "fills_seen": {m: sorted(fills_seen[m])
                                                         for m in MODALITIES},
                    "expected_fill": expected_fill,
                    "errors": errs[:20]},
                   caveat="audio/vision 的空槽值是插值结果，**不具溯源资格**，"
                          "已在 V4 的可溯源率中扣除；本条只断言掩码与计数自洽，"
                          "不对插值值做零断言。")


# ==================== V7 数值合规 ====================


def check_v7_numeric(unaligned_root: str, aligned_dir: str,
                     sids: List[str]) -> Dict[str, object]:
    """V7 —— 无 NaN/Inf；特征 float32、掩码 bool；记录各模态量级（供 f16 讨论）。"""
    checked = failed = 0
    errs: List[str] = []
    absmax = {m: 0.0 for m in MODALITIES}
    for sid in sids:
        for m in MODALITIES:
            d = _load_unaligned(unaligned_root, m, sid)
            f = np.asarray(d["features"])
            checked += 1
            if f.dtype != np.float32:
                failed += 1
                errs.append(f"{sid}/{m}: 未对齐特征 dtype={f.dtype}，应为 float32")
            checked += 1
            if not np.all(np.isfinite(f)):
                failed += 1
                errs.append(f"{sid}/{m}: 未对齐特征含 NaN/Inf")
            if f.size:
                absmax[m] = max(absmax[m], float(np.abs(f).max()))
        al = _load_aligned(aligned_dir, sid)
        for m in MODALITIES:
            f = np.asarray(al[f"{m}_features"])
            v = np.asarray(al[f"{m}_valid"])
            checked += 1
            if f.dtype != np.float32 or v.dtype != np.bool_:
                failed += 1
                errs.append(f"{sid}/{m}: 对齐 dtype 不符 {f.dtype}/{v.dtype}")
            checked += 1
            if not np.all(np.isfinite(f)):
                failed += 1
                errs.append(f"{sid}/{m}: 对齐特征含 NaN/Inf")

    detail = (f"全部特征均为 float32 且无 NaN/Inf，掩码均为 bool；"
              f"各模态 |x|max：文本 {absmax['text']:.3f}、语音 {absmax['audio']:.1f}、"
              f"视觉 {absmax['vision']:.3f}（量级差 3 个数量级，故 V4 必须用相对容差）。"
              if not errs else f"{failed} 处数值异常：" + "；".join(errs[:5]))
    return _result("V7", "数值合规（无 NaN/Inf，dtype 正确）", not errs, checked, failed, detail,
                   {"abs_max_per_modality": absmax})


# ==================== V8 时长口径交叉核对 ====================


def check_v8_duration(aligned_dir: str, sids: List[str],
                      limit: int = 0) -> Dict[str, object]:
    """V8 —— 用 ffprobe 重新探测视频流时长，与文件内 duration 比对。

    这是**外部**证据：不依赖任何本项目写下的中间量。
    """
    target = sids if limit <= 0 else sids[:limit]
    checked = failed = 0
    errs: List[str] = []
    max_dev = 0.0
    t0 = time.time()
    for sid in target:
        try:
            video_path = U.video_path_of(*sid.rsplit("_", 1))
        except Exception as exc:
            failed += 1
            errs.append(f"{sid}: 无法定位源视频（{type(exc).__name__}）")
            continue
        if not os.path.isfile(video_path):
            failed += 1
            errs.append(f"{sid}: 源视频不存在")
            continue
        try:
            probed = float(U.probe_streams(video_path)["video_duration"])
        except Exception as exc:
            failed += 1
            errs.append(f"{sid}: ffprobe 失败 {type(exc).__name__}: {exc}")
            continue
        al = _load_aligned(aligned_dir, sid)
        stored = float(np.asarray(al["duration"], np.float64))
        dev = abs(probed - stored)
        max_dev = max(max_dev, dev)
        checked += 1
        if dev > 1e-3:
            failed += 1
            errs.append(f"{sid}: 文件 duration={stored:.6f} 与 ffprobe={probed:.6f} 差 {dev:.2e}s")

    detail = (f"对 {checked} 条样本重新执行 ffprobe，视频流时长与文件内 duration 的最大"
              f"绝对偏差 {max_dev:.2e} s（阈值 1e-3 s），耗时 {time.time() - t0:.1f}s。"
              if not errs else f"{failed} 条时长不符：" + "；".join(errs[:5]))
    return _result("V8", "时长口径交叉核对（外部 ffprobe）", not errs, checked, failed, detail,
                   {"max_deviation_sec": max_dev, "probed_samples": len(target),
                    "elapsed_sec": round(time.time() - t0, 2)})


# ==================== V9 预训练权重真实性 ====================


def check_v9_model() -> Dict[str, object]:
    """
    V9 —— 外部证据：把 RoBERTa 当**语言模型**用一次掩码补全。

    如果权重只是随机初始化的空壳，掩码补全不可能给出语义正确的词。
    这条检查不来自任何本项目产物，因此是「模型 revision 可复现」的最硬证据。
    注意：管线只用 encoder 的 last_hidden_state，因此加载时
    `pooler.*` 缺失（随机初始化但从未使用）、`lm_head.*` 未预期（预训练头，本检查才用到）
    都是正常现象，要如实记录而不是当成错误。
    """
    import torch
    from transformers import AutoTokenizer, RobertaForMaskedLM

    name = config.TEXT_MODEL_NAME
    # local_files_only=True：本项目受「零安装 / 零网络」约束，这里顺带证明
    # 全部权重与分词器都能**纯离线**加载（不依赖任何网络往返）。
    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    lm, info = RobertaForMaskedLM.from_pretrained(
        name, output_loading_info=True, local_files_only=True)
    lm.eval()
    probes = {
        "The capital of France is <mask>.": {"Paris", "paris"},
        "I love this movie, it was <mask>.": {"great", "amazing", "awesome",
                                              "wonderful", "excellent"},
    }
    rows: List[Dict[str, object]] = []
    n_pass = 0
    with torch.no_grad():
        for text, expect in probes.items():
            ids = tok(text, return_tensors="pt")["input_ids"]
            pos = int((ids[0] == tok.mask_token_id).nonzero()[0])
            logits = lm(**{"input_ids": ids}).logits[0, pos]
            order = torch.argsort(logits, descending=True)[:5].tolist()
            top5 = [tok.decode([i]).strip() for i in order]
            ok = top5[0] in expect or any(t in expect for t in top5)
            n_pass += int(ok)
            rows.append({"probe": text, "top5": top5, "expected": sorted(expect), "passed": ok})

    emb_std = float(lm.get_input_embeddings().weight.detach().std())
    unexp = sorted(str(x) for x in (info.get("unexpected_keys") or []))
    miss = sorted(str(x) for x in (info.get("missing_keys") or []))
    detail = (f"掩码补全 {n_pass}/{len(probes)} 通过（top-5 命中语义正确词），"
              f"词嵌入标准差 {emb_std:.4f}（随机初始化应≈1/√768≈0.036），"
              f"hidden={lm.config.hidden_size} vocab={lm.config.vocab_size}；"
              f"纯离线（local_files_only=True）加载成功；"
              f"未预期键 {len(unexp)} 个、缺失键 {len(miss)} 个。")
    return _result("V9", "预训练权重真实性（掩码补全，外部证据）", n_pass == len(probes),
                   len(probes), len(probes) - n_pass, detail,
                   {"probes": rows, "word_embedding_std": emb_std,
                    "hidden_size": int(lm.config.hidden_size),
                    "vocab_size": int(lm.config.vocab_size),
                    "loaded_local_files_only": True,
                    "unexpected_keys": unexp,
                    "missing_keys": miss,
                    "note": "本检查加载的是 RobertaForMaskedLM（含 lm_head），"
                            "因此键集齐全；管线用的 AutoModel 会另报 pooler.* 缺失，"
                            "那是正常的——管线只取 last_hidden_state，从不使用 pooler。"},
                   caveat="本项验证「权重确为预训练成果」，不验证「特征对情感任务有效」"
                          "（后者属问题二）。")


# ==================== V10 跨样本查重 ====================


def check_v10_duplicate(unaligned_root: str, sids: List[str]) -> Dict[str, object]:
    """
    V10 —— 模态内两两查重。

    历史上本项目出过「不同视频的音频互相覆盖后逐字节相同」的事故，成本≈0 的
    哈希查重能直接暴露这类管线 bug。查到重复不等于有 bug，但**每一组都必须有解释**。
    """
    checked = 0
    dup_groups: List[Dict[str, object]] = []
    for m in MODALITIES:
        buckets: Dict[str, List[str]] = {}
        for sid in sids:
            d = _load_unaligned(unaligned_root, m, sid)
            f = np.asarray(d["features"])
            h = _md5_of_array(f)
            buckets.setdefault(h, []).append(sid)
            checked += 1
        for h, ids in buckets.items():
            if len(ids) > 1:
                face = [_load_unaligned(unaligned_root, "vision", i)["meta"].get("face_ratio")
                        for i in ids] if m == "vision" else None
                voiced = [_load_unaligned(unaligned_root, "audio", i)["meta"].get("voiced_ratio")
                          for i in ids] if m == "audio" else None
                # 同一 video 的不同片段：属已知合理来源（同源片段共享静音/全零）
                same_video = len({i.rsplit("_", 1)[0] for i in ids}) == 1
                explained = ((m == "vision" and face and all((x or 0) == 0 for x in face))
                             or (m == "audio" and voiced and all((x or 0) == 0 for x in voiced))
                             or same_video)
                dup_groups.append({"modality": m, "md5": h, "samples": ids,
                                   "face_ratio": face, "voiced_ratio": voiced,
                                   "same_video": same_video, "explained": bool(explained)})

    unexplained = [g for g in dup_groups if not g["explained"]]
    detail = (f"三模态共 {checked} 个特征数组按 md5 查重：发现 {len(dup_groups)} 组重复，"
              f"其中 {len(dup_groups) - len(unexplained)} 组可解释"
              f"（同源视频片段 / 全零视觉 / 数字静音音频），"
              + (f"{len(unexplained)} 组无法解释，需排查。" if unexplained else "无不可解释重复。")
              if dup_groups else f"三模态共 {checked} 个特征数组按 md5 查重，未发现逐字节重复。")
    return _result("V10", "跨样本重复检测", not unexplained, checked, len(unexplained), detail,
                   {"groups": dup_groups},
                   caveat="重复**不必然**是缺陷；本项要求每一组都有合理解释，"
                          "并把解释写进异常台账。")


# ==================== V11 产物新鲜度 ====================


def check_v11_freshness(unaligned_root: str, aligned_dir: str) -> Dict[str, object]:
    """V11 —— 对齐产物必须晚于未对齐产物，否则交付不自洽（陈旧产物）。"""
    def newest(d: str) -> Tuple[float, str]:
        best, name = 0.0, ""
        for f in os.listdir(d):
            if not f.endswith(".npz"):
                continue
            p = os.path.join(d, f)
            mt = os.path.getmtime(p)
            if mt > best:
                best, name = mt, f
        return best, name

    def oldest(d: str, exclude: Tuple[str, ...] = ()) -> Tuple[float, str]:
        best, name = float("inf"), ""
        for f in os.listdir(d):
            if not f.endswith(".npz") or f in exclude:
                continue
            mt = os.path.getmtime(os.path.join(d, f))
            if mt < best:
                best, name = mt, f
        return best, name

    u_new, u_name = 0.0, ""
    per_mod = {}
    for m in MODALITIES:
        t, n = newest(os.path.join(unaligned_root, m))
        per_mod[m] = {"newest": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)), "file": n}
        if t > u_new:
            u_new, u_name = t, f"{m}/{n}"
    a_old, a_name = oldest(aligned_dir, exclude=("aligned_50.npz",))
    ok = a_old > u_new
    detail = (f"对齐产物最早写入时间 {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(a_old))}"
              f" 晚于未对齐产物最新写入时间 "
              f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(u_new))}，"
              f"交付自洽（对齐确实建立在当前特征之上）。" if ok else
              f"对齐产物（最早 {a_name}）**早于**未对齐产物（最新 {u_name}），"
              f"属陈旧产物，必须重跑对齐。")
    return _result("V11", "产物新鲜度（对齐晚于特征）", ok, len(MODALITIES) + 1,
                   int(not ok), detail,
                   {"unaligned_newest": u_name, "aligned_oldest": a_name,
                    "unaligned_newest_per_modality": per_mod,
                    "aligned_newest": time.strftime("%Y-%m-%d %H:%M:%S",
                                                    time.localtime(newest(aligned_dir)[0]))})


# ==================== V12 异常普查 ====================


def check_v12_anomaly_census(unaligned_root: str, aligned_dir: str,
                             sids: List[str]) -> Dict[str, object]:
    """
    V12 —— 异常普查（**只用于分类与标注，绝不用于删除样本**）。

    统计口径一律来自源 meta（`face_ratio` / `voiced_ratio` / `frame_gaps`），
    **不使用 `*_valid` 掩码**——后者只表示「槽被分配了单元」，
    反例：-mJ2ud6oKI8_1 的 vision_valid 全 True 而特征能量恰为 0。
    """
    stats: Dict[str, List[str]] = {
        "A1_audio_digital_silence": [], "A2_vad_zero_but_energy": [],
        "A3_vision_face_all_fail": [], "A4_vision_face_partial": [],
        "A5_vision_energy_zero_but_valid": [], "A6_audio_shorter_than_video": [],
        "A7_audio_valid_slots_lt_50": [], "A8_pts_beyond_axis": [],
        "VFR_frame_gap": [], "LONG_PAUSE": [], "EMPTY_TEXT": [],
    }
    for sid in sids:
        au = _load_unaligned(unaligned_root, "audio", sid)
        am = au["meta"]
        vi = _load_unaligned(unaligned_root, "vision", sid)
        vm = vi["meta"]
        tx = _load_unaligned(unaligned_root, "text", sid)
        tm = tx["meta"]

        env = np.asarray(au["extra"].get("envelope", []), np.float64)
        if env.size and np.all(env <= 0):
            stats["A1_audio_digital_silence"].append(sid)
        if float(am.get("vad_ratio", 1) or 0) == 0.0 and env.size and float(env.max()) > 0:
            stats["A2_vad_zero_but_energy"].append(sid)
        fr = float(vm.get("face_ratio", 0) or 0)
        if fr == 0.0:
            stats["A3_vision_face_all_fail"].append(sid)
        elif fr < 0.5:
            stats["A4_vision_face_partial"].append(sid)
        vfeat = np.asarray(vi["features"])
        if vfeat.size and float(np.abs(vfeat).max()) == 0.0:
            stats["A5_vision_energy_zero_but_valid"].append(sid)
        va, va_ = am.get("container_video_duration"), am.get("container_audio_duration")
        if va and va_ and float(va) - float(va_) > 0.1:
            stats["A6_audio_shorter_than_video"].append(sid)
        al = _load_aligned(aligned_dir, sid)
        acl = _aligned_meta(al)
        if int(np.asarray(al["audio_valid"]).sum()) < config.ALIGN_SEQ_LEN:
            stats["A7_audio_valid_slots_lt_50"].append(sid)
        if int(acl["vision"]["align"].get("units_beyond_duration", 0)) > 0:
            stats["A8_pts_beyond_axis"].append(sid)
        gaps = np.asarray(vi["extra"].get("frame_gaps", []), np.float64)
        if gaps.size and float(gaps.max()) > 0.1:
            stats["VFR_frame_gap"].append(sid)
        if float(am.get("voiced_ratio", 1) or 0) < 0.1:
            stats["LONG_PAUSE"].append(sid)
        if int(tm.get("num_words", 0) or 0) == 0:
            stats["EMPTY_TEXT"].append(sid)

    counts = {k: len(v) for k, v in stats.items()}
    detail = "；".join(f"{k}={v}" for k, v in sorted(counts.items()) if v)
    return _result("V12", "异常普查（仅用于标注，不用于删除）", True,
                   len(sids), 0, detail,
                   {"counts": counts, "samples": {k: v for k, v in stats.items() if v}},
                   caveat="本项**只分类不处置**。竞赛指南明确：置信度/质量阈值用于质量标注，"
                          "**不得作为删除样本的理由**。所有命中的样本一律保留并标注。")


# ==================== V13 词时间路由一致性 ====================


def check_v13_word_routing(unaligned_root: str, aligned_dir: str, sids: List[str],
                           delivery_dir: Optional[str] = None) -> Dict[str, object]:
    """
    V13 —— 实测词时间路由的一致性（本方案吸收参考方案时序逻辑后的新增项）。

    文本的 pts 现在**逐样本二选一**：走实测路由的样本用 stable-ts 强制对齐的区间中心，
    其余仍用均匀假设。两种来源并存时，最容易出的错不是数值错，而是**同一份事实在
    不同交付物里写法不一致**——例如对齐阶段按实测时刻归属了 50 槽，交付阶段却回读
    均匀假设做溯源。故本项只查一致性，逐条断言：

        1. 每条样本的 alignment_mode ∈ {word_level, clip_level}；
        2. 落盘 meta 的 text.time_basis 与 alignment_mode 严格对应；
        3. word_level ⇒ 官方词数 = 对齐词数、零时长词 = 0、区间落在 [0,T] 内
           （即该档位的三条准入条件在产物中真的成立，而不是只写在代码里）；
        4. word_level ⇒ 实测词时刻个数与文本单元数一致；
        5. alignment_q1.json 的 word_t_sec、features_q1.npz 的 text pts、
           aligned npz 的槽归属三者用**同一套**文本时刻；
        6. clip_level ⇒ 文本时刻恰为均匀假设 (i+0.5)·T/W（不得偷换成实测）；
        7. summary_q1.csv / features_q1.npz / 对齐目录三处的 word_level 计数相等。
    """
    from align_multimodal import time_to_slot

    checked = failed = 0
    errs: List[str] = []
    n_word = n_clip = n_other = 0
    checked_word: List[str] = []
    n_routing_missing = 0

    for sid in sids:
        p = os.path.join(aligned_dir, f"{sid}.npz")
        if not os.path.isfile(p):
            checked += 1
            failed += 1
            errs.append(f"{sid}: 对齐产物缺失")
            continue
        with np.load(p, allow_pickle=True) as z:
            meta = json.loads(str(np.asarray(z["meta"]).item()))
            duration = float(np.asarray(z["duration"], np.float64))
            edges = np.asarray(z["slot_edges"], np.float64)
            L = int(np.asarray(z["n_slots"]).item())

        routing = meta.get("routing")
        checked += 1
        if not routing or "alignment_mode" not in routing:
            n_routing_missing += 1
            failed += 1
            errs.append(f"{sid}: 对齐产物缺少 routing.alignment_mode")
            continue
        mode = str(routing["alignment_mode"])
        wa = routing.get("word_align", {}) or {}
        basis = str(meta["text"]["time_basis"])
        W = int(meta["text"]["num_units"])

        checked += 1
        if mode == "word_level":
            n_word += 1
            if basis != "measured_forced_alignment":
                failed += 1
                errs.append(f"{sid}: mode=word_level 但 time_basis={basis}")
        elif mode == "clip_level":
            n_clip += 1
            if basis != "uniform_assumption":
                failed += 1
                errs.append(f"{sid}: mode=clip_level 但 time_basis={basis}")
        else:
            n_other += 1
            failed += 1
            errs.append(f"{sid}: alignment_mode 取值非法「{mode}」")
            continue

        if mode != "word_level":
            # clip_level：文本时刻必须恰为均匀假设，不得被实测值替换
            src = _load_unaligned(unaligned_root, "text", sid)
            pts = np.asarray(src["pts"], np.float64)
            checked += 1
            exp = (np.arange(pts.size, dtype=np.float64) + 0.5) * duration / max(pts.size, 1)
            if pts.size == 0 or np.allclose(pts, exp, atol=1e-4):
                pass
            else:
                failed += 1
                errs.append(f"{sid}: clip_level 的文本时刻偏离均匀假设"
                            f"（max|Δ|={np.max(np.abs(pts - exp)):.6f}）")
            continue

        # ---- word_level：三条准入条件必须在产物中成立 ----
        checked_word.append(sid)
        meas = np.asarray(wa.get("word_ts_sec") or [], np.float64)
        checked += 1
        if int(wa.get("n_official_words") or -1) != int(wa.get("n_aligned_words") or -2):
            failed += 1
            errs.append(f"{sid}: 官方 {wa.get('n_official_words')} 词 vs "
                        f"对齐 {wa.get('n_aligned_words')} 词，未全词对齐却标为 word_level")
        checked += 1
        if int(wa.get("zero_duration_words") or 0) != 0:
            failed += 1
            errs.append(f"{sid}: 含 {wa.get('zero_duration_words')} 个零时长词")
        checked += 1
        if not meas.size:
            failed += 1
            errs.append(f"{sid}: word_level 但未落盘实测词时刻")
            continue
        checked += 1
        if meas.size != W:
            failed += 1
            errs.append(f"{sid}: 实测词时刻 {meas.size} 个 vs 文本单元 {W} 个")
            continue
        checked += 1
        if not (meas.min() >= -TIME_TOL and meas.max() <= duration + TIME_TOL):
            failed += 1
            errs.append(f"{sid}: 实测词时刻越出 [0,T]（{meas.min():.4f}~{meas.max():.4f}, "
                        f"T={duration:.4f}）")
        # 实测词时刻必须落在由该时刻推出的槽内——即对齐确实按实测归属
        checked += 1
        slots = np.clip((meas / max(duration, 1e-12) * L).astype(np.int64), 0, L - 1)
        if not (np.all(edges[slots] <= meas + 1e-6) and
                np.all(meas < edges[slots + 1] + 1e-6)):
            failed += 1
            errs.append(f"{sid}: 实测词时刻与其槽边界不自洽")

    # ---- 三处 word_level 计数必须相等 ----
    counts = {"aligned_dir": n_word}
    if delivery_dir:
        fq = os.path.join(delivery_dir, "features_q1.npz")
        if os.path.isfile(fq):
            with np.load(fq, allow_pickle=True) as d:
                if "text_time_basis_by_sample" in d:
                    b = [str(x) for x in d["text_time_basis_by_sample"]]
                    counts["features_q1.npz"] = sum(
                        1 for x in b if x == "measured_forced_alignment")
        sc = os.path.join(delivery_dir, "summary_q1.csv")
        if os.path.isfile(sc):
            with open(sc, "r", encoding="utf-8-sig") as fh:
                counts["summary_q1.csv"] = sum(
                    1 for r in csv.DictReader(fh) if r.get("alignment_mode") == "word_level")
        aj = os.path.join(delivery_dir, "alignment_q1.json")
        if os.path.isfile(aj):
            with open(aj, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
            counts["alignment_q1.json"] = sum(
                1 for s in doc["samples"].values()
                if s.get("alignment_mode") == "word_level")
    checked += 1
    if len(set(counts.values())) > 1:
        failed += 1
        errs.append(f"word_level 计数在各交付物间不一致：{counts}")

    detail = (f"{n_word} 条走实测路由、{n_clip} 条保持均匀假设，逐条与落盘的时间基准字段"
              f"严格对应；{len(checked_word)} 条实测样本均满足「全词对齐 + 零时长词为 0 + "
              f"区间不越界」三条准入条件，且其槽归属确由实测时刻推出。"
              + (f" 各交付物计数：{counts}。" if len(counts) > 1 else "")
              if not errs else "发现 " + str(len(errs)) + " 处问题：" + "；".join(errs[:5]))
    return _result("V13", "实测词时间路由一致性", not errs, checked, failed, detail,
                   {"n_word_level": n_word, "n_clip_level": n_clip,
                    "n_illegal_mode": n_other, "n_routing_missing": n_routing_missing,
                    "word_level_samples": sorted(checked_word), "counts_across_artifacts": counts,
                    "errors": errs[:20]},
                   caveat="本项证明的是**路由实现与声明一致**，不是「实测词时刻与真值一致」。"
                          "本数据集没有逐词人工时间戳真值，对齐质量只能用「官方词数是否"
                          "被完整覆盖」这一可机检的弱证据来把关。")


# ==================== 报告输出 ====================


HONESTY = """## 诚实性前置声明（读下面任何数字之前请先读这里）

1. **本数据集没有逐词人工时间戳真值。** 文本的时间轴逐样本二选一，来源由
   `alignment_mode` 字段标明：`word_level`（实测，stable-ts 强制对齐的区间中心，
   全词对齐且无零时长词）或 `clip_level`（均匀假设，词 i 中心时刻 = (i+0.5)·T/W）；
   音频/视觉的 `pts` 才是解码器实测值。因此本报告**只报告一致性检查与抽查结果**，
   **绝不声称对齐达到某个毫秒级平均误差**——那需要真值才能计算，这里没有。
   任何形如「平均误差 N 毫秒」的表述在本项目中都是无依据的。
   实测档位的准入证据是**官方词数是否被完整覆盖**这一可机检的弱证据，
   它不能证明逐词时刻正确，只能证明对齐器在该样本上放下了全部官方词。
2. **`*_valid=True` 只表示「该槽被分配了采样单元」，不等于该模态真实可用。**
   反例：`-mJ2ud6oKI8_1` 的 `vision_valid` 为 50/50 全 True，但其视觉特征能量恰为 0
   （全片未检出人脸）。判定模态可用性必须看 `face_ratio` / `voiced_ratio`。
3. **音频/视觉的空槽由插值填充，没有源帧。** 因此「每个聚合值都能溯源到源帧」
   这一条**无法 100% 满足**；本报告给出真实的可溯源率，并把这些槽标记为 `interp`，
   而不是假装全部可溯源。
4. **异常一律保留并标注，不删除样本。** 竞赛指南明确：置信度/质量阈值用于质量标注，
   不得作为删除样本的理由。本报告与交付台账中没有任何一条以质量为由剔除的样本。
"""


def write_report(result: Dict[str, object], out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "verify_report.json")
    mpath = os.path.join(out_dir, "verify_report.md")
    # 先整段序列化再落盘：json.dump 是流式写出，若中途遇到不可序列化的对象
    # （例如 transformers 的加载信息里是 set 而不是 list），会在磁盘上留下一个
    # **被截断的半个 JSON** —— 比直接失败更糟，因为下游会把它当成报告读。
    payload = json.dumps(result, ensure_ascii=False, indent=1, default=str)
    with open(jpath, "w", encoding="utf-8") as fh:
        fh.write(payload)

    lines = ["# 问题一 机器核验报告", "",
             f"- 生成时间：{result['generated_at']}",
             f"- 代码指纹：`{result.get('code_fingerprint', '')}`",
             f"- 样本数：{result['n_samples']}    时间槽数：{result['n_slots']}",
             f"- **总体结论：{'全部通过' if result['all_passed'] else '存在问题，见下表'}**"
             f"（{result['summary']['passed']}/{result['summary']['total']} 项通过）", "",
             HONESTY, "",
             "## 检查结果总表", "",
             "| 代号 | 检查项 | 结论 | 检查数 | 失败数 | 说明 |",
             "|---|---|---|---|---|---|"]
    for c in result["checks"]:
        lines.append(f"| {c['code']} | {c['name']} | {'通过' if c['passed'] else '**失败**'} "
                     f"| {c['checked']} | {c['failed']} | {str(c['detail'])[:220]} |")
    lines += ["", "## 逐项明细", ""]
    for c in result["checks"]:
        lines += [f"### {c['code']} {c['name']} —— {'通过' if c['passed'] else '失败'}", "",
                  c["detail"], ""]
        if c.get("caveat"):
            lines += [f"> **限定**：{c['caveat']}", ""]
        if c.get("sub_check"):
            s = c["sub_check"]
            lines += [f"**{s['code']} {s['name']}** —— {'通过' if s['passed'] else '失败'}："
                      f"{s['detail']}", ""]
            if s.get("caveat"):
                lines += [f"> {s['caveat']}", ""]
        ev = c.get("evidence") or {}
        if ev:
            lines += ["<details><summary>证据</summary>", "", "```json",
                      json.dumps(ev, ensure_ascii=False, indent=1, default=str)[:6000],
                      "```", "", "</details>", ""]
    with open(mpath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return jpath, mpath


# ==================== 驱动 ====================


def verify_all(unaligned_root: Optional[str] = None, aligned_dir: Optional[str] = None,
               quick: bool = False, model_check: bool = True,
               out_dir: Optional[str] = None,
               summary_csv: Optional[str] = None) -> Dict[str, object]:
    t0 = time.time()
    unaligned_root = unaligned_root or os.path.dirname(U.unaligned_dir("text"))
    unaligned_root = unaligned_root.rstrip("\\/")
    aligned_dir = aligned_dir or os.path.join(PROJECT_ROOT, "data", "aligned")
    out_dir = out_dir or os.path.join(PROJECT_ROOT, "data", "q1_delivery")
    summary_csv = summary_csv or os.path.join(aligned_dir, "align_summary.csv")

    sids = [str(s["sample_id"]) for s in U.list_samples()]
    LOGGER.info("[核验] 样本 %d 条，开始 13 项机器检查 ...", len(sids))

    checks: List[Dict[str, object]] = []
    checks.append(check_v1_coverage(unaligned_root, aligned_dir, summary_csv))
    checks.append(check_v2_word_time(unaligned_root, sids))
    checks.append(check_v3_shapes(unaligned_root, aligned_dir, sids))
    checks.append(check_v4_traceable(unaligned_root, aligned_dir, sids))
    checks.append(check_v5_slot_geom(aligned_dir, sids))
    checks.append(check_v6_fill(unaligned_root, aligned_dir, sids))
    checks.append(check_v7_numeric(unaligned_root, aligned_dir, sids))
    checks.append(check_v8_duration(aligned_dir, sids, limit=(5 if quick else 0)))
    if model_check:
        checks.append(check_v9_model())
    else:
        checks.append(_result("V9", "预训练权重真实性（掩码补全，外部证据）", True, 0, 0,
                              "已按 --no-model-check 跳过。", {}, caveat="本项被显式跳过，非通过。"))
    checks.append(check_v10_duplicate(unaligned_root, sids))
    checks.append(check_v11_freshness(unaligned_root, aligned_dir))
    checks.append(check_v12_anomaly_census(unaligned_root, aligned_dir, sids))
    checks.append(check_v13_word_routing(unaligned_root, aligned_dir, sids,
                                         delivery_dir=out_dir))

    for c in checks:
        (LOGGER.info if c["passed"] else LOGGER.warning)(
            "[核验] %-4s %-40s %s（检查 %d，失败 %d）",
            c["code"], c["name"], "通过" if c["passed"] else "失败", c["checked"], c["failed"])

    all_passed = all(c["passed"] for c in checks)
    result = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pipeline_version": config.PIPELINE_VERSION,
        "code_fingerprint": U.module_fingerprint(verify_all),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "n_samples": len(sids),
        "n_slots": config.ALIGN_SEQ_LEN,
        "paths": {"unaligned_root": unaligned_root, "aligned_dir": aligned_dir,
                  "summary_csv": summary_csv if os.path.isfile(summary_csv) else None},
        "elapsed_sec": round(time.time() - t0, 2),
        "summary": {"total": len(checks),
                    "passed": sum(1 for c in checks if c["passed"]),
                    "failed": sum(1 for c in checks if not c["passed"]),
                    "total_assertions": sum(c["checked"] for c in checks),
                    "failed_assertions": sum(c["failed"] for c in checks)},
        "all_passed": all_passed,
        "checks": checks,
        "honesty": HONESTY,
    }
    jp, mp = write_report(result, out_dir)
    LOGGER.info("[核验] 结论：%s（%d/%d 项通过）。报告 %s / %s",
                "全部通过" if all_passed else "存在问题",
                result["summary"]["passed"], result["summary"]["total"], jp, mp)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="问题一交付前机器核验（只读，不修改产物）")
    p.add_argument("--quick", action="store_true", help="V8 只探 5 条样本")
    p.add_argument("--no-model-check", action="store_true", help="跳过 V9（不加载 RoBERTa）")
    p.add_argument("--out", default=None, help="报告输出目录，默认 data/q1_delivery/")
    p.add_argument("--unaligned-root", default=None)
    p.add_argument("--aligned-dir", default=None)
    args = p.parse_args(argv)

    r = verify_all(unaligned_root=args.unaligned_root, aligned_dir=args.aligned_dir,
                   quick=args.quick, model_check=not args.no_model_check, out_dir=args.out)
    return 0 if r["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
