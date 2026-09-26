# -*- coding: utf-8 -*-
"""
align_multimodal.py —— 三模态「按秒对齐」模块（本文件是新文件，承接三个未对齐提取器）

总体方案
--------
    unaligned_text / unaligned_audio / unaligned_vision 三个提取器
    各自产出「特征 (T,D) + 每单元时间戳 pts」，
    它们的时间轴彼此独立、采样率也不同（文本 0 Hz 无实测时间、
    音频 20 Hz、视觉 15 Hz）。

    本模块只做一件事：**把三条独立时间轴统一到同一根「片段秒轴」上**，
    再离散成 L=50 个时间槽，输出与附件2 完全同构的 (50, D) 张量。

对齐规则（一句话）
------------------
    以片段真实时长 T 为公共时间轴，把 [0, T] 等分为 L 个槽（槽宽 = T/L）；
    每个未对齐单元按其**实测时间戳**归属到所在槽，同槽内单元特征取**均值**即该槽特征；
    空槽按策略补齐并在 valid_mask 中标记。文本因无逐词时间戳，
    采用「词在 [0,T] 内均匀分布」的显式假设（meta.time_basis 会如实标注）。

为什么必须用「秒」而不是「数组下标」
------------------------------------
    三个模态的采样率不同（20 Hz vs 15 Hz vs 词），片段时长也不同（3~29 s）。
    若按数组下标等分（既有 utils.resample_feature_sequence 的做法），
    1 个音频帧与 1 个视觉帧被强行当作等长，会把 50 ms 与 66.7 ms 混为一谈，
    且丢帧/VFR 造成的间隔不均会被完全抹平。按秒归属则天然正确。

槽 ↔ 时间 的可逆映射（问题3 定位证据的前提）
-------------------------------------------
    槽是 [0,T] 上的等分区间，因此映射是闭式的、双向可逆：
        slot -> time : [k*T/L, (k+1)*T/L)
        time -> slot : min(L-1, floor(t*L/T))
    每个样本的 T 随文件落盘，故拿到任一槽号即可还原其真实时间区间；
    反之给定时间戳也能定位到唯一槽。这正是问题3「找出模型关注的时间片段」
    能投影回问题1 时间轴上的依据。

    比「闭式公式」更强的是**溯源**：每槽同时记录「哪些未对齐单元落进了它」，
    于是 槽 → 单元下标 → 该单元的原文词 / 该帧的时间戳 这条链是完整的，
    可以一路追到原始证据（见 extra.slot_unit_index）。

输出
----
    data/aligned/<sample_id>.npz   单样本：三模态 (50,D) + valid_mask + 长度 + 槽时间
    data/aligned/aligned_50.npz    汇总：附件2 口径，data[modality] 为 (N,50,D) 定长张量 + lengths
    data/aligned/slot_time_map.csv 槽↔时间映射与每槽来源单元数（问题1 要求的时间轴对应表）
    data/aligned/align_summary.csv 每样本每模态：原始有效时长 / 单元数 / 特征维度 / 有效槽数
"""

from __future__ import annotations

import argparse
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
import unaligned_common as U  # noqa: E402
from utils import LOGGER  # noqa: E402

MODALITIES = ("text", "audio", "vision")

# 空槽补齐策略（按模态的疏密特性给不同默认值，理由见下方 _FILL_DEFAULT 注释）
_FILL_DEFAULT = {
    # 文本：一段 5 s 的话只有十几个词，天然填不满 50 槽。
    # 没有词的槽「就是没有文本内容」，补零并在 valid_mask 标 False 才诚实，
    # 若插值会把相邻词的语义抹到空白处，制造出原文并不存在的「词」。
    "text": "zero",
    # 音频/视觉是稠密信号（20 Hz / 15 Hz 远快于 10 Hz 槽率），
    # 空槽只可能出现在信号短于片段之处，用插值补齐不引入虚假语义。
    "audio": "interpolate",
    "vision": "interpolate",
}


# ==================== 时间轴与槽 ====================


def slot_edges(duration: float, n_slots: int) -> np.ndarray:
    """把 [0, duration] 等分为 n_slots 个槽，返回 n_slots+1 个边界时刻。"""
    return np.linspace(0.0, float(duration), int(n_slots) + 1, dtype=np.float64)


def slot_centers(duration: float, n_slots: int) -> np.ndarray:
    """各槽中心时刻（用于按时间插值的方法）。"""
    e = slot_edges(duration, n_slots)
    return (e[:-1] + e[1:]) / 2.0


def time_to_slot(t, duration: float, n_slots: int) -> np.ndarray:
    """时间戳 → 槽号（闭式，可逆）。边界外按最近端截断。"""
    t = np.asarray(t, dtype=np.float64)
    idx = np.floor(t / float(duration) * n_slots).astype(np.int64)
    return np.clip(idx, 0, n_slots - 1)


def slot_to_interval(k: int, duration: float, n_slots: int) -> Tuple[float, float]:
    """槽号 → 时间区间（闭式，可逆）。time_to_slot 的逆。"""
    e = slot_edges(duration, n_slots)
    return float(e[k]), float(e[k + 1])


# ==================== 核心：单模态对齐 ====================


def align_series(features: np.ndarray, pts: np.ndarray, duration: float,
                 n_slots: int, method: str = "time_bin",
                 fill: str = "interpolate") -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    把一个模态的未对齐序列对齐到 n_slots 个槽。

    Args:
        features: (T, D) float32
        pts:      (T,)  每个单元的时间戳（秒）
        duration: 该片段的公共时间轴长度 T（三模态共用同一个值）
        n_slots:  槽数 L，默认 50
        method:   "time_bin" —— 按时间戳归属 + 同槽均值（默认，尊重实际采样）
                  "interp"   —— 把特征视作时间的函数，线性插值到槽中心
        fill:     空槽处理 "zero" / "interpolate" / "nearest"

    Returns:
        aligned (L, D) float32
        valid   (L,)   bool，该槽是否由真实单元直接支撑（插值补齐的槽为 False）
        meta    本步统计：来源单元数、空槽数、每槽单元数等
    """
    L, D = int(n_slots), (features.shape[1] if features.ndim == 2 else 0)
    aligned = np.zeros((L, D), dtype=np.float32)
    valid = np.zeros(L, dtype=bool)
    counts = np.zeros(L, dtype=np.int64)
    slot_units: List[List[int]] = [[] for _ in range(L)]

    T_ = features.shape[0] if features.ndim == 2 else 0
    if T_ == 0 or D == 0 or duration <= 0:
        return aligned, valid, {"num_units": int(T_), "empty_slots": L, "unit_counts": counts.tolist(),
                                "slot_unit_index": slot_units, "note": "输入为空或时长无效，整段置零",
                                "units_clipped_to_last_slot": 0, "units_clipped_to_first_slot": 0,
                                "units_beyond_duration": 0}

    pts = np.asarray(pts, dtype=np.float64).reshape(-1)
    if pts.shape[0] != T_:
        raise ValueError(f"pts 长度 {pts.shape[0]} 与特征帧数 {T_} 不一致")

    # ---- 越界记账（A8）----
    # time_to_slot 用 np.clip(floor(t/T*L), 0, L-1) 把轴外的单元并进端点槽。这是有意为之
    # （并入端点总比丢弃单元诚实），但必须显式计数而不是让它静默发生。
    # 实测有 5 条样本的视觉末帧 PTS 超出公共时间轴 0.033~0.067 s。
    #
    # 这里直接复用 time_to_slot 的算式来判定，而不是另立判据（如 t > T）：
    # pts 以 float32 落盘，在 T≈9.3 处 float32 的舍入步长已达 9.5e-7，用 t > T 会把
    # 「精度表示误差」误报成「越界」。按 floor 的结果判定则与真实归类严格一致。
    raw_idx = np.floor(pts / float(duration) * L).astype(np.int64)
    n_clipped = int((raw_idx > L - 1).sum())      # 右越界，被并入末槽
    n_underflow = int((raw_idx < 0).sum())        # 左越界，被并入首槽
    # 另报一个与算法无关的物理事实：pts 严格超出 T 且超出量远大于 float32 精度（1e-5 s，
    # 比实测最小越界量 0.033 s 小三个数量级），供评审交叉核对。
    n_beyond = int((pts > duration + 1e-5).sum())

    if method == "time_bin":
        slot_of = time_to_slot(pts, duration, L)
        acc = np.zeros((L, D), dtype=np.float64)
        np.add.at(acc, slot_of, features.astype(np.float64))
        np.add.at(counts, slot_of, 1)
        nz = counts > 0
        aligned[nz] = (acc[nz] / counts[nz][:, None]).astype(np.float32)
        valid[nz] = True
        for u, s in enumerate(slot_of):
            slot_units[int(s)].append(int(u))
    elif method == "interp":
        order = np.argsort(pts, kind="stable")
        p_sorted, f_sorted = pts[order], features[order]
        # 去重：重复时间戳会让 np.interp 结果依赖顺序
        keep = np.concatenate([[True], np.diff(p_sorted) > 1e-9])
        p_sorted, f_sorted = p_sorted[keep], f_sorted[keep]
        centers = slot_centers(duration, L)
        for d in range(D):
            aligned[:, d] = np.interp(centers, p_sorted, f_sorted[:, d]).astype(np.float32)
        # 插值结果是「构造」出来的：只有落在采样时间范围内的槽才算有效
        valid = (centers >= p_sorted[0]) & (centers <= p_sorted[-1])
        slot_of = time_to_slot(pts, duration, L)
        np.add.at(counts, slot_of, 1)
        for u, s in enumerate(slot_of):
            slot_units[int(s)].append(int(u))
    else:
        raise ValueError(f"未知对齐方法 {method}，可选 time_bin / interp")

    # ---- 空槽补齐 ----
    empty = ~valid
    n_empty = int(empty.sum())
    if n_empty and fill != "zero":
        idx = np.arange(L)
        filled_from_neighbor = False
        if fill == "interpolate" and valid.sum() >= 2:
            for d in range(D):
                aligned[empty, d] = np.interp(idx[empty], idx[valid], aligned[valid, d]).astype(np.float32)
            filled_from_neighbor = True
        elif fill == "nearest" or (fill == "interpolate" and valid.sum() >= 1):
            nearest = idx[valid][np.argmin(np.abs(idx[empty][:, None] - idx[valid][None, :]), axis=1)]
            aligned[empty] = aligned[nearest]
            filled_from_neighbor = True
        if not filled_from_neighbor:
            LOGGER.debug("[对齐] 无有效槽可参考，空槽保持零向量（L=%d, 空槽=%d）", L, n_empty)

    meta = {
        "method": method,
        "fill": fill,
        "num_units": int(T_),
        "empty_slots": n_empty,
        "valid_slots": int(valid.sum()),
        "valid_ratio": round(float(valid.mean()), 4),
        "units_per_slot_mean": round(float(counts.mean()), 3),
        "units_per_slot_max": int(counts.max()),
        "unit_counts": counts.tolist(),
        "slot_unit_index": slot_units,
        # A8 越界记账（见上方注释）。三者口径不同，分开报以免被混为一谈：
        #   units_clipped_to_last_slot  被 time_to_slot 并入末槽的单元数（算法后果，权威）
        #   units_clipped_to_first_slot 被并入首槽的单元数
        #   units_beyond_duration       pts 实测越出 T 且超出量 > 1e-5 s（物理事实，与算法无关）
        "units_clipped_to_last_slot": n_clipped,
        "units_clipped_to_first_slot": n_underflow,
        "units_beyond_duration": n_beyond,
    }
    return aligned, valid, meta


# ==================== 单样本三模态对齐 ====================


def resolve_duration(loaded: Dict[str, Dict[str, object]], policy: str = "video") -> float:
    """
    确定该片段的公共时间轴长度 T。

    policy="video"  : 取提取阶段写入的 ffprobe 视频流时长（最稳，与容器一致）
    policy="max_pts": 取三模态末帧时间戳的最大值（信号口径，能覆盖尾帧）

    注意不要用 cv2 的 帧数/fps：本批 mp4 该值虚高 1.5~2.6 倍（见 unaligned_common 文档）。

    本函数只读未对齐特征文件里已落盘的元数据，**不需要重新探测原始视频**——
    时长在提取阶段就已经由 ffprobe 测准并写进 meta，对齐阶段直接复用即可，
    这样「对齐」这一步是自洽的：给定未对齐特征文件就能独立重跑。
    """
    last_pts = [float(d["pts"][-1]) for d in loaded.values() if len(d["pts"])]
    if policy == "max_pts" and last_pts:
        return float(max(last_pts))

    # 视觉/音频提取阶段都会写入 duration_used（= ffprobe 视频流时长）
    for m in ("vision", "audio", "text"):
        meta = loaded.get(m, {}).get("meta", {})
        for key in ("duration_used", "container_video_duration"):
            try:
                v = float(meta.get(key) or 0.0)
            except (TypeError, ValueError):
                v = 0.0
            if v > 0:
                return v
    return float(max(last_pts)) if last_pts else 0.0


def resolve_text_pts(sample_id: str, n_units: int,
                     word_align_dir: Optional[str] = None
                     ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, object]]]:
    """
    取文本模态该用的时间戳：实测优先，否则 None（调用方保持原均匀假设时间）。

    只有证据路由判为 word_level 的样本才返回实测词时间。判据与理由见
    ``word_align.decide_mode``；此处额外校验词数与该样本文本特征行数一致，
    不一致说明词对齐记录与未对齐特征不同源（例如只重跑了其中一步），
    此时宁可回退均匀假设，也不能让时间戳与特征行错位。
    """
    import word_align as WA

    rec = WA.load_word_align(sample_id, word_align_dir)
    if not rec:
        return None, None
    if rec.get("mode") != WA.MODE_WORD_LEVEL:
        return None, rec
    ts = rec.get("word_ts_sec") or []
    if len(ts) != int(n_units):
        LOGGER.warning("[对齐] %s 实测词数 %d 与文本特征行 %d 不符，回退均匀假设",
                       sample_id, len(ts), int(n_units))
        return None, rec
    return np.asarray(ts, dtype=np.float64), rec


def align_sample(sample_id: str, unaligned_root: str, n_slots: int = 50,
                 method: str = "time_bin", duration_policy: str = "video",
                 fills: Optional[Dict[str, str]] = None,
                 word_align_dir: Optional[str] = None,
                 word_align_enabled: bool = True) -> Optional[Dict[str, object]]:
    """
    对一条样本做三模态对齐。

    三模态共用同一个 duration，因此共用同一根秒轴——这是「共享时间轴」的落地方式。

    文本模态的时间戳按证据路由决定来源（见 word_align 模块）：
        word_level  —— 用 stable-ts 强制对齐得到的实测词区间中心
        clip_level  —— 保持均匀假设 t_j=(j+0.5)*T/W
    两条路径都只改「词落在哪个槽」，不改特征、不改槽数与维度，故
    附件2/3/4 的接口形状与问题二/三的输入完全不受影响。
    """
    fills = fills or _FILL_DEFAULT
    loaded: Dict[str, Dict[str, object]] = {}
    for m in MODALITIES:
        path = os.path.join(unaligned_root, m, f"{sample_id}.npz")
        if not os.path.isfile(path):
            LOGGER.warning("[对齐] 缺少 %s 模态文件，跳过该样本: %s", m, path)
            return None
        loaded[m] = U.load_unaligned(path)

    duration = resolve_duration(loaded, duration_policy)
    if duration <= 0:
        LOGGER.error("[对齐] %s 无法确定片段时长，跳过", sample_id)
        return None

    # ---- 文本时间戳来源：实测词时间优先 ----
    n_text_units = int(loaded["text"]["features"].shape[0])
    text_pts_measured, wa_rec = (None, None)
    if word_align_enabled:
        try:
            text_pts_measured, wa_rec = resolve_text_pts(sample_id, n_text_units, word_align_dir)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("[对齐] %s 读取词对齐记录失败，回退均匀假设: %s", sample_id, exc)

    edges = slot_edges(duration, n_slots)
    out: Dict[str, object] = {"sample_id": sample_id, "duration": float(duration),
                              "slot_edges": edges, "duration_policy": duration_policy,
                              "modalities": {}}
    for m in MODALITIES:
        d = loaded[m]
        pts = np.asarray(d["pts"], np.float32)
        time_basis = d["meta"].get("time_basis", "unknown")
        if m == "text" and text_pts_measured is not None:
            pts = text_pts_measured
            time_basis = "measured_forced_alignment"
        aligned, valid, meta = align_series(np.asarray(d["features"], np.float32),
                                           pts, duration, n_slots, method, fills.get(m, "zero"))
        out["modalities"][m] = {
            "aligned": aligned, "valid": valid,
            "num_units": int(d["features"].shape[0]),
            "feature_dim": int(d["features"].shape[1]) if d["features"].ndim == 2 else 0,
            "src_duration": float(d["pts"][-1]) if len(d["pts"]) else 0.0,
            "time_basis": time_basis,
            "extra": d["extra"],
            "align_meta": meta,
        }

    # ---- 证据路由与词对齐摘要（落进交付物，供核验与论文引用）----
    align_mode = (wa_rec or {}).get("mode") or "not_run"
    out["alignment_mode"] = align_mode
    out["word_align"] = {
        "mode": align_mode,
        "text_time_basis": out["modalities"]["text"]["time_basis"],
        "n_official_words": (wa_rec or {}).get("n_official_words"),
        "n_aligned_words": (wa_rec or {}).get("n_aligned_words"),
        "zero_duration_words": (wa_rec or {}).get("zero_duration_words"),
        "coverage": (wa_rec or {}).get("coverage"),
        "first_t": (wa_rec or {}).get("first_t"),
        "last_t": (wa_rec or {}).get("last_t"),
        "aligner": (wa_rec or {}).get("aligner"),
        "aligner_version": (wa_rec or {}).get("aligner_version"),
        "whisper_model": (wa_rec or {}).get("whisper_model"),
        "interval_convention": (wa_rec or {}).get("interval_convention"),
        # 实测词时刻落进对齐产物，使交付阶段「只读 unaligned + aligned」即可
        # 复现 slot_src_pts / word_t_sec，无需再依赖 data/word_align/ 目录。
        "word_ts_sec": (list(wa_rec.get("word_ts_sec") or [])
                        if align_mode == "word_level" else []),
    }
    return out


# ==================== 落盘 ====================


def save_sample_aligned(out_dir: str, res: Dict[str, object]) -> str:
    """保存单样本对齐结果：三模态 (50,D) + valid_mask + 槽边界。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{res['sample_id']}.npz")
    # duration 与 slot_edges 必须存 float64：槽归属 slot_unit_index 是用 float64 的
    # duration 算出来的，若这里降成 float32，读回文件后重算归属会在「恰好落在槽边界」
    # 的单元上差一个槽（实测 12026 个非空槽里有 39 个不可复现，全为短句文本）。
    # 特征数组与 valid 掩码仍为 float32/bool，不受影响。
    payload = {"sample_id": np.array(str(res["sample_id"])),
               "duration": np.array(float(res["duration"]), np.float64),
               "slot_edges": np.asarray(res["slot_edges"], np.float64),
               "n_slots": np.array(int(len(res["slot_edges"]) - 1))}
    meta_all: Dict[str, object] = {}
    for m in MODALITIES:
        md = res["modalities"][m]
        payload[f"{m}_features"] = np.asarray(md["aligned"], np.float32)
        payload[f"{m}_valid"] = np.asarray(md["valid"], bool)
        meta_all[m] = {"num_units": md["num_units"], "feature_dim": md["feature_dim"],
                       "src_duration": md["src_duration"], "time_basis": md["time_basis"],
                       "align": {k: v for k, v in md["align_meta"].items() if k != "slot_unit_index"},
                       "slot_unit_index": md["align_meta"]["slot_unit_index"]}
    # 证据路由写在顶层 routing 键下（与 text/audio/vision 并列），
    # 使「这条样本的文本时间戳是实测还是假设」无需读 word_align 目录即可判定。
    meta_all["routing"] = {"alignment_mode": res.get("alignment_mode", "not_run"),
                           "align_version": config.ALIGN_VERSION,
                           "word_align": res.get("word_align", {})}
    payload["meta"] = np.array(json.dumps(meta_all, ensure_ascii=False))
    np.savez_compressed(path, **payload)
    return path


def save_aggregate(out_dir: str, results: List[Dict[str, object]]) -> str:
    """
    汇总成附件2 口径的定长张量。

    附件2 的约定：data[modality] 为 (N, 50, D) 的填充张量，长度另存一个列表
    （对齐后长度恒为 50，但若某些样本某模态为空仍需要单独记录）。
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "aligned_50.npz")
    payload: Dict[str, object] = {
        "sample_id": np.array([str(r["sample_id"]) for r in results]),
        # 与单样本文件口径一致，见 save_sample_aligned 的注释
        "duration": np.array([float(r["duration"]) for r in results], np.float64),
        "n_slots": np.array(int(len(results[0]["slot_edges"]) - 1) if results else 0),
    }
    for m in MODALITIES:
        dims = [r["modalities"][m]["feature_dim"] for r in results]
        D = max(dims) if dims else 0
        N = len(results)
        arr = np.zeros((N, payload["n_slots"], D), np.float32)
        mask = np.zeros((N, payload["n_slots"]), bool)
        for i, r in enumerate(results):
            a = r["modalities"][m]["aligned"]
            arr[i, :, :a.shape[1]] = a
            mask[i] = r["modalities"][m]["valid"]
        payload[f"{m}_features"] = arr
        payload[f"{m}_valid"] = mask
        payload[f"{m}_features_dim"] = np.array(D)
        payload[f"{m}_lengths"] = np.array([payload["n_slots"]] * N, np.int64)
    np.savez_compressed(path, **payload)
    return path


def save_slot_time_map(out_dir: str, results: List[Dict[str, object]], n_slots: int) -> str:
    """
    槽↔时间映射表（问题1 要求「给出文本片段/语音时间范围/视频帧的对应关系」）。

    逐样本逐槽给出：槽号、时间区间、以及三模态各自有多少单元落在该槽。
    有了这张表，问题3 定位到的槽号可以直接读回真实时间区间。
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "slot_time_map.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "duration_sec", "slot_index",
                    "t_start_sec", "t_end_sec", "t_center_sec",
                    "text_units", "audio_units", "vision_units",
                    "text_valid", "audio_valid", "vision_valid"])
        for r in results:
            dur = float(r["duration"])
            edges = np.asarray(r["slot_edges"], np.float64)
            for k in range(n_slots):
                row = [r["sample_id"], round(dur, 4), k,
                       round(float(edges[k]), 4), round(float(edges[k + 1]), 4),
                       round(float((edges[k] + edges[k + 1]) / 2), 4)]
                vals = []
                for m in MODALITIES:
                    md = r["modalities"][m]
                    vals.append(len(md["align_meta"]["slot_unit_index"][k]))
                row += vals
                row += [int(r["modalities"][m]["valid"][k]) for m in MODALITIES]
                w.writerow(row)
    return path


def save_summary_csv(out_dir: str, results: List[Dict[str, object]]) -> str:
    """问题1 要求的汇总表：样本 ID / 模态 / 原始有效时长 / 特征维度 / 对齐粒度。"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "align_summary.csv")
    rows = []
    for r in results:
        sid, dur = str(r["sample_id"]), float(r["duration"])
        for m in MODALITIES:
            md = r["modalities"][m]
            n_slots = len(r["slot_edges"]) - 1
            rows.append({
                "sample_id": sid,
                # 必须按**最后一个**下划线切分：本批有 **4 个 video_id 自身就含下划线**
                # （-I_e4mIh0yE、-tANM6ETl_M、-wMB_hJL-3o、-lzEya4AM_4），这 4 个 video_id
                # 下辖 **6 条样本**，用 replace("_", "$_$", 1) 会把 video_id 内部的第一个
                # 下划线当成分隔符，写出 -I$_$e4mIh0yE_1（应为 -I_e4mIh0yE$_$1），共波及 18/300 行。
                # ⚠️「4」是 video_id 数、「6」是样本数，两者别混——本注释曾写成「6 个 video_id」。
                "official_id": U.official_id(*sid.rsplit("_", 1)),
                "modality": m,
                "clip_duration_sec": round(dur, 3),                 # 公共时间轴长度
                "modality_span_sec": round(float(md["src_duration"]), 3),  # 该模态实际覆盖时长
                "raw_units": int(md["num_units"]),                  # 未对齐单元数
                "feature_dim": int(md["feature_dim"]),
                "valid_slots": int(np.asarray(md["valid"]).sum()),
                "n_slots": int(n_slots),
                "align_granularity_sec": round(dur / n_slots, 4),    # 对齐粒度 = 槽宽
                "units_per_slot": round(float(md["align_meta"]["units_per_slot_mean"]), 3),
                "time_basis": md["time_basis"],
            })
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


# ==================== 驱动 ====================


def run(samples: List[Dict[str, object]], unaligned_root: str, out_dir: str,
        n_slots: int = config.ALIGN_SEQ_LEN, method: str = "time_bin",
        duration_policy: str = "video", fills: Optional[Dict[str, str]] = None,
        overwrite: bool = False, word_align_dir: Optional[str] = None,
        word_align_enabled: bool = True) -> List[Dict[str, object]]:
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict[str, object]] = []
    n_fail = 0
    n_word_level = 0
    for i, s in enumerate(samples, 1):
        sid = str(s["sample_id"])
        try:
            res = align_sample(sid, unaligned_root, n_slots, method, duration_policy, fills,
                               word_align_dir=word_align_dir,
                               word_align_enabled=word_align_enabled)
            if res is None:
                n_fail += 1
                continue
            save_sample_aligned(out_dir, res)
            results.append(res)
            n_word_level += int(res.get("alignment_mode") == "word_level")
            LOGGER.info("[对齐] (%d/%d) %s 时长=%.2fs 槽宽=%.4fs | 有效槽 文本%d/语音%d/视觉%d | %s",
                        i, len(samples), sid, float(res["duration"]),
                        float(res["duration"]) / n_slots,
                        int(res["modalities"]["text"]["valid"].sum()),
                        int(res["modalities"]["audio"]["valid"].sum()),
                        int(res["modalities"]["vision"]["valid"].sum()),
                        res.get("alignment_mode"))
        except Exception as exc:
            n_fail += 1
            LOGGER.error("[对齐] (%d/%d) 失败 %s: %s: %s", i, len(samples), sid, type(exc).__name__, exc)

    if not results:
        LOGGER.error("[对齐] 没有任何样本成功，跳过汇总输出")
        return results

    LOGGER.info("[对齐] 汇总输出 ...")
    LOGGER.info("[对齐]   %s", save_aggregate(out_dir, results))
    LOGGER.info("[对齐]   %s", save_slot_time_map(out_dir, results, n_slots))
    LOGGER.info("[对齐]   %s", save_summary_csv(out_dir, results))
    LOGGER.info("[对齐] 完成：成功 %d / 失败 %d / 槽数 %d / 方法 %s | 实测词时间 %d 条，均匀假设 %d 条",
                len(results), n_fail, n_slots, method, n_word_level, len(results) - n_word_level)
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="三模态按秒对齐（未对齐特征 → (50,D) 定长张量）")
    p.add_argument("--only", default=None, help="只处理 sample_id 含该子串的样本")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 条")
    p.add_argument("--slots", type=int, default=config.ALIGN_SEQ_LEN, help="时间槽数，默认 50")
    p.add_argument("--method", choices=["time_bin", "interp"], default="time_bin",
                   help="time_bin=按时间戳归属+均值池化（默认）；interp=按时间插值")
    p.add_argument("--duration", choices=["video", "max_pts"], default="video",
                   help="公共时间轴取 ffprobe 视频流时长（默认）或各模态末帧最大值")
    p.add_argument("--fill-text", choices=["zero", "interpolate", "nearest"], default=_FILL_DEFAULT["text"])
    p.add_argument("--fill-av", choices=["zero", "interpolate", "nearest"], default="interpolate",
                   help="音频与视觉的空槽补齐策略")
    p.add_argument("--unaligned-root", default=None, help="未对齐特征根目录，默认 data/unaligned_features")
    p.add_argument("--out", default=None, help="输出目录，默认 data/aligned")
    p.add_argument("--word-align-dir", default=None,
                   help="词对齐记录目录，默认 data/word_align（由 code/word_align.py 生成）")
    p.add_argument("--no-word-align", action="store_true",
                   help="关闭实测词时间，文本模态一律用均匀假设（消融/对照用）")
    args = p.parse_args(argv)

    root = args.unaligned_root or os.path.join(os.path.dirname(_CODE_DIR), "data", "unaligned_features")
    out_dir = args.out or os.path.join(os.path.dirname(_CODE_DIR), "data", "aligned")
    samples = U.list_samples(only=args.only, limit=args.limit)
    LOGGER.info("[对齐] 待处理样本 %d 条；未对齐根目录 %s", len(samples), root)
    run(samples, root, out_dir, n_slots=args.slots, method=args.method,
        duration_policy=args.duration,
        fills={"text": args.fill_text, "audio": args.fill_av, "vision": args.fill_av},
        word_align_dir=args.word_align_dir,
        word_align_enabled=not args.no_word_align)
    return 0


if __name__ == "__main__":
    sys.exit(main())
