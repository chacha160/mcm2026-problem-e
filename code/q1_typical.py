# -*- coding: utf-8 -*-
"""
q1_typical.py —— 五类典型样本的人工核对包（问题一「手工抽查」那一项）

题目要求人工抽查覆盖**五类**样本，每类给出「词—秒—语音段—视频帧」的对应关系：

    长句 / 短句 / 停顿明显 / 多人或远景 / 面部检测失败

本模块的作用是把「挑哪条、为什么挑它」变成**可复查的规则**（写进
`typical_samples.csv` 的判据列），再复用 `timeline_visualize` 出图与出对应表。
这样评审看到的不是「作者手选的一条」，而是「按明确极值判据自动选出的一条」。

诚实性
------
「多人或远景」这一类依赖人脸探测：本项目只对每条样本**均匀抽 3 帧**做纯检测，
所以 `max_face_count` 是**抽样下界**，不能声称是全片最大人脸数。
若 3 帧抽样未发现多人（本批即如此），报告必须直说是**以远景/小脸实例呈现**，
而不是硬造一个「多人」样本充数。探测失败或超时的样本写「未探测到」，
**不因此剔除样本**，该类的选样退化为 `face_ratio` 证据。
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
import unaligned_common as U  # noqa: E402
from utils import LOGGER  # noqa: E402

MODALITIES = ("text", "audio", "vision")

CATEGORIES = ("长句", "短句", "停顿明显", "多人或远景", "面部检测失败")


def _candidates(unaligned_root: str, aligned_dir: str, sids: List[str]
                ) -> List[Dict[str, object]]:
    """逐样本收集五类判据所需的全部统计量（逐条读、用完即关）。"""
    rows: List[Dict[str, object]] = []
    for sid in sids:
        au = U.load_unaligned(os.path.join(unaligned_root, "audio", f"{sid}.npz"))
        tm = U.load_unaligned(os.path.join(unaligned_root, "text", f"{sid}.npz"))["meta"]
        am = au["meta"]
        vm = U.load_unaligned(os.path.join(unaligned_root, "vision", f"{sid}.npz"))["meta"]
        with np.load(os.path.join(aligned_dir, f"{sid}.npz"), allow_pickle=True) as z:
            duration = float(np.asarray(z["duration"], np.float64))
        # 音轨包络：判「该样本有没有音频内容」的唯一依据（与 q1_verify.py 的 A1 同口径，
        # 全为 0 即数字静音）。**不能**用 `voiced_ratio==0` 代替——
        # 数字静音与「有能量但 VAD 判零」（A2）的 voiced_ratio 都是 0，
        # 前者是**没有音频**，后者是**有音频但都不成浊音**，混为一谈会选错典型样本。
        env = np.asarray(au["extra"].get("envelope", []), np.float64)
        rows.append({
            "sample_id": sid,
            "num_words": int(tm.get("num_words", 0) or 0),
            "duration_sec": round(duration, 3),
            "voiced_ratio": float(am.get("voiced_ratio", 0) or 0),
            "vad_ratio": float(am.get("vad_ratio", 0) or 0),
            "face_ratio": float(vm.get("face_ratio", 0) or 0),
            "vision_frames": int(vm.get("num_frames", 0) or 0),
            "audio_silent": bool(env.size and np.all(env <= 0)),
            "audio_has_content": bool(env.size and float(env.max()) > 0),
        })
    return rows


def pick_typical(unaligned_root: str, aligned_dir: str, sids: List[str],
                 face_probe: Optional[Dict[str, Dict[str, object]]] = None
                 ) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """
    按**明确极值判据**为五类各选一条，并尽量不重复同一条样本。

    返回 (选中列表, 该类别的取证说明)。
    """
    rows = _candidates(unaligned_root, aligned_dir, sids)
    probe = face_probe or {}
    notes: Dict[str, object] = {}
    picked: List[Dict[str, object]] = []
    used: set = set()

    def take(pool: List[Dict[str, object]], rule: str) -> Optional[Dict[str, object]]:
        """按 `pool` 已排好的顺序取第一条未被占用的样本。"""
        for r in pool:
            if r["sample_id"] not in used:
                used.add(r["sample_id"])
                return r
        return pool[0] if pool else None

    # 1) 长句 —— 词数最多
    pool = sorted(rows, key=lambda r: (-r["num_words"], r["sample_id"]))
    r = take(pool, "词数最多")
    if r:
        picked.append({**r, "category": "长句",
                       "criterion": "文本词数最多（num_words 最大）",
                       "criterion_value": f"num_words={r['num_words']}, duration={r['duration_sec']}s"})

    # 2) 短句 —— 词数最少，且优先时长短（短句的槽宽更细，能看出文本大量空槽）
    pool = sorted(rows, key=lambda r: (r["num_words"], r["duration_sec"], r["sample_id"]))
    r = take(pool, "词数最少")
    if r:
        picked.append({**r, "category": "短句",
                       "criterion": "文本词数最少（num_words 最小），同词数取时长短者",
                       "criterion_value": f"num_words={r['num_words']}, duration={r['duration_sec']}s"})

    # 3) 停顿明显 —— 有声占比最低，且时长足够长以容纳停顿结构。
    #
    # **候选池必须先剔除数字静音样本**：`voiced_ratio` 最小的样本往往是音轨
    # 完全没有内容的 A1 样本（如 -mJ2ud6oKI8_1，包络恒为 0）。它的 voiced_ratio=0
    # 不是「说话断断续续」，而是「根本没有音频」——拿它当「停顿明显」的例子，
    # 图上会呈现一条平直的空音轨，评审只会读成「这组数据没有声音」，
    # 而真正的「停顿明显」（有内容、有声有停）反而没被展示。
    # 注意：这里**不是删样本**——该样本仍在 100 行汇总、特征与异常台账里，
    # 只是不当「停顿」这一类别的示例（它在第 6 节以 A1/A3/A5 的身份被如实标注）。
    with_audio = [r for r in rows if r.get("audio_has_content")]
    silent = [r for r in rows if r.get("audio_silent")]
    if len(with_audio) < 1:
        LOGGER.warning("[典型] 没有任何样本的音频有内容，「停顿明显」无法回避数字静音样本")
        with_audio = rows
    pool = sorted(with_audio, key=lambda r: (r["voiced_ratio"], -r["duration_sec"],
                                             r["sample_id"]))
    r = take(pool, "voiced_ratio 最小")
    if r:
        picked.append({**r, "category": "停顿明显",
                       "criterion": "在有音频内容的样本中，有声帧占比最小"
                                    "（voiced_ratio 最小），同值取时长更长者；"
                                    "数字静音样本不参与本类别（判据见 README 第 7 节）",
                       "criterion_value": f"voiced_ratio={r['voiced_ratio']}, "
                                          f"duration={r['duration_sec']}s"})
    notes["n_audio_silent"] = len(silent)
    notes["audio_silent_samples"] = [r["sample_id"] for r in silent]

    # 4) 多人或远景 —— 优先人脸探测的「框面积占比」极值
    probed = [r for r in rows if str(probe.get(r["sample_id"], {}).get("detector_ok", 0)) == "1"
              or probe.get(r["sample_id"], {}).get("detector_ok") == 1]
    multi = []
    if probed:
        multi = [r for r in probed
                 if int(probe[r["sample_id"]].get("max_face_count", 0) or 0) >= 2]
    if multi:
        pool = sorted(multi, key=lambda r: (-int(probe[r["sample_id"]]["max_face_count"]),
                                            r["sample_id"]))
        r = take(pool, "3 帧抽样中最多人脸数最大")
        criterion = "3 帧抽样中检出的人脸数最大（max_face_count 最大）"
        cval = (f"max_face_count={probe[r['sample_id']]['max_face_count']}, "
                f"框面积占比={float(probe[r['sample_id']]['max_face_area_ratio']):.4f}")
        notes["multi_person_found"] = True
    else:
        # 抽样未发现多人：退化为「远景/小脸」——框面积占比最小的已探测样本
        cand = probed or rows
        if probed:
            pool = sorted(cand, key=lambda r: (float(probe[r["sample_id"]]["max_face_area_ratio"]),
                                               r["sample_id"]))
            cval = (f"框面积占比最小="
                    f"{float(probe[pool[0]['sample_id']]['max_face_area_ratio']):.4f}"
                    f"（3 帧抽样），face_ratio={pool[0]['face_ratio']}")
        else:
            pool = sorted(cand, key=lambda r: (r["face_ratio"], r["sample_id"]))
            cval = f"人脸探测不可用，退化为 face_ratio 最小={pool[0]['face_ratio']}"
        r = take(pool, "远景/小脸")
        criterion = ("**本批 3 帧抽样未发现多人场景**，故该类别以「远景/小脸」实例呈现，"
                     "判据为抽样框面积占比最小（见 README 的诚实说明）")
        notes["multi_person_found"] = False
    if r:
        picked.append({**r, "category": "多人或远景", "criterion": criterion,
                       "criterion_value": cval})

    # 5) 面部检测失败 —— face_ratio 最小（0 即全失败）。
    #
    # 同值（都为 0）时**优先取其余模态正常的样本**。理由：本批有样本是
    # 「音轨数字静音 + 全片无人脸」两个模态一起坏（-mJ2ud6oKI8_1），
    # 拿它当「面部检测失败」的例子，图上从语音到视觉全是空的，
    # 评审看不出究竟是哪一种模态失效，也**看不出这个类别想说明什么**。
    # 而「音频正常、只有人脸检测失败」的样本（-ri04Z7vwnc_0）是一个
    # **单模态失效的对照**：同一段素材里语音时间轴清清楚楚、视觉一片空白，
    # 一眼就能看出是视觉模态的问题。这类样本才是这一类别该展示的对象。
    pool = sorted(rows, key=lambda r: (r["face_ratio"],
                                       0 if r.get("audio_has_content") else 1,
                                       -r["duration_sec"], r["sample_id"]))
    r = take(pool, "face_ratio 最小")
    if r:
        picked.append({**r, "category": "面部检测失败",
                       "criterion": "人脸检出比例最小（face_ratio 最小，0 表示全部帧失败）；"
                                    "同值时优先取其余模态正常的样本（单模态失效对照）",
                       "criterion_value": f"face_ratio={r['face_ratio']}, "
                                          f"vision_frames={r['vision_frames']}, "
                                          f"voiced_ratio={r['voiced_ratio']}"})

    notes["n_probed"] = len(probed)
    notes["n_unprobed"] = len(rows) - len(probed)
    notes["probe_sample"] = (
        "人脸探测只对每条样本均匀抽取 3 帧做 MTCNN 纯检测，"
        "因此 max_face_count 是**抽样下界**，不是全片最大人脸数；"
        "探测器与特征提取所用的 MTCNN 同参同权重，仅 keep_all 不同。")
    return picked, notes


def build_typical_pack(unaligned_root: str, aligned_dir: str, out_dir: str,
                       sids: List[str],
                       face_probe: Optional[Dict[str, Dict[str, object]]] = None,
                       make_figures: bool = True) -> Dict[str, object]:
    """
    产出 `typical_samples.csv` + 五类图（figures/）+ 逐槽对应表（correspondence/）。
    """
    picked, notes = pick_typical(unaligned_root, aligned_dir, sids, face_probe)
    fig_dir = os.path.join(out_dir, "figures")
    cor_dir = os.path.join(out_dir, "correspondence")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(cor_dir, exist_ok=True)

    # 先清掉上一次运行留下的图与对应表。**必须清**：五类是「按极值现选」的，
    # 判据或数据一变，选中的样本就变；旧图不会被覆盖，而是**留在目录里**，
    # 于是交付里出现第六张图、第六张表，指向一个已经不属于任何类别的样本。
    # 本项目真踩过：修正「停顿明显」的候选池后重跑，`correspondence/` 里
    # 同时留着新旧两个样本，6 张表配 5 个类别。
    # 只删本模块生成的两种文件名（`*_timeline.png` / `*_correspondence.csv`），
    # 不整个清空目录，以免误删人工放进去的东西。
    for d, suffix in ((fig_dir, "_timeline.png"), (cor_dir, "_correspondence.csv")):
        for f in sorted(os.listdir(d)):
            if f.endswith(suffix):
                try:
                    os.remove(os.path.join(d, f))
                except OSError as exc:
                    LOGGER.warning("[典型] 清理旧文件失败 %s：%s", f, exc)

    figures: List[str] = []
    correspondences: List[str] = []
    if make_figures:
        import timeline_visualize as TV
        TV.setup_chinese_font()
        for r in picked:
            sid = str(r["sample_id"])
            try:
                fig = TV.make_figure(sid, unaligned_root, aligned_dir,
                                     video_path=U.video_path_of(*sid.rsplit("_", 1)),
                                     out_dir=fig_dir)
                if fig:
                    figures.append(fig)
            except Exception as exc:
                LOGGER.warning("[典型] %s 出图失败（不阻断）：%s: %s",
                               sid, type(exc).__name__, exc)
            try:
                cor = TV.export_correspondence(sid, unaligned_root, aligned_dir, cor_dir)
                if cor:
                    correspondences.append(cor)
                    _annotate_correspondence(cor, unaligned_root, aligned_dir, sid)
            except Exception as exc:
                LOGGER.warning("[典型] %s 对应表失败（不阻断）：%s: %s",
                               sid, type(exc).__name__, exc)

    csv_path = os.path.join(out_dir, "typical_samples.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=["category", "sample_id", "official_id",
                                           "criterion", "criterion_value", "num_words",
                                           "duration_sec", "voiced_ratio", "face_ratio",
                                           "correspondence_csv"])
        w.writeheader()
        by_sid = {r["sample_id"]: r for r in picked}
        for r in picked:
            sid = str(r["sample_id"])
            w.writerow({
                "category": r["category"], "sample_id": sid,
                "official_id": U.official_id(*sid.rsplit("_", 1)),
                "criterion": r["criterion"], "criterion_value": r["criterion_value"],
                "num_words": r["num_words"], "duration_sec": r["duration_sec"],
                "voiced_ratio": r["voiced_ratio"], "face_ratio": r["face_ratio"],
                "correspondence_csv": next(
                    (os.path.relpath(c, out_dir).replace("\\", "/") for c in correspondences
                     if os.path.basename(c).startswith(sid)), ""),
            })
    LOGGER.info("[典型] 写出 typical_samples.csv（%d 类）", len(picked))
    return {"csv": csv_path, "picked": picked, "notes": notes,
            "figures": figures, "correspondences": correspondences}


def _annotate_correspondence(path: str, unaligned_root: str, aligned_dir: str,
                             sid: str) -> None:
    """
    给 `export_correspondence` 的逐槽表**补三列**，使交付要求的「原词、字符范围、
    词元索引」与「填充及有效掩码」在同一张表上完整可查。

    原表（由 timeline_visualize 维护）已含：slot / t_start_sec / t_end_sec /
    text_words / text_valid / audio_valid / audio_unit_range / audio_t_range /
    vision_valid / vision_unit_range / vision_pts_range
    ——「词—秒—语音段—视频帧」四要素其实已经齐了。本函数只补原表没有的：
        char_span      该槽内的词在**规范化文本**中的字符区间
        token_index    该槽内的词占用的 input_ids 下标（分号分隔）
        fill_flag      该槽是否由插值填充（无源帧），以及是哪个模态
    纯加法，不改原表任何一列的含义。
    """
    tx = U.load_unaligned(os.path.join(unaligned_root, "text", f"{sid}.npz"))
    ex = tx["extra"]
    spans = ex.get("char_spans", [])
    spans_o = ex.get("char_spans_original", [])
    toks = ex.get("token_index", [])

    with np.load(os.path.join(aligned_dir, f"{sid}.npz"), allow_pickle=True) as z:
        meta = json.loads(str(np.asarray(z["meta"]).item()))
        valid = {m: np.asarray(z[f"{m}_valid"]) for m in MODALITIES}
        counts = {m: np.asarray(meta[m]["align"]["unit_counts"], np.int64)
                  for m in MODALITIES}
        slot_idx = meta["text"]["slot_unit_index"]

    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
        names = list(rows[0].keys()) if rows else []

    def fmt(v, j):
        if 0 <= j < len(v):
            return "-".join(str(int(x)) for x in v[j])
        return ""

    for r in rows:
        k = int(r["slot"])
        wi_list = [int(x) for x in slot_idx[k]]
        r["char_span"] = " / ".join(fmt(spans, j) for j in wi_list if fmt(spans, j))
        r["char_span_original"] = " / ".join(fmt(spans_o, j) for j in wi_list
                                             if fmt(spans_o, j))
        r["token_index"] = ";".join(",".join(str(t) for t in toks[j])
                                    for j in wi_list if j < len(toks))
        filled = [m for m in MODALITIES if int(counts[m][k]) == 0 and not bool(valid[m][k])]
        r["fill_flag"] = ("插值填充(无源帧):" + ",".join(filled)) if filled else ""

    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=names + ["char_span", "char_span_original",
                                                   "token_index", "fill_flag"])
        w.writeheader()
        w.writerows(rows)
