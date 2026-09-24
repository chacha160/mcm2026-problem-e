# -*- coding: utf-8 -*-
"""
timeline_visualize.py —— 三模态共享时间轴可视化（本文件是新文件）

用途
----
    把一条样本的三种模态画在**同一根秒轴**上，直接回答题目「在同一时间轴上展示
    文本片段 / 语音时间范围 / 视频帧 以及三类特征对应关系」：

        图顶   视频帧条    ：在各自实测时刻贴出该时刻的真实视频帧
        第二段 语音振幅    ：由 100 Hz RMS 包络画出的振幅曲线，底色标出有声区间
        第三段 文本片段    ：每个词按其实测/假设时刻排布，分泳道避免重叠
        第四段 对齐槽      ：3×50 格，显示各模态每个槽是否被真实单元支撑
        第五段 三类特征    ：三模态对齐后的 (50, D) 特征热力图，与上面各段共用 x 轴
        底部   x 轴刻度（秒），并贯穿全图画出 50 个对齐槽的边界

    因此「某一秒」是一条竖线，穿过视频帧、振幅、词、槽和全部三维特征——
    这就是「共享时间轴」的可视化落地。

可选的额外输出
-------------
    --correspondence  同时导出一份逐槽对应表 CSV：
                      槽号 / 时间区间 / 该槽的词 / 音频帧时刻 / 视觉帧时刻

典型样本自动选取
---------------
    --pick typical  不指定 sample_id，自动挑一条「最能说明问题」的样本：
                    文本非空、人脸检出率高、有声比例适中、时长接近中位数。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import unaligned_common as U  # noqa: E402
from utils import LOGGER  # noqa: E402

MODALITIES = ("text", "audio", "vision")
MODALITY_CN = {"text": "文本", "audio": "语音", "vision": "视觉"}


# ==================== 字体与画图基础 ====================


def setup_chinese_font() -> bool:
    """
    配置中文字体。图中含中文标签（模态名、轴名），若字体缺失会显示成方框。

    返回是否成功设置到某个明确存在的中文字体。字体缺失时不报错、只告警，
    因为即使退化成英文标签，图的信息量也不受影响。
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt

    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("Microsoft YaHei", "SimHei", "DengXian", "SimSun",
                 "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS"):
        if cand in names:
            plt.rcParams["font.sans-serif"] = [cand, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            LOGGER.info("[可视化] 中文字体: %s", cand)
            return True
    LOGGER.warning("[可视化] 未找到中文字体，图中中文可能显示为方框（不影响数据）")
    plt.rcParams["axes.unicode_minus"] = False
    return False


# ==================== 数据装载 ====================


def load_vision_frames(video_path: str, target_times: List[float]
                       ) -> Tuple[List[Tuple[float, np.ndarray]], Dict[str, object]]:
    """
    单遍流式解码视频，取出最接近 target_times 的若干真实帧。

    为什么要「单遍流式 + 就近取」而不是按时刻 seek：
        按时刻 seek 依赖容器索引，本批 mp4 的索引元数据不可靠（帧数虚高）；
        流式解码的 PTS 才是解码器实测值。为了不让内存随视频长度增长，
        只保留需要的几帧，其余帧读完即丢。
    """
    if not target_times:
        return [], {}
    want = sorted(target_times)
    best: List[Optional[Tuple[float, np.ndarray, float]]] = [None] * len(want)
    picked: Dict[float, np.ndarray] = {}
    for pts, frame in U.iter_video_frames(video_path, target_fps=None):
        for i, t in enumerate(want):
            d = abs(pts - t)
            cur = best[i]
            if cur is None or d < cur[2]:
                best[i] = (float(pts), frame, d)
    out: List[Tuple[float, np.ndarray]] = []
    for i, item in enumerate(best):
        if item is None:
            continue
        # 同一时刻可能被多个目标同时选中，去重避免重复贴图
        key = round(item[0], 4)
        if key in picked:
            out.append((item[0], picked[key]))
        else:
            picked[key] = item[1]
            out.append((item[0], item[1]))
    return out, {}


def voiced_intervals(pts: np.ndarray, vad: np.ndarray, hop: float) -> List[Tuple[float, float]]:
    """把逐帧 VAD 标志压成连续的有声时间区间，用于在振幅图上打底色。"""
    if len(vad) == 0:
        return []
    flag = np.asarray(vad) > 0.5
    spans: List[Tuple[float, float]] = []
    start = None
    for i, f in enumerate(flag):
        if f and start is None:
            start = pts[i]
        elif not f and start is not None:
            spans.append((float(start), float(pts[i - 1] + hop)))
            start = None
    if start is not None:
        spans.append((float(start), float(pts[-1] + hop)))
    return spans


def _pack_lanes(words: List[str], times: List[float], duration: float,
                char_w: float = 0.011, pad: float = 0.012) -> Tuple[List[int], int]:
    """
    给每个词分配一条「泳道」，使同一泳道内相邻词的时间区间不重叠。

    词在时间轴上的宽度按字符数估算（英文词长与渲染宽度近似线性）。
    贪心：把词放进第一条「上一个词的右边界已过」的泳道，否则新开一条。
    """
    lanes: List[float] = []      # 每条泳道当前的右边界（秒）
    assign: List[int] = []
    for w, t in zip(words, times):
        width = char_w * max(len(w), 2) + pad
        left = max(0.0, t - width / 2)
        right = min(duration, t + width / 2)
        placed = False
        for li, r in enumerate(lanes):
            if left >= r + 0.004:
                lanes[li] = right
                assign.append(li)
                placed = True
                break
        if not placed:
            lanes.append(right)
            assign.append(len(lanes) - 1)
    return assign, max(1, len(lanes))


def _downsample_cols(mat: np.ndarray, target_cols: int = 80) -> np.ndarray:
    """把 (L, D) 沿特征维平均池化到 target_cols 列，便于热力图显示而不失真过多。"""
    L, D = mat.shape
    if D <= target_cols:
        return mat
    nb = target_cols
    edges = np.linspace(0, D, nb + 1).astype(int)
    out = np.zeros((L, nb), np.float32)
    for j in range(nb):
        a, b = edges[j], max(edges[j] + 1, edges[j + 1])
        out[:, j] = mat[:, a:b].mean(axis=1)
    return out


# ==================== 主图 ====================


def make_figure(sample_id: str, unaligned_root: str, aligned_dir: str,
                video_path: Optional[str] = None, n_thumbs: int = 8,
                rep_time: float = 0.35, dpi: int = 150,
                out_dir: Optional[str] = None) -> str:
    """生成一条样本的三模态共享时间轴图，返回 PNG 路径。"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # ---- 读未对齐 ----
    ut = U.load_unaligned(os.path.join(unaligned_root, "text", f"{sample_id}.npz"))
    ua = U.load_unaligned(os.path.join(unaligned_root, "audio", f"{sample_id}.npz"))
    uv = U.load_unaligned(os.path.join(unaligned_root, "vision", f"{sample_id}.npz"))
    apath = os.path.join(aligned_dir, f"{sample_id}.npz")
    al = np.load(apath, allow_pickle=False)
    duration = float(al["duration"])
    L = int(al["n_slots"])
    edges = np.asarray(al["slot_edges"], np.float64)

    words = list(ut["extra"].get("words", []))
    word_t = np.asarray(ut["pts"], np.float64)
    env = np.asarray(ua["extra"].get("envelope", []), np.float64)
    env_t = np.asarray(ua["extra"].get("envelope_times", []), np.float64)
    vad = np.asarray(ua["features"], np.float32)[:, 73] if ua["features"].shape[1] >= 74 else np.zeros(0)
    hop = float(ua["meta"].get("hop_length", 800)) / float(ua["meta"].get("sample_rate", 16000))
    spans = voiced_intervals(np.asarray(ua["pts"], np.float64), vad, hop)

    vis_pts = np.asarray(uv["pts"], np.float64)
    face_flags = np.asarray(uv["extra"].get("face_flags", []), bool)

    # ---- 取视频帧 ----
    if video_path is None:
        video_path = U.video_path_of(*sample_id.rsplit("_", 1))
    thumbs: List[Tuple[float, np.ndarray]] = []
    if os.path.isfile(video_path):
        # 均匀分布的帧（避开首尾各 5%，避免黑帧），再视间距决定是否插入代表帧。
        # 必须保证相邻取帧时刻的间隔不小于缩略图宽度，否则两张图会互相压住看不清——
        # 这是「按秒贴图」的固有约束：时刻挨得近，图就必然重叠。
        t0, t1 = 0.05 * duration, 0.95 * duration
        targets = [float(t) for t in np.linspace(t0, t1, n_thumbs)]
        rep = rep_time * duration
        min_gap = (t1 - t0) / max(n_thumbs - 1, 1) * 0.9
        if all(abs(rep - t) > min_gap for t in targets):
            targets.append(rep)
        targets.sort()
        thumbs, _ = load_vision_frames(video_path, targets)
    else:
        LOGGER.warning("[可视化] 找不到视频，跳过视频帧条: %s", video_path)

    # ---- 文本泳道（须在建画布前算好，才能按泳道数决定面板高度）----
    lanes, n_lanes = _pack_lanes(words, [float(x) for x in word_t], duration)

    # ---- 画布 ----
    # 文本面板高度随泳道数变化：只有 1 条泳道时留 1.15 的固定高度会大片空白。
    text_h = min(1.15, 0.5 + 0.3 * n_lanes)
    fig = plt.figure(figsize=(15.0, 11.6), dpi=dpi)
    gs = GridSpec(5, 1, height_ratios=[1.0, 1.25, text_h, 0.75, 2.1], hspace=0.16,
                  left=0.075, right=0.975, top=0.925, bottom=0.055)

    ax_v = fig.add_subplot(gs[0])
    ax_a = fig.add_subplot(gs[1], sharex=ax_v)
    ax_t = fig.add_subplot(gs[2], sharex=ax_v)
    ax_s = fig.add_subplot(gs[3], sharex=ax_v)
    ax_f = fig.add_subplot(gs[4], sharex=ax_v)

    def slot_lines(ax):
        """画出 50 槽边界。每槽都画会太密，故只在每 5 槽画一条，另用浅色画全部。"""
        for k in range(1, L):
            ax.axvline(edges[k], color="#d9d9d9", lw=0.5, zorder=0)
        for k in range(5, L, 5):
            ax.axvline(edges[k], color="#9a9a9a", lw=0.8, ls="--", zorder=0)

    # ---------- 段一：视频帧条 ----------
    # 底部留出 -0.14~0 画「抽帧齿条」，故缩略图区域取 [0, 1] 之上不再压缩
    ax_v.set_ylim(-0.14, 1.0)
    ax_v.set_yticks([])
    ax_v.set_ylabel("视频帧", fontsize=11)
    if thumbs:
        pos = ax_v.get_position()
        fig_w_in, fig_h_in = fig.get_size_inches()
        avail_w_in = pos.width * fig_w_in
        panel_h_in = pos.height * fig_h_in * (1.0 / (1.0 + 0.14))   # 齿条占了 14% 高度
        # 缩略图宽度要被「相邻帧的间距」夹住，否则 9 张图会互相压在一起看不清。
        # 先按 16:9 和面板高度算理想宽度，再与 1/n 的间距上限取小，最后反推高度保持比例。
        w_frac_aspect = (0.80 * panel_h_in * 16.0 / 9.0) / avail_w_in
        w_frac = min(w_frac_aspect, 0.88 / max(len(thumbs), 1))
        h_frac = min(0.86, w_frac * avail_w_in * 9.0 / 16.0 / panel_h_in)
        y0 = (1.0 - h_frac) / 2.0
        for t, img in thumbs:
            xc = min(max(t / duration, w_frac / 2), 1 - w_frac / 2)
            try:
                iax = ax_v.inset_axes([xc - w_frac / 2, y0, w_frac, h_frac])
                iax.imshow(img[:, :, ::-1])          # BGR -> RGB
                iax.set_xticks([]); iax.set_yticks([])
                for sp in iax.spines.values():
                    sp.set_edgecolor("#333333"); sp.set_linewidth(0.6)
                iax.set_title(f"{t:.2f}s", fontsize=6.5, pad=1.2)
            except Exception as exc:
                LOGGER.debug("[可视化] 贴帧失败 t=%.2f: %s", t, exc)
        ax_v.axvline(rep_time * duration, color="#d62728", lw=1.6, zorder=5)
        # 标签放在面板底部与「抽帧齿条」同一行，避开缩略图上方的时刻标注
        ax_v.text(duration, 0.015, f"红色竖线 = 代表帧 {rep_time * duration:.2f}s",
                  transform=ax_v.transData, color="#d62728", fontsize=7.5,
                  ha="right", va="bottom",
                  bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0), zorder=6)
    else:
        ax_v.text(0.5, 0.5, "（无视频帧）", ha="center", va="center", transform=ax_v.transAxes)

    # 抽帧齿条：逐帧画出视觉模态实际采样的时刻（底部一行细刻度）。
    # 齿距均匀 = 恒定帧率；出现空档 = 解码器丢帧/VFR。人脸缺失的帧标红，
    # 让「哪些时刻没有可用人脸」在时间轴上一眼可见，而不是藏在 valid_mask 里。
    if len(vis_pts):
        n_face_missing = int((~face_flags).sum()) if len(face_flags) == len(vis_pts) else 0
        ax_v.vlines(vis_pts, -0.12, 0.0, color="#666666", lw=0.5, zorder=3)
        if n_face_missing:
            miss = vis_pts[~face_flags] if len(face_flags) == len(vis_pts) else np.zeros(0)
            ax_v.vlines(miss, -0.12, 0.0, color="#d62728", lw=0.9, zorder=4)
        ax_v.text(0.0, 0.015, f"视觉抽帧 {len(vis_pts)} 帧 @ 15 Hz"
                              + (f"，其中 {n_face_missing} 帧未检出人脸（红）" if n_face_missing
                                 else "，人脸全部检出"),
                  transform=ax_v.transData, fontsize=7.5, color="#555555",
                  va="bottom", ha="left",
                  bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0), zorder=6)
    slot_lines(ax_v)
    ax_v.set_title(
        f"样本 {sample_id}    片段时长 {duration:.2f}s    "
        f"对齐槽数 L={L}    槽宽（对齐粒度）{duration / L * 1000:.0f} ms    "
        f"三模态共享同一根秒轴",
        fontsize=12.5, pad=10)

    # ---------- 段二：语音振幅 ----------
    for i, (s, e) in enumerate(spans):
        ax_a.axvspan(s, e, color="#ffe8cc", zorder=0,
                     label="有声区间（VAD）" if i == 0 else None)
    if len(env):
        ax_a.fill_between(env_t, 0, env, color="#1f77b4", alpha=0.55, lw=0,
                          label="语音振幅包络（100 Hz RMS）")
        ax_a.plot(env_t, env, color="#0d4c73", lw=0.7)
    ax_a.set_ylabel("语音振幅", fontsize=11)
    ax_a.set_ylim(0, max(float(env.max()) * 1.35, 1e-6) if len(env) else 1.0)
    ax_a.set_yticks([])
    slot_lines(ax_a)
    ax_a.legend(loc="upper right", fontsize=8, framealpha=0.85, ncol=2)
    ax_a.text(0.005, 0.86,
              f"音频 {ua['features'].shape[0]} 帧 @ {ua['meta'].get('frame_rate_hz', 20):.0f} Hz，"
              f"覆盖 0~{float(ua['pts'][-1]):.2f}s",
              transform=ax_a.transAxes, fontsize=8, color="#0d4c73")

    # ---------- 段三：文本 ----------
    ax_t.set_ylim(-0.2, n_lanes - 0.8 + 0.4)
    ax_t.set_yticks([])
    ax_t.set_ylabel("文本词", fontsize=11)
    ax_t.set_facecolor("#fafafa")
    for w, t, ln in zip(words, word_t, lanes):
        y = n_lanes - 1 - ln
        ax_t.text(float(t), y, w, fontsize=7.5, ha="center", va="center",
                  bbox=dict(boxstyle="round,pad=0.18", fc="#fff3cd", ec="#e0b400", lw=0.5))
    slot_lines(ax_t)
    ax_t.text(0.005, 0.88,
              f"文本 {len(words)} 词（无逐词时间戳，按词在 [0,T] 内均匀分布假设定位；"
              f"该假设随文件落盘于 meta.time_basis）",
              transform=ax_t.transAxes, fontsize=8, color="#8a6d00")

    # ---------- 段四：对齐槽掩码 ----------
    # 行序：从下往上依次是 文本(y=0) / 语音(y=1) / 视觉(y=2)，
    # 刻度标签必须与「实际绘制的 y 值」对齐——这里显式按 y 值给标签，避免标签与色块错位。
    ax_s.set_ylim(-0.6, 2.6)
    ax_s.set_yticks([0, 1, 2])
    ax_s.set_yticklabels([MODALITY_CN[MODALITIES[0]], MODALITY_CN[MODALITIES[1]],
                          MODALITY_CN[MODALITIES[2]]], fontsize=10)
    for row, m in enumerate(MODALITIES):
        mask = np.asarray(al[f"{m}_valid"], bool)
        y = row
        for k in range(L):
            ax_s.add_patch(plt.Rectangle((edges[k], y - 0.34), edges[k + 1] - edges[k], 0.68,
                                         facecolor="#2b7bba" if mask[k] else "#e9e9e9",
                                         edgecolor="white", lw=0.3))
    slot_lines(ax_s)
    ax_s.set_ylabel("对齐槽", fontsize=11)
    ax_s.text(0.005, 0.88, "深色 = 该槽由真实单元直接支撑；浅色 = 空槽（已按策略补齐，valid_mask 置 False）",
              transform=ax_s.transAxes, fontsize=8, color="#333333")

    # ---------- 段五：三类对齐特征热力图 ----------
    ax_f.set_ylabel("对齐特征", fontsize=11)
    ax_f.set_yticks([])
    mats = []
    for m in MODALITIES:
        f = np.asarray(al[f"{m}_features"], np.float32)
        v = np.asarray(al[f"{m}_valid"], bool)
        fd = _downsample_cols(f, 80)
        # 逐维度标准化到 [0,1]，否则不同模态量纲差异极大，图会糊成一片
        lo, hi = np.percentile(fd, 2, axis=0), np.percentile(fd, 98, axis=0)
        nrm = np.clip((fd - lo) / np.maximum(hi - lo, 1e-9), 0, 1)
        # 无效槽用 NaN 标记：viridis 里 0.5 是正常的中间色，若用常数填会被误读成真实取值，
        # NaN 会走 cmap.set_bad 的专用颜色（浅灰），与任何真实取值都不会混淆。
        nrm[~v] = np.nan
        mats.append(nrm.T)
    gap = np.full((1, mats[0].shape[1]), np.nan, np.float32)
    stack = np.vstack([mats[0], gap, mats[1], gap, mats[2]])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e0e0e0")
    ax_f.imshow(stack, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                extent=[0, duration, 0, stack.shape[0]], interpolation="nearest")
    # 热力图各段之间标注模态名
    y0 = 0
    for m, mat in zip(MODALITIES, mats):
        ax_f.text(-0.006 * duration, y0 + mat.shape[0] / 2, f"{MODALITY_CN[m]}\n{mat.shape[0]}列",
                  transform=ax_f.transData, fontsize=9, ha="right", va="center", color="#222222")
        y0 += mat.shape[0] + 1
    ax_f.set_xlabel("时间（秒）—— 同一竖线穿过视频帧、振幅、词、槽与三类特征", fontsize=11)
    slot_lines(ax_f)

    for ax in (ax_v, ax_a, ax_t, ax_s):
        plt.setp(ax.get_xticklabels(), visible=False)
    ax_f.set_xlim(0, duration)

    out_dir = out_dir or os.path.join(os.path.dirname(_CODE_DIR), "data", "timeline")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{sample_id}_timeline.png")
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    LOGGER.info("[可视化] 已输出 %s", path)
    return path


# ==================== 逐槽对应表 ====================


def export_correspondence(sample_id: str, unaligned_root: str, aligned_dir: str,
                          out_dir: str) -> str:
    """
    导出「槽 ↔ 时间 ↔ 各模态原始单元」的对应表。

    这张表就是题目要的「对应关系」的文字形态：任给一个槽，可读出它的真实时间区间、
    落在其中的词、以及音频/视觉帧的时刻范围；反过来给定时间也能定位到槽。
    """
    import json
    ut = U.load_unaligned(os.path.join(unaligned_root, "text", f"{sample_id}.npz"))
    ua = U.load_unaligned(os.path.join(unaligned_root, "audio", f"{sample_id}.npz"))
    uv = U.load_unaligned(os.path.join(unaligned_root, "vision", f"{sample_id}.npz"))
    al = np.load(os.path.join(aligned_dir, f"{sample_id}.npz"), allow_pickle=False)
    edges = np.asarray(al["slot_edges"], np.float64)
    L = int(al["n_slots"])
    meta = json.loads(str(al["meta"]))
    words = list(ut["extra"].get("words", []))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{sample_id}_correspondence.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["slot", "t_start_sec", "t_end_sec",
                    "text_words", "text_valid", "audio_valid",
                    "audio_unit_range", "audio_t_range", "vision_valid",
                    "vision_unit_range", "vision_pts_range"])
        for k in range(L):
            t_idx = meta["text"]["slot_unit_index"][k]
            a_idx = meta["audio"]["slot_unit_index"][k]
            v_idx = meta["vision"]["slot_unit_index"][k]
            row = [k, round(float(edges[k]), 4), round(float(edges[k + 1]), 4),
                   " ".join(words[i] for i in t_idx if i < len(words)),
                   int(np.asarray(al["text_valid"], bool)[k]),
                   int(np.asarray(al["audio_valid"], bool)[k])]
            row += [f"{a_idx[0]}-{a_idx[-1]}" if a_idx else ""]
            row += [f"{float(ua['pts'][a_idx[0]]):.3f}-{float(ua['pts'][a_idx[-1]]):.3f}" if a_idx else ""]
            row += [int(np.asarray(al["vision_valid"], bool)[k])]
            row += [f"{v_idx[0]}-{v_idx[-1]}" if v_idx else ""]
            row += [f"{float(uv['pts'][v_idx[0]]):.3f}-{float(uv['pts'][v_idx[-1]]):.3f}" if v_idx else ""]
            w.writerow(row)
    LOGGER.info("[可视化] 已输出对应表 %s", path)
    return path


# ==================== 典型样本自动选取 ====================


def pick_typical(unaligned_root: str, samples: List[Dict[str, object]]) -> Optional[str]:
    """
    自动挑一条「最能说明问题」的样本：

    评分偏好：文本非空且有 20~45 个词（词多但泳道仍可读）、人脸检出率 ≥0.9、
    有声比例 0.2~0.7（既有声也有静音，振幅图才有起伏）、时长接近全体中位数。
    """
    cands = []
    for s in samples:
        sid = str(s["sample_id"])
        try:
            ut = U.load_unaligned(os.path.join(unaligned_root, "text", f"{sid}.npz"))
            ua = U.load_unaligned(os.path.join(unaligned_root, "audio", f"{sid}.npz"))
            uv = U.load_unaligned(os.path.join(unaligned_root, "vision", f"{sid}.npz"))
        except Exception:
            continue
        nw = ut["features"].shape[0]
        fr = float(uv["meta"].get("face_ratio") or 0.0)
        vr = float(ua["meta"].get("voiced_ratio") or 0.0)
        dur = float(uv["meta"].get("duration_used") or uv["meta"].get("duration_last_pts") or 0.0)
        cands.append({"sid": sid, "nw": nw, "fr": fr, "vr": vr, "dur": dur})
    if not cands:
        return None
    med = float(np.median([c["dur"] for c in cands]))
    def score(c):
        s = 0.0
        s += 1.0 if c["nw"] > 0 else -5.0
        s += 1.0 if 18 <= c["nw"] <= 48 else -abs(c["nw"] - 33) / 40
        s += 2.0 * c["fr"]
        s += 1.5 * (1 - abs(c["vr"] - 0.4) / 0.4) if c["vr"] > 0 else -1.0
        s += 1.5 * (1 - min(abs(c["dur"] - med) / max(med, 1e-6), 1.0))
        return s
    best = max(cands, key=score)
    LOGGER.info("[可视化] 自动选取典型样本 %s（词数=%d 人脸率=%.2f 有声比=%.2f 时长=%.2fs）",
                best["sid"], best["nw"], best["fr"], best["vr"], best["dur"])
    return best["sid"]


# ==================== 驱动 ====================


def _normalize_argv(argv: List[str]) -> List[str]:
    """
    把「以横线开头的样本 ID」改写成 --id=... 形式。

    本批 MOSEI 的 video_id 形如 -3g5yACwYnA（YouTube ID 可能以 '-' 开头），
    直接作为位置参数传给 argparse 会被当成选项而报 "unrecognized arguments"。
    这里只改写「以单个 - 开头、且含下划线」的 token，不会误伤真正的选项。
    """
    out: List[str] = []
    for a in argv:
        if len(a) > 2 and a[0] == "-" and a[1] != "-" and "_" in a:
            out.append(f"--id={a}")
        else:
            out.append(a)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = _normalize_argv(list(sys.argv[1:] if argv is None else argv))
    p = argparse.ArgumentParser(description="三模态共享时间轴可视化（文本词 + 语音振幅 + 视频帧 + 对齐槽）")
    p.add_argument("sample_id", nargs="?", default=None, help="样本 ID，如 -3g5yACwYnA_13")
    p.add_argument("--id", dest="sample_id_opt", default=None,
                   help="样本 ID（当 ID 以 '-' 开头时用这个传，或直接写 --id=-xxx_1）")
    p.add_argument("--pick", choices=["typical"], default=None,
                   help="不指定样本时自动选取典型样本")
    p.add_argument("--thumbs", type=int, default=8, help="视频帧条贴图数量")
    p.add_argument("--rep-time", type=float, default=0.35,
                   help="代表帧的相对时刻（0~1，默认 0.35）")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--correspondence", action="store_true", help="同时导出逐槽对应表 CSV")
    p.add_argument("--unaligned-root", default=None)
    p.add_argument("--aligned-dir", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    setup_chinese_font()
    root = args.unaligned_root or os.path.join(os.path.dirname(_CODE_DIR), "data", "unaligned_features")
    adir = args.aligned_dir or os.path.join(os.path.dirname(_CODE_DIR), "data", "aligned")
    odir = args.out or os.path.join(os.path.dirname(_CODE_DIR), "data", "timeline")

    sid = args.sample_id or args.sample_id_opt
    if not sid:
        samples = U.list_samples()
        sid = pick_typical(root, samples) if args.pick == "typical" else str(samples[0]["sample_id"])
    if not sid:
        LOGGER.error("[可视化] 无法确定要画的样本")
        return 1

    make_figure(sid, root, adir, n_thumbs=args.thumbs, rep_time=args.rep_time,
                dpi=args.dpi, out_dir=odir)
    if args.correspondence:
        export_correspondence(sid, root, adir, odir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
