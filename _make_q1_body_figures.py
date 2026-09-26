# -*- coding: utf-8 -*-
"""
_make_q1_body_figures.py —— 生成《问题一论文正文》所用的图

只读 data/q1_v2/ 下的成品（NPZ / results_100.csv / verify_report.json /
体积台账 / 异常台账 / 分阶段报告），不写任何交付物。输出为仓库根目录下的
png，供 _make_q1_body_docx.py 按文件名引用。

两条纪律：

1. **图里只画数据里有的数**。每一张图的取值都能追到某个成品文件，
   不在图上写任何推算值或示意数字；纯示意的图（如归属判定）明写在标题里。
2. **图内不出现与生成方式有关的字样**，也不出现本机路径、用户名、样本之外的身份串。
   自动化生成不改变产物应为中性技术图这件事。

用法：
    python _make_q1_body_figures.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
Q1 = os.path.join(ROOT, "data", "q1_v2")

# 中文字体：按可用性依次回退（与 _make_flowchart.py 同一条链）
for _f in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 9

C_TEXT = "#2b6cb0"
C_AUDIO = "#c05621"
C_VISION = "#2f855a"
C_GRAY = "#9aa0a6"
C_LIGHT = "#e8eaed"
EDGE = "#1a1a1a"


def save(fig, name: str) -> None:
    """存图。PNG 元数据里不写生成工具，故显式清空 Software 字段。"""
    path = os.path.join(ROOT, name)
    kw = dict(dpi=300, bbox_inches="tight", facecolor="white")
    for md in ({"Software": None}, {}):
        try:
            fig.savefig(path, metadata=md, **kw)
            break
        except Exception:
            continue
    plt.close(fig)
    size = os.path.getsize(path)
    print(f"  ✓ {name}  ({size:,} B)")


def _npz(name: str):
    return np.load(os.path.join(Q1, "features", name), allow_pickle=False)


def _results():
    with open(os.path.join(Q1, "results_100.csv"), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _corr(stem: str):
    with open(os.path.join(Q1, "correspondence", stem), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _stage(name: str):
    with open(os.path.join(Q1, "reports", f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------- 图 1-2

def fig_granularity() -> None:
    """三模态观测粒度与共享时间基准（典型样本 -THoVjtIkeU$_$6 的真实时间标签）。"""
    z = _npz("-THoVjtIkeU___6.npz")
    ws = z["word_start_sec"]
    we = z["word_end_sec"]
    words = [str(w) for w in z["words"]]
    wt = z["word_time_valid"]
    lld = z["raw_audio_lld_center_sec"]
    pts = z["raw_video_pts_sec"]
    fv = z["raw_video_face_feature_valid"].astype(bool)
    acov = z["audio_presentation_coverage_sec"]
    vcov = z["video_presentation_coverage_sec"]

    fig, ax = plt.subplots(figsize=(9.2, 3.5))
    lane_w = (1.62, 2.42)      # 词区间
    lane_a = (0.88, 1.26)      # 语音窗
    lane_v = (0.12, 0.50)      # 视频帧

    for i in range(len(words)):
        if not wt[i] or not np.isfinite(ws[i]):
            continue
        ax.add_patch(Rectangle((ws[i], lane_w[0]), we[i] - ws[i],
                               lane_w[1] - lane_w[0], facecolor=C_TEXT,
                               alpha=0.30, edgecolor=C_TEXT, lw=0.7))
        if we[i] - ws[i] >= 0.20:
            ax.text((ws[i] + we[i]) / 2, lane_w[0] + 0.05, words[i],
                    rotation=90, ha="center", va="bottom", fontsize=4.8,
                    color="#1a365d")

    ax.vlines(lld, lane_a[0], lane_a[1], color=C_AUDIO, lw=0.35, alpha=0.55)
    ax.vlines(pts[fv], lane_v[0], lane_v[1], color=C_VISION, lw=0.6, alpha=0.9)
    ax.vlines(pts[~fv], lane_v[0], lane_v[1], color="#c53030", lw=0.9, alpha=0.9)

    ax.axvline(0.0, color=C_GRAY, lw=0.9, ls="--")
    ax.text(0.04, lane_w[1] + 0.26,
            f"共享时间零点 t0 = {z['shared_t0_sec']:.1f} s；"
            f"音频覆盖 {acov[0]:.2f}–{acov[1]:.2f} s，视频覆盖 {vcov[0]:.2f}–{vcov[1]:.2f} s",
            ha="left", va="center", fontsize=6.8, color="#444")

    ax.text(-0.13, sum(lane_w) / 2, "词区间", ha="right", va="center", fontsize=7.5)
    ax.text(-0.13, sum(lane_a) / 2, "语音窗", ha="right", va="center", fontsize=7.5)
    ax.text(-0.13, sum(lane_v) / 2, "视频帧", ha="right", va="center", fontsize=7.5)
    ax.text(acov[1], sum(lane_a) / 2 + 0.42,
            f"LLD 窗 {lld.size} 个，步长 {z['lld_frame_step_sec'] * 1000:.0f} ms",
            ha="right", va="bottom", fontsize=6.6, color=C_AUDIO)
    ax.text(acov[1], lane_v[1] + 0.06,
            f"视频帧 {pts.size} 帧，人脸有效 {int(fv.sum())} 帧",
            ha="right", va="bottom", fontsize=6.6, color=C_VISION)

    n_bad = int((wt == 0).sum())
    ax.set_xlim(-0.35, max(acov[1], vcov[1]) + 0.05)
    ax.set_ylim(-0.05, lane_w[1] + 0.72)
    ax.set_xlabel("相对共享零点的时间 / s")
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("三模态观测粒度与共享时间基准（典型样本 -THoVjtIkeU 第 6 段）",
                 fontsize=9.5, pad=16)
    ax.text(0.0, -0.30, f"该样本 {len(words)} 个词单元，其中 {n_bad} 个无合法区间，"
                        "在时间轴上不占位置",
            transform=ax.transAxes, fontsize=6.8, color="#666")
    save(fig, "图1-2_三模态观测粒度与时间基准.png")


# --------------------------------------------------------------- 图 1-3

def fig_attribution() -> None:
    """词区间归属与词级聚合判定（示意，坐标非实测值）。"""
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    words = [("词 i", 0.5, 2.0), ("词 i+1", 2.0, 3.5), ("词 i+2", 3.5, 5.0)]
    cols = [C_TEXT, "#805ad5", C_VISION]
    yb = 2.55
    for (lab, s, e), c in zip(words, cols):
        ax.add_patch(Rectangle((s, yb), e - s, 0.42, facecolor=c, alpha=0.22,
                               edgecolor=c, lw=1.1))
        ax.text((s + e) / 2, yb + 0.21, lab, ha="center", va="center", fontsize=8.5, color=c)
        ax.text(s, yb - 0.10, f"$s$={s:.1f}", ha="center", va="top", fontsize=7, color="#444")
        ax.text(e, yb - 0.10, f"$e$={e:.1f}", ha="center", va="top", fontsize=7, color="#444")

    # 语音 LLD 窗：步长 0.2（示意），落点判定 s<=c<e
    c = np.round(np.arange(0.5, 5.01, 0.2), 2)
    for x in c:
        k = next((idx for idx, (_, s, e) in enumerate(words) if s <= x < e), None)
        col = cols[k] if k is not None else C_GRAY
        ax.vlines(x, 1.72, 2.12, color=col, lw=1.0)
    ax.text(-0.05, 1.92, "语音窗", ha="right", va="center", fontsize=8)
    # 视频帧：步长 0.33（示意）
    for x in np.round(np.arange(0.5, 5.01, 0.33), 2):
        k = next((idx for idx, (_, s, e) in enumerate(words) if s <= x < e), None)
        col = cols[k] if k is not None else C_GRAY
        ax.vlines(x, 1.00, 1.40, color=col, lw=1.6)
    ax.text(-0.05, 1.20, "视频帧", ha="right", va="center", fontsize=8)

    ax.text(-0.05, 1.56, "落在区间内即参与聚合：$s_i\\leq c_k<e_i$",
            ha="left", va="center", fontsize=7.5, color="#333")
    ax.annotate("起点计入", xy=(2.0, 2.12), xytext=(2.30, 3.42), ha="center",
                fontsize=7.5, color=C_AUDIO,
                arrowprops=dict(arrowstyle="-|>", color=C_AUDIO, lw=0.9))
    ax.annotate("终点不计入\n（归入下一个词）", xy=(3.5, 1.40), xytext=(4.15, 0.55),
                ha="center", fontsize=7.5, color="#c53030",
                arrowprops=dict(arrowstyle="-|>", color="#c53030", lw=0.9))
    ax.plot([2.0], [1.92], marker="o", ms=6, mfc="white", mec=cols[0], mew=1.4)
    ax.plot([2.0], [1.20], marker="o", ms=6, mfc="white", mec=cols[0], mew=1.4)
    ax.annotate("$c_k=e_i$ 处归右不归左", xy=(2.0, 2.12), xytext=(1.05, 3.42),
                ha="center", fontsize=7.5, color=cols[0],
                arrowprops=dict(arrowstyle="-|>", color=cols[0], lw=0.9))

    ax.annotate("", xy=(0.5, 0.06), xytext=(5.0, 0.06),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.2))
    ax.text(5.05, 0.06, "时间", va="center", fontsize=8)
    ax.set_xlim(-1.35, 5.6)
    ax.set_ylim(-0.12, 3.85)
    ax.axis("off")
    ax.set_title("词区间归属与词级聚合的判定规则（示意图）", fontsize=9.5, pad=6)
    ax.text(0.0, -0.06, "每个词在该模态上取其区间内全部观测的均值与总体标准差拼接："
                        "语音 25×2=50 维，视觉 52×2=104 维；区间内无有效观测则该模态判为无效。",
            transform=ax.transAxes, fontsize=6.8, color="#666")
    save(fig, "图1-3_词区间归属与聚合判定.png")


# --------------------------------------------------------------- 图 1-4

def fig_dims() -> None:
    """三模态特征维度构成：原始观测 → 词级聚合 → 逐词维度。

    三段维数相差一个数量级（768 / 25 / 52），按长度画条会把语音与视觉压成一条线，
    故改用框线示意，长度不承载数量信息，数值全部写在框里。
    """
    fig, ax = plt.subplots(figsize=(9.2, 3.0))
    rows = [("文本", "官方文本经 RoBERTa-base 编码\nsubword 隐状态", "按词取均值",
             768, C_TEXT),
            ("语音", "eGeMAPSv02 低层描述符\n25 维 / 窗", "按词取 [mean, std]\n（ddof=0）",
             50, C_AUDIO),
            ("视觉", "面部 blendshape\n52 维 / 帧", "按词取 [mean, std]\n（ddof=0）",
             104, C_VISION)]
    xs = [(0.010, 0.115), (0.140, 0.455), (0.475, 0.665), (0.685, 0.800)]
    ys = [0.700, 0.415, 0.130]
    bh = 0.215
    for (m, raw, op, dim, col), y in zip(rows, ys):
        for (x0, x1), txt, fc, ec in ((xs[0], m, col, col),
                                      (xs[1], raw, "#ffffff", "#c8ccd0"),
                                      (xs[2], op, "#ffffff", "#c8ccd0"),
                                      (xs[3], f"{dim} 维", "#ffffff", col)):
            ax.add_patch(FancyBboxPatch((x0, y), x1 - x0, bh,
                                        boxstyle="round,pad=0.004,rounding_size=0.01",
                                        facecolor=fc, edgecolor=ec,
                                        alpha=0.16 if fc == col else 1.0,
                                        lw=1.1, mutation_aspect=0.32))
            ax.text((x0 + x1) / 2, y + bh / 2, txt, ha="center", va="center",
                    fontsize=7.6, color=col if fc == col else "#333")
        ax.annotate("", xy=(xs[3][0] - 0.006, y + bh / 2),
                    xytext=(xs[2][1] + 0.006, y + bh / 2),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1))
    ax.plot([0.845, 0.845], [ys[-1] + bh / 2, ys[0] + bh / 2], color=EDGE, lw=1.0)
    ax.annotate("", xy=(0.885, ys[0] + bh / 2), xytext=(0.845, ys[0] + bh / 2),
                arrowprops=dict(arrowstyle="-", color=EDGE, lw=1.0))
    ax.annotate("", xy=(0.885, ys[-1] + bh / 2), xytext=(0.845, ys[-1] + bh / 2),
                arrowprops=dict(arrowstyle="-", color=EDGE, lw=1.0))
    ax.text(0.900, (ys[0] + ys[-1]) / 2 + bh / 2,
            "逐词维度合计\n768 + 50 + 104\n= 922 维", ha="left", va="center",
            fontsize=8.2, color=EDGE)
    ax.text(0.0, 0.045, "三路特征都是词级、长度同为该样本的词单元数 L；"
                        "每一路各有一条独立掩码，未有效观测处特征行为零。",
            fontsize=6.8, color="#666")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.97)
    ax.axis("off")
    ax.set_title("三模态特征的定义与维度构成", fontsize=9.5, pad=2)
    save(fig, "图1-4_三模态特征维度构成.png")


# --------------------------------------------------------------- 图 2-2

def fig_word_counts() -> None:
    """典型样本逐词源观测计数：参与聚合的观测与时段内全部观测。"""
    stem = "-THoVjtIkeU___6_correspondence.csv"
    rows = _corr(stem)
    n = len(rows)
    idx = np.arange(n)
    words = [r["word"] for r in rows]
    na = np.array([int(r["n_audio_windows"]) for r in rows])
    nf = np.array([int(r["n_face_frames"]) for r in rows])
    ns = np.array([int(r["n_frames_in_span"]) for r in rows])
    tv = np.array([int(r["word_time_valid"]) for r in rows])

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 4.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1, 1], hspace=0.16))
    ax = axes[0]
    ax.bar(idx, na, color=C_AUDIO, alpha=0.85, width=0.72)
    ax.set_ylabel("语音窗数", fontsize=8.5)
    ax.bar(idx[tv == 0], na[tv == 0], color="#c53030", alpha=0.85, width=0.72)
    ax.text(0.995, 0.92, f"参与聚合的 LLD 窗，全样本合计 {na.sum()}",
            transform=ax.transAxes, ha="right", fontsize=7, color=C_AUDIO)

    ax = axes[1]
    ax.bar(idx, ns, color=C_LIGHT, edgecolor=C_GRAY, lw=0.6, width=0.72,
           label="时段内全部帧")
    ax.bar(idx, nf, color=C_VISION, alpha=0.9, width=0.72, label="人脸有效帧")
    ax.set_ylabel("视频帧数", fontsize=8.5)
    ax.legend(fontsize=7, loc="upper right", frameon=False, ncol=2)
    ax.text(0.995, 0.62, f"人脸有效帧合计 {nf.sum()}，时段内帧合计 {ns.sum()}",
            transform=ax.transAxes, ha="right", fontsize=7, color="#444")

    bad = np.where(tv == 0)[0]
    for a in axes:
        for b in bad:
            a.axvspan(b - 0.45, b + 0.45, color="#c53030", alpha=0.08, zorder=0)
    if bad.size:
        axes[0].annotate("该词无合法区间\n三路特征行全为零",
                         xy=(bad[0], na[bad[0]]), xytext=(bad[0] + 3.2, na.max() * 0.72),
                         fontsize=7, color="#c53030",
                         arrowprops=dict(arrowstyle="-|>", color="#c53030", lw=0.9))
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(words, rotation=90, fontsize=5.6)
    axes[1].set_xlabel("词单元（按文本顺序）", fontsize=8.5)
    axes[1].set_xlim(-0.8, n - 0.2)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    axes[0].set_title("典型样本逐词源观测计数（-THoVjtIkeU 第 6 段，30 个词单元）",
                      fontsize=9.5, pad=6)
    save(fig, "图2-2_典型样本逐词源观测计数.png")


# --------------------------------------------------------------- 图 3-1

def fig_verify() -> None:
    """十六项机器核验的断言数。"""
    with open(os.path.join(Q1, "reports", "verify_report.json"), encoding="utf-8") as f:
        rep = json.load(f)
    order = [c for c in rep["checks"] if not c.get("evidence", {}).get("skipped")]
    skip = [c for c in rep["checks"] if c.get("evidence", {}).get("skipped")]
    codes = [c["code"] for c in order]
    vals = [c["checked"] for c in order]

    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    x = np.arange(len(codes))
    cols = [C_TEXT if v >= 1000 else "#6b7f95" for v in vals]
    ax.bar(x, vals, color=cols, width=0.66)
    for xi, v in zip(x, vals):
        ax.text(xi, v * 1.12, f"{v:,}", ha="center", fontsize=6.6, color="#333")
    if skip:
        ax.bar([len(codes)], [1], color=C_GRAY, alpha=0.5, width=0.66, hatch="//")
        ax.text(len(codes), 1.6, "0（按设计跳过）", ha="center", fontsize=6.6, color="#666",
                rotation=90, va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(0.6, max(vals) * 4)
    ax.set_xticks(list(x) + [len(codes)])
    ax.set_xticklabels(codes + [skip[0]["code"] if skip else ""], fontsize=8)
    ax.set_ylabel("断言条数（对数轴）", fontsize=8.5)
    ax.set_xlabel("核验项编号", fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"十六项机器核验的断言条数（执行 {rep['n_executed']} 项，"
                 f"合计 {rep['total_assertions_checked']:,} 条，失败 "
                 f"{rep['total_assertions_failed']} 条）", fontsize=9.5, pad=8)
    save(fig, "图3-1_机器核验断言分布.png")


# --------------------------------------------------------------- 图 3-2

def fig_stage_time() -> None:
    """分阶段实名耗时。"""
    st = [("文本 text", _stage("stage_text")["runtime_sec"]),
          ("对齐 align", _stage("stage_align")["runtime_sec"]),
          ("媒体 media", _stage("stage_media")["runtime_sec"])]
    fig, ax = plt.subplots(figsize=(9.2, 2.5))
    y = np.arange(len(st))[::-1]
    cols = [C_TEXT, C_AUDIO, C_VISION]
    ax.barh(y, [v for _, v in st], color=cols, alpha=0.85, height=0.58)
    for yi, (lab, v) in zip(y, st):
        ax.text(v + 12, yi, f"{v:.1f} s", va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _ in st], fontsize=8.5)
    ax.set_xlim(0, max(v for _, v in st) * 1.22)
    ax.set_xlabel("实名耗时 / s", fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    tot = sum(v for _, v in st)
    ax.set_title(f"分阶段实名耗时（三段合计 {tot:.1f} s）", fontsize=9.5, pad=6)
    ax.text(0.0, -0.34, "prepare、tables、verify 三段不单独计时；媒体阶段含逐帧解码与"
                        "面部特征提取，是链路的主要耗时来源。",
            transform=ax.transAxes, fontsize=6.8, color="#666")
    save(fig, "图3-2_分阶段实名耗时.png")


# --------------------------------------------------------------- 图 4-1

def fig_route() -> None:
    """对齐粒度、结构闸门与路由类别的取值分布。"""
    rows = _results()
    def cnt(key):
        out = {}
        for r in rows:
            out[r[key]] = out.get(r[key], 0) + 1
        return sorted(out.items(), key=lambda kv: -kv[1])

    groups = [("对齐粒度 alignment_granularity", cnt("alignment_granularity"), C_TEXT),
              ("结构闸门 text_av_time_mapping_status", cnt("text_av_time_mapping_status"), C_AUDIO),
              ("路由类别 alignment_mode", cnt("alignment_mode"), C_VISION)]
    # 三栏都数 100 条样本，故共用一个横轴刻度，条形长度才可直接互比
    xmax = max(v for _, items, _ in groups for _, v in items) * 1.36
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 4.2))
    for ax, (title, items, col) in zip(axes, groups):
        y = np.arange(len(items))[::-1]
        vals = [v for _, v in items]
        ax.barh(y, vals, color=col, alpha=0.82, height=0.5)
        for yi, (lab, v) in zip(y, items):
            ax.text(v + xmax * 0.012, yi, f"{lab}  {v}", va="center", fontsize=7.6)
        ax.set_xlim(0, xmax)
        ax.set_yticks([])
        ax.set_title(title, fontsize=8.5, loc="left", pad=2)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlabel("样本条数（100 条）", fontsize=8.5)
    fig.suptitle("100 条样本的对齐粒度、结构闸门与路由分布", fontsize=9.5, y=0.99)
    fig.text(0.0, -0.06, "word_partial 表示官方词的时间区间只有部分可用，clip_only 表示只能给到片段级；"
                         "两类不发出的路由类别不在图中出现。",
             fontsize=6.8, color="#666")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "图4-1_对齐粒度与路由分布.png")


# --------------------------------------------------------------- 图 4-2

def fig_words_valid() -> None:
    """逐样本词单元数与共同有效词数。"""
    rows = _results()
    nw = np.array([int(r["n_official_words"]) for r in rows])
    vl = np.array([int(r["valid_length"]) for r in rows])
    dur = np.array([float(r["original_effective_duration_sec"]) for r in rows])
    o = np.argsort(-nw)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 4.2),
                             gridspec_kw=dict(height_ratios=[2, 1.15], hspace=0.30))
    ax = axes[0]
    x = np.arange(len(rows))
    ax.bar(x, nw[o], color=C_LIGHT, edgecolor=C_GRAY, lw=0.4, width=0.9,
           label="词单元数")
    ax.bar(x, vl[o], color=C_TEXT, alpha=0.9, width=0.9, label="共同有效词数")
    ax.set_xlim(-1, len(rows))
    ax.set_ylabel("词数", fontsize=8.5)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper right")
    ax.text(0.005, 0.86, f"词单元合计 {nw.sum():,}，共同有效词合计 {vl.sum():,}",
            transform=ax.transAxes, fontsize=7.5, color="#333")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    ax.hist(nw, bins=np.arange(0, 70, 5), color=C_TEXT, alpha=0.75, edgecolor="white")
    ax.axvline(nw.mean(), color="#c53030", lw=1.2, ls="--")
    ax.text(nw.mean() + 1.0, ax.get_ylim()[1] * 0.86,
            f"均值 {nw.mean():.1f} 词\n区间 {nw.min()}–{nw.max()} 词",
            fontsize=7.5, color="#c53030")
    ax.set_xlabel("每条样本的词单元数", fontsize=8.5)
    ax.set_ylabel("样本条数", fontsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"有效时长 {dur.min():.1f}–{dur.max():.1f} s，"
                 f"逐样本变长、不做填充", fontsize=8, loc="right", pad=3, color="#444")
    fig.suptitle("100 条样本的词单元数与共同有效词数（按词单元数降序）",
                 fontsize=9.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "图4-2_逐样本词单元数与共同有效词数.png")


# --------------------------------------------------------------- 图 4-3

def fig_size() -> None:
    """交付体积构成。"""
    with open(os.path.join(Q1, "reports", "q1v2_size_ledger.json"), encoding="utf-8") as f:
        led = json.load(f)
    v2 = led["v2"]
    items = sorted(v2["counted"].items(), key=lambda kv: -kv[1]["bytes"])
    names = [k for k, _ in items]
    vals = [v["bytes"] / 1024 / 1024 for _, v in items]
    other = [k for k, _ in v2["not_counted"].items()]
    ovals = [v2["not_counted"][k]["bytes"] / 1024 / 1024 for k in other]

    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    y = np.arange(len(items) + len(other))[::-1]
    cols = [C_TEXT] * len(items) + [C_GRAY] * len(other)
    alphas = [0.85] * len(items) + [0.45] * len(other)
    for yi, v, c, a in zip(y, vals + ovals, cols, alphas):
        ax.barh(yi, v, color=c, alpha=a, height=0.6,
                hatch="//" if a < 0.5 else None)
    for yi, v in zip(y, vals + ovals):
        ax.text(v + 0.18, yi, f"{v:.2f} MB", va="center", fontsize=7.6)
    ax.set_yticks(y)
    ax.set_yticklabels(names + [k + "（不计入）" for k in other], fontsize=7.8)
    ax.set_xlabel("体积 / MB", fontsize=8.5)
    ax.set_xlim(0, max(vals) * 1.30)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    cb = v2["counted_bytes"] / 1024 / 1024
    ax.set_title(f"交付体积构成（计入合计 {cb:.2f} MB，占 50 MB 上限的 "
                 f"{cb / 50 * 100:.1f}%）", fontsize=9.5, pad=6)
    ax.text(0.0, -0.30, "斜纹部分为可由一次完整重跑再生的中间态，不计入提交体积；"
                        "上图为计入部分的逐项构成。",
            transform=ax.transAxes, fontsize=6.8, color="#666")
    save(fig, "图4-3_交付体积构成.png")


# --------------------------------------------------------------- 图 4-4

def fig_anomaly() -> None:
    """异常台账的成因分布。"""
    with open(os.path.join(Q1, "audit", "anomaly_ledger.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    causes = ["no_face_detected", "word_coverage_below_0.5",
              "TRI_MODAL_WITH_AUDIO_CONTENT_INVALID"]
    labels = ["未检出人脸", "词覆盖低于 0.5", "音频内容无效"]
    cols = [C_VISION, C_AUDIO, C_TEXT]
    mat = np.zeros((len(rows), len(causes)))
    for i, r in enumerate(rows):
        for j, c in enumerate(causes):
            mat[i, j] = 1.0 if c in r["notes"] else 0.0
    # 各成因分别计数、并列画出：同一行可同时命中多条成因，堆叠会把总数画成 38
    fig, ax = plt.subplots(figsize=(9.2, 2.5))
    y = np.arange(len(causes))[::-1]
    vals = [int(mat[:, j].sum()) for j in range(len(causes))]
    ax.barh(y, vals, color=cols, alpha=0.84, height=0.52)
    for yi, lab, v in zip(y, labels, vals):
        ax.text(v + 0.4, yi, f"{lab}  {v} 条", va="center", fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(0, max(vals) * 1.55)
    ax.set_xlabel("命中该成因的异常条数", fontsize=8.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"异常台账的成因分布（台账共 {len(rows)} 行）", fontsize=9.5, pad=8)
    ax.text(0.0, -0.42, "同一行可同时命中多条成因，故各成因条数之和大于台账行数；"
                        "全部 29 行的 counts_as_deletion 均为 0，异常只登记不剔除。",
            transform=ax.transAxes, fontsize=6.8, color="#666")
    save(fig, "图4-4_异常成因分布.png")


def main() -> int:
    print("[生成] 只读 data/q1_v2 成品，写出 png 到仓库根目录")
    fig_granularity()
    fig_attribution()
    fig_dims()
    fig_word_counts()
    fig_verify()
    fig_stage_time()
    fig_route()
    fig_words_valid()
    fig_size()
    fig_anomaly()
    print("[完成] 共 10 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
