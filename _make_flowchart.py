# -*- coding: utf-8 -*-
"""
_make_flowchart.py —— 生成问题一的分析总流程图（论文用图）

输出（均为 png 300 dpi + 同名 pdf 矢量）：
    图1_三问总体分析流程图   对应模板「图1 问题的总分析」，只画分析过程，不写结论
    图2_问题一分析总流程图   问题一的执行链路，供 5.1 节引用

准则：图里只画代码里真实存在的步骤，不写代码里没有的环节。
    输入      label-100.xlsx + <video_id>/<clip_id>.mp4         见 unaligned_common.list_samples()
    ①文本     unaligned_text.py     RoBERTa-base，pts 为均匀假设
    ②语音     unaligned_audio.py    ffmpeg 16 kHz + librosa，74 维，pts 实测
    ③视觉     unaligned_vision.py   cv2 + MTCNN + ResNet50 + 固定随机投影，35 维，pts 实测
    ④对齐     align_multimodal.align_series()   time_bin / 50 槽 / 分模态空槽策略
    ⑤交付     q1_delivery.py        summary_q1.csv 等 9 项
    ⑥核验     q1_verify.py          V1–V12，17436 条断言
    ⑦融合     q1_fusion.py / q1_typical.py / q1_readme.py
    三问编排  run_all.py            audit → q1 → q2 → q3 → pack
    Q2 实验   q2_train.py           4 类型 × 4 位置 × 3 时长 = 48 组受控缺失
    Q3 解释   q3_explain.py + q3_timebase.py   归因与因果双口径 + 三类证据

用法：python _make_flowchart.py [--which all|overview|q1]
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

# 中文字体：按可用性依次回退
for _f in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
    try:
        matplotlib.font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# 画布比例须与各自 viewport 同比，否则 set_aspect("equal") 下文字会被压扁。
W, H = 9.30, 14.56       # 问题一详图：viewport 10.35 × 16.2  （比 0.639）
OW, OH = 11.00, 6.90     # 三问总览图：viewport 15.60 × 9.79（比 1.593）
EDGE = "#1a1a1a"
FILL_IN = "#ececec"        # 输入
FILL_MOD = "#ffffff"       # 分模态提取
FILL_CORE = "#d9d9d9"      # 核心（对齐）
FILL_OUT = "#f4f4f4"       # 输出


def box(ax, x, y, w, h, lines, fill="#ffffff", lw=1.4, fs=10.0,
        bold_first=True, note=""):
    """在 (x,y) 左下角画一个圆角框。

    lines[0] 为标题（加粗）；note 若非空，则以小号灰斜体画在框底内侧
    （把路径/文件名等注记收进框内，避免与连线、汇合线重叠）。
    """
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=lw, edgecolor=EDGE, facecolor=fill, zorder=2))
    n = len(lines)
    note_h = 0.235 if note else 0.0
    top = y + h - 0.030
    step = (h - 0.055 - note_h) / max(n, 1)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, top - i * step - step * 0.28, t,
                ha="center", va="center", fontsize=fs, color=EDGE,
                fontweight="bold" if (i == 0 and bold_first) else "normal",
                zorder=3)
    if note:
        ax.text(x + w / 2, y + 0.030 + note_h * 0.50, note,
                ha="center", va="center", fontsize=fs - 1.9,
                color="#555555", style="italic", zorder=3)


def arrow(ax, p, q, style="-|>", lw=1.4, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p, q, arrowstyle=style, mutation_scale=15, linewidth=lw,
        color=EDGE, zorder=1, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=0.0, shrinkB=0.0))


def make_overview(out_png: str, out_pdf: str) -> None:
    """三问总体分析流程图（对应模板「图1 问题的总分析」）。

    只画分析过程与建模路线，不写结论、不写指标数值——评审看的是思路是否成立，
    结果另有正文与交付物支撑。
    """
    fig, ax = plt.subplots(figsize=(OW, OH))
    ax.set_xlim(0, 15.60)
    ax.set_ylim(3.91, 13.70)
    ax.set_aspect("equal")
    ax.axis("off")

    CW, GAP = 4.70, 0.42
    X0 = 0.33
    col_x = [X0 + i * (CW + GAP) for i in range(3)]
    col_cx = [x + CW / 2 for x in col_x]
    FULL_X, FULL_W = X0, 3 * CW + 2 * GAP          # 0.33 .. 15.27
    CX = FULL_X + FULL_W / 2                        # 7.80

    # ---------- 公共基础与约束 ----------
    box(ax, FULL_X, 11.60, FULL_W, 1.85, [
        "数据基础与公共约束",
        "附件1 原始视频与文本标注　附件2 训练/验证　附件3 缺失推断　附件4 解释推断",
        "唯一数据来源，不引入任何外部情感数据集",
        "三问共用同一特征版本与同一输入接口（aligned_50）",
        "有效词区 = [1, L−1)，填充位不承载情感",
        "红线　异常只登记、不删除、不剔除",
    ], fill=FILL_IN, fs=9.0)

    arrow(ax, (CX, 11.60), (CX, 11.28))
    ax.plot([col_cx[0], col_cx[2]], [11.28, 11.28], color=EDGE, lw=1.4, zorder=1)
    for cx in col_cx:
        arrow(ax, (cx, 11.28), (cx, 10.98))

    # ---------- 三问标题 ----------
    heads = [
        ["【问题一】", "原始多模态特征提取与时序对齐"],
        ["【问题二】", "缺失模态下的情感鲁棒预测"],
        ["【问题三】", "可解释的情感预测与证据落地"],
    ]
    for x, cx, head in zip(col_x, col_cx, heads):
        box(ax, x, 10.03, CW, 0.95, head, fill=FILL_CORE, fs=9.6)
        arrow(ax, (cx, 10.03), (cx, 9.82))

    # ---------- 分析要点 + 建模路线 ----------
    # 三栏行数必须一致（1 表头 + 5 要点 + 1 空行 + 1 路线表头 + 3 路线），
    # 否则同样框高下三栏的行基线会错开，看上去像排版事故。
    bodies = [
        ["【分析要点】",
         "· 三模态采样率与时间粒度各异",
         "· 须同构化到同一离散时间轴",
         "· 对齐粒度不可自选",
         "· 须先切出有效区",
         "· 文本无逐词时间戳，词—时段只能近似",
         "",
         "【建模路线】",
         "分模态提特征 → 统一到 50 等宽槽",
         "→ 逐槽回捞源帧建立溯源链",
         "→ 异常只记不删"],
        ["【分析要点】",
         "· 缺失是核心变量，不是噪声",
         "· 零填充等于读成「情感为零」",
         "· 缺失是区间性的",
         "· 掩码须逐模态",
         "· 跨模态取并集会使实验失效",
         "",
         "【建模路线】",
         "三路结构信号 → 掩码注意力融合",
         "→ 双头输出（极性 + 强度）",
         "→ 主动制造缺失以测其影响"],
        ["【分析要点】",
         "· 解释分归因与因果两类口径",
         "· 二者不可互替",
         "· 因果会被冗余模态补偿",
         "· 归因有争议，故两口径并行",
         "· 附件4 无逐词时间戳",
         "",
         "【建模路线】",
         "归因与三类因果度量并行输出",
         "→ 报一致率、背离不隐藏",
         "→ 由 mp4 重建时间轴并落地三类证据"],
    ]
    for x, body in zip(col_x, bodies):
        box(ax, x, 6.12, CW, 3.70, body, fill="#ffffff", fs=8.0)

    # ---------- 公共口径 ----------
    for cx in col_cx:
        arrow(ax, (cx, 6.12), (cx, 5.91))
    ax.plot([col_cx[0], col_cx[2]], [5.91, 5.91], color=EDGE, lw=1.4, zorder=1)
    arrow(ax, (CX, 5.91), (CX, 5.66))

    box(ax, FULL_X, 4.26, FULL_W, 1.40, [
        "贯穿三问的统一口径",
        "· 三阶段同一特征版本与同一输入接口",
        "· 三问可各自独立重跑，逐级落盘、互不覆盖",
        "· 异常只登记不剔除，不符处显式标注",
    ], fill=FILL_OUT, fs=9.0)

    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已生成：{out_png}")
    print(f"已生成：{out_pdf}")


def make(out_png: str, out_pdf: str) -> None:
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, 10.35)
    ax.set_ylim(0, 16.2)
    ax.set_aspect("equal")
    ax.axis("off")

    LX, MW, GAP = 0.30, 3.05, 0.30          # 左侧起点 / 每栏宽 / 栏间距
    CX = LX + (3 * MW + 2 * GAP) / 2.0      # 整图竖直主轴 = 5.175
    LANE_R = LX + 3 * MW + 2 * GAP          # 第三栏右边界 = 10.05
    xs = [LX + i * (MW + GAP) for i in range(3)]
    lane_cx = [x + MW / 2 for x in xs]

    # ================= 输入 =================
    box(ax, CX - 3.875, 15.00, 7.75, 1.15, [
        "输入（赛题附件 1）",
        "label-100.xlsx：100 条清单 video_id / clip_id / text / label",
        "<video_id>/<clip_id>.mp4：100 条原始视频",
        "附件 2 的标准张量 (50,768)/(50,74)/(50,35) 仅作接口对照，不参与本问提取",
    ], fill=FILL_IN, fs=10.4)
    arrow(ax, (CX, 15.00), (CX, 14.30))
    ax.text(CX + 0.12, 14.86, "list_samples()　遍历样本清单", ha="left",
            va="center", fontsize=9.0, color=EDGE)

    # ================= 三路分叉 =================
    ax.plot([LX + 0.15, LANE_R - 0.15], [14.30, 14.30], color=EDGE, lw=1.4,
            zorder=1)
    for cx in lane_cx:
        arrow(ax, (cx, 14.30), (cx, 13.90))

    lane_titles = ["① 文本支路", "② 语音支路", "③ 视觉支路"]
    lane_files = ["unaligned_text.py", "unaligned_audio.py", "unaligned_vision.py"]
    lane_body = [
        ["RoBERTa-base（离线）",
         "子词隐状态 → 按词求均值",
         "维度 768",
         "",
         "时间基准：均匀假设",
         "pts = (i+0.5)·T/W",
         "← 非实测，仅用于共轴"],
        ["ffmpeg 解码 → 16 kHz",
         "librosa 逐帧",
         "20 维 MFCC + 差分",
         "+ 韵律 + 频谱 → 74 维",
         "帧率 20 Hz",
         "",
         "时间基准：实测 pts",
         "← ffmpeg 解码时间戳"],
        ["cv2 抽帧（按 POS_MSEC）",
         "MTCNN 检人脸",
         "ResNet-50 池化 → 2048 维",
         "固定 seed=42 随机投影",
         "→ 35 维（100 条共用）",
         "帧率 15 Hz（实测 15.18）",
         "时间基准：实测 pts",
         "← 抽帧实测时间戳"],
    ]
    for x, t, f, body in zip(xs, lane_titles, lane_files, lane_body):
        box(ax, x, 11.35, MW, 2.55, [t] + body, fill=FILL_MOD, fs=8.5,
            note=f)

    # ================= 汇合 =================
    ax.plot([LX + 0.15, LANE_R - 0.15], [10.95, 10.95], color=EDGE, lw=1.4,
            zorder=1)
    for cx in lane_cx:
        arrow(ax, (cx, 11.35), (cx, 10.95))
    arrow(ax, (CX, 10.95), (CX, 10.65))

    box(ax, CX - 3.475, 9.70, 6.95, 0.95, [
        "未对齐特征 (T, D) + pts + meta + extra",
    ], fill=FILL_OUT, fs=9.6,
        note="data/unaligned_features/<模态>/<sample_id>.npz　·　三支各自独立、可单独重跑")
    arrow(ax, (CX, 9.70), (CX, 9.40))

    # ================= 对齐核心 =================
    box(ax, CX - 4.425, 6.80, 8.85, 2.60, [
        "④ 统一对齐　align_series()",
        "公共时间轴 T = 视频真实时长（ffprobe，duration_policy = video）",
        "槽归属：slot = clip(floor(t / T · 50), 0, 49)　—— 闭式，与 slot_to_interval 互为逆",
        "槽内聚合：同槽求均值（time_bin，尊重各模态实际采样率）",
        "空槽补齐：文本 zero　│　音频 / 视觉 interpolate　—— 分模态，不一律插值",
        "越界记账 A8：并入端点槽，但显式计数（不静默发生）",
        "溯源留痕：逐槽记录 slot_unit_index（该槽由哪些源帧聚合而来）",
    ], fill=FILL_CORE, fs=8.9, lw=2.0)
    arrow(ax, (CX, 6.80), (CX, 6.50))

    box(ax, CX - 3.875, 5.55, 7.75, 0.95, [
        "对齐张量 (50,768) / (50,74) / (50,35) + valid 掩码 + slot_unit_index",
    ], fill=FILL_OUT, fs=9.4,
        note="data/aligned/<sample_id>.npz　·　aligned_50.npz　·　align_summary.csv")
    arrow(ax, (CX, 5.55), (CX, 5.25))

    # ================= 五、六 分叉 =================
    AXX, BXX = CX - 2.925, CX + 2.925        # 2.25 / 8.10
    ax.plot([AXX, BXX], [5.25, 5.25], color=EDGE, lw=1.4, zorder=1)
    arrow(ax, (AXX, 5.25), (AXX, 4.95))
    arrow(ax, (BXX, 5.25), (BXX, 4.95))

    box(ax, AXX - 1.60, 2.85, 3.20, 2.10, [
        "⑤ 交付物生成",
        "q1_delivery.py",
        "",
        "summary_q1.csv（100×32）",
        "alignment_q1.json",
        "anomaly_ledger.csv（151 条）",
        "extract_config.json ← 复现凭据",
    ], fill=FILL_OUT, fs=8.8)
    box(ax, BXX - 1.60, 2.85, 3.20, 2.10, [
        "⑥ 12 项机器核验",
        "q1_verify.py",
        "",
        "V1–V12　17 436 条断言",
        "V4 回捞源帧重算 5.95e-8",
        "V8 外部 ffprobe 交叉核对",
        "V9 掩码补全 2/2 命中",
    ], fill=FILL_OUT, fs=8.8)

    ax.plot([AXX, BXX], [2.60, 2.60], color=EDGE, lw=1.4, zorder=1)
    arrow(ax, (AXX, 2.85), (AXX, 2.60))
    arrow(ax, (BXX, 2.85), (BXX, 2.60))
    arrow(ax, (CX, 2.60), (CX, 2.30))

    box(ax, CX - 2.10, 1.88, 4.20, 0.42, ["data/q1_delivery/"],
        fill="#ffffff", fs=9.4, lw=1.2)
    arrow(ax, (CX, 1.88), (CX, 1.62))

    box(ax, CX - 4.525, 0.95, 9.05, 0.67, [
        "⑦ 三层有效性（存在性 / 观测有效 / 内容有效）　＋　文本与语音的三态对应",
        "＋　五类典型样本　＋　交付 README 与复现说明",
    ], fill=FILL_MOD, fs=8.8)

    # ================= 全局红线（底部横幅，无箭头，独立于流程） =================
    box(ax, CX - 4.525, 0.10, 9.05, 0.65, [
        "全局红线约束（赛题「数据说明」的注）：异常只登记、不删除、不剔除",
        "anomaly_ledger.csv 共 151 条记录（15 类码，涉及 72 个样本），counts_as_deletion ≡ 0；",
        "任何样本都不因异常被移出训练 / 验证 / 推理集合",
    ], fill="#ffffff", fs=8.5, lw=1.4)

    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已生成：{out_png}")
    print(f"已生成：{out_pdf}")


def main(argv=None) -> int:
    import argparse
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="生成论文流程图")
    ap.add_argument("--which", default="all",
                    choices=["all", "overview", "q1"],
                    help="all=两张都生成（默认）；overview=三问总览；q1=问题一详图")
    a = ap.parse_args(argv)

    if a.which in ("all", "overview"):
        make_overview(os.path.join(here, "图1_三问总体分析流程图.png"),
                      os.path.join(here, "图1_三问总体分析流程图.pdf"))
    if a.which in ("all", "q1"):
        make(os.path.join(here, "图2_问题一分析总流程图.png"),
             os.path.join(here, "图2_问题一分析总流程图.pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
