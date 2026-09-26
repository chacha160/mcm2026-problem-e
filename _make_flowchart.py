# -*- coding: utf-8 -*-
"""
_make_flowchart.py —— 生成问题一的分析总流程图（论文用图）

输出（均为 png 300 dpi + 同名 pdf 矢量）：
    图1_三问总体分析流程图   对应模板「图1 问题的总分析」，只画分析过程，不写结论
    图2_问题一分析总流程图   问题一的执行链路，供 5.1 节引用
    图1_问题一v2总体流程          论文引用的 v2 流程图：横排结构图（三路汇合）
    图1_问题一v2总体流程_详细版    同一条链路的竖排逐阶段详图（带代码路径，留底用）
    图2_典型样本v2对齐时间线       典型样本的逐词对齐时间线

准则：图里只画代码里真实存在的步骤，不写代码里没有的环节。
    输入      label-100.xlsx + <video_id>/<clip_id>.mp4         见 unaligned_common.list_samples()
    ①文本     unaligned_text.py     RoBERTa-base，时间基准按证据路由（实测/均匀）
    ②语音     unaligned_audio.py    ffmpeg 16 kHz + librosa，74 维，pts 实测
    ③视觉     unaligned_vision.py   cv2 + MTCNN + ResNet50 + 固定随机投影，35 维，pts 实测
    ④对齐     align_multimodal.align_series()   time_bin / 50 槽 / 分模态空槽策略
    ⑤交付     q1_delivery.py        summary_q1.csv 等 9 项
    ⑥核验     q1_verify.py          V1–V13，逐项机器核验
    ⑦融合     q1_fusion.py / q1_typical.py / q1_readme.py
    编排      run_all.py            q1raw → q1 → pack（只编问题一）

用法：python _make_flowchart.py [--which all|overview|q1]
"""

from __future__ import annotations

import os
import re
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


def _mt(text: str) -> str:
    """转义 matplotlib 的数学定界符 `$`。

    样本编号形如 `-THoVjtIkeU$_$6`，其中的 `$_$` 会被 matplotlib 当成行内公式，
    于是 `_` 变成下标算子并抛 ParseSyntaxException。图上显示时把 `$` 转义掉，
    编号本身不变（`\\$` 渲染出来仍是 `$`）。
    """
    return text.replace("$", r"\$")


def box(ax, x, y, w, h, lines, fill="#ffffff", lw=1.4, fs=10.0,
        bold_first=True, note="", vcenter=False):
    """在 (x,y) 左下角画一个圆角框。

    lines[0] 为标题（加粗）；note 若非空，则以小号灰斜体画在框底内侧
    （把路径/文件名等注记收进框内，避免与连线、汇合线重叠）。

    vcenter=True 把文字块按框的几何中心垂直居中。默认的 top 排版是为两三行
    的块设计的，单行框套它会显得顶在框上半部；骨架图那种一框一字的图必须
    显式传 vcenter=True（且 note 为空，否则文字块会压到注记上）。
    """
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=lw, edgecolor=EDGE, facecolor=fill, zorder=2))
    n = len(lines)
    note_h = 0.235 if note else 0.0
    step = (h - 0.055 - note_h) / max(n, 1)
    if vcenter:
        # 让 i=0 与 i=n-1 两行关于 y+h/2 对称：top 下移半个块高再加回基线偏移。
        top = y + h / 2.0 + (n - 1) * step / 2.0 + step * 0.28
    else:
        top = y + h - 0.030
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
         "时间基准：证据路由",
         "实测 7 条 → 词区间中心",
         "其余 93 条 → (i+0.5)·T/W"],
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
        "summary_q1.csv（100×39）",
        "alignment_q1.json",
        "anomaly_ledger.csv（151 条）",
        "extract_config.json ← 复现凭据",
    ], fill=FILL_OUT, fs=8.8)
    box(ax, BXX - 1.60, 2.85, 3.20, 2.10, [
        "⑥ 13 项机器核验",
        "q1_verify.py",
        "",
        "V1–V13　17 772 条断言",
        "V4 回捞源帧重算 5.95e-8",
        "V8 外部 ffprobe 交叉核对",
        "V9 掩码补全 2/2 命中",
        "V13 词时间路由 7/93",
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


# ===========================================================================
# 问题一 v2（词级多粒度）的两张图
#
# 与上面的 `make_overview` / `make` 的分工：那两张画的是**旧 50 等分槽架构**，
# 服务于三问稿。本节的图只画 v2，且框内每个数字都**从 data/q1_v2/ 的产物现场读**，
# 不写常数——旧流程图的毛病正是数字写死在脚本里，产物一变图就不实话。
# ===========================================================================

ROOT = os.path.dirname(os.path.abspath(__file__))
Q1V2 = os.path.join(ROOT, "data", "q1_v2")
QW, QH = 10.60, 13.40      # v2 总流程图画布（按内容定，等比）
TW, TH = 13.20, 8.20       # 典型样本时间线画布（上：三带时间线；下：逐词计数）

# 维度常量与核验同源：图里写的 768 / 25 / 52 必须就是 NPZ 第二维被断言的那个数。
sys.path.insert(0, os.path.join(ROOT, "code"))
import q1v2_contract as C  # noqa: E402


def _q1v2_facts() -> dict:
    """把两张图要用的数一次读齐。来源全是产物文件，任一缺失即抛错（不静默降级）。"""
    import csv
    import json

    def _json(*parts):
        with open(os.path.join(Q1V2, *parts), "r", encoding="utf-8") as fh:
            return json.load(fh)

    with open(os.path.join(Q1V2, "results_100.csv"), "r",
              encoding="utf-8-sig", newline="") as fh:
        results = list(csv.DictReader(fh))
    verify = _json("reports", "verify_report.json")
    checks = {c["code"]: c for c in verify["checks"]}
    ledger = _json("reports", "q1v2_size_ledger.json")
    assets = _json("metadata", "model_assets.json")
    corr = _json("correspondence", "_meta.json")

    modes = {}
    for r in results:
        modes[r["alignment_mode"]] = modes.get(r["alignment_mode"], 0) + 1

    return {
        "n_samples": len(results),
        "word_units": checks["V3"]["evidence"]["expected_word_units"],
        "ws_chunks": checks["V3"]["evidence"]["expected_whitespace_chunks"],
        "sum_words": checks["V13"]["evidence"]["sum_word_units"],
        "sum_slots": checks["V13"]["evidence"]["sum_aligner_slots"],
        "valid_words": checks["V13"]["evidence"]["sum_structurally_valid_words"],
        "zero_words": checks["V13"]["evidence"]["sum_zero_duration_word_count"],
        "frames": checks["M1"]["evidence"]["sum_video_frame_count"],
        "face_frames": checks["M1"]["evidence"]["sum_face_detected_frame_count"],
        "no_face": checks["M1"]["evidence"]["n_samples_without_face"],
        "n_checks": verify["n_checks"], "n_executed": verify["n_executed"],
        "n_assert": verify["total_assertions_checked"],
        "n_failed": verify["total_assertions_failed"],
        "all_passed": verify["all_passed"],
        "bytes_v2": ledger["accounting"]["v2_counted_bytes"],
        "modes": modes,
        "assets": assets, "corr": corr,
    }


# 六个阶段的标题：只有这一处定义，详细版与简版都从这里取，标题不会各自漂。
_Q1V2_STAGE_TITLES = (
    "S0  prepare　登记与冻结",
    "S1  text　文本特征",
    "S2  align　逐词时间",
    "S3  media　语音与视觉特征",
    "S4  tables　词级聚合与落盘",
    "S5  verify　机器核验",
)


def make_q1v2_pipeline(out_png: str, out_pdf: str) -> None:
    """图：问题一 v2 总体流程（S0–S5），数字取自 data/q1_v2/ 的产物。"""
    f = _q1v2_facts()
    fig, ax = plt.subplots(figsize=(QW, QH))
    ax.set_xlim(0, 10.60)
    ax.set_ylim(0, 13.40)
    ax.set_aspect("equal")
    ax.axis("off")

    X, BW = 0.55, 9.50
    CX = X + BW / 2.0
    T = _Q1V2_STAGE_TITLES

    box(ax, X, 12.35, BW, 0.92, [
        "输入：赛题附件 1（CMU-MOSEI 族）",
        f"label-100.xlsx + 100 条原始 mp4；锁定 {f['n_samples']} 个样本",
        "样本清单、标签、官方转写文本三者同源；不删样本、不改标签",
    ], fill=FILL_IN, fs=10.2)

    stages = [
        (T[0], [
            "枚举样本 → registry.json（键集合 = features = manifest = results）",
            "冻结资产 SHA（RoBERTa / whisper / MediaPipe / openSMILE）与配置摘要",
        ], "code/run_q1v2_all.py::stage_prepare"),
        (T[1], [
            f"RoBERTa-base 末层隐状态（{C.ROBERTA_HIDDEN_DIM} 维），按词池化",
            f"词单元 {f['word_units']} 个 / 空白块 {f['ws_chunks']} 个（official_text 的纯函数）",
        ], "code/q1v2_text.py"),
        (T[2], [
            f"冻结 stable-ts 强制对齐；槽 {f['sum_slots']} 个 → 并集得词区间",
            "不丢瞬时词（remove_instant_words=False），零时长词如实保留",
        ], "code/q1v2_align.py"),
        (T[3], [
            f"openSMILE eGeMAPSv02：逐窗 {C.OPENMILE_LLD_DIM} 维 → 词级 "
            f"[均值, 总体标准差] = {C.OPENMILE_LLD_DIM * 2} 维",
            f"MediaPipe FaceLandmarker：逐帧 {C.FACE_MODEL_DIM} 维 → 词级 = "
            f"{C.FACE_MODEL_DIM * 2} 维；人脸帧 {f['face_frames']:,}/{f['frames']:,}",
        ], "code/q1v2_media.py"),
        (T[4], [
            "按词区间 [s,e) 回捞源观测 → 变长、不填充；四条 *_valid 掩码标记有效位",
            f"文本 {C.ROBERTA_HIDDEN_DIM} + 语音 {C.OPENMILE_LLD_DIM * 2} + 视觉 "
            f"{C.FACE_MODEL_DIM * 2} = {C.ROBERTA_HIDDEN_DIM + C.OPENMILE_LLD_DIM * 2 + C.FACE_MODEL_DIM * 2} 维/词",
        ], "code/q1v2_build.py　→　features/*.npz"),
        (T[5], [
            f"{f['n_checks']} 项检查（执行 {f['n_executed']}），"
            f"{f['n_assert']:,} 条断言，失败 {f['n_failed']} 条",
            f"路由 UNCERTAIN_REVIEW {f['modes'].get('UNCERTAIN_REVIEW', 0)}　/　"
            f"TRI_MODAL_WITH_AUDIO_CONTENT_INVALID "
            f"{f['modes'].get('TRI_MODAL_WITH_AUDIO_CONTENT_INVALID', 0)}",
            "另两类需人工听辨证据，本套代码不伪造该结论，故一条不发",
        ], "code/q1v2_verify.py　→　reports/verify_report.json"),
    ]

    y = 11.55
    for i, (title, body, note) in enumerate(stages):
        h = 1.42
        box(ax, X, y - h, BW, h, [title] + body, fill=FILL_MOD, fs=9.6, note=note)
        # 箭头画在**框与框之间的空隙**里（框占 [y-h, y]，空隙是它下面那 0.30）。
        # 从 y 起画会落进框内、被框盖住（框 zorder=2 > 箭头 zorder=1），看上去就没有箭头。
        arrow(ax, (CX, y - h), (CX, y - h - 0.30))
        y -= h + 0.30

    box(ax, X, 0.10, BW, 1.28, [
        "输出与红线",
        f"交付体积 {f['bytes_v2']:,} B（约 {f['bytes_v2'] / 1024 / 1024:.2f} MiB）"
        f"，含 features/ 100 个 NPZ 与两张表 + 台账 + 元数据",
        f"异常只登记不删除（counts_as_deletion ≡ 0）；"
        f"{f['n_samples']} 个样本全部保留（其中 {f['no_face']} 条无检出人脸，仍保留）",
        "运行期零网络：资产身份由 SHA-256 锁定，核验时逐字节比对",
    ], fill=FILL_OUT, fs=9.4)

    _save(fig, out_png, out_pdf)
    print(f"已生成：{out_png}")
    print(f"已生成：{out_pdf}")


def _stage_lines(i: int) -> list:
    """从共用的阶段标题表取第 i 阶段，拆成横排框要的两行。

    _Q1V2_STAGE_TITLES 的元素形如 `S2  align　逐词时间`：双空格是给竖排详图
    对齐用的，横排框里居中显示会偏，故折叠成一个空格；全角空格拆成第二行。
    这样两版流程图的 S 码与阶段名仍只有一处定义，不会各自漂。
    """
    head, _, cn = _Q1V2_STAGE_TITLES[i].partition("　")
    return [re.sub(r"\s+", " ", head), cn]


def _save(fig, out_png: str, out_pdf: str) -> None:
    """存 png/pdf。png 元数据里不写生成工具，故显式清 Software 字段。"""
    kw = dict(bbox_inches="tight", facecolor="white")
    for md in ({"Software": None}, {}):
        try:
            fig.savefig(out_png, dpi=300, metadata=md, **kw)
            break
        except Exception:
            continue
    fig.savefig(out_pdf, **kw)
    plt.close(fig)


def _assert_fits(fig, ax, items, pad=0.05):
    """框内文字必须放得下：逐行量宽，超宽即报错并指出是哪一行。

    图靠 `add_axes([0, 0, 1, 1])` 铺满画布，又用 `set_aspect("equal")`，故
    1 个数据单位就是 1 英寸，窗口宽度除以 dpi 即英寸数，可直接与框宽比。
    图一旦印进论文就不再看脚本了，文字溢出边框靠肉眼发现并不可靠，让这里挡住。
    """
    r = fig.canvas.get_renderer()
    over = []
    for w, lines, fs in items:
        for i, t in enumerate(lines):
            obj = ax.text(0, 0, t, fontsize=fs,
                          fontweight="bold" if i == 0 else "normal")
            wid = obj.get_window_extent(renderer=r).width / fig.dpi
            obj.remove()
            if wid > w - 2 * pad:
                over.append((t, round(wid, 2), round(w - 2 * pad, 2)))
    if over:
        raise AssertionError("框内文字超出框宽（文字 / 需宽 / 可用）：" + "；".join(
            f"{t!r} {a} in > {b} in" for t, a, b in over))


def make_q1v2_pipeline_overview(out_png: str, out_pdf: str) -> None:
    """图：问题一 v2 总体流程的结构视图（论文引用版）。

    与 `make_q1v2_pipeline`（竖排 _详细版，一阶段一框、带代码路径）画的是同一条
    链路，差别在画法与信息取舍：

    * 横排，宽高比约 1.45，把 15.9 cm 的页面宽度用满（竖排详图印出来是一整页高）；
    * 把**逻辑结构**画出来：登记冻结后分出三路——文本特征、逐词时间、音视频
      特征——三路取到的观测在 tables 阶段按词区间汇合，得 922 维词级特征；
    * 只留各阶段的关键产物与口径，代码路径留在 _详细版。

    阶段名取自 _Q1V2_STAGE_TITLES，维度取自 q1v2_contract，计数取自
    data/q1_v2/ 的产物（_q1v2_facts），框内没有一个手写的数。

    画布按 1 数据单位 = 1 英寸设计（add_axes 铺满 + 等比），印成 15.9 cm 宽时
    缩到 0.78 倍，故字号写 10.2 磅、实得约 8.0 磅——取字号的依据是印出来的磅值，
    不是画布上的磅值。
    """
    f = _q1v2_facts()
    d_txt, d_aud, d_vis = C.ROBERTA_HIDDEN_DIM, C.OPENMILE_LLD_DIM, C.FACE_MODEL_DIM
    d_word = d_txt + 2 * d_aud + 2 * d_vis

    fig = plt.figure(figsize=(8.00, 5.50))
    ax = fig.add_axes([0, 0, 1, 1])          # 铺满画布：1 数据单位 = 1 英寸
    ax.set_xlim(0, 8.00)
    ax.set_ylim(0, 5.50)
    ax.set_aspect("equal")
    ax.axis("off")

    # 三行网格：左列（输入 / S0 / 口径）、中列（三路）、右列（S4 / S5 / 输出）。
    # 同一行同高、横平竖直；列间与行间都留走廊，连线一律走走廊，不穿任何框。
    LX, LW = 0.14, 1.94
    AX_, AW = 2.36, 3.12
    RX, RW = 5.76, 2.10
    BUS = 5.62                       # 汇合竖线落在中列与右列之间的走廊里
    ROWS = ((4.24, 5.44), (2.78, 3.98), (1.30, 2.50))
    H, FS = 1.20, 10.2

    def mid(row: int) -> float:
        lo, hi = ROWS[row]
        return (lo + hi) / 2.0

    measured = []                    # (框宽, 行, 字号)，画完统一量宽核验

    def put(x, y, w, lines, fill, lw=1.4, h=None):
        """画一个框并登记量宽核验。h 缺省取网格行高。"""
        hh = H if h is None else h
        box(ax, x, y, w, hh, lines, fill=fill, lw=lw, fs=FS)
        measured.append((w, lines, FS))

    # ---------------- 左列
    put(LX, ROWS[0][0], LW, [
        "输入",
        "赛题附件 1：文本与标签表",
        f"{f['n_samples']} 条原始 mp4 剪辑",
    ], FILL_IN)

    put(LX, ROWS[1][0], LW, [
        *_stage_lines(0),
        "样本清单登记，键集合一致",
        "资产 SHA-256 与配置冻结",
    ], FILL_MOD)

    # 口径框没有 S 码、边框细一档，避免被读成流程里的一个阶段。
    put(LX, ROWS[2][0], LW, [
        "对齐口径",
        "词区间取半开 [s, e)",
        "零时长词如实保留",
        "对齐只给结构信息",
        "不判定文本与讲话一致",
    ], FILL_MOD, lw=1.0)

    # ---------------- 中列：三路（自上而下与 S1 / S2 / S3 同序）
    put(AX_, ROWS[0][0], AW, [
        *_stage_lines(1),
        f"官方文本切出 {f['word_units']} 个词单元",
        f"含 {f['ws_chunks']} 个空白块（纯函数）",
        f"末层隐状态 {d_txt} 维/词",
    ], FILL_MOD)

    put(AX_, ROWS[1][0], AW, [
        *_stage_lines(2),
        "音频重采样至 16 kHz 单声道",
        "与视频 PTS 共享时间零点",
        f"冻结强制对齐器，{f['sum_slots']} 个槽",
        "槽并集得逐词区间 [s, e)",
    ], FILL_MOD)

    put(AX_, ROWS[2][0], AW, [
        *_stage_lines(3),
        f"eGeMAPSv02 低层描述符 {d_aud} 维/窗",
        f"人脸混合系数 {d_vis} 维/帧",
        f"人脸帧 {f['face_frames']:,} / {f['frames']:,}",
    ], FILL_MOD)

    # ---------------- 右列
    put(RX, ROWS[0][0], RW, [
        *_stage_lines(4),
        "按词区间回捞三路观测",
        f"均值+标准差 = {d_word} 维/词",
        "变长、不填充，四条掩码",
    ], FILL_CORE)

    put(RX, ROWS[1][0], RW, [
        *_stage_lines(5),
        f"{f['n_checks']} 项检查，执行 {f['n_executed']} 项",
        f"{f['n_assert']:,} 条断言，失败 {f['n_failed']} 条",
    ], FILL_MOD)

    put(RX, ROWS[2][0], RW, [
        "输出",
        f"features/ {f['n_samples']} 个 NPZ",
        "manifest.csv 等两张表",
        "典型样本逐词对应表",
    ], FILL_OUT)

    # ---------------- 贯穿全程的约束：横贯整幅，不与流程连箭头。
    # 高度必须显式给：缺省是网格行高 1.20，会顶到第三行框（其下沿 1.30）里去。
    put(LX, 0.13, 7.72, [
        "贯穿全程的三条约束",
        "运行期零网络（资产 SHA-256 锁定）· 不删样本与标签（异常只登记）",
        "交付物不含身份信息",
    ], FILL_IN, h=0.80)

    # ---------------- 连线
    arrow(ax, (LX + LW / 2, ROWS[0][0]), (LX + LW / 2, ROWS[1][1]))      # 输入→S0
    for r in range(3):                                                     # S0→三路
        arrow(ax, (LX + LW, mid(1)), (AX_, mid(r)))
    for r in range(3):                                                     # 三路→汇合线
        arrow(ax, (AX_ + AW, mid(r)), (BUS, mid(r)), style="-")
    arrow(ax, (BUS, mid(2)), (BUS, mid(0)), style="-")                     # 竖汇合线
    arrow(ax, (BUS, mid(0)), (RX, mid(0)))                                 # 汇合→S4
    arrow(ax, (RX + RW / 2, ROWS[0][0]), (RX + RW / 2, ROWS[1][1]))        # S4→S5
    arrow(ax, (RX + RW / 2, ROWS[1][0]), (RX + RW / 2, ROWS[2][1]))        # S5→输出

    _assert_fits(fig, ax, measured)
    _save(fig, out_png, out_pdf)
    print(f"已生成：{out_png}")
    print(f"已生成：{out_pdf}")


def make_q1v2_timeline(out_png: str, out_pdf: str, sample_index: int = 0) -> None:
    """图：典型样本的逐词对齐时间线。

    横轴为秒。每个词画三行：词区间 / 语音 LLD 窗的**中心时刻跨度** / 人脸帧的
    PTS 跨度。画的是**交付的 correspondence CSV 里写明的跨度与计数**，
    不按计数在区间内均分取点——那会给出 CSV 并不支持的精确位置。
    """
    import csv

    f = _q1v2_facts()
    entry = f["corr"]["files"][sample_index]
    path = os.path.join(Q1V2, "correspondence", entry["csv"])
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    def _span(text):
        if not text or "–" not in text:
            return None
        a, b = text.split("–")
        return float(a), float(b)

    valid, skipped, audio, vision = [], [], [], []
    for r in rows:
        if r["word_start_sec"]:
            valid.append((float(r["word_start_sec"]), float(r["word_end_sec"]),
                          r["char_span_original"], int(r["n_audio_windows"]),
                          int(r["n_face_frames"]), int(r["n_frames_in_span"])))
            audio.append(_span(r["audio_center_span_sec"]))
            vision.append(_span(r["face_pts_span_sec"]))
        else:
            skipped.append((r["char_span_original"], r["note"]))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(TW, TH), sharex=True,
        gridspec_kw={"height_ratios": [1.9, 1.25], "hspace": 0.16})

    tmax = max(w[1] for w in valid)
    ax1.set_xlim(-tmax * 0.02, tmax * 1.04)
    ax1.set_ylim(0.0, 3.55)
    ax1.set_yticks([0.62, 1.62, 2.62])
    ax1.set_yticklabels(["视频帧", "语音窗", "词区间"], fontsize=10)
    ax1.tick_params(axis="y", length=0)
    for sp in ("top", "right", "left"):
        ax1.spines[sp].set_visible(False)

    WY, AY, VY, BH = 2.62, 1.62, 0.62, 0.30
    for i, (s, e, word, nw, nf, nfall) in enumerate(valid):
        # 词区间：灰条 + 竖排词形（横轴是秒，词宽不足以横排文字）
        ax1.add_patch(FancyBboxPatch(
            (s, WY - BH / 2), max(e - s, tmax * 0.0012), BH,
            boxstyle="round,pad=0.002,rounding_size=0.02",
            linewidth=1.0, edgecolor=EDGE, facecolor="#d9d9d9", zorder=3))
        ax1.text((s + e) / 2, WY + BH / 2 + 0.06, _mt(word), rotation=90,
                 ha="center", va="bottom", fontsize=8.0, color=EDGE, zorder=4)
        if audio[i]:
            a0, a1 = audio[i]
            ax1.plot([a0, a1], [AY, AY], color="#1f6fb4", lw=4.2,
                     solid_capstyle="butt", zorder=3)
        if vision[i]:
            v0, v1 = vision[i]
            ax1.plot([v0, v1], [VY, VY], color="#b4581f", lw=4.2,
                     solid_capstyle="butt", zorder=3)

    note = ("未参与聚合的词（CSV 中 word_time_valid=0，不凑观测）："
            + "、".join(_mt(w) for w, _ in skipped)) if skipped else ""
    ax1.set_title(
        f"典型样本 {_mt(entry['sample_key'])} 的逐词对应（共 {len(rows)} 个词单元）\n"
        f"灰＝词区间　蓝＝语音 LLD 窗中心跨度　橙＝人脸帧 PTS 跨度"
        + (f"\n{note}" if note else ""),
        fontsize=11.0, pad=10)

    # 下 panel：逐词源观测计数。**横轴仍是秒**（与上 panel 同一条时间轴），
    # 柱位取词区间的中点、柱宽按词时长缩放——若改用词序号当横轴，两 panel 就
    # 不在同一个坐标系里了，sharex 会把上 panel 挤到左侧一角。
    centers = [(v[0] + v[1]) / 2 for v in valid]
    widths = [max((v[1] - v[0]) * 0.40, tmax * 0.0035) for v in valid]
    ax2.bar([c - w / 2 for c, w in zip(centers, widths)],
            [v[3] for v in valid], width=widths,
            color="#1f6fb4", label="语音 LLD 窗数")
    ax2.bar([c + w / 2 for c, w in zip(centers, widths)],
            [v[4] for v in valid], width=widths,
            color="#b4581f", label="有效人脸帧数")
    ax2.set_ylabel("源观测计数", fontsize=10)
    ax2.set_xlabel("共享时间轴（秒）　—　柱位即词区间中点", fontsize=10.5)
    ax2.set_xticks(centers)
    ax2.set_xticklabels([_mt(v[2]) for v in valid], rotation=90, fontsize=8.0)
    ax2.legend(fontsize=9.0, frameon=False, ncol=2, loc="upper right")
    ax2.grid(axis="y", lw=0.5, alpha=0.35, zorder=0)
    ax2.set_axisbelow(True)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    # xlim 在**两张都画完后**统一设定：sharex 下后画的那个若触发了 autoscale，
    # 会把共享范围撑到自己的数据范围，先设的 xlim 会被覆盖。
    for ax in (ax1, ax2):
        ax.set_xlim(-tmax * 0.02, tmax * 1.04)

    _save(fig, out_png, out_pdf)
    print(f"已生成：{out_png}")
    print(f"已生成：{out_pdf}")


def main(argv=None) -> int:
    import argparse
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="生成论文流程图")
    ap.add_argument("--which", default="all",
                    choices=["all", "overview", "q1", "q1v2"],
                    help="all=全生成（默认）；overview=三问总览；q1=问题一旧架构详图；"
                         "q1v2=问题一 v2 总流程图（横排结构图 + 竖排_详细版）"
                         "+ 典型样本时间线")
    a = ap.parse_args(argv)

    if a.which in ("all", "overview"):
        make_overview(os.path.join(here, "图1_三问总体分析流程图.png"),
                      os.path.join(here, "图1_三问总体分析流程图.pdf"))
    if a.which in ("all", "q1"):
        make(os.path.join(here, "图2_问题一分析总流程图.png"),
             os.path.join(here, "图2_问题一分析总流程图.pdf"))
    if a.which in ("all", "q1v2"):
        # 论文引用的 图1 是横排结构图（三路汇合）；带代码路径的竖排详图另存
        # _详细版，两者由同一次运行一起重生成，不会各自变旧。
        make_q1v2_pipeline_overview(os.path.join(here, "图1_问题一v2总体流程.png"),
                                    os.path.join(here, "图1_问题一v2总体流程.pdf"))
        make_q1v2_pipeline(os.path.join(here, "图1_问题一v2总体流程_详细版.png"),
                           os.path.join(here, "图1_问题一v2总体流程_详细版.pdf"))
        make_q1v2_timeline(os.path.join(here, "图2_典型样本v2对齐时间线.png"),
                           os.path.join(here, "图2_典型样本v2对齐时间线.pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
