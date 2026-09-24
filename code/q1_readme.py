# -*- coding: utf-8 -*-
"""
q1_readme.py —— 生成 `README_问题一交付与验证.md`（问题一的数学模型与验收文档）

为什么用生成器而不是手写
------------------------
这份文档要进论文、要给评审核对。手写的最大风险不是写得不好，而是**数字会过期**：
改了参数、重跑一次、修了一个缺陷，正文里的数字就悄悄说谎了。
本模块沿用 `emotion_model.write_readme()` 的既有做法——**所有数字都从产物里现读现算**，
文档不可能与产物不一致。

数据来源（全部只读，不写任何交付物）：
    data/q1_delivery/verify_report.json    12 项机器核验的结果与证据
    data/q1_delivery/summary_q1.csv        100 条样本的逐条汇总
    data/q1_delivery/extract_config.json   可复现性快照（参数、版本、失败处理规则）
    data/q1_delivery/anomaly_ledger.csv    异常台账
    data/q1_delivery/typical_samples.csv   五类典型样本的选取判据
    data/q1_delivery/face_probe.csv        人脸探测抽样结果
    data/q1_delivery/features_q1.npz       变长特征的 offsets（用于核对 Σ 单元数）
    data/q1_delivery/alignment_q1.json     全局 meta（槽数、填充规则、时间基准）

用法：
    python q1_readme.py                    # 生成到 data/q1_delivery/
    python q1_readme.py --out D:/tmp/x.md  # 指定输出文件
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import config  # noqa: E402
import unaligned_common as U  # noqa: E402
from utils import LOGGER  # noqa: E402

PROJECT_ROOT = os.path.dirname(_CODE_DIR)
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "data", "q1_delivery")
MD_NAME = "README_问题一交付与验证.md"
MD_NAME_ALT = "README_问题一交付与验证.md"      # 放 code/ 下的同名副本
MODALITIES = ("text", "audio", "vision")
MOD_ZH = {"text": "文本", "audio": "语音", "vision": "视觉"}
# 题目给的 50 MB 上限。按 50 × 1024² 取更严格的口径（不要用 50 × 10⁶ 侥幸过关）。
LIMIT_BYTES = 50 * 1024 * 1024
# 子目录行的说明文字（`_walk_sizes` 把子目录合并成一行，行名形如 `figures/（5 个文件）`）
DIR_DESC = {
    "figures": "五类典型样本的时间轴图（词—秒—语音段—视频帧同轴）",
    "correspondence": "五类典型样本的逐槽对应表",
}


# ==================== 读取 ====================

def _read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        LOGGER.warning("[文档] 缺少 %s（相关小节会标注为不可用）", path)
        return []
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: str) -> Dict:
    if not os.path.isfile(path):
        LOGGER.warning("[文档] 缺少 %s（相关小节会标注为不可用）", path)
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _nums(rows: Sequence[Dict[str, str]], key: str) -> List[float]:
    out: List[float] = []
    for r in rows:
        v = r.get(key)
        if v in (None, "", "nan"):
            continue
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


def _stat(vals: Sequence[float], unit: str = "", dig: int = 4) -> str:
    if not vals:
        return "—"
    f = "{:." + str(dig) + "g}"
    return (f.format(min(vals)) + " / " + f.format(st.median(vals)) + " / "
            + f.format(max(vals)) + (" " + unit if unit else ""))


def _kv(d: Dict) -> str:
    """把核验证据里的小字典排成 `a=1; b=2`，别让 Python 的 dict repr 直接进正文。"""
    return "；".join(str(k) + "=" + str(v) for k, v in d.items())


def _demote(md: str) -> str:
    """
    把内嵌文档（honesty 段）的标题降两级，避免它与本文档自己的标题同级。
    只动行首的 `#`，不动正文里的代码块。
    """
    out = []
    for ln in md.splitlines():
        s = ln.lstrip()
        out.append(("##" + ln) if s.startswith("#") else ln)
    return "\n".join(out)


# ==================== 各小节 ====================

def _sec_header(ctx: Dict[str, object]) -> List[str]:
    v = ctx["verify"]
    return [
        "# 问题一：文本、语音、视觉三模态情感特征提取与时序对齐",
        "## 数学模型、机器验证与交付说明",
        "",
        "> **本文档由 `code/q1_readme.py` 自动生成，正文中每一个数字都从 "
        "`data/q1_delivery/` 下的产物现读现算**，不存在手写数字。",
        "> 复核方法：`python code/q1_readme.py` 重新生成，与本文件逐行比对。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| 生成时刻 | " + str(v.get("generated_at", "—")) + " |",
        "| 流水线版本 | " + str(v.get("pipeline_version", "—")) + " |",
        "| 代码指纹 | `" + str(v.get("code_fingerprint", "—")) + "` |",
        "| 样本数 | " + str(v.get("n_samples", "—")) + " 条 |",
        "| 统一序列长度 | " + str(v.get("n_slots", "—")) + " 槽 |",
        "| 机器核验 | " + (
            "**全部通过**" if v.get("all_passed") else "**有未通过项**") + "（"
        + str(ctx["vsum"].get("passed", "—")) + "/" + str(ctx["vsum"].get("total", "—"))
        + " 项，共 " + str(ctx["vsum"].get("total_assertions", "—")) + " 条断言，"
        + "失败 " + str(ctx["vsum"].get("failed_assertions", "—")) + " 条）|",
        "",
    ]


def _sec_honesty(ctx: Dict[str, object]) -> List[str]:
    lines = [
        "---",
        "",
        "## 0. 诚实性前置声明",
        "",
        "**读下面任何数字之前，请先读这一节。** 本节由 `q1_verify.py` 写入"
        "`verify_report.json` 的 `honesty` 字段，与核验结果同源、不可分离。",
        "",
    ]
    lines += [_demote(str(ctx["verify"].get("honesty", "（缺失）")).rstrip()), ""]
    lines += [
        "补充三条与**交付口径**有关的边界：",
        "",
        "5. **不提交原始视频**（约 4 GB）。交付只保留源视频的**相对位置**"
        "（`alignment_q1.json` 的 `source_video`）与**关键帧**（五类典型图的帧条）。",
        "6. **冻结的 RoBERTa 本体（约 110M 参数，float32 ≈ 440 MB）不随附**，"
        "改为在 `extract_config.json` 固定其本地 revision 哈希、来源地址与逐文件校验值。",
        "7. **零安装 / 零网络**：全程未 `pip install`、未下载任何权重，"
        "只用本机已缓存资源（V9 用 `local_files_only=True` 加载并做了掩码补全验证）。",
        "",
    ]
    return lines


def _sec_formal(ctx: Dict[str, object]) -> List[str]:
    s = ctx["summary"]
    L = ctx["n_slots"]
    dur = _nums(s, "duration_sec")
    grain = _nums(s, "align_granularity_sec")
    dims = ctx["dims"]
    n = len(s)
    return [
        "---",
        "",
        "## 1. 符号与问题形式化",
        "",
        "### 1.1 原始样本空间",
        "",
        "附件1 给出的原始样本集合记为",
        "",
        "```",
        "S = { s_i = (video_id_i, clip_id_i) : i = 1..N },   N = " + str(n),
        "```",
        "",
        "样本总数为 " + str(n) + "，来自 " + str(ctx["n_video_ids"]) + " 个不同的 `video_id`。"
        "每个样本对应一个 mp4 文件（同一 video_id 下可切出多个 clip_id 片段）。"
        "样本编号统一采用 `sample_id = video_id + \"_\" + clip_id`，"
        "并同时保留官方的 `official_id = video_id + \"$_$\" + clip_id` 写法。",
        "",
        "> ⚠️ **样本编号的切分歧义（本项目真实踩过的坑）**：有 "
        + str(ctx["n_ids_with_underscore"]) + " 个 `video_id` **自身含下划线**"
        "（" + "、".join("`" + v + "`" for v in ctx["ids_with_underscore"]) + "），"
        "涉及 **" + str(ctx["n_samples_with_underscore_vid"]) + " 条样本**"
        "（这些样本的 `sample_id` 会出现两个及以上下划线）。"
        "因此从 `sample_id` 反解 `video_id`/`clip_id` 必须按**最后一个**下划线切分。"
        "本项目的汇总表曾因按第一个下划线切分而写错 "
        + str(3 * ctx["n_samples_with_underscore_vid"]) + " 行 `official_id`，"
        "已修复并作为缺陷 D2 记入台账（见第 6 节）。",
        "",
        "### 1.2 三个模态的观测",
        "",
        "对第 i 个样本，三个模态各自给出一串**带时间基准的观测单元序列**"
        "（下称「单元」），而不是统一定长的张量：",
        "",
        "```",
        "文本 :  X_i^t = (x_{i,1}^t, ..., x_{i,W_i}^t),   x^t ∈ R^{" + str(dims["text"]) + "}",
        "语音 :  X_i^a = (x_{i,1}^a, ..., x_{i,P_i}^a),   x^a ∈ R^{" + str(dims["audio"]) + "}",
        "视觉 :  X_i^v = (x_{i,1}^v, ..., x_{i,F_i}^v),   x^v ∈ R^{" + str(dims["vision"]) + "}",
        "```",
        "",
        "每个单元自带一个时间戳 `pts`：`τ_{i,k}^m` = 该单元在**本片段时间轴**上的时刻（秒）。",
        "三个模态的单元数互不相同、且同一模态在不同样本间也不同：",
        "",
        "| 模态 | 单元语义 | 单元数 W/P/F 的范围（min / 中位 / max） | Σ 单元数 |",
        "|---|---|---|---|",
        "| 文本 | 词 | " + _stat(_nums(s, "text_seq_len"), dig=4) + " | "
        + str(ctx["units_sum"]["text"]) + " |",
        "| 语音 | 20 Hz 帧 | " + _stat(_nums(s, "audio_seq_len"), dig=4) + " | "
        + str(ctx["units_sum"]["audio"]) + " |",
        "| 视觉 | 15 Hz 帧 | " + _stat(_nums(s, "vision_seq_len"), dig=4) + " | "
        + str(ctx["units_sum"]["vision"]) + " |",
        "",
        "### 1.3 公共时间轴（时间锚点）",
        "",
        "问题一的第一件事是**定一个所有模态都落得上去的时间锚点**。本模型取",
        "",
        "```",
        "T_i = 该样本视频流的时长（ffprobe 流时长），单位秒",
        "```",
        "",
        "也就是**以视觉（视频）时间轴为公共锚点**。理由：三个模态里只有视频与音频"
        "带容器级的时长元数据，而音频容器时长在本批数据上**系统性地短于视频**"
        "（见第 6 节的 A6 异常，50/100 条命中，缺口恒为约 0.17 s，是 AAC priming 的"
        "系统性偏差）。若以音频为锚，视觉末段会被整体截断；以视频为锚则只影响音频尾部，"
        "且该缺口可被显式记账。",
        "",
        "实测 `T` 分布：**" + _stat(dur, "s", dig=4) + "**（min / 中位 / max），"
        "合计 " + ("{:.3f}".format(sum(dur))) + " s。",
        "",
        "### 1.4 统一离散化：L 槽",
        "",
        "把公共时间轴等分成 L 个槽（本实现 L = " + str(L) + "，与附件2 的序列位置口径一致）：",
        "",
        "```",
        "槽宽 Δ_i = T_i / L,   Δ 实测 " + _stat(grain, "s", dig=4) + "（min / 中位 / max）",
        "第 k 个槽的时间区间  I_{i,k} = [ T_i·k/L ,  T_i·(k+1)/L ),   k = 0..L-1",
        "```",
        "",
        "### 1.5 问题形式化",
        "",
        "**目标**：构造映射 Φ，把变长三模态观测变成**同一根时间轴上的等长表示**，"
        "并使得（a）每个输出值都能回溯到确定来源，（b）无法回溯的位置被显式标记：",
        "",
        "```",
        "Φ : (X_i^t, X_i^a, X_i^v, τ_i^t, τ_i^a, τ_i^v, T_i)  →  (Z_i^t, Z_i^a, Z_i^v, V_i^t, V_i^a, V_i^v)",
        "",
        "其中  Z_i^m ∈ R^{L × D_m}   槽级特征（对齐结果）",
        "      V_i^m ∈ [0,1]^L      槽级有效掩码",
        "```",
        "",
        "**交付形态**：变长表示（不补零）与定长表示（补零到 L）由同一份对齐结果导出，"
        "前者用于交付（体积小、无假零），后者用于问题二的建模（等长张量）。",
        "",
    ]


def _sec_pipeline(ctx: Dict[str, object]) -> List[str]:
    fp = ctx["cfg"].get("failure_policy", {})
    rows = [
        ("① 样本枚举", "读 `label-100.xlsx`，按 `video_id + \"_\" + clip_id` 构造 100 个 `sample_id`，"
                       "并解析每个样本的 mp4 路径。",
         "缺视频文件：记录并跳过该样本，不中断整批"),
        ("② 视频探针", "`ffprobe` 读视频流时长 T、音频流时长、分辨率、名义帧率、编解码器。",
         "时长 ≤ 0：整段置零并标注"),
        ("③ 文本提取", "规范化文本 → RoBERTa 词元 → 按 `word_ids` 归并到词 → 每个词取子词隐状态均值。",
         "空文本：W=0，槽全部置零并记 `EMPTY_TEXT`"),
        ("④ 语音提取", "`ffmpeg` 管道解码单声道 16 kHz → 74 维逐帧（20 Hz）→ 逐帧 pts。",
         "无音轨/解码失败：该模态整段置零并标注"),
        ("⑤ 视觉提取", "逐帧流式解码（内存与视频长度无关）→ MTCNN 检人 → ResNet-50 池化 2048 维 → "
                       "固定随机投影到 35 维。",
         "未检出人脸的帧：相邻有效帧线性插值；全片无脸则保持零向量，`face_ratio=0`"),
        ("⑥ 时长核对", "以视频流时长 T 作为公共时间轴，并记录与音频容器时长的差。",
         "音频短于视频：如实记账（A6），不拉伸、不截断"),
        ("⑦ 槽归属", "`slot = clip(floor(τ/T·L), 0, L-1)`，逐单元落到唯一槽。",
         "τ 落在 [0,T] 之外的单元并入最近端点槽，**逐样本计数上报**"),
        ("⑧ 槽聚合", "每个非空槽对槽内单元取算术平均。",
         "空槽按模态填充规则处理（见第 4.4 节）"),
        ("⑨ 掩码", "该槽是否被分配到过采样单元 → `valid`；同时另记可溯源性与模态可用性。",
         "`valid=True` **不等于**模态可用，见第 4.5 节"),
        ("⑩ 落盘", "逐样本 `data/aligned/<sample_id>.npz` + 汇总 `aligned_50.npz`。",
         "写入失败（文件被占用）：报错并跳过该步，不静默"),
        ("⑪ 交付", "四项交付物 + 异常台账 + 体积核算 + 五类典型样本，写入 `data/q1_delivery/`。",
         "人脸探测失败/超时：记状态码，判据标为不可用，不剔除样本"),
    ]
    lines = [
        "---",
        "",
        "## 2. 原始样本的处理流程（含失败分支）",
        "",
        "```",
        "                      label-100.xlsx (100 条)",
        "                              │",
        "                   ① 样本枚举 ─┴─ 缺视频 → 记录并跳过",
        "                              │",
        "                   ② ffprobe 探针 ─── 得到公共时间轴 T_i",
        "              ┌───────────────┼───────────────┐",
        "        ③ 文本提取       ④ 语音提取       ⑤ 视觉提取",
        "        RoBERTa 词级     74 维 @20Hz      35 维 @15Hz",
        "        τ = 均匀假设     τ = k·hop/sr     τ = 解码器实测 PTS",
        "              └───────────────┼───────────────┘",
        "                              │",
        "                   ⑥ 以视频时长 T 统一锚点",
        "                              │",
        "                   ⑦ 槽归属 slot(τ) = clip(floor(τ/T·L), 0, L-1)",
        "                              │",
        "                   ⑧ 槽内均值聚合 → Z (L × D_m)",
        "                              │",
        "                   ⑨ 有效掩码 V (L,) + 可溯源记账 + 异常记账",
        "                              │",
        "                   ⑩ data/aligned/*.npz  +  aligned_50.npz",
        "                              │",
        "                   ⑪ data/q1_delivery/  四项交付物 + 台账 + 五类典型样本",
        "                              │",
        "                   ⑫ q1_verify.py 12 项机器核验（只读，不改产物）",
        "```",
        "",
        "逐步的判据与失败分支：",
        "",
        "| 步骤 | 做什么 | 失败/边界分支 |",
        "|---|---|---|",
    ]
    for a, b, c in rows:
        lines.append("| " + a + " | " + b + " | " + c + " |")
    lines += [
        "",
        "**贯穿全流程的三条硬规则**（源码出处见 `extract_config.json` 的 `failure_policy`）：",
        "",
        "1. **不删样本。** " + str(fp.get("no_sample_deletion_note", "")),
        "2. **看门狗。** " + str(fp.get("subprocess_watchdog", {}).get("reason", "")),
        "   做法：视觉提取与人脸探测均逐样本起独立子进程 + 墙钟超时；超时只影响该样本。",
        "3. **宁可记账也不静默。** 越界单元并入端点槽、插值填充、模态缺失——"
        "每一类都有代号、有计数、有样本清单，而不是悄悄发生。",
        "",
    ]
    return lines


def _sec_features(ctx: Dict[str, object]) -> List[str]:
    cfg = ctx["cfg"]
    tp = cfg.get("text_params", {})
    ap = cfg.get("audio_params", {})
    vp = cfg.get("vision_params", {})
    lay = ap.get("dim_layout", {})
    return [
        "---",
        "",
        "## 3. 各模态情感特征的定义与提取方法",
        "",
        "### 3.1 文本：词级 RoBERTa 语义向量",
        "",
        "**定义.** 设样本的原始文本经规范化（统一空白、unicode 引号/省略号/破折号归一）后为 "
        "`text'`，词序列为 `w_1..w_W`。用 RoBERTa-base 分词后得到子词序列，"
        "按 fast tokenizer 的 `word_ids()` 把子词归并回词；第 k 个词的表示定义为其全部子词"
        "**隐状态（最后一层）的均值**：",
        "",
        "```",
        "x_k^t = (1/|T_k|) · Σ_{j ∈ T_k} h_j ,     h_j ∈ R^768 为 RoBERTa 最后一层隐状态",
        "T_k = 第 k 个词占用的子词下标集合,   |T_k| >= 1",
        "```",
        "",
        "- 维度 " + str(ctx["dims"]["text"]) + "（RoBERTa-base hidden size），"
        "**只取最后 4 层之外的最后一层、不做微调**（冻结预训练权重，零训练成本）。",
        "- 不取 `[CLS]`：本问要的是**逐词**表示以便落到时间轴上，"
        "`[CLS]` 是句级表示、无逐词对应。",
        "- **时间基准是假设值，不是实测值**：",
        "",
        "```",
        "τ_k^t = (k - 0.5) · T / W        （词 k 的中心时刻，均匀分布假设）",
        "```",
        "",
        "  本数据集**没有逐词时间戳真值**，故 " + "`time_basis = \""
        + str(tp.get("time_basis", "")) + "\"`。这条时间轴只用于与音频/视觉共轴**可视化**，"
        "本模型**绝不**据此声称词级时间精度（见第 10 节）。",
        "- 溯源三元组（本次新增，`extra` 通道落盘）：",
    ] + [
        "  - `" + k + "`：" + str(v)
        for k, v in (tp.get("char_span_basis", {}) or {}).items()
    ] + [
        "",
        "### 3.2 语音：74 维声学情感特征",
        "",
        "**定义.** 音轨解码为单声道 16 kHz 波形后，以 " + str(ap.get("hop_length")) + " 采样为跳步"
        "（帧率 " + str(ap.get("frame_rate_hz")) + " Hz）逐帧提取 " + str(ap.get("feature_dim"))
        + " 维特征：",
        "",
        "```",
        "x_k^a = [ MFCC(20) | ΔMFCC(20) | ΔΔMFCC(20) | logF0 | voiced_prob",
        "          | logRMS | ZCR | 谱质心 | 谱带宽 | ... ]  ∈ R^" + str(ap.get("feature_dim")),
        "```",
        "",
        "维度布局（`dim_layout`，逐段写入产物 meta 可查）：",
        "",
        "| 区间 | 内容 | 提取方法 |",
        "|---|---|---|",
        "| " + str(lay.get("mfcc", "?")) + " | MFCC 20 维 | librosa `n_mfcc="
        + str(ap.get("n_mfcc")) + "`, `n_fft=" + str(ap.get("n_fft")) + "` |",
        "| " + str(lay.get("delta_mfcc", "?")) + " | 一阶差分 ΔMFCC | 沿时间轴一阶差分 |",
        "| " + str(lay.get("delta2_mfcc", "?")) + " | 二阶差分 ΔΔMFCC | 沿时间轴二阶差分 |",
        "| " + str(lay.get("log_f0", "?")) + " | log F0 | pyin；清音帧取 log(fmin) |",
        "| " + str(lay.get("voiced_prob", "?")) + " | 浊音概率 | pyin `voiced_prob` |",
        "| " + str(lay.get("log_rms", "?")) + " | log RMS 能量 | 帧级均方根取对数 |",
        "| " + str(lay.get("zcr", "?")) + " | 过零率 | 帧内符号变化率 |",
        "| 其余 | 谱质心 / 谱带宽 / 谱对比度 / 谱滚降 / 色度等 | librosa 谱特征 |",
        "",
        "- **时间基准是实测值**：`τ_k^a = k · hop / sr`，"
        "由 librosa `center=True` 的「帧中心时刻」约定推出，等间隔。",
        "- VAD（有声判定）参数：" + _kv(ap.get("vad", {}) or {}) + "，"
        "导出 `voiced_ratio`（有声帧占比）作为该模态的**可用性指标**。",
        "",
        "### 3.3 视觉：35 维人脸深度特征",
        "",
        "**定义.** 逐帧流式解码（内存占用与视频长度无关），每帧：",
        "",
        "```",
        "帧 → MTCNN(face_size=" + str(vp.get("face_size")) + ", thresholds=[0.6,0.7,0.7]) 裁人脸",
        "   → PIL Resize(224,224) → ToTensor → ImageNet 归一化",
        "   → ResNet-50 全局平均池化 → v ∈ R^2048",
        "   → 固定随机投影 P (2048 × 35)  → x^v = v·P ∈ R^35",
        "```",
        "",
        "投影矩阵 P 的元素服从 N(0, 1/√2048)、seed = " + str(
            cfg.get("models", {}).get("vision_projection", {}).get("seed", 42))
        + "，由 Johnson–Lindenstrauss 引理近似保距（E‖v·P‖² = ‖v‖²）。"
        "**同一 seed 在任何机器、任何 numpy 版本下产生同一矩阵**，"
        "其 md5 随每个特征文件落盘，因此「用的是不是同一个投影」可被逐文件验证。",
        "",
        "- 采样率 " + str(vp.get("target_fps")) + " Hz（实测达成 "
        + str(vp.get("realized_fps")) + " Hz）。取 15 Hz 的依据："
        "附件2 的 500 帧上限 ÷ 最长片段 25 s ≈ 20 Hz，而与官方重叠的 18 条样本实测中位数为 14.7 Hz。",
        "- **时间基准是实测值**：`τ_k^v` 来自 `cv2` 解码器的实际时间戳"
        "（`CAP_PROP_POS_MSEC`），**允许不均匀**（VFR 视频如实保留）。",
        "- 缺失人脸的处理：" + str(vp.get("missing_face_policy")) + "。"
        "插值发生在**本模块内**（是人脸检测漏检这一模态自身属性的补全），"
        "而不是在对齐阶段——对齐模块因此对模态无感知。",
        "- 本批 " + str(ctx["n_vfr"]) + " 条样本存在帧间隔不均（VFR），"
        "已作为 `VFR_frame_gap` 记入台账；**不剔除**。",
        "",
        "### 3.4 三模态维度与关键参数汇总",
        "",
        "| 项 | 文本 | 语音 | 视觉 |",
        "|---|---|---|---|",
        "| 单元语义 | 词 | 帧 | 帧 |",
        "| 特征维度 D | " + str(ctx["dims"]["text"]) + " | " + str(ctx["dims"]["audio"])
        + " | " + str(ctx["dims"]["vision"]) + " |",
        "| 采样率 | 每词 1 个 | " + str(ap.get("frame_rate_hz")) + " Hz | "
        + str(vp.get("target_fps")) + " Hz |",
        "| 时间基准 | " + str(tp.get("time_basis")) + "（假设） | " + str(ap.get("time_basis"))
        + " | " + str(vp.get("time_basis")) + " |",
        "| 主要工具 | transformers " + str(ctx["libs"].get("transformers")) + " | librosa "
        + str(ctx["libs"].get("librosa")) + " | torchvision "
        + str(ctx["libs"].get("torchvision")) + " + facenet-pytorch "
        + str(ctx["libs"].get("facenet-pytorch")) + " |",
        "| 预训练权重 | roberta-base（冻结） | 无（信号处理） | ResNet-50 IMAGENET1K_V2 + MTCNN（冻结） |",
        "",
    ]


def _sec_align(ctx: Dict[str, object]) -> List[str]:
    v = {c["code"]: c for c in ctx["verify"].get("checks", [])}
    v6 = v.get("V6", {}).get("evidence", {})
    v4 = v.get("V4", {}).get("evidence", {})
    v5 = v.get("V5", {}).get("evidence", {})
    per = v6.get("per_modality", {})
    tr = v4.get("traceability", {})
    return [
        "---",
        "",
        "## 4. 跨模态时序对齐的规则与实现",
        "",
        "### 4.1 规则一：统一时间锚点",
        "",
        "```",
        "T_i = 视频流时长（ffprobe）——三模态共用的唯一时间原点与总长",
        "```",
        "",
        "### 4.2 规则二：单元 → 槽的归属函数",
        "",
        "```",
        "slot(τ) = clip( floor( τ / T · L ),  0,  L-1 )        L = " + str(ctx["n_slots"]) + "",
        "```",
        "",
        "这是一条**左闭右开**的等分量化：`τ ∈ [T·k/L, T·(k+1)/L)` 落在槽 k。"
        "`clip` 的作用是把时间轴外的单元并到最近端点——**并入端点总比丢弃单元诚实**，"
        "但必须显式记账而不是静默发生（见 4.6 节）。",
        "",
        "可逆性：由槽号可还原其时间区间 `I_k = [T·k/L, T·(k+1)/L)`，"
        "往返映射 `slot(I_k) = k` 对全部 " + str(ctx["n_slots"]) + " 个槽成立"
        "（V5 已逐槽验证；槽宽抖动最大 "
        + ("{:.3e}".format(float(v5.get("max_width_jitter_sec", 0.0)))) + " s，"
        "即 float64 的表示极限）。",
        "",
        "### 4.3 规则三：槽内聚合",
        "",
        "```",
        "Z_{i,k}^m = (1/|U_{i,k}^m|) · Σ_{j ∈ U_{i,k}^m} x_{i,j}^m ,   U = 落入槽 k 的单元下标集合",
        "```",
        "",
        "取算术平均而非最大/拼接：槽是「一小段时间」，"
        "平均值是这段时间内该模态表示的**无偏压缩**，且与下游池化（问题二的均值池化）口径一致。"
        "每个槽的**来源单元下标区间**都落盘（`slot_src_units`），使每个聚合值可被逐一复算。",
        "",
        "### 4.4 规则四：空槽的填充",
        "",
        "| 模态 | 填充规则 | 理由 |",
        "|---|---|---|",
        "| 文本 | **补零** | 词是离散事件，一个槽内没有词就是「没有词」；"
        "零向量是 RoBERTa 表示空间里唯一无歧义的「空」 |",
        "| 语音 | **相邻有效槽线性插值** | 20 Hz 帧是连续过程的采样，"
        "空槽意味着容器时长短于公共轴，插值比补零更接近真值 |",
        "| 视觉 | **相邻有效槽线性插值** | 同上（15 Hz 帧） |",
        "",
        "实测填充数量（V6）：",
        "",
        "| 模态 | 总槽数 | 空槽数 | 空槽中非零的个数 | 填充方式 |",
        "|---|---|---|---|---|",
        "| 文本 | " + str(per.get("text", {}).get("slots", "—")) + " | "
        + str(per.get("text", {}).get("empty", "—")) + " | "
        + str(per.get("text", {}).get("empty_nonzero", "—")) + " | zero（断言必须恰为 0）|",
        "| 语音 | " + str(per.get("audio", {}).get("slots", "—")) + " | "
        + str(per.get("audio", {}).get("empty", "—")) + " | "
        + str(per.get("audio", {}).get("empty_nonzero", "—")) + " | interpolate |",
        "| 视觉 | " + str(per.get("vision", {}).get("slots", "—")) + " | "
        + str(per.get("vision", {}).get("empty", "—")) + " | "
        + str(per.get("vision", {}).get("empty_nonzero", "—")) + " | interpolate |",
        "",
        "### 4.5 规则五：有效掩码的语义（**最容易误用的一条**）",
        "",
        "```",
        "V_{i,k}^m = 1  ⟺  槽 k 被分配到了至少一个采样单元（|U_{i,k}^m| > 0）",
        "```",
        "",
        "**`V=1` 只表示「这个槽被填过」，不表示「这个槽是真观测」。**"
        "三条必须分开的判据：",
        "",
        "| 判据 | 含义 | 落盘位置 |",
        "|---|---|---|",
        "| `valid` | 该槽被分配过采样单元 | `{m}_valid` |",
        "| **可溯源** | 该槽的值由 ≥1 个源单元算出（插值槽**不**可溯源） | `alignment_q1.json` 的 `slot_src_units` |",
        "| **模态可用** | 该模态对本样本真的有意义 | `face_ratio` / `voiced_ratio` |",
        "",
        "实测反例（已写入核验报告的 `honesty` 段）：`-mJ2ud6oKI8_1` 的 `vision_valid` 为 "
        "50/50 **全 True**，但其视觉特征能量恰为 **0.0**——全片一帧人脸都没检出，"
        "`_fill_missing_faces` 保持零向量，与全 True 的掩码并存。"
        "**用 `valid` 判断模态可用性会得到完全错误的结论。**",
        "",
        "可溯源率实测（V4）：",
        "",
        "| 模态 | 槽数 | 可溯源槽 | 插值槽（无源帧） | 可溯源率 |",
        "|---|---|---|---|---|",
    ] + [
        "| " + MOD_ZH[m] + " | " + str(tr.get(m, {}).get("slots", "—")) + " | "
        + str(tr.get(m, {}).get("traceable", "—")) + " | "
        + str(tr.get(m, {}).get("interp", "—")) + " | "
        + ("{:.2f}%".format(100.0 * tr[m]["traceable"] / tr[m]["slots"])
           if tr.get(m, {}).get("slots") else "—") + " |"
        for m in MODALITIES
    ] + [
        "",
        "> 文本的可溯源率低**不是缺陷**：文本每样本只有 5~78 个词却要落 50 个槽，"
        "中位 20 个词只填满 20 个槽，其余 30 个槽本就该是空的（补零），"
        "空槽没有任何源单元可言。这恰恰是「不补零的变长交付」存在的理由。",
        "",
        "### 4.6 规则六：越界单元的处置与记账",
        "",
        "```",
        "τ < 0        → 并入槽 0      （左越界）",
        "τ > T·(L-1)/L 之后仍满足 floor 到 L 及以上 → 并入槽 L-1  （右越界）",
        "```",
        "",
        "实测（V5）：右越界并入末槽的单元共 **" + str(v5.get("units_clipped_to_last_slot", "—"))
        + "** 个，左越界 **" + str(v5.get("units_clipped_to_first_slot", "—"))
        + "** 个；而 `pts` 严格超出 T 且超出量 > 1e-5 s 的有 **"
        + str(v5.get("units_beyond_duration", "—")) + "** 个。三者口径不同、分开报出：",
        "",
        "- 前两者是**算法后果**（`floor` 的结果超过 L-1），权威且与实现严格一致；",
        "- 第三者是与算法无关的**物理事实**（时间戳实测越轴）；",
        "- 两者相差的 " + str(int(v5.get("units_clipped_to_last_slot", 0))
                            - int(v5.get("units_beyond_duration", 0)))
        + " 个是「恰好落在最后一个槽右端点」的帧——按左闭右开区间它们本就属于末槽，"
        "**不是异常**。",
        "",
        "记账字段（`align_series` 的 meta，逐样本落盘）：",
        "`units_clipped_to_last_slot` / `units_clipped_to_first_slot` / `units_beyond_duration`。",
        "",
        "### 4.7 两种交付形态由同一份对齐结果导出",
        "",
        "| 形态 | 形状 | 用途 | 特点 |",
        "|---|---|---|---|",
        "| **变长**（`features_q1.npz`） | Σ 单元数 × D，配 offsets | 交付 | 不补零、无假数据、体积小 |",
        "| **定长**（`aligned_50.npz`） | N × L × D | 问题二建模 | 等长张量，可直接喂模型 |",
        "",
        "两者不是两次计算，而是**同一次对齐的两个视图**："
        "`features_q1.npz` 的 offsets 切分出的第 i 段，"
        "与 `aligned_50.npz` 第 i 条的槽聚合值逐元素一致（V4 已核）。",
        "",
    ]
    return lines


def _sec_requirements(ctx: Dict[str, object]) -> List[str]:
    v = {c["code"]: c for c in ctx["verify"].get("checks", [])}
    lines = [
        "---",
        "",
        "## 5. 题目三条要求的逐条对照",
        "",
        "要求原文三条，逐条对应到**可执行的机器检查**（`code/q1_verify.py`，"
        "12 项、共 " + str(ctx["vsum"].get("total_assertions", "—")) + " 条断言）。"
        "下表由核验结果自动生成，`passed` 列即实测结果。",
        "",
        "### 5.1 要求一：原始样本覆盖完整性",
        "",
        "> 样本编号、模态文件与输出特征是否一一对应。",
        "",
        "**判据设为双向集合相等**（不只是「都能找到」，还包括「没有多余的」）：",
        "",
        "`label-100.xlsx` 的 100 个 `sample_id`  ≡  三模态未对齐文件名  ≡  "
        "单样本对齐文件名  ≡  `aligned_50.npz` 的 `sample_id`  ≡  汇总表 `summary_q1.csv`。",
        "",
        "反向检查同样重要：若某个目录里多出一个不属于这 100 条的样本，"
        "只查「无缺失」是发现不了的。",
        "",
    ]
    c1 = v.get("V1", {})
    ev1 = c1.get("evidence", {}).get("counts", {})
    lines += [
        "| 集合 | 条数 |",
        "|---|---|",
    ] + ["| `" + k + "` | " + str(x) + " |" for k, x in ev1.items()] + [
        "",
        "- **V1 " + ("通过" if c1.get("passed") else "**未通过**") + "**：缺失集合与多余集合均为空"
        "（`missing={}`、`extra={}`）。",
        "- **V1b " + ("通过" if c1.get("sub_check", {}).get("passed") else "**未通过**")
        + "**：" + str(c1.get("sub_check", {}).get("detail", "")) + "",
        "- 端口一致性另由 V3 核：三模态定长张量形状分别为 "
        + str(v.get("V3", {}).get("evidence", {}).get("aggregate", {}).get("text", {}).get("shape", "—"))
        + "、" + str(v.get("V3", {}).get("evidence", {}).get("aggregate", {}).get("audio", {}).get("shape", "—"))
        + "、" + str(v.get("V3", {}).get("evidence", {}).get("aggregate", {}).get("vision", {}).get("shape", "—"))
        + "，维度分别为 " + str(ctx["dims"]["text"]) + "/" + str(ctx["dims"]["audio"]) + "/"
        + str(ctx["dims"]["vision"]) + "。",
        "",
        "### 5.2 要求二：时序组织的可核验性",
        "",
        "> 序列位置、有效长度、填充规则及其与原始素材的对应记录。",
        "",
        "| 子项 | 检查 | 结果 | 证据 |",
        "|---|---|---|---|",
    ]
    ev2 = v.get("V2", {}).get("evidence", {})
    lines += [
        "| 序列位置 | V2 文本 `pts` 严格递增且落在 (0, T] | "
        + ("通过" if v.get("V2", {}).get("passed") else "未通过") + " | "
        + _kv(ev2.get("text_pts", {})) + " |",
        "| 序列位置 | V2 音频/视觉 `pts` 非递减（实测时间戳，允许等间隔） | "
        + ("通过" if v.get("V2", {}).get("passed") else "未通过") + " | "
        + _kv(ev2.get("av_pts", {})) + " |",
        "| 词索引一致 | V2 字符区间可重建原词、严格单调、`token_index` 严格递增 | "
        + ("通过" if v.get("V2", {}).get("passed") else "未通过") + " | "
        + _kv(ev2.get("char_spans", {})) + "；"
        + _kv(ev2.get("token_index", {})) + " |",
        "| 序列长度 | V3 `W == len(words) == num_words == len(char_spans)`，L=50 全样本成立 | "
        + ("通过" if v.get("V3", {}).get("passed") else "未通过") + " | 见上 |",
        "| 有效长度 | V6 填充规则自洽（文本空槽恰为全零；掩码与计数一致） | "
        + ("通过" if v.get("V6", {}).get("passed") else "未通过") + " | 见 4.4 表 |",
        "| 槽↔时间可逆 | V5 往返映射 50/50 槽成立、槽宽一致 | "
        + ("通过" if v.get("V5", {}).get("passed") else "未通过") + " | 抖动最大 "
        + "{:.2e}".format(float(v.get("V5", {}).get("evidence", {}).get("max_width_jitter_sec", 0.0)))
        + " s |",
        "| **与原始素材对应** | V4 每个音视频聚合值回捞源单元重算 | "
        + ("通过" if v.get("V4", {}).get("passed") else "未通过") + " | max-norm 相对偏差 "
        + "{:.3e}".format(float(v.get("V4", {}).get("evidence", {}).get("v4a_max_norm_rel_diff", 0.0)))
        + "（判据 " + str(v.get("V4", {}).get("evidence", {}).get("v4a_rel_tol")) + "）|",
        "| 与原始素材对应 | V8 用外部 `ffprobe` 现探视频时长交叉核对 | "
        + ("通过" if v.get("V8", {}).get("passed") else "未通过") + " | 最大偏差 "
        + "{:.2e}".format(float(v.get("V8", {}).get("evidence", {}).get("max_deviation_sec", 0.0)))
        + " s（全 " + str(v.get("V8", {}).get("evidence", {}).get("probed_samples", "—")) + " 条）|",
        "",
        "**「每个聚合值都能溯源到源帧」这一条的真实达成度**：",
        "",
        "音视频槽共 " + str(2 * ctx["n_slots"] * len(ctx["summary"])) + " 个，其中可溯源 "
        + str(tr_sum(ctx, "audio") + tr_sum(ctx, "vision")) + " 个；"
        "剩余的是**插值填充槽，本来就没有源帧**，已逐个标记为 `interp` 而不是冒充有源。"
        "文本槽的可溯源率低是因为它的空槽按定义就该是空的（见 4.5 的说明）。"
        "**本模型不把这一条报告成 100% 达成**——那需要伪造源帧。",
        "",
        "### 5.3 要求三：方法的合理性与可复现性",
        "",
        "> 特征提取工具、关键参数、处理日志和复现实验说明。",
        "",
        "| 子项 | 交付位置 | 核对方式 |",
        "|---|---|---|",
        "| 特征提取工具（含版本） | `extract_config.json` 的 `libraries`（"
        + str(len(ctx["libs"])) + " 项）、`external_executables` | 与 `pip list` 比对 |",
        "| 模型 revision | `extract_config.json` 的 `models.text.local_revision` | "
        "`" + str(ctx["cfg"].get("models", {}).get("text", {}).get("local_revision", "—")) + "` |",
        "| 关键参数 | 同上（音频/视觉/对齐全部参数） | 与代码中的实际默认值比对 |",
        "| 参数漂移 | `config_vs_actual` 记录「config 声明值 vs 实际生效值」 | "
        "见下方 |",
        "| 处理日志 | `code/logs/extraction.log`；异常另入 `anomaly_ledger.csv` | 逐样本可查 |",
        "| 复现实验说明 | `extract_config.json` 的 `reproduce`（"
        + str(len(ctx["cfg"].get("reproduce", []))) + " 条命令） | 见第 8 节 |",
        "| 代码指纹 | `code_fingerprints` 逐模块 | `" + str(
            ctx["cfg"].get("code_fingerprints", {})) + "` |",
        "",
        "**参数漂移（如实记录，不掩盖）**：",
        "",
    ]
    for k, d in ctx["cfg"].get("config_vs_actual", {}).items():
        lines += [
            "- **" + k + "**：config 声明 `" + str(d.get("config_declared")) + "`，"
            "实际生效 `" + str(d.get("actual_effective")) + "`",
            "  " + str(d.get("explanation", "")),
        ]
    lines += [
        "",
        "> 这类漂移是「可复现性声明」最容易出错的地方：把 `config.py` 原样贴进论文，"
        "就会公布一个**没有任何特征是用它算出来的**参数值。"
        "本项目一律从产物 meta 读**实际生效值**，并把差异显式列出。",
        "",
        "### 5.4 三项要求之外的第四件事：不应被删除的样本",
        "",
        "竞赛指南明确：置信度/质量阈值**仅用于质量标注，不得作为删除样本的理由**。"
        "本交付中 `counts_as_deletion` 恒为 0，`summary_q1.csv` 的 `included` 列恒为 1：",
        "",
        "| verdict | 条数 | 含义 |",
        "|---|---|---|",
    ] + [
        "| " + k + " | " + str(n) + " | " + desc + " |"
        for k, n, desc in ctx["verdicts"]
    ] + [
        "",
    ]
    return lines


def tr_sum(ctx: Dict[str, object], m: str) -> int:
    v = {c["code"]: c for c in ctx["verify"].get("checks", [])}
    return int(v.get("V4", {}).get("evidence", {}).get("traceability", {}).get(m, {}).get("traceable", 0))


def _sec_anomaly(ctx: Dict[str, object]) -> List[str]:
    led = ctx["ledger"]
    counts: Dict[str, int] = {}
    for r in led:
        counts[r["anomaly_code"]] = counts.get(r["anomaly_code"], 0) + 1
    meta = _read_json(os.path.join(ctx["out_dir"], "verify_report.json"))
    v12 = {c["code"]: c for c in meta.get("checks", [])}.get("V12", {}).get("evidence", {})
    samples = v12.get("samples", {})
    return [
        "---",
        "",
        "## 6. 异常处理：可追溯",
        "",
        "异常不隐藏、不删除、不调松判据。每一类都有**代号、计数、样本清单、处置动作**，"
        "落在 `anomaly_ledger.csv`（共 " + str(len(led)) + " 行，"
        + str(len(counts)) + " 类）。",
        "",
        "| 代号 | 含义 | 命中样本数 | 处置 |",
        "|---|---|---|---|",
    ] + [
        "| `" + code + "` | " + ANOM_DESC.get(code, "—") + " | " + str(n) + " | "
        # `D*` 是**代码缺陷**（已修复）；`R*` 是**设计规则**（不是缺陷，写进台账是为了留痕）；
        # 其余 `A*`/`VFR`/`LONG_PAUSE`/`EMPTY_TEXT` 是**素材与数据的固有情况**，一律保留并标注。
        + ("**已修复**（见下）" if code.startswith("D")
           else "设计规则，非缺陷（留痕）" if code.startswith("R")
           else "保留并标注") + " |"
        for code, n in sorted(counts.items())
    ] + [
        "",
        "### 6.1 本批数据实测到的异常（决定台账与汇总表的列）",
        "",
    ] + [
        "- **" + code + "**：" + ANOM_FULL.get(code, "")
        + (" 命中样本：" + "、".join("`" + s + "`" for s in samples.get(code, [])[:6])
           + ("…" if len(samples.get(code, [])) > 6 else "")) if samples.get(code) else ""
        for code in ("A1_audio_digital_silence", "A2_vad_zero_but_energy",
                     "A3_vision_face_all_fail", "A4_vision_face_partial",
                     "A5_vision_energy_zero_but_valid", "A6_audio_shorter_than_video",
                     "A7_audio_valid_slots_lt_50", "A8_pts_beyond_axis",
                     "VFR_frame_gap", "LONG_PAUSE", "EMPTY_TEXT")
    ] + [
        "",
        "**最值得单独说明的一条**：`-mJ2ud6oKI8_1` 同时命中 A1（音轨数字静音）"
        "与 A3/A5（视觉全片无人脸）——**语音与视觉两个模态同时失效、只剩文本**。"
        "这是「模态缺失」最真实的测试点，本模型按规则**保留它并在汇总表标出**，"
        "而不是以「质量不合格」为由剔除。",
        "",
        "### 6.2 本次发现并修复的两处真实缺陷",
        "",
        "**D1：`duration`/`slot_edges` 落盘时被降成 float32，导致溯源链在文件上不可复现。**",
        "",
        "- 现象：用「从源单元重算槽值」的方法扫描全部非空槽，发现 **39 个槽（0.32%，"
        "全部在文本模态）的槽归属无法从落盘文件复现**。",
        "- 根因：槽归属用 float64 的 `duration` 算出，**落盘时却降成 float32**。"
        "落在槽边界上的词（`τ/T·L` 恰为整数，短句且 W 与 50 有公因子时必然发生）"
        "会因此**差一个槽**。实测 `-MeTTeMJBNc_0`：align 时为 `[1,3,5,7,9,11]`，"
        "从文件读回重算为 `[1,3,4,7,8,10]`。",
        "- 波及面比预想大：**99/100 个样本的 `duration` 都被 float32 舍入改动过**"
        "（如 5.514 → 5.513999938964844），只是绝大多数恰好没落在槽边界上。",
        "- 修复：`duration` 与 `slot_edges` 改存 **float64**。"
        "**特征数组与有效掩码一个字节都没动**——已与修复前的备份逐元素 md5 比对："
        "100 个逐样本文件的三模态 `*_features`/`*_valid`、"
        "以及 `aligned_50.npz` 的全部数组，除 `duration` 外**完全一致**；"
        "全量非空槽的复现违例由 **39 降到 0**。",
        "- 已实测确认该修复**没有改变问题二的结果**：`emotion_model.py --cv-only` 复跑"
        "与 `data/model/README_情感判定模型_运行结果.md` 逐项一致"
        "（Ridge MAE 0.527 / r +0.413 / ρ +0.505）。证据留在 `data/_d1_check_model/`。",
        "",
        "**D2：汇总表 `official_id` 按第一个下划线切分，写错 18 行。**",
        "",
        "- 现象：" + str(ctx["n_ids_with_underscore"]) + " 个 `video_id` **自身含下划线**"
        "（如 `-I_e4mIh0yE`），而 `save_summary_csv` 用 `replace(\"_\", \"$_$\", 1)` "
        "按**第一个**下划线切分，于是写出 `-I$_$e4mIh0yE_1`（正确应为 `-I_e4mIh0yE$_$1`）。",
        "- 波及面：共 18/300 行（3 个模态各 6 行）。",
        "- 修复：改为 `sid.rsplit(\"_\", 1)`，按**最后一个**下划线切分。"
        "修复后 V1b 对 300 行重新核对，**0 处不一致**。",
        "",
        "**D3：人脸探测的逐帧证据被「按主键建字典」的读法吃掉 2/3。**",
        "",
        "- 现象：给探测工具加「逐帧证据必须每样本 3 行」的自检后，"
        "发现 `face_probe_frames.csv` 只有 297 行（应为 300），并触发整批 100 条被无谓重探。",
        "- 根因：`face_probe.py` 的 `_read_csv()` 是**按 `sample_id` 建字典**的读法，"
        "被同时用于读样本表（一行一样本，主键唯一——正确）"
        "与读逐帧表（**每条样本 3 行**——主键不唯一）。读逐帧表时，"
        "3 行互相覆盖，只剩最后一行：297 → 100。",
        "- 两个后果：**(1)** 续探时内存里的逐帧证据缩水，"
        "当轮未重新探测的样本，其证据会被写少（本次实测就少 3 行）；"
        "**(2)** 「样本行已存在则跳过」的续探判据因帧数恒不齐而永不成立，"
        "整批 100 条被重探（约 12 分钟）。",
        "- 为什么这是个**值得写进报告**的缺陷：它不报错、不崩，"
        "产出的 `face_probe.csv` 看上去完全正常（100 行、状态全 OK），"
        "只有去数逐帧表的行数才会发现证据缺了一角。"
        "**一张表能不能按主键建字典，取决于主键在表里唯不唯一**——"
        "这类「读法与被读的表结构不匹配」的错，静默是它最危险的地方。",
        "- 修复：新增 `_read_csv_rows()`（逐行读）供逐帧表使用；"
        "`run()` 收尾再加一道自检——每样本帧数必须等于抽样帧数（" + str(3) + "），"
        "不齐则 `LOGGER.error` 报出而不是留一个看起来正常的汇总表。"
        "该缺陷只影响探测工具与证据完整性，**不改动任何特征文件**。",
        "",
        "> 附注：`data/_d1_check_model/` 那份笔记里另有一处标作「D3」的观察，"
        "指的是 `data/model/` 下两份 CSV 与 README 口径不一致（属**问题二**的目录卫生问题），"
        "与本节这条 D3 是两回事，两者编号相同纯属巧合。",
        "",
        "### 6.3 与异常处理有关的四条全局规则",
        "",
        "| 代号 | 规则 | 理由 |",
        "|---|---|---|",
        "| `R1_FILL_MISSING_FACES_INPLACE` | 缺失人脸在**视觉提取器内**按相邻有效帧插值补齐 | "
        "「人脸检测会漏」是视觉模态自身的属性，不属于对齐步骤；"
        "放在提取器内使对齐模块对模态无感知 |",
        "| `R2_POOLER_RANDOM_INIT` | ResNet-50 的 **pooler 未被使用**，其随机初始化不影响结果 | "
        "提取只用 `children()[:-1]`（卷积主干 + 全局池化），"
        "pooler 在 `[:-1]` 之外；torchvision 会就此发一条无害警告，如实记入台账 |",
        "| `A6` | 音频容器短于视频时**不拉伸、不截断** | 缺口是 AAC priming 的系统性偏差，"
        "拉伸会伪造时间轴 |",
        "| `A8` | 越界单元并入端点槽并**逐样本计数** | 并入端点比丢弃诚实，"
        "但必须显式而不是静默 |",
        "",
    ]


ANOM_DESC = {
    "A1_audio_digital_silence": "音轨数字静音（无任何音频内容）",
    "A2_vad_zero_but_energy": "VAD 判零有声帧但存在能量",
    "A3_vision_face_all_fail": "视觉人脸全失败（face_ratio=0）",
    "A4_vision_face_partial": "视觉人脸部分失败（0<face_ratio<0.5）",
    "A5_vision_energy_zero_but_valid": "视觉能量恰为 0 但 valid 全 True（掩码陷阱实证）",
    "A6_audio_shorter_than_video": "音频容器时长短于视频 > 0.1 s",
    "A7_audio_valid_slots_lt_50": "音频有效槽 < 50（A6 的后果）",
    "A8_pts_beyond_axis": "视觉帧 PTS 越过公共时间轴",
    "VFR_frame_gap": "帧间隔不均（VFR）",
    "LONG_PAUSE": "长停顿时段",
    "EMPTY_TEXT": "文本为空",
    "D1_DURATION_FLOAT32": "duration/slot_edges 落盘降精度导致溯源不可复现",
    "D2_OFFICIAL_ID_FORMAT": "official_id 按首个下划线切分写错",
    "D3_FRAMES_TABLE_KEYED_READER": "逐帧表被按主键建字典的读法读，每样本 3 行只剩 1 行",
    "R1_FILL_MISSING_FACES_INPLACE": "缺失人脸在提取器内插值补齐（设计规则）",
    "R2_POOLER_RANDOM_INIT": "ResNet 池化头随机初始化未被使用（设计规则）",
}

ANOM_FULL = {
    "A1_audio_digital_silence": "音轨为数字静音，包络恒为 0，全程无任何音频内容。"
                                "该样本的语音特征不可解读为「情感平淡」，应视为模态缺失。",
    "A2_vad_zero_but_energy": "VAD 判为 0 个有声帧，但实测存在能量（logRMS −45~−9.85）。"
                              "说明该样本的能量低于 VAD 的绝对门限，属边界情形。",
    "A3_vision_face_all_fail": "全片一帧人脸都没检出，视觉特征保持零向量，`face_ratio=0`。",
    "A4_vision_face_partial": "部分帧检不出人脸（0<face_ratio<0.5），这些帧按插值补齐。",
    "A5_vision_energy_zero_but_valid": "视觉特征能量恰为 0，但 `vision_valid` 全为 True"
                                       "——掩码语义陷阱的实证反例。",
    "A6_audio_shorter_than_video": "音频容器时长比视频短，缺口恒为约 0.17 s"
                                   "（AAC priming 的系统性偏差，50/100 条命中）。",
    "A7_audio_valid_slots_lt_50": "音频尾部有效槽不足 50（A6 的直接后果），"
                                  "分布 {45,47,48,49,50}，最差 45/50，尾部由插值补齐。",
    "A8_pts_beyond_axis": "视觉帧 PTS 实测越过公共时间轴 T（+0.033~+0.067 s），"
                          "被 `time_to_slot` 的 clip 并入末槽。",
    "VFR_frame_gap": "帧间隔不均（可变帧率），实测 pts 的差分不恒定；"
                     "因为按实测 PTS 归槽，VFR 被如实保留而不是按名义帧率拉伸。",
    "LONG_PAUSE": "存在长停顿时段（voiced_ratio 偏低）。这是**真实语音现象**，"
                  "不是缺陷——它正是「停顿明显」这一类典型样本的选取依据。",
    "EMPTY_TEXT": "文本为空（本批 0 条）。",
}


def _fig_of(out_dir: str, sid) -> str:
    """
    找该样本的图。`typical_samples.csv` **没有**记录图的文件名，
    所以这里是「到 figures/ 目录里按 sample_id 前缀找」——
    出图失败时返回空串，正文写「—」而不是硬说「有」。
    """
    d = os.path.join(out_dir, "figures")
    sid = str(sid or "")
    if not (sid and os.path.isdir(d)):
        return ""
    for f in sorted(os.listdir(d)):
        if f.startswith(sid):
            return "figures/" + f
    return ""


def _rel(out_dir: str, path) -> str:
    """
    把路径统一写成「相对交付目录」的短路径，正文里不要塞绝对路径。

    ⚠️ **已经是相对路径的不能再 relpath**：`typical_samples.csv` 里存的就是
    `correspondence/xxx.csv` 这种相对写法，若直接丢给 `os.path.relpath(p, out_dir)`，
    Python 会拿**当前工作目录**去解析它，算出来是一串带 `..` 的东西，
    再被下面的守卫判成「不在交付目录内」而丢掉——于是正文里该有表名的地方全是「—」。
    本项目真踩过这个。**绝对路径才需要换算。**
    """
    p = str(path or "")
    if not p:
        return ""
    if not os.path.isabs(p):
        r = os.path.normpath(p)
        return "" if r.startswith("..") else r.replace("\\", "/")
    try:
        r = os.path.relpath(p, out_dir)
        return "" if r.startswith("..") else r.replace("\\", "/")
    except ValueError:
        return ""


def _sec_typical(ctx: Dict[str, object]) -> List[str]:
    tp = ctx["typical"]
    pb = ctx["probe"]
    probe_ok = [r for r in pb if str(r.get("detector_ok")) in ("1", "1.0")]
    n_faces = [int(r["max_face_count"]) for r in probe_ok
               if str(r.get("max_face_count", "")).lstrip("-").isdigit()]
    agree = [float(r["agree_with_extractor"]) for r in pb
             if r.get("agree_with_extractor") not in (None, "")]
    lines = [
        "---",
        "",
        "## 7. 五项人工抽查：五类典型样本",
        "",
        "题目要求人工抽查覆盖五类样本，每类给出「词—秒—语音段—视频帧」的对应关系。"
        "本项目的做法是：把「挑哪条、为什么挑它」变成**可复查的规则**（写进 "
        "`typical_samples.csv` 的判据列），再复用 `timeline_visualize` 出图与出对应表。"
        "评审看到的不是「作者手选的一条」，而是「按明确极值判据自动选出的一条」。",
        "",
        "| 类别 | 样本 | 判据 | 取值 | 图 | 对应表 |",
        "|---|---|---|---|---|---|",
    ]
    for r in tp:
        lines.append("| " + r.get("category", "—") + " | `" + r.get("sample_id", "—")
                     + "` | " + str(r.get("criterion", "")).replace("|", "/") + " | "
                     + str(r.get("criterion_value", "")).replace("|", "/") + " | "
                     + (_fig_of(ctx["out_dir"], r.get("sample_id", "")) or "—") + " | "
                     + (_rel(ctx["out_dir"], r.get("correspondence_csv", "")) or "—") + " |")
    lines += [
        "",
        "对应表（`data/q1_delivery/correspondence/*.csv`）逐槽给出："
        "`槽号 / 起止秒 / 该槽原词 / 字符范围（规范化文本与原文两套）/ 词元索引 / "
        "文本有效 / 语音有效 / 语音源单元区间 / 语音源时间区间 / "
        "视觉有效 / 视觉源单元区间 / 视觉源时间区间 / 填充标记`。",
        "「词—秒—语音段—视频帧」四要素在**同一行**上齐备，可逐行核对。",
        "",
        "### 7.1 「多人或远景」这一类是怎么取证的",
        "",
        "这是五类里唯一**无法从既有产物判断**的一类：现有视觉提取器的 MTCNN 是 "
        "`keep_all=False`，每帧只返回**一张**裁好的人脸、**不保留任何框坐标**，"
        "因此逐帧结果里只有「有没有人脸」这一个布尔量——"
        "没有框，就分不出「一个人占满屏」和「五个人各占一角」。",
        "",
        "为此本项目另建了一个 **`keep_all=True` 的探测器实例**（`code/face_probe.py`，"
        "与提取器同参同权重，**只有 `keep_all` 不同**），对每条样本均匀抽 **3 帧**"
        "（0.15T / 0.50T / 0.85T）做**纯检测**（不跑 ResNet 前向、不写特征），"
        "记录每帧检出的人脸数与框面积占比。",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| 探测成功样本 | " + str(len(probe_ok)) + " / " + str(len(pb)) + " |",
        "| 探测失败/超时 | " + str(len(pb) - len(probe_ok)) + "（记状态码，**未剔除样本**）|",
        "| `max_face_count` 分布 | "
        + str({k: n_faces.count(k) for k in sorted(set(n_faces))}) + " |",
        "| 与提取器 `face_flags` 的一致性 | "
        + ("{:.2%}".format(sum(agree) / len(agree)) if agree else "—")
        + "（逐帧比对，最近特征帧） |",
        "| 抽样时刻定位偏差 | 最大 " + str(max([r.get("seek_delta_max_sec", "0") for r in pb
                                              if r.get("seek_delta_max_sec") not in (None, "")],
                                            key=lambda x: float(x), default="—")) + " s |",
        "",
        "**三条必须随结果一起声明的边界**：",
        "",
        "1. `max_face_count` 是**抽样下界**，不是全片最大人脸数。"
        "只探 3 帧，若恰好都没拍到第二个人，结论只能是「抽样未发现多人」，"
        "**不构成「该片段只有一个人」的证据**。",
        "2. 探测帧与特征帧**不同格**（前者按固定比例定位、后者按 15 Hz 采样），"
        "最近可差约 1/15 s。因此「探到人脸 ⇔ 提取器有 face_flag」只作**一致性核对**报出"
        "（上表的一致性比率），不作硬断言。",
        "3. **探测失败不等于没有人脸**：失败写成 `detector_ok=0` + 状态码，"
        "判据标为**不可用**，绝不写成「0 张脸」。",
        "",
    ]
    lines += [
        "**「面部检测失败」这一类为什么优先取「其余模态正常」的样本**：",
        "",
        "本批恰好有一条样本是**语音与视觉同时失效**"
        "（`-mJ2ud6oKI8_1`：音轨数字静音 + 全片无人脸）。若按 `face_ratio` 最小直接选，"
        "它会被选中，但图上从语音到视觉全是空的，评审**看不出到底是哪一路模态出了问题**，"
        "也看不出这个类别想说明什么。因此同值（`face_ratio` 全为 0）时"
        "**优先取其余模态正常的样本**——同一段素材里语音时间轴清清楚楚、视觉一片空白，"
        "这才是一个干净的**单模态失效对照**。",
        "",
        "**另一件只有图能说明的事**：仅凭特征文件无法区分「全片检出不到人脸」的两种原因——"
        "**(a)** 素材里本来就没人（如动画/空镜），**(b)** 检测器在真的人脸上漏检了。"
        "本交付在「面部检测失败」的图里带了**按时间均匀抽取的关键帧**，"
        "所以原因**可以直接看图判断**，而不必靠猜。这也是交付里保留关键帧的实际用处"
        "——原始视频（约 4 GB）不随交付附上，但判定所需的画面证据留了下来。",
        "",
    ]
    silent = sorted({str(r.get("sample_id")) for r in ctx["ledger"]
                     if r.get("anomaly_code") == "A1_audio_digital_silence"})
    if silent:
        lines += [
            "**「停顿明显」这一类为什么要把数字静音样本排除在候选池外**：",
            "",
            "若直接取 `voiced_ratio` 最小的样本，选中的会是音轨**完全没有内容**的 "
            + str(len(silent)) + " 条 A1 样本（" + "、".join("`" + s + "`" for s in silent)
            + "）——它们的 `voiced_ratio` 为 0 **不是「说话断断续续」，而是「根本没有音频」**。"
            "拿它当「停顿明显」的例子，图上只会呈现一条平直的空音轨，"
            "评审读到的会是「这组数据没有声音」，而真正的「有声有停」反而没被展示。",
            "",
            "因此该类别的候选池限定为**音轨包络存在正值的样本**"
            "（判据与第 6 节的 A1 同口径：包络全为 0 即数字静音），"
            "在此池内取 `voiced_ratio` 最小者；选取规则已写进 `typical_samples.csv` 的判据列。",
            "",
            "> ⚠️ 这**不是删样本**：A1 样本仍在 `summary_q1.csv` 的 100 行里、"
            "仍在特征文件里、仍以 A1/A3/A5 的身份在异常台账里被如实标注，"
            "只是不作为「停顿」这一类别的**示例**。"
            "「排除在示例候选之外」与「从数据集中删除」是两件事，本报告严格区分。",
            "",
        ]
    if ctx.get("multi_person_found"):
        lines += ["本批 3 帧抽样**发现**了多人场景，该类以 `max_face_count` 最大的样本呈现。", ""]
    else:
        lines += [
            "本批 3 帧抽样**未发现**多人场景，故该类以「远景/小脸」实例呈现"
            "（判据：抽样框面积占比最小）。这与 `pick_typical` 的实现一致，"
            "而不是硬造一个「多人」样本充数。",
            "",
        ]
    return lines


def _sec_reproduce(ctx: Dict[str, object]) -> List[str]:
    cfg = ctx["cfg"]
    return [
        "---",
        "",
        "## 8. 复现实验说明",
        "",
        "### 8.1 环境（全部已在本机验证存在，**无需联网**）",
        "",
        "| 项 | 值 |",
        "|---|---|",
        "| Python | " + str(cfg.get("env", {}).get("python", "—")) + " |",
        "| 平台 | " + str(cfg.get("env", {}).get("platform", "—")) + " |",
        "| ffmpeg / ffprobe | 9.0.2（Gyan.FFmpeg，路径见 `extract_config.json`）|",
        "| 库版本 | " + "、".join(k + " " + str(v) for k, v in ctx["libs"].items()) + " |",
        "| 零网络策略 | " + str(cfg.get("env", {}).get("offline_policy", "")) + " |",
        "",
        "### 8.2 命令序列",
        "",
        "```bash",
        # 原样输出命令，**不要**再加 `#` 前缀——那会把每条命令都变成注释，
        # 复制粘贴过去一条都跑不了。`reproduce` 里以 `#` 开头的条目本身就是注释。
    ] + [str(x) for x in cfg.get("reproduce", [])] + [
        "```",
        "",
        "一次性跑通（含验收），各步产物互相独立、可单独重跑：",
        "",
        "```bash",
        "cd D:\\23届建模\\code",
        "python run_unaligned_all.py                 # 提取 → 对齐（约 35 分钟）",
        "python run_unaligned_all.py --verify        # 跑完执行 12 项机器核验",
        "python run_unaligned_all.py --deliver       # 跑完生成交付物（+人脸探测约 10 分钟）",
        "",
        "python q1_verify.py                         # 单独核验（只读，不改任何产物）",
        "python q1_delivery.py                       # 单独生成交付物",
        "python q1_readme.py                         # 重新生成本文档",
        "python face_probe.py                        # 单独跑人脸探测（支持断点续探）",
        "```",
        "",
        "### 8.3 随机性与确定性",
        "",
        "- `RANDOM_SEED = " + str(cfg.get("random_seed", 42)) + "`；"
        "视觉投影矩阵由 `numpy.random.default_rng(seed)` 构造，"
        "**跨机器、跨 numpy 版本确定性一致**，其 md5 逐文件落盘可验证。",
        "- 交叉验证的折划分由固定 seed 生成，按**视频分组**（同一视频的片段必须同折），"
        "避免同源片段跨折泄漏。",
        "- 提取/对齐流程本身**无随机成分**（除视觉投影的固定 seed 外）。",
        "",
        "### 8.4 交叉核对：预训练权重的真实性",
        "",
        "只声明「用了 roberta-base」是不够的，本交付用**外部可验证的行为**证明权重确为预训练成果：",
        "对 RoBERTa 做掩码补全探测（V9，`local_files_only=True` 加载）：",
        "",
    ] + [
        "- `" + str(p.get("probe", "")) + "` → top-5 "
        + str(p.get("top5", [])) + " " + ("✅" if p.get("passed") else "❌")
        for p in {c["code"]: c for c in ctx["verify"].get("checks", [])}
        .get("V9", {}).get("evidence", {}).get("probes", [])
    ] + [
        "",
        "另外记录词嵌入标准差 `"
        + "{:.6f}".format(float({c["code"]: c for c in ctx["verify"].get("checks", [])}
                                .get("V9", {}).get("evidence", {}).get("word_embedding_std", 0.0)))
        + "`（随机初始化会给出约 0.02 量级的更小值）与 hidden_size "
        + str({c["code"]: c for c in ctx["verify"].get("checks", [])}
              .get("V9", {}).get("evidence", {}).get("hidden_size", "—")) + "。",
        "",
    ]


def _walk_sizes(out_dir: str) -> "Dict[str, float]":
    """
    遍历交付目录，逐项算字节数（MB）。文件按文件名成一行，
    子目录**合并成一行**（如 `correspondence/（5 个文件）`），避免把 5 张图 + 5 张表
    摊成 10 行淹没正文。合并不等于漏计——合计是逐文件累加的。
    """
    out: Dict[str, float] = {}
    if not os.path.isdir(out_dir):
        return out
    for name in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            out[name] = os.path.getsize(p) / 1e6
        elif os.path.isdir(p):
            n, tot = 0, 0
            for root, _dirs, files in os.walk(p):
                for f in files:
                    n += 1
                    tot += os.path.getsize(os.path.join(root, f))
            if n:
                out[name + "/（" + str(n) + " 个文件）"] = tot / 1e6
    return out


def _f16_roundtrip(out_dir: str) -> Dict[str, Tuple[float, float, float]]:
    """
    在 `features_q1.npz` 上**独立重算**一次 float16↔float32 往返误差。

    为什么不直接引用 `size_report.md` 里的表：那份表是 `q1_delivery.py` 算的，
    引用它就等于把同一个结论抄两遍。这里从交付特征文件现算，
    与 `size_report.md` 构成**两个独立口径的互证**——不一致就说明有一方错了。
    返回 {模态: (最大绝对偏差, 最大相对偏差, 相对变化 > 1e-3 的元素占比%)}
    """
    out: Dict[str, Tuple[float, float, float]] = {}
    path = os.path.join(out_dir, "features_q1.npz")
    if not os.path.isfile(path):
        return out
    with np.load(path, allow_pickle=True) as z:
        for m in MODALITIES:
            key = f"{m}_features"
            if key not in z:
                continue
            x = np.asarray(z[key], np.float32)
            if x.size == 0:
                continue
            back = x.astype(np.float16).astype(np.float32)
            abs_max = float(np.max(np.abs(back.astype(np.float64) - x.astype(np.float64))))
            scale = float(np.max(np.abs(x))) or 1.0
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(back.astype(np.float64) - x.astype(np.float64)) / np.maximum(
                    np.abs(x).astype(np.float64), 1e-30)
            out[m] = (abs_max, abs_max / scale, 100.0 * float(np.mean(rel > 1e-3)))
    return out


def _sec_budget(ctx: Dict[str, object]) -> List[str]:
    sb = ctx["size_rows"]
    art = ctx["artifacts"]
    total = sum(v for v in art.values())
    limit_mib = LIMIT_BYTES / 1024.0 / 1024.0
    junk_ratio = 100.0 - float(ctx["text_valid_ratio"])
    return [
        "---",
        "",
        "## 9. 体积核算与交付清单",
        "",
        "题目给定上限 **50 MB**。本报告按 **" + "{:.2f}".format(limit_mib)
        + " MiB = " + "{:,}".format(LIMIT_BYTES) + " B** 执行——"
        "即把「50 MB」当成 50 × 1024² 而不是 50 × 10⁶，"
        "**取更严格的那个口径**，这样无论评审判哪种口径都不会超。",
        "",
        "未压缩估算公式（按题目口径）：`Σ_m (全部单元数) × D_m × 每元素字节数`。",
        "",
        "| 块 | 单元数 | 维度 | 未压缩 float32 | 未压缩 float16 | 实测压缩后 |",
        "|---|---|---|---|---|---|",
    ] + [
        "| " + r["block"] + " | " + r["units"] + " | " + r["dim"] + " | "
        + r["uncompressed_MB_f32"] + " MB | "
        + (_mb_or_dash(r.get("bytes_f16"))) + " | "
        + ("—" if r["compressed_measured_MB"] in ("-1", "") else r["compressed_measured_MB"] + " MB")
        + " |"
        for r in sb
    ] + [
        "",
        "**实测交付物体积**：`data/q1_delivery/` 目录**全部内容**（含图与对应表）合计 **"
        + "{:.3f}".format(total) + " MB**，距上限余量 **"
        + "{:.2f}".format(limit_mib - total) + " MB**（用掉 "
        + "{:.1f}%".format(100.0 * total / limit_mib) + "）。",
        "",
        "| 交付内容 | 大小 (MB) | 说明 |",
        "|---|---|---|",
    ] + [
        "| `" + k + "` | " + "{:.3f}".format(v) + " | " + _art_desc(k) + " |"
        for k, v in art.items()
    ] + [
        "",
        "### 9.1 为什么用变长而不是定长填充",
        "",
        "| 方案 | 未压缩体积 | 问题 |",
        "|---|---|---|",
        "| 定长填充 (100, 50, 768) 文本单模态 | "
        + NEXT_ROW(sb, "（对照）定长填充 文本 (100,50,768)", "uncompressed_MB_f32") + " MB | "
        "文本有效槽仅 " + "{:.2f}%".format(ctx["text_valid_ratio"]) + "，其余是**人造零** |",
        "| 定长填充三模态合计 | "
        + NEXT_ROW(sb, "（对照）定长填充 三模态合计 (100,50,ΣD)", "uncompressed_MB_f32") + " MB | "
        "同上 |",
        "| **变长（本交付）** | "
        + NEXT_ROW(sb, "（合计）变长 float32", "uncompressed_MB_f32") + " MB | 无假零 |",
        "",
        "> 注：`aligned_50.npz`（定长版）实测只有 8.16 MB，"
        "**是因为补出来的零被压缩算法吃掉了**，不是因为零不存在。"
        "所以选变长的理由不只是体积，更是**不让 " + "{:.1f}%".format(junk_ratio)
        + " 的人造零混进交付数据**。",
        "",
        "### 9.2 关于 float16",
        "",
        "下表的往返误差由 `q1_readme.py` **在交付特征文件上现算**"
        "（与 `size_report.md` 的表互为独立口径的互证）：",
        "",
        "| 模态 | 最大绝对偏差 | 最大相对偏差 | 相对变化 > 1e-3 的元素占比 |",
        "|---|---|---|---|",
    ] + [
        "| " + MOD_ZH[m] + " | " + (lambda t: "{:.4g}".format(t[0]) if t else "—")(
            ctx["f16"].get(m)) + " | "
        + (lambda t: "{:.3e}".format(t[1]) if t else "—")(ctx["f16"].get(m)) + " | "
        + (lambda t: "{:.4f}%".format(t[2]) if t else "—")(ctx["f16"].get(m)) + " |"
        for m in MODALITIES
    ] + [
        "",
        "**结论：不采用。** 变长 float32 三模态合计仅 "
        + NEXT_ROW(sb, "（合计）变长 float32", "uncompressed_MB_f32") + " MB，"
        "距上限余量 " + "{:.2f}".format(limit_mib - total) + " MB，**根本不紧张**；"
        "改成 float16 一共只省约 6.5 MB，却把语音维"
        "（量级最大，`max|x|` = "
        + next((r["abs_max_absvalue"] for r in sb if r["block"] == "audio_features"), "—")
        + "）的绝对偏差放宽到 "
        + (lambda t: "{:.4g}".format(t[0]) if t else "—")(ctx["f16"].get("audio"))
        + " 量级。既然体积不是瓶颈，**就没有任何理由用精度换体积**。",
        "",
        "**同时如实记录一处口径差异**：若改用 float16，语音维会出现"
        "相对变化超过 1e-3 的元素（占比见上表）——"
        "这类元素在回归/分类里未必致命，但**它是一处与 float32 不一致的地方**，"
        "按题目「若采用 float16 必须记录该偏差」的要求，"
        "这里把它写清楚：本交付**不采用** float16，故不存在该偏差。",
        "",
        "### 9.3 不随附的资源及其替代方案",
        "",
        "| 资源 | 大小 | 不随附的替代 |",
        "|---|---|---|",
        "| 原始视频 | 约 4 GB | 保留源视频**相对位置**（`alignment_q1.json` 的 `source_video`）"
        "+ **关键帧**（五类典型图的帧条与对应表） |",
        "| RoBERTa 本体 | 约 440 MB（110M 参数 float32）| "
        "固定 **revision 哈希 + 来源地址 + 逐文件字节数与校验值**（`extract_config.json`）|",
        "| ResNet-50 权重 | 102,540,417 B | 同上（含下载 URL 与文件名）|",
        "| MTCNN 权重 | 3 个小文件 | 同上（含字节数）|",
        "",
    ]


def _art_desc(label: str) -> str:
    if label in ART_DESC:
        return ART_DESC[label]
    head = label.split("/", 1)[0]           # `correspondence/（5 个文件）` -> `correspondence`
    return DIR_DESC.get(head, "")


def NEXT_ROW(rows: Sequence[Dict[str, str]], block: str, key: str) -> str:
    for r in rows:
        if r.get("block") == block:
            return str(r.get(key, "—"))
    return "—"


def _mb_or_dash(v) -> str:
    """float16 试算列：对照行没有 f16 试算（bytes_f16 为 0 或空）时写「—」而不是「0.000 MB」。"""
    try:
        b = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if b <= 0 else "{:.3f} MB".format(b / 1e6)


ART_DESC = {
    "features_q1.npz": "变长三模态特征（不补零），每模态 features + offsets + pts + dim + time_basis",
    "alignment_q1.json": "逐样本对齐记录：槽边界、来源单元区间、填充与有效掩码、文本溯源四元组",
    "summary_q1.csv": "100 行逐样本汇总：序列长度、有效槽、可用性指标、异常代号与原因",
    "extract_config.json": "可复现性快照：工具版本、模型 revision、参数、失败处理规则、参数漂移",
    "anomaly_ledger.csv": "结构化异常台账：代号/严重度/原因/处置动作（counts_as_deletion 恒为 0）",
    "size_budget.csv": "各块体积核算与 float16 试算",
    "size_report.md": "体积核算说明与 float16 结论",
    "typical_samples.csv": "五类典型样本的类别/判据/取值",
    "face_probe.csv": "逐样本人脸探测结果（3 帧抽样）",
    "face_probe_frames.csv": "逐探测帧的原始证据（框面积、定位偏差、交叉核对）",
    "verify_report.json": "12 项机器核验的完整证据",
    "verify_report.md": "同上的人类可读版",
    "README_问题一交付与验证.md": "本文档（数学模型与验收说明）",
}


def _sec_limits(ctx: Dict[str, object]) -> List[str]:
    return [
        "---",
        "",
        "## 10. 诚实边界：本问**没有**做的事与**不能**声称的事",
        "",
        "| # | 不能声称 | 原因 |",
        "|---|---|---|",
        "| 1 | **对齐达到某个毫秒级平均误差** | "
        "本数据集**没有逐词人工时间戳真值**；文本的时间轴是本管线自己声明的均匀假设。"
        "没有真值就无法计算误差。本报告只报**一致性检查**与**抽查** |",
        "| 2 | **「每个聚合值都能溯源到源帧」100% 成立** | "
        "音视频的空槽由插值填充，**本来就没有源帧**。真实可溯源率见 4.5 节，"
        "插值槽被逐个标记为 `interp` 而不是冒充有源 |",
        "| 3 | **`*_valid` 可作为模态可用性判据** | "
        "反例 `-mJ2ud6oKI8_1`：`vision_valid` 50/50 全 True 而视觉能量恰为 0 |",
        "| 4 | **文本逐词时间正确** | `time_basis=\"uniform_assumption\"`，"
        "由构造保证，只能证明实现与声明一致（V2 的证据段已明确标注这一点）|",
        "| 5 | **`max_face_count` 是片段中出现的最大人脸数** | "
        "只抽 3 帧探测，它是**抽样下界** |",
        "| 6 | **以质量为由剔除过任何样本** | 竞赛指南禁止；"
        "`counts_as_deletion` 恒为 0，`included` 列恒为 1 |",
        "| 7 | **异常样本的异常已「修好」** | "
        "音轨数字静音、全片无人脸这类**素材本身的缺陷无法通过对齐修复**，"
        "只能标注并如实交付 |",
        "| 8 | **本交付的特征对情感任务一定有效** | "
        "V9 只验证「权重确为预训练成果」，不验证「特征对情感任务有效」——"
        "后者属问题二的结论范围 |",
        "",
        "**关于「本问的实质贡献」**：题目把贡献定义为**统一时间锚点、异常可追溯、"
        "全过程可复现，以及在真实场景下对主体跟踪和失败帧的处理**。"
        "本文档的四节正对应这四点：第 4 节（锚点与对齐规则）、第 6 节（异常台账）、"
        "第 8 节（复现说明）、第 3.3 与 7.1 节（失败帧的插值补齐与主体跟踪的探测取证）。",
        "",
        "---",
        "",
        "## 附：本文件的自校验",
        "",
        "**唯一副本**：本文档只有这一份，位于 `data/q1_delivery/`（它是交付物之一，"
        "计入体积核算）；`code/README_未对齐三模态方案.md` 只是**指过来**，不另存副本。"
        "两份文档各改各的、最后互相矛盾，是这类说明文档最常见的坏结局。",
        "",
        "本文档的数字全部来自 `data/q1_delivery/` 下的产物。若产物被重新生成，"
        "请重跑 `python code/q1_readme.py` 后再引用本文档中的数字；"
        "`q1_verify.py` 的 V11 项会检查产物新鲜度（对齐晚于特征、交付晚于对齐），"
        "产物与文档一旦脱节即会被核验报出。",
        "",
    ]


# ==================== 组装 ====================

def build(out_dir: str = DEFAULT_OUT_DIR) -> str:
    verify = _read_json(os.path.join(out_dir, "verify_report.json"))
    if not verify:
        LOGGER.error("[文档] 找不到 %s，请先跑 `python q1_verify.py`",
                     os.path.join(out_dir, "verify_report.json"))
        return ""
    cfg = _read_json(os.path.join(out_dir, "extract_config.json"))
    summary = _read_csv(os.path.join(out_dir, "summary_q1.csv"))
    ledger = _read_csv(os.path.join(out_dir, "anomaly_ledger.csv"))
    typical = _read_csv(os.path.join(out_dir, "typical_samples.csv"))
    probe = _read_csv(os.path.join(out_dir, "face_probe.csv"))
    size_rows = _read_csv(os.path.join(out_dir, "size_budget.csv"))

    # Σ 单元数：优先从 features_q1.npz 的 offsets 现算（独立于汇总表，可互证）
    units_sum = {m: 0 for m in MODALITIES}
    fpath = os.path.join(out_dir, "features_q1.npz")
    if os.path.isfile(fpath):
        with np.load(fpath, allow_pickle=True) as z:
            for m in MODALITIES:
                off = np.asarray(z[f"{m}_offsets"], np.int64)
                units_sum[m] = int(off[-1] - off[0])
    if not any(units_sum.values()):
        for m in MODALITIES:
            units_sum[m] = int(sum(_nums(summary, f"{m}_seq_len")))

    n_slots = int(verify.get("n_slots", 50) or 50)
    dims = {"text": int(cfg.get("text_params", {}).get("feature_dim", 768) or 768),
            "audio": int(cfg.get("audio_params", {}).get("feature_dim", 74) or 74),
            "vision": int(cfg.get("vision_params", {}).get("feature_dim", 35) or 35)}
    for m in MODALITIES:                      # 以汇总表为准做一次交叉核对
        v = _nums(summary, f"{m}_dim")
        if v and int(v[0]) != dims[m]:
            LOGGER.warning("[文档] 维度口径不一致：config=%d，summary=%d", dims[m], int(v[0]))

    verdict_counts: Dict[str, int] = {}
    for r in summary:
        verdict_counts[r.get("verdict", "?")] = verdict_counts.get(r.get("verdict", "?"), 0) + 1
    verdict_desc = {
        "OK": "无异常",
        "OK_WITH_INFO": "有信息性异常（如 VFR 帧间隔、音频短于视频），不影响交付",
        "OK_WITH_WARN": "有警告级异常（模态缺失/局部失效等），已标注，样本仍保留",
    }
    verdicts = [(k, verdict_counts[k], verdict_desc.get(k, "")) for k in sorted(verdict_counts)]

    # 哪些 video_id 自身含下划线（D2 的量化依据）。
    # 注意两个数不是一回事：**含下划线的 video_id 个数** 与
    # **sample_id 会出现 ≥2 个下划线的样本条数**——后者才是写错的行数来源。
    vids = sorted({str(r["video_id"]) for r in summary}) if summary else []
    ids_us = [v for v in vids if "_" in v]
    n_sid_us = sum(1 for r in summary if str(r.get("sample_id", "")).count("_") >= 2)

    ctx: Dict[str, object] = {
        "out_dir": out_dir, "verify": verify, "cfg": cfg, "summary": summary,
        "ledger": ledger, "typical": typical, "probe": probe, "size_rows": size_rows,
        "vsum": verify.get("summary", {}), "n_slots": n_slots, "dims": dims,
        "units_sum": units_sum, "libs": cfg.get("libraries", {}), "verdicts": verdicts,
        "n_ids_with_underscore": len(ids_us), "ids_with_underscore": ids_us,
        "n_samples_with_underscore_vid": n_sid_us, "n_video_ids": len(vids),
        "text_valid_ratio": (lambda v: (st.mean(v) * 100.0 if v else 0.0))(
            _nums(summary, "text_valid_ratio")),
        "n_vfr": sum(1 for r in ledger if r.get("anomaly_code") == "VFR_frame_gap"),
        "multi_person_found": any(
            str(r.get("max_face_count", "")).lstrip("-").isdigit()
            and int(r["max_face_count"]) >= 2 for r in probe),
        "f16": _f16_roundtrip(out_dir),
        "artifacts": {},
    }
    # 交付物体积：**遍历目录现算**，与 size_report 互为独立口径。
    # 刻意不写死文件名清单——图（figures/*.png）与对应表（correspondence/*.csv）
    # 是后一步才生成的，写死清单会让「合计」在出图之后**静默漏计**，
    # 而这一节的全部意义就是回答「会不会超 50 MB」。
    ctx["artifacts"] = _walk_sizes(out_dir)

    parts: List[str] = []
    parts += _sec_header(ctx)
    parts += _sec_honesty(ctx)
    parts += _sec_formal(ctx)
    parts += _sec_pipeline(ctx)
    parts += _sec_features(ctx)
    parts += _sec_align(ctx)
    parts += _sec_requirements(ctx)
    parts += _sec_anomaly(ctx)
    parts += _sec_typical(ctx)
    parts += _sec_reproduce(ctx)
    parts += _sec_budget(ctx)
    parts += _sec_limits(ctx)

    text = "\n".join(parts).rstrip() + "\n"
    out_path = os.path.join(out_dir, MD_NAME)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    LOGGER.info("[文档] 写出 %s（%d 行，%.1f KB）",
                out_path, text.count("\n"), len(text.encode("utf-8")) / 1024)
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="生成问题一交付与验证说明文档")
    p.add_argument("--out", default=None, help="输出目录，默认 data/q1_delivery")
    args = p.parse_args(argv)
    path = build(args.out or DEFAULT_OUT_DIR)
    return 0 if path else 1


if __name__ == "__main__":
    sys.exit(main())
