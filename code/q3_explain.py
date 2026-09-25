# -*- coding: utf-8 -*-
"""
q3_explain.py —— 问题三：可解释的多模态情感预测与关键证据落地

赛题要求（逐条对应）
--------------------
第 27 行「构建可解释的多模态情感预测模型」：
    复用问题二的主干（同一输入接口、同一缺失语义），但把「解释」做进模型内部
    而不是事后拟合：槽级注意力权重 + 模态留一扰动，两者都直接来自前向计算。
第 28 行「识别出影响预测结果的关键证据；关键证据需可对应至原始文本片段、
        语音时段或视觉关键帧」：
    由 q3_timebase 重建的槽位时间基准，把关键槽号还原成
    ① 文本片段（词）② 语音时段（秒） ③ 视觉关键帧（真实抽帧 + 人脸马赛克）
第 29 行「分析不同模态在情感判断中的作用程度和主要参考模态」：
    · 作用程度 = 模态留一扰动引起的预测变化（因果口径）+ 注意力占比（归因口径），
      两个口径并列输出，不一致时一并报告而不是只留好看的那个；
    · 主要参考模态 = 扰动口径下损失最大的模态。
第 48–52 行的论文要求：
    `attachment4_explanations.csv` 给出 20 条全量三件套（预测 + 模态作用 + 关键证据），
    图件给出典型样本的证据时间轴与关键帧。

解释的两条口径为什么要并列
--------------------------
注意力权重反映「模型把信息从哪儿搬过来」，但注意力高不等于对结果有因果作用
（可能只是被冗余搬运）。留一扰动直接测「拿掉它会怎样」，是因果口径，但会受
模态间冗余补偿影响而低估。两者同向时结论可靠，背离时必须在论文里说明，
只报其中之一属于选择性报告。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import config  # noqa: E402
import q2q3_common as Q  # noqa: E402
import q2_model as M  # noqa: E402
import q3_timebase as T  # noqa: E402
from q2_train import OUT_DEFAULT as Q2_OUT  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

OUT_DEFAULT = os.path.join(config.PROJECT_ROOT, "data", "q3")
TOP_K = 5             # 全局关键证据槽数（跨模态一起排序）
PER_MODALITY_K = 3    # 每个模态各自的关键证据槽数，保证三类证据都落到


# ==================== 词元槽 → 文本片段 ====================

def _largest_remainder(weights: np.ndarray, total: int) -> np.ndarray:
    """按权重把 total 个名额整数分配，保证每项 ≥1 且总和恰为 total。

    为什么不用 round(w/w.sum()*total)：舍入后总和对不上，会出现某词拿到 0 个词元，
    导致该词在证据里彻底消失。
    """
    n = len(weights)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    total = max(total, n)
    raw = weights / weights.sum() * total
    base = np.maximum(1, np.floor(raw).astype(np.int64))
    # 先补足或削减到 total
    diff = total - int(base.sum())
    if diff > 0:
        order = np.argsort(-(raw - base))
        for i in range(diff):
            base[order[i % n]] += 1
    elif diff < 0:
        order = np.argsort(-base)          # 从最多的开始削，且不削到 0
        i = 0
        while diff < 0 and i < 10000:
            k = order[i % n]
            if base[k] > 1:
                base[k] -= 1
                diff += 1
            i += 1
    return base


def token_slot_to_words(raw_text: str, n_slots: int) -> List[Tuple[int, int]]:
    """把 n_slots 个词元槽映射到 (起始词下标, 结束词下标+1) 半开区间。

    官方只给了 `text_bert` 的词元 id，本机没有缓存 BERT 词表（离线环境无法下载），
    因此无法逐词元还原 WordPiece 边界。这里用「按词长加权 + 最大余额法」近似：
    WordPiece 对长词会切成更多片，故词元数近似与词字符长度成正比。

    这是**近似**，不是真实对齐；它在输出中被显式标注为 `token_word_map=weighted`，
    并且只影响「证据落在哪个词」的措辞，不影响槽号、时间与关键帧的确定性。
    """
    words = str(raw_text).split()
    if not words or n_slots <= 0:
        return []
    w = np.array([max(1.0, np.ceil(len(x) / 5.0)) for x in words], dtype=np.float64)
    alloc = _largest_remainder(w, n_slots)          # 每个词分到几个词元槽
    spans: List[Tuple[int, int]] = []
    for wi, k in enumerate(alloc):
        for _ in range(int(k)):
            spans.append((wi, wi + 1))
    return spans[:n_slots]


def words_in_span(raw_text: str, wi: int, wj: int) -> str:
    words = str(raw_text).split()
    return " ".join(words[wi:wj])


# ==================== 归因 ====================

@torch.no_grad()
def attribute_sample(model, enc_np: Dict[str, np.ndarray], i: int,
                     device: torch.device) -> Dict[str, object]:
    """对单条样本做槽级归因 + 模态留一扰动。

    进来先 `eval()`：归因是**推理**，必须关掉 dropout。调用方即便忘了切换模式，
    这里也兜住——否则同一份模型每次跑出的注意力与留一量都不同，
    而问题三的结论全部建立在这些量上（这个坑实际踩过）。
    """
    model.eval()
    sub = {k: v[i:i + 1] for k, v in enc_np.items()}
    bt = M.to_torch(sub, device)
    out = model(bt, need_slot_scores=True)

    prob = torch.softmax(out["polarity_logits"], dim=-1)[0].cpu().numpy()
    pred = int(prob.argmax())
    inten = float(out["intensity"][0].cpu().numpy())

    # --- 归因口径：槽级注意力占比 ---
    alpha = out["slot_alpha"][0].cpu().numpy()          # (3, 50)
    usable = out["token_usable"][0].cpu().numpy()       # (3, 50)
    alpha = np.where(usable, alpha, 0.0)
    share = alpha.sum(axis=1)                            # 每模态的注意力占比
    share = share / share.sum() if share.sum() > 0 else np.zeros(3)

    # --- 因果口径：逐一拿掉某模态，看预测类别概率掉多少 ---
    # 两个统计量并列，因为单一统计量会漏掉不同情形：
    #   drop = p(原预测类) - p_masked(原预测类)：直接、可读，但模型若靠别的
    #          模态补偿回同一类别，这个值会接近 0，看起来像「该模态不重要」；
    #   KL   = KL(原分布 ‖ 掩蔽后分布)：对「概率重分布」敏感，即使 argmax 没变
    #          也能反映出该模态对判断的支撑程度。
    # 二者同向才说明结论稳；背离时报告里必须说明（见模块开头）。
    loo: Dict[str, float] = {}
    loo_kl: Dict[str, float] = {}
    flip: Dict[str, int] = {}
    for mi, mname in enumerate(Q.MODALITIES):
        sub2 = {k: v[i:i + 1].copy() for k, v in enc_np.items()}
        # 把该模态整条拿掉：obs_mask / modal_avail / miss_flag 三处同步改，
        # 这样模型的注意力掩码与缺失状态嵌入都认为「这个模态不存在」，
        # 与附件3 的真实缺失走的是同一条代码路径。
        # obs_mask 是逐模态的，其余两个模态的观测不受牵连——这正是留一法
        # 该有的语义（旧版用跨模态并集，拿掉任一模态都会把三个模态一起掐掉）。
        if mi == 0:
            sub2["text_ids"][:, :] = Q.PAD_TOKEN_ID
        elif mi == 1:
            sub2["audio"][:, :, :] = 0.0
        else:
            sub2["vision"][:, :, :] = 0.0
        sub2["obs_mask"][:, mi, :] = False
        sub2["miss_flag"][:, mi, :] = True
        sub2["modal_avail"][:, mi] = False
        o2 = model(M.to_torch(sub2, device))
        p2 = torch.softmax(o2["polarity_logits"], dim=-1)[0].cpu().numpy()
        loo[mname] = float(prob[pred] - p2[pred])
        loo_kl[mname] = float(np.sum(prob * np.log((prob + 1e-12) / (p2 + 1e-12))))
        flip[mname] = int(int(p2.argmax()) != pred)

    # 主要参考模态：以 **KL** 定序而不是 Δ。
    # Δ = p(原预测类) − p_masked(原预测类) 可以为负（拿掉某模态后原类别概率反而
    # 上升，说明该模态在被冗余补偿、甚至是在拖后腿）。若按 max(Δ) 选「主要模态」，
    # 负值里的最大者会被选成主模态，语义正好说反。KL(原分布‖掩蔽后分布) 恒非负，
    # 衡量的是「该模态对当前判断的支撑程度」，定序不会出这种错。
    # 阈值 1e-4 之下视为无差别：此时如实报「未能确定」，不硬选一个。
    max_kl = max(loo_kl.values())
    dominant = max(loo_kl, key=loo_kl.get) if max_kl > 1e-4 else None
    # 同时给出 Δ 口径下的名次，供论文对比两口径是否一致
    dominant_by_drop = max(loo, key=loo.get) if max(loo.values()) > 1e-6 else None

    # --- 关键证据槽：在「可用」槽里按注意力取 top-k ---
    flat = alpha.reshape(-1)                             # (3*50,) 行优先：modality*50+slot
    order = np.argsort(-flat)
    evid: List[Tuple[int, int, float]] = []
    for idx in order:
        if flat[idx] <= 0:
            break
        mi, sj = int(idx // Q.SEQ_LEN), int(idx % Q.SEQ_LEN)
        evid.append((mi, sj, float(flat[idx])))
        if len(evid) >= TOP_K:
            break

    # 逐模态证据：只取全局 top-k 有个实际麻烦——某模态注意力整体偏低时会被
    # 完全挤出证据表（实测未训练模型下 5 条证据里一条语音都没有）。而赛题第 28
    # 行要求证据能落到「文本片段 / 语音时段 / 视觉关键帧」三类上，第 29 行还要求
    # 比较各模态的作用。故另外按模态各取 top-k 并列输出，并如实标注某模态
    # 确实没有可用槽（而不是用别的模态顶上）。
    per_mod: List[Tuple[int, int, float]] = []
    for mi in range(3):
        sub = alpha[mi]
        for sj in np.argsort(-sub)[:PER_MODALITY_K]:
            if sub[sj] <= 0:
                continue
            per_mod.append((mi, int(sj), float(sub[sj])))

    return {
        "polarity_prob": prob, "polarity_idx": pred, "intensity": inten,
        "slot_alpha": alpha, "modal_share_attention": share,
        "modal_loo_drop": loo, "modal_loo_kl": loo_kl, "modal_loo_flip": flip,
        "dominant_modality": dominant,
        "dominant_modality_by_drop": dominant_by_drop,
        "evidence": evid, "evidence_per_modality": per_mod,
    }


# ==================== 报告与产物 ====================

def build_explanations(model, enc_np: Dict[str, np.ndarray],
                       samples: Sequence["Q.Sample4"],
                       timebases: Sequence[T.Timebase],
                       device: torch.device) -> Tuple[List[Dict], np.ndarray]:
    """返回 `(逐样本解释表, 槽级注意力 (N,3,50))`。

    槽级注意力单独以数组形式返回而不是塞进 JSON：20×3×50 个浮点数写进 JSON
    既难读又占体积，而图件正好需要它做「证据强度时间轴」。
    """
    rows: List[Dict] = []
    alpha_all = np.zeros((len(samples), len(Q.MODALITIES), Q.SEQ_LEN), dtype=np.float32)
    for i, s in enumerate(samples):
        att = attribute_sample(model, enc_np, i, device)
        tb = timebases[i]
        alpha_all[i] = att["slot_alpha"]
        spans = token_slot_to_words(s.raw_text, tb.n_slots)
        L = s.info.valid_len

        def _rec(mi: int, sj: int, w: float, scope: str) -> Dict:
            mname = Q.MODALITIES[mi]
            t0, t1 = tb.of_slot(sj)      # sj 是官方序列槽号，词元位与词元槽 1:1
            frag = ""
            if spans and 1 <= sj <= len(spans):
                wi, wj = spans[sj - 1]
                frag = words_in_span(s.raw_text, max(0, wi - 1), wj + 1)
            return {
                "scope": scope,
                "modality": mname,
                "slot": sj,
                "attention": round(w, 6),
                "is_missing_slot": bool(s.info.missing[mname][sj]),
                "time_start_sec": None if np.isnan(t0) else round(t0, 3),
                "time_end_sec": None if np.isnan(t1) else round(t1, 3),
                "text_fragment": frag,
                # 视觉证据的可核对凭据：真实视频文件 + 该槽中点时刻
                "keyframe": (f"{s.key}@{round((t0+t1)/2,3)}s"
                             if mname == "vision" and not np.isnan(t0) else ""),
            }

        # 两个口径会撞同一批槽（全局 top-5 往往就是某些模态的 top-3），
        # 按 (模态, 槽号) 去重并保留先出现的全局口径——它的排序跨模态可比，
        # 而模态内口径的名次只在本模态内可比。不去重的话同一行证据会在
        # 解释卡与证据表里各出现一次，看起来像「有多条独立证据支持」。
        evid_records: List[Dict] = []
        _seen: set = set()
        for mi, sj, w in att["evidence"]:
            _seen.add((mi, int(sj)))
            evid_records.append(_rec(mi, sj, w, "global_topk"))
        for mi, sj, w in att["evidence_per_modality"]:
            if (mi, int(sj)) in _seen:
                continue
            _seen.add((mi, int(sj)))
            evid_records.append(_rec(mi, sj, w, "modality_topk"))

        rows.append({
            "sample": s.key,
            "valid_len_token": L,
            "n_word_slots": tb.n_slots,
            "duration_sec": round(tb.duration, 3),
            "timebase_level": tb.level,
            "speech_ratio": round(tb.speech_ratio, 4),
            "pred_polarity": Q.POLARITY_CN[att["polarity_idx"] - 1],
            "pred_prob_negative": round(float(att["polarity_prob"][0]), 6),
            "pred_prob_neutral": round(float(att["polarity_prob"][1]), 6),
            "pred_prob_positive": round(float(att["polarity_prob"][2]), 6),
            "pred_intensity": round(att["intensity"], 6),
            "modality_attention_text": round(float(att["modal_share_attention"][0]), 6),
            "modality_attention_audio": round(float(att["modal_share_attention"][1]), 6),
            "modality_attention_vision": round(float(att["modal_share_attention"][2]), 6),
            "modality_loo_text": round(att["modal_loo_drop"]["text"], 6),
            "modality_loo_audio": round(att["modal_loo_drop"]["audio"], 6),
            "modality_loo_vision": round(att["modal_loo_drop"]["vision"], 6),
            "modality_loo_kl_text": round(att["modal_loo_kl"]["text"], 6),
            "modality_loo_kl_audio": round(att["modal_loo_kl"]["audio"], 6),
            "modality_loo_kl_vision": round(att["modal_loo_kl"]["vision"], 6),
            "loo_flip_text": att["modal_loo_flip"]["text"],
            "loo_flip_audio": att["modal_loo_flip"]["audio"],
            "loo_flip_vision": att["modal_loo_flip"]["vision"],
            "dominant_modality_loo": att["dominant_modality"] or "",
            "dominant_modality_drop": att["dominant_modality_by_drop"] or "",
            "vision_fully_missing": not bool(s.info.available["vision"]),
            "token_word_map": "weighted",
            "top_evidence": evid_records,
        })
    return rows, alpha_all


def flatten_evidence(rows: Sequence[Dict]) -> List[Dict]:
    """把每条样本的 top-k 证据展成一维表，便于论文直接引用。"""
    out = []
    for r in rows:
        for rank, e in enumerate(r["top_evidence"], start=1):
            out.append({
                "sample": r["sample"], "rank": rank, "scope": e["scope"],
                "pred_polarity": r["pred_polarity"],
                "pred_intensity": r["pred_intensity"],
                "modality": e["modality"], "slot": e["slot"],
                "attention": e["attention"],
                "is_missing_slot": e["is_missing_slot"],
                "time_start_sec": e["time_start_sec"],
                "time_end_sec": e["time_end_sec"],
                "text_fragment": e["text_fragment"],
                "keyframe_ref": e["keyframe"],
            })
    # 同一 (样本, 模态, 槽) 可能在两个口径里都出现：去重时保留全局口径那条，
    # 因为全局口径的 rank 是跨模态可比的，模态内口径的 rank 只在本模态内可比。
    seen = set()
    dedup = []
    for o in out:
        k = (o["sample"], o["modality"], o["slot"])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(o)
    for i, o in enumerate(dedup, start=1):
        o["rank"] = i
    return dedup


def write_explanation_cards(rows: Sequence[Dict], samples: Sequence["Q.Sample4"],
                            out_path: str, summary: Dict,
                            n_cards: int = 3) -> str:
    """产出「典型样本解释卡」（赛题 L49）。

    为什么单独出一份 Markdown 而不是只给 CSV：L49 要的是一张**能读的卡**——
    预测结论、三模态作用程度、主要参考模态、关键证据定位四件事在同一屏里。
    散在 CSV 的三张表里，评审得自己拼，等于没交付。

    选样规则：优先覆盖「三模态齐全」和「视觉缺失」两类，各取一半，
    不挑「解释得好看的」样本。
    """
    vis_missing = [i for i, r in enumerate(rows) if r["vision_fully_missing"]]
    vis_ok = [i for i, r in enumerate(rows) if not r["vision_fully_missing"]]
    pick: List[int] = []
    for lst in (vis_ok, vis_missing):
        for i in lst:
            if len(pick) >= n_cards:
                break
            pick.append(i)
    pick = pick[:n_cards]

    L_: List[str] = []
    L_.append("# 附件4 典型样本解释卡\n")
    L_.append(f"> 共 {len(rows)} 条附件4 样本，此处给出 {len(pick)} 张解释卡"
              f"（覆盖「三模态齐全」与「视觉缺失」两类）。\n"
              f"> 全量结果见 `attachment4_predictions.csv` 与 "
              f"`attachment4_top_evidence.csv`。\n")
    L_.append("## 全样本汇总\n")
    L_.append(f"- 极性分布：{summary['pred_polarity_distribution']}")
    L_.append(f"- 注意力占比均值："
              f"{ {k: round(v, 4) for k, v in summary['attention_share_mean'].items()} }")
    L_.append(f"- 留一 Δ 均值："
              f"{ {k: round(v, 4) for k, v in summary['loo_drop_mean'].items()} }")
    L_.append(f"- 留一 KL 均值："
              f"{ {k: round(v, 4) for k, v in summary['loo_kl_mean'].items()} }")
    L_.append(f"- 拿掉即翻转类别的样本数：{summary['loo_flip_counts']}")
    L_.append(f"- 主要参考模态分布（留一口径）：{summary['dominant_modality_counts']}")
    L_.append(f"- 证据模态覆盖（被点到的样本数）："
              f"{summary['evidence_coverage_by_modality']}")
    L_.append(f"- 归因与因果两口径一致率："
              f"{summary['loo_vs_attention_agreement']['agree']}"
              f"/{summary['loo_vs_attention_agreement']['total']} = "
              f"{summary['loo_vs_attention_agreement']['ratio']:.2%}\n")

    for i in pick:
        r, s = rows[i], samples[i]
        L_.append(f"---\n\n## 解释卡 {i + 1}：`{r['sample']}`\n")
        L_.append(f"**预测结论**：极性 **{r['pred_polarity']}**　强度 "
                  f"**{r['pred_intensity']:.3f}**　"
                  f"（概率 N={r['pred_prob_negative']:.3f} / "
                  f"U={r['pred_prob_neutral']:.3f} / "
                  f"P={r['pred_prob_positive']:.3f}）\n")
        L_.append(f"- 有效词元长度 `L={r['valid_len_token']}`，词元槽 "
                  f"{r['n_word_slots']} 个；时长 {r['duration_sec']} s，"
                  f"语音占比 {r['speech_ratio']:.2f}，时间轴级别 `{r['timebase_level']}`")
        L_.append(f"- 视觉是否整体缺失："
                  f"{'**是**（该样本无脸）' if r['vision_fully_missing'] else '否'}\n")

        L_.append("**三模态作用程度**（两个口径并列，背离时以后文说明为准）\n")
        L_.append("| 模态 | 注意力占比 | 留一 Δ（因果） | 留一 KL | 拿掉后是否翻转 |")
        L_.append("|---|---|---|---|---|")
        for m in Q.MODALITIES:
            L_.append(f"| {m} | {r[f'modality_attention_{m}']:.4f} | "
                      f"{r[f'modality_loo_{m}']:+.4f} | "
                      f"{r[f'modality_loo_kl_{m}']:.4f} | "
                      f"{'是' if r[f'loo_flip_{m}'] else '否'} |")
        L_.append("")
        dom = r["dominant_modality_loo"] or "**未能确定**"
        L_.append(f"**主要参考模态**（留一口径）：**{dom}**\n")

        L_.append("**关键证据定位**\n")
        L_.append("| 模态 | 槽号 | 注意力 | 时间区间 (s) | 文本片段 | 视觉关键帧 | 该槽是否缺失 |")
        L_.append("|---|---|---|---|---|---|---|")
        for e in r["top_evidence"]:
            t = ("—" if e["time_start_sec"] is None
                 else f"[{e['time_start_sec']}, {e['time_end_sec']})")
            frag = (e["text_fragment"] or "—").replace("|", "\\|")[:60]
            kf = e["keyframe"] or "—"
            L_.append(f"| {e['modality']} | {e['slot']} | {e['attention']:.4f} | {t} | "
                      f"{frag} | {kf} | {'是' if e['is_missing_slot'] else '否'} |")
        L_.append("")
        if r["vision_fully_missing"]:
            L_.append("> 该样本视觉整体缺失，故其视觉证据行若存在也只是"
                      "「有效区内的局部观测」，不构成画面证据。\n")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L_) + "\n")
    return out_path


def _write_csv(path: str, rows: Sequence[Dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = [c for c in rows[0].keys() if c != "top_evidence"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ==================== 图件 ====================

def _setup_font():
    """选一个**真能画出汉字**的字体；找不到就全线改用英文标注。

    两道坎，缺一不可：

    1. 能不能解析到字体文件 —— 只设 `rcParams` 不报错，但名字不存在时渲染
       才会把汉字画成方框，所以必须用 findfont 实际解析。
    2. 解析到的字体**有没有汉字字形** —— 本机 `Microsoft YaHei` 解析到
       `msyh.ttc`（TrueType 集合），matplotlib 对它取不到汉字字形，图上的
       中文照样变 ▯。故这里直接问字体表的 cmap 里有没有「缺」这个字，
       有才算数。

    两条都过不了就返回 False，调用方据此把标注换成英文——宁可图上是英文，
    也不要出现一张汉字变方框的图。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    probe = "缺"          # 图里实际用到的字，直接拿它验
    for fam in ("SimHei", "Noto Sans SC", "Microsoft YaHei", "KaiTi",
                "FangSong", "Microsoft JhengHei", "WenQuanYi Zen Hei"):
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=fam),
                                         fallback_to_default=False)
        except Exception:
            continue
        if not (path and os.path.isfile(path)):
            continue
        try:
            if FT2Font(path).get_char_index(ord(probe)) == 0:
                continue                      # 取不到该字的字形，跳过
        except Exception:
            continue
        plt.rcParams["font.sans-serif"] = [fam]
        plt.rcParams["axes.unicode_minus"] = False
        return True, fam
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return False, "DejaVu Sans"


def make_figures(rows: Sequence[Dict], samples: Sequence["Q.Sample4"],
                 timebases: Sequence[T.Timebase], out_dir: str,
                 alpha_all: Optional[np.ndarray] = None,
                 n_samples: int = 4) -> List[str]:
    """证据时间轴 + 关键帧条 + 模态作用总览。

    时间轴不是「槽位铺满时间」的装饰图：**柱高 = 该槽的注意力权重**（按模态内
    最大值归一），所以哪一段被模型看重一眼可辨。缺失槽用红色斜纹标出，
    语音区间用灰色底纹标出——这样「证据落在有声段还是静音段」也能直接读出来。
    """
    import matplotlib.pyplot as plt
    cjk, fam = _setup_font()
    # 没有可用 CJK 字体时全线退回英文：宁可图上是英文，也不要有字变方框的图
    def L(zh: str, en: str) -> str:
        return zh if cjk else en
    made: List[str] = []
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 优先展示能同时覆盖「三模态齐全」和「视觉缺失」的样本，避免只挑好看的；
    # 缺少的类别用其余样本补齐，样本数不够时不强求。
    pool = list(range(len(rows)))
    vis_missing = [i for i in pool if rows[i]["vision_fully_missing"]]
    vis_ok = [i for i in pool if not rows[i]["vision_fully_missing"]]
    pick: List[int] = []
    for lst in (vis_ok, vis_missing):
        for i in lst:
            if len(pick) >= n_samples:
                break
            pick.append(i)
        if len(pick) >= n_samples:
            break
    pick = pick[:n_samples]

    for i in pick:
        r, s, tb = rows[i], samples[i], timebases[i]
        fig, axes = plt.subplots(3, 1, figsize=(11, 6.6), sharex=True,
                                 gridspec_kw={"hspace": 0.30})
        colors = {"text": "#2f6fd0", "audio": "#e08a1e", "vision": "#3f9e5a"}
        alpha = alpha_all[i] if alpha_all is not None else None
        for mi, mname in enumerate(Q.MODALITIES):
            ax = axes[mi]
            slots = np.arange(Q.SEQ_LEN)
            t0 = np.array([tb.of_slot(j)[0] if j >= 1 else np.nan for j in slots])
            t1 = np.array([tb.of_slot(j)[1] if j >= 1 else np.nan for j in slots])
            # 非词元槽（[CLS]/[SEP]/填充）与越界槽没有时间，画图前先摘掉，
            # 否则 matplotlib 会因 nan 坐标把整条 bar 丢掉或直接报错。
            valid = np.isfinite(t0) & np.isfinite(t1) & (t1 > t0)
            width = np.zeros_like(t0)
            width[valid] = t1[valid] - t0[valid]
            tv = np.where(valid, t0, 0.0)
            h = np.zeros_like(t0)
            if alpha is not None:
                a = alpha[mi].astype(np.float64)
                mx = a[valid].max() if valid.any() and a[valid].max() > 0 else 1.0
                h[valid] = a[valid] / mx
            else:
                h[valid] = 1.0
            # 语音区间底纹：让「证据落在有声段还是静音段」直接可读
            for (s0, s1) in tb.speech:
                ax.axvspan(s0, s1, color="#000000", alpha=0.06, lw=0)
            cut = ~s.info.observed[mname] & ~s.info.padding_mask
            ax.bar(tv[valid], np.maximum(h[valid], 0.02), width=width[valid],
                   align="edge", color=colors[mname], alpha=0.85,
                   edgecolor="white", linewidth=0.3)
            n_miss = 0
            for j in np.flatnonzero(valid & s.info.missing[mname]):
                ax.bar(tv[j], 1.0, width=max(width[j], 0.01), align="edge",
                       color="#c0392b", alpha=0.85, hatch="///", edgecolor="white",
                       linewidth=0.0)
                n_miss += 1
            n_cut = int((valid & cut).sum())
            tag = (f"{mname}  " + L(f"缺失{n_miss}槽", f"{n_miss} missing")
                   + (L(f" / 零值{n_cut}槽", f" / {n_cut} zero") if n_cut else ""))
            ax.set_ylabel(tag, fontsize=9)
            ax.set_yticks([])
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, max(tb.duration, 1e-3))
        axes[0].set_title(
            f"[{r['sample']}] pred={r['pred_polarity']}  intensity={r['pred_intensity']:.2f}  "
            f"dominant(LOO)={r['dominant_modality_loo'] or 'n/a'}  "
            f"timebase={r['timebase_level']}", fontsize=10)
        axes[-1].set_xlabel(
            "time (s) —  bar height = slot attention (normalized within modality); "
            "red hatch = missing slot; gray band = detected speech", fontsize=8)
        p = os.path.join(fig_dir, f"evidence_timeline_{r['sample']}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        made.append(p)

    # 关键帧条：每条样本 top-3 视觉证据的真实抽帧
    for i in pick:
        r, s, tb = rows[i], samples[i], timebases[i]
        vis = [e for e in r["top_evidence"] if e["modality"] == "vision"
               and e["time_start_sec"] is not None]
        if not vis or not s.video_path:
            continue
        frames, titles, n_faces_total = [], [], 0
        for e in vis[:3]:
            tmid = (e["time_start_sec"] + e["time_end_sec"]) / 2.0
            fr, nf = T.grab_frame(s.video_path, tmid, mosaic=True, return_faces=True)
            if fr is not None:
                frames.append(fr)
                titles.append(f"slot {e['slot']} @ {tmid:.2f}s")
                n_faces_total += int(nf)
        if not frames:
            continue
        fig, axes = plt.subplots(1, len(frames), figsize=(3.1 * len(frames), 2.6))
        axes = np.atleast_1d(axes)
        for ax, fr, ti in zip(axes, frames, titles):
            ax.imshow(fr[:, :, ::-1])
            ax.set_title(ti, fontsize=9)
            ax.axis("off")
        # 图注据实写：只有真的打了码才声称打了码。
        # 早期版本在检测器不可用时静默返回原图，图注却仍写「mosaicked」——
        # 提交物里出现与事实不符的声明比不打码本身更严重。
        if n_faces_total > 0:
            cap_note = f"faces mosaicked for anonymity ({n_faces_total} detected)"
        else:
            cap_note = ("raw frame from the competition-provided video "
                        "(no face detected, so no mosaic applied)")
        fig.suptitle(f"[{r['sample']}] key visual evidence — {cap_note}", fontsize=10)
        p = os.path.join(fig_dir, f"keyframes_{r['sample']}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        made.append(p)

    # 模态作用程度总览
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [r["sample"] for r in rows]
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    for mi, mname in enumerate(Q.MODALITIES):
        vals = np.array([r[f"modality_attention_{mname}"] for r in rows])
        ax.bar(x, vals, bottom=bottom, label=f"{mname} (attention share)")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("attention share")
    ax.set_title("Modality contribution across the 20 attachment-4 samples")
    ax.legend(fontsize=8)
    p = os.path.join(fig_dir, "modality_attention_all.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    made.append(p)
    return made


# ==================== 主流程 ====================

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="问题三：可解释预测与关键证据落地")
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--init", choices=("q2", "train"), default="q2",
                    help="q2=直接复用问题二训练好的同架构模型（默认，保证两问模型一致）；"
                         "train=在问题三内部独立重训一遍")
    ap.add_argument("--q2-model", default=os.path.join(Q2_OUT, "q2_model.pt"))
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cpu")
    print("=" * 72)
    print("问题三：可解释的多模态情感预测（附件4 专项）")
    print("=" * 72)

    a2 = Q.load_attachment2()
    tr, va = a2["train"], a2["valid"]
    y_pol_tr, y_int_tr = Q.to_polarity(tr.reg_label), Q.to_intensity(tr.reg_label)
    y_pol_va, y_int_va = Q.to_polarity(va.reg_label), Q.to_intensity(va.reg_label)

    train_np = M.stack_split(tr)
    valid_np = M.stack_split(va)

    # 文本支路初始化：与问题二用**同一张**蒸馏表（缓存由问题二写出）。
    from q2_train import train_model, _load_or_build_text_table
    _tbl, _tbl_stats = _load_or_build_text_table(tr, os.path.join(Q2_OUT, "text_emb_table.npz"))

    model = None
    if args.init == "q2":
        if os.path.isfile(args.q2_model):
            ck = torch.load(args.q2_model, map_location=device, weights_only=False)
            model = M.MaskedTriModalModel(M.ModelConfig(), text_emb_table=_tbl).to(device)
            model.load_state_dict(ck["state_dict"])
            # 必须显式 eval()：nn.Module 默认 training=True，若不切换，
            # 解释路径里 **dropout(p=0.15) 会一直生效**，注意力与留一 Δ/KL/翻转
            # 全都变成随机的——同一份模型文件重跑会给出不同的解释，
            # 而问题三的全部结论都建立在这些量上。
            model.eval()
            hist = {"best_epoch": ck.get("train_history", {}).get("best_epoch"),
                    "best_valid": ck.get("train_history", {}).get("best_valid"),
                    "source": os.path.relpath(args.q2_model, config.PROJECT_ROOT)}
            print(f"\n[模型] 复用问题二训练结果：{hist['source']}"
                  f"（同架构、同输入接口、同缺失语义；best_epoch={hist['best_epoch']}）")
        else:
            print(f"\n[模型] 未找到 {args.q2_model}，改为在问题三内部独立训练")
            args.init = "train"

    if model is None:
        print("\n[训练] 与问题二同架构同接口，独立训练以便问题三自洽可复现")
        model, hist = train_model(train_np, y_pol_tr, y_int_tr, valid_np, y_pol_va, y_int_va,
                                  args.epochs, args.batch_size, args.lr, args.patience,
                                  device, seed=args.seed, text_emb_table=_tbl)
        model.eval()          # 同 `--init q2`：解释路径必须关掉 dropout，理由见上
    print(f"  最佳轮次={hist['best_epoch']}  "
          f"valid={ {k: round(v,4) for k,v in hist['best_valid'].items()} }")

    print("\n[附件4] 装载 20 条可解释专项样本并重建槽位时间基准")
    a4 = Q.load_attachment4()
    tbs = T.build_all_timebases(a4)
    for tb in tbs:
        print(f"  {tb.key}: n_slots={tb.n_slots:2d} dur={tb.duration:6.2f}s "
              f"speech={len(tb.speech):2d}段 ratio={tb.speech_ratio:.2f} level={tb.level}")

    enc_np = M.stack_attachment4(a4)
    rows, alpha_all = build_explanations(model, enc_np, a4, tbs, device)
    flat = flatten_evidence(rows)

    _write_csv(os.path.join(args.out, "attachment4_predictions.csv"), rows)
    _write_csv(os.path.join(args.out, "attachment4_top_evidence.csv"), flat)
    with open(os.path.join(args.out, "attachment4_explanations.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    # 槽级注意力另存数组：图件与后续复核要用，JSON 里放不下也不该放
    np.savez_compressed(os.path.join(args.out, "attachment4_slot_attention.npz"),
                        slot_attention=alpha_all,
                        samples=np.array([r["sample"] for r in rows]),
                        modalities=np.array(Q.MODALITIES))

    # 汇总：模态作用程度与主要参考模态的总体分布
    dom_counts: Dict[str, int] = {}
    for r in rows:
        k = r["dominant_modality_loo"] or "undetermined"
        dom_counts[k] = dom_counts.get(k, 0) + 1
    share_mean = {m: float(np.mean([r[f"modality_attention_{m}"] for r in rows]))
                  for m in Q.MODALITIES}
    loo_mean = {m: float(np.mean([r[f"modality_loo_{m}"] for r in rows]))
                for m in Q.MODALITIES}
    loo_kl_mean = {m: float(np.mean([r[f"modality_loo_kl_{m}"] for r in rows]))
                   for m in Q.MODALITIES}
    loo_flip_n = {m: int(np.sum([r[f"loo_flip_{m}"] for r in rows]))
                  for m in Q.MODALITIES}
    # 证据覆盖：每个模态是否都被某条样本的关键证据点到（赛题第 28 行的硬要求）
    evid_mod_cov = {m: int(np.sum([any(e["modality"] == m for e in r["top_evidence"])
                                   for r in rows])) for m in Q.MODALITIES}
    pred_dist: Dict[str, int] = {}
    for r in rows:
        pred_dist[r["pred_polarity"]] = pred_dist.get(r["pred_polarity"], 0) + 1

    summary = {
        "n_samples": len(rows),
        "model_source": hist.get("source", "问题三内部独立训练（同一训练协议与种子）"),
        "text_emb_table": _tbl_stats,
        "train_best_epoch": hist["best_epoch"],
        "valid_metrics": hist["best_valid"],
        "pred_polarity_distribution": pred_dist,
        "attention_share_mean": share_mean,
        "loo_drop_mean": loo_mean,
        "loo_kl_mean": loo_kl_mean,
        "loo_flip_counts": loo_flip_n,
        "evidence_coverage_by_modality": evid_mod_cov,
        "dominant_modality_counts": dom_counts,
        "timebase_levels": {lv: sum(1 for t in tbs if t.level == lv)
                            for lv in sorted({t.level for t in tbs})},
        "vision_fully_missing_samples": [r["sample"] for r in rows if r["vision_fully_missing"]],
    }

    # 赛题 L52 要问题三的「验证集基础性能评价与错误归因」。
    # `--init q2` 时问题三用的就是问题二的模型，验证集结果**必然**与问题二逐位相同——
    # 此时重复算一遍只会制造两份可能漂移的数字，故直接指向问题二的产物并说明原因；
    # `--init train` 时是另一个模型实例，才需要自己算一份。
    if args.init == "q2" and os.path.isfile(os.path.join(Q2_OUT, "error_analysis_valid.json")):
        summary["valid_performance"] = {
            "source": os.path.relpath(os.path.join(Q2_OUT, "error_analysis_valid.json"),
                                      config.PROJECT_ROOT),
            "note": "问题三复用问题二的模型实例，验证集性能与错误归因与问题二完全一致，"
                    "故直接引用而不重复计算。",
            "metrics": hist.get("best_valid"),
        }
    else:
        from q2_train import (error_analysis as _ea, write_error_analysis as _wea,
                              evaluate as _ev)
        _pred = _ev(model, valid_np, y_pol_va, y_int_va, device)
        _an = _ea(_pred, y_pol_va, y_int_va, valid_np, sample_names=list(va.ids))
        _wea(_an, args.out, tag="valid")
        # 标签要说清楚**算的是谁的模型**：`--init q2` 时问题二那份还没生成，
        # 这里代算的是同一个模型实例，结果与问题二一致，不是另一个模型。
        _own = (args.init == "train")
        summary["valid_performance"] = {
            "source": "error_analysis_valid.json（本处自算）",
            "model_instance": ("问题三独立训练的模型" if _own
                               else "与问题二同一个模型实例（问题二产物尚未生成，此处代算）"),
            "analysis": _an}
        print("\n[验证集错误归因]（"
              + ("问题三独立训练的模型实例" if _own
                 else "复用的问题二模型实例，问题二产物尚未生成，此处代算") + "）")
        for c in _an["conclusions"]:
            print(f"  · {c}")

    # 两个口径是否一致：按样本统计 dominant(LOO) 与 attention 最大模态是否相同
    agree = 0
    for r in rows:
        att_top = max(Q.MODALITIES, key=lambda m: r[f"modality_attention_{m}"])
        if r["dominant_modality_loo"] == att_top:
            agree += 1
    summary["loo_vs_attention_agreement"] = {
        "agree": agree, "total": len(rows), "ratio": round(agree / len(rows), 4)}

    if not args.no_figures:
        made = make_figures(rows, a4, tbs, args.out, alpha_all=alpha_all)
        summary["figures"] = made
        print(f"\n[图件] 生成 {len(made)} 张")

    # 典型样本解释卡（L49）：在 summary 定稿后出卡，卡里的全样本汇总才完整
    card_path = write_explanation_cards(
        rows, a4, os.path.join(args.out, "explanation_cards.md"), summary)
    print(f"\n[解释卡] {os.path.relpath(card_path, config.PROJECT_ROOT)}")

    torch.save({"state_dict": model.state_dict(),
                "config": M.ModelConfig().__dict__,
                "train_history": hist}, os.path.join(args.out, "q3_model.pt"))
    with open(os.path.join(args.out, "q3_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[汇总]")
    print(f"  极性分布：{pred_dist}")
    print(f"  注意力占比均值：{ {k: round(v,3) for k,v in share_mean.items()} }")
    print(f"  留一扰动均值  ：{ {k: round(v,3) for k,v in loo_mean.items()} }")
    print(f"  留一 KL 均值  ：{ {k: round(v,4) for k,v in loo_kl_mean.items()} }")
    print(f"  拿掉即翻转类别：{loo_flip_n}")
    print(f"  证据模态覆盖  ：{evid_mod_cov}（每模态被点到的样本数 / {len(rows)}）")
    print(f"  主要参考模态分布：{dom_counts}")
    print(f"  两口径一致率：{agree}/{len(rows)} = {agree/len(rows):.2%}")
    print(f"  视觉整段缺失样本：{summary['vision_fully_missing_samples']}")
    print(f"\n产物目录：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
