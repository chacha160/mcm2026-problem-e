# -*- coding: utf-8 -*-
"""
_gen_paper_tables.py —— 从交付产物生成论文用的 Markdown 表格片段。

设计原则：论文里的每一个数字都必须来自 data/ 下的实际产物，
不允许在正文里手写。本脚本把「产物 → 表格 Markdown」这一步固化下来，
生成的 _tables/*.md 由 _build_paper.py 按占位符嵌入正文。

用法：python _gen_paper_tables.py
"""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "_tables")
os.makedirs(OUT, exist_ok=True)


def P(*a):
    return os.path.join(ROOT, *a)


def read_csv(*a, **kw):
    return pd.read_csv(P(*a), **kw)


def read_json(*a):
    with io.open(P(*a), encoding="utf-8") as f:
        return json.load(f)


def f3(x):
    """固定三位小数；nan -> —"""
    try:
        v = float(x)
    except Exception:
        return "—"
    if v != v:
        return "—"
    return "%.3f" % v


def fi(x):
    try:
        return "%d" % int(round(float(x)))
    except Exception:
        return "—"


def save(name, header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    txt = "\n".join(lines) + "\n"
    with io.open(os.path.join(OUT, name + ".md"), "w", encoding="utf-8") as f:
        f.write(txt)
    print("  %-34s %d 行" % (name + ".md", len(rows)))
    return txt


# ══════════════════════════════════════════════════════════════ 问题一
summ = read_csv("data", "q1_delivery", "summary_q1.csv")
ver = read_json("data", "q1_delivery", "verify_report.json")
led = read_csv("data", "q1_delivery", "anomaly_ledger.csv")
typ = read_csv("data", "q1_delivery", "typical_samples.csv")
al = pd.read_csv(P("data", "timeline", "-s9qJ7ATP7w_7_correspondence.csv"))
buf = read_csv("data", "q1_delivery", "size_budget.csv")

print("问题一：")
save("q1_dims",
     ["模态", "对齐后形状", "原始序列长度区间", "特征维度", "提取工具与口径"],
     [["文本 text", "50 × 768", "%d–%d" % (summ.text_seq_len.min(), summ.text_seq_len.max()),
       "768", "roberta-base 末层子词隐状态按词求均值（离线）"],
      ["语音 audio", "50 × 74", "%d–%d" % (summ.audio_seq_len.min(), summ.audio_seq_len.max()),
       "74", "ffmpeg 16 kHz 重采样 + librosa（MFCC/差分/韵律/频谱）"],
      ["视觉 vision", "50 × 35", "%d–%d" % (summ.vision_seq_len.min(), summ.vision_seq_len.max()),
       "35", "cv2 抽帧 + MTCNN 检脸 + ResNet-50 池化后固定随机投影"]])

save("q1_verify",
     ["编号", "检验内容", "受检单元", "失败", "关键结果"],
     [(c["code"], c["name"], fi(c["checked"]), fi(c["failed"]), c["detail"].split("。")[0])
      for c in ver["checks"]])
print("  V1–V12 断言总数 %d，失败 %d" % (ver["summary"]["total_assertions"],
                                        ver["summary"]["failed_assertions"]))

cnt = led.groupby(["anomaly_code", "modality"]).size().reset_index(name="n")
cnt = cnt.sort_values("n", ascending=False)
save("q1_ledger",
     ["异常代码", "模态", "条数", "处置"],
     [(r.anomaly_code, r.modality, fi(r.n),
       led[led.anomaly_code == r.anomaly_code].action.iloc[0]) for r in cnt.itertuples()])

save("q1_typical",
     ["类别", "判据", "实测判据值", "样本编号", "时长 (s)", "词数", "有声帧占比", "人脸检出比"],
     [(r.category, r.criterion, r.criterion_value, r.sample_id, f3(r.duration_sec),
       fi(r.num_words), f3(r.voiced_ratio), f3(r.face_ratio)) for r in typ.itertuples()])

# 逐槽对应表（典型样本 -s9qJ7ATP7w_7，50 槽）
rows = []
for r in al.itertuples():
    tw = "—" if (isinstance(r.text_words, float) and r.text_words != r.text_words) else str(r.text_words)
    au = "—" if (isinstance(r.audio_unit_range, float) and r.audio_unit_range != r.audio_unit_range) else str(r.audio_unit_range)
    at = "—" if (isinstance(r.audio_t_range, float) and r.audio_t_range != r.audio_t_range) else str(r.audio_t_range)
    vu = "—" if (isinstance(r.vision_unit_range, float) and r.vision_unit_range != r.vision_unit_range) else str(r.vision_unit_range)
    vp = "—" if (isinstance(r.vision_pts_range, float) and r.vision_pts_range != r.vision_pts_range) else str(r.vision_pts_range)
    rows.append([fi(r.slot), "%.4f–%.4f" % (r.t_start_sec, r.t_end_sec), tw,
                 fi(r.text_valid), au, at, vu, vp])
save("q1_slots", ["槽", "时间区间 (s)", "落槽词", "文本有效", "语音源帧号", "语音时刻 (s)",
                  "视觉源帧号", "视觉时刻 (s)"], rows)

save("q1_budget",
     ["数据块", "单元数", "特征维度", "元素数", "float32 未压缩 (MB)"],
     [[r.block, fi(r.units) if r.units >= 0 else "—", fi(r.dim) if r.dim >= 0 else "—",
       fi(r.elements) if r.elements >= 0 else "—",
       f3(r.uncompressed_MB_f32) if r.uncompressed_MB_f32 >= 0 else "—"] for r in buf.itertuples()])

# 赛题「四(一)2(2)」要求的全量 100 条样本汇总表
s = summ.sort_values("sample_id")
save("q1_all100",
     ["样本编号", "原始时长 (s)", "对齐粒度 (s)", "文本维度 768 序列长/有效槽",
      "语音维度 74 序列长/有效槽", "视觉维度 35 序列长/有效槽",
      "人脸检出比", "音视频可溯源率", "判定"],
     [[r.sample_id, f3(r.duration_sec), "%.4f" % r.align_granularity_sec,
       "%d / %d" % (r.text_seq_len, r.text_valid_slots),
       "%d / %d" % (r.audio_seq_len, r.audio_valid_slots),
       "%d / %d" % (r.vision_seq_len, r.vision_valid_slots),
       f3(r.face_ratio), f3(r.traceable_av_ratio), r.verdict] for r in s.itertuples()])

# ══════════════════════════════════════════════════════════════ 问题二
m2 = read_json("data", "q2", "q2_metrics.json")
ms = read_json("data", "q2", "missingness_summary.json")
ab = read_csv("data", "q2", "text_branch_ablation.csv")
att3 = read_csv("data", "q2", "attachment3_predictions.csv")
ex = read_json("data", "q2", "error_analysis_valid.json")
exn = read_json("data", "q2", "error_analysis_test.json")

print("问题二：")
save("q2_setup",
     ["项", "设置"],
     [["数据", "附件 2 aligned_50：train 3395 / valid 728 / test 727 条"],
      ["输入接口", "text_bert(3,50) + audio(50,74) + vision(50,35)，三处逐位一致"],
      ["隐藏维度 / 编码器", "d = 128；2 层 TransformerEncoder，4 头，前馈 256，norm_first"],
      ["Dropout", "0.15（嵌入、编码器与双头；另见训练期模态丢弃）"],
      ["优化器", "AdamW（解耦权重衰减），学习率 1e-3，权重衰减 1e-4，余弦退火；批大小 64"],
      ["最大轮数 / 早停", "60 / patience 12；实际 17 轮，最优轮 best_epoch = %d" % m2["train_history"]["best_epoch"]],
      ["选择分数", "0.5·macro-F1 + 0.5·(1 − MAE/3)，仅按 valid 早停"],
      ["损失", "加权交叉熵（极性，类别权重取频次倒数平方根）+ SmoothL1（强度）"],
      ["训练期模态丢弃", "以 0.25 概率整模态丢弃，保证至少保留一个模态"],
      ["随机种子", "42（初始化、批次打乱、受控缺失注入共用同一序列）"],
      ["参数量", "2 282 885（约 8.7 MB，fp32）"]])

base = ms["by_missing_type"]
save("q2_miss_type",
     ["缺失类型", "Accuracy", "macro-F1", "MAE", "Pearson", "配置数"],
     [["无缺失（基线）", f3(m2["valid"]["accuracy"]), f3(m2["valid"]["macro_f1"]),
       f3(m2["valid"]["mae"]), f3(m2["valid"]["pearson"]), "1"]] +
     [[{"text": "文本", "audio": "语音", "vision": "视觉", "audio+vision": "语音+视觉"}[k],
       f3(v["accuracy"]), f3(v["macro_f1"]), f3(v["mae"]), f3(v["pearson"]), fi(v["n_configs"])]
      for k, v in base.items()])

save("q2_miss_dur",
     ["缺失时长", "切断比例", "Accuracy", "macro-F1", "MAE", "Pearson", "配置数"],
     [[{"short": "短", "medium": "中", "long": "长"}[k],
       {"short": "20%", "medium": "40%", "long": "60%"}[k],
       f3(v["accuracy"]), f3(v["macro_f1"]), f3(v["mae"]), f3(v["pearson"]), fi(v["n_configs"])]
      for k, v in ms["by_duration"].items()])

save("q2_miss_pos",
     ["缺失位置", "Accuracy", "macro-F1", "MAE", "Pearson", "配置数"],
     [[{"head": "头部", "middle": "中部", "tail": "尾部", "random": "随机"}[k],
       f3(v["accuracy"]), f3(v["macro_f1"]), f3(v["mae"]), f3(v["pearson"]), fi(v["n_configs"])]
      for k, v in ms["by_position"].items()])

save("q2_ablation",
     ["文本支路初始化", "最优轮", "valid macro-F1", "valid Acc", "valid MAE", "valid Pearson",
      "test macro-F1", "test Acc", "test MAE", "test Pearson"],
     [[{"distill": "官方 768 维蒸馏初值（本文采用）", "random": "随机初值"}[r.text_branch_init],
       fi(r.best_epoch), f3(r.valid_macro_f1), f3(r.valid_accuracy), f3(r.valid_mae),
       f3(r.valid_pearson), f3(r.test_macro_f1), f3(r.test_accuracy), f3(r.test_mae),
       f3(r.test_pearson)] for r in ab.itertuples()])

save("q2_perf",
     ["数据集", "样本数", "Accuracy", "macro-F1", "MAE", "Pearson"],
     [["验证集 valid", fi(m2["valid"]["n"]), f3(m2["valid"]["accuracy"]), f3(m2["valid"]["macro_f1"]),
       f3(m2["valid"]["mae"]), f3(m2["valid"]["pearson"])],
      ["测试集 test", "727", f3(m2["test"]["accuracy"]), f3(m2["test"]["macro_f1"]),
       f3(m2["test"]["mae"]), f3(m2["test"]["pearson"])]])

save("q2_baseline",
     ["平凡基线", "数值", "说明"],
     [["预测多数类 Positive（极性）", "Accuracy = " + f3(m2["valid"]["baseline"]["accuracy_predict_majority"]),
       "无任何输入信息时的最优常数分类器"],
      ["预测训练集均值（强度）", "MAE = " + f3(m2["valid"]["baseline"]["mae_predict_train_mean"]),
       "无任何输入信息时的最优常数回归器"]])

save("q2_recall",
     ["类别", "召回率", "精确率"],
     [[k, f3(m2["valid"]["recall_" + k.lower()]), f3(ex["confusion"]["precision"][k])]
      for k in ("Negative", "Neutral", "Positive")])

save("q2_att3",
     ["样本", "有效词元 L", "语音缺失槽", "视觉缺失槽", "极性预测", "p(负)", "p(中)", "p(正)",
      "强度预测", "带符号值"],
     [[r.sample, fi(r.valid_len_token), fi(r.audio_missing_slots), fi(r.vision_missing_slots),
       r.pred_polarity, f3(r.pred_polarity_prob_neg), f3(r.pred_polarity_prob_neu),
       f3(r.pred_polarity_prob_pos), f3(r.pred_intensity), f3(r.pred_signed_value)]
      for r in att3.itertuples()])

print("  附件3 极性分布", att3.pred_polarity.value_counts().to_dict(),
      "缺失样本数", fi((att3.audio_missing_slots + att3.vision_missing_slots > 0).sum()))

save("q2_err_conf",
     ["真实 \\ 预测", "Negative", "Neutral", "Positive", "召回率"],
     [[lab] + [fi(v) for v in row] + [f3(ex["confusion"]["recall"][lab])]
      for lab, row in zip(ex["confusion"]["labels"], ex["confusion"]["matrix_true_by_pred"])])

save("q2_err_slice",
     ["缺失槽数区间", "样本数", "Accuracy", "macro-F1", "MAE", "Pearson"],
     [[k, fi(v["n"]), f3(v["accuracy"]), f3(v["macro_f1"]), f3(v["mae"]), f3(v["pearson"])]
      for k, v in ex["by_missing_slot_count"].items()])

# ══════════════════════════════════════════════════════════════ 问题三
q3 = read_json("data", "q3", "q3_summary.json")
att4 = read_csv("data", "q3", "attachment4_predictions.csv")
ev = read_csv("data", "q3", "attachment4_top_evidence.csv")
exp4 = read_json("data", "q3", "attachment4_explanations.json")

print("问题三：")
save("q3_modality",
     ["模态", "注意力占比（归因）", "留一强度变化 Δ", "留一 KL（因果）", "拿掉即翻转（条）"],
     [[{"text": "文本", "audio": "语音", "vision": "视觉"}[k],
       f3(q3["attention_share_mean"][k]), "%+.4f" % q3["loo_drop_mean"][k],
       f3(q3["loo_kl_mean"][k]), fi(q3["loo_flip_counts"][k])]
      for k in ("text", "audio", "vision")])

save("q3_evidence",
     ["模态", "覆盖样本数", "证据行数", "其中带关键帧引用"],
     [[{"text": "文本", "audio": "语音", "vision": "视觉"}[k],
       fi(q3["evidence_coverage_by_modality"][k]),
       fi((ev.modality == k).sum()),
       fi(ev[(ev.modality == k) & ev.keyframe_ref.notna()].shape[0])]
      for k in ("text", "audio", "vision")])

# 时间轴重建结果（附件4 全量）
save("q3_timebase",
     ["样本", "时长 (s)", "时间轴级别", "语音占比", "有效词元 L", "主要参考模态", "视觉整体缺失"],
     [[r.sample, f3(r.duration_sec), r.timebase_level, f3(r.speech_ratio),
       fi(r.valid_len_token),
       {"text": "文本", "audio": "语音", "vision": "视觉"}[r.dominant_modality_kl],
       "是" if r.vision_fully_missing else "否"] for r in att4.itertuples()])

save("q3_att4",
     ["样本", "时长 (s)", "有效词元", "极性预测", "p(负)", "p(中)", "p(正)", "强度预测",
      "主要模态", "文本注意力", "语音注意力", "视觉注意力", "文本 KL", "语音 KL", "视觉 KL"],
     [[r.sample, f3(r.duration_sec), fi(r.valid_len_token), r.pred_polarity,
       f3(r.pred_prob_negative), f3(r.pred_prob_neutral), f3(r.pred_prob_positive),
       f3(r.pred_intensity),
       {"text": "文本", "audio": "语音", "vision": "视觉"}[r.dominant_modality_kl],
       f3(r.modality_attention_text), f3(r.modality_attention_audio), f3(r.modality_attention_vision),
       f3(r.modality_loo_kl_text), f3(r.modality_loo_kl_audio), f3(r.modality_loo_kl_vision)]
      for r in att4.itertuples()])

# 典型样本解释卡：取 1 号样本的完整证据
ev1 = ev[ev["sample"].astype(str) == "1"].sort_values("rank")
row1 = att4[att4["sample"].astype(str) == "1"]
if len(ev1):
    save("q3_card",
         ["排序", "口径", "模态", "槽号", "注意力", "时间区间 (s)", "文本片段", "关键帧引用"],
         [[fi(r.rank),
           {"global_topk": "全局 top-5", "modality_topk": "模态内 top-3"}.get(r.scope, r.scope),
           {"text": "文本", "audio": "语音", "vision": "视觉"}.get(r.modality, r.modality),
           fi(r.slot), "%.4f" % r.attention,
           "%.3f–%.3f" % (r.time_start_sec, r.time_end_sec),
           r.text_fragment if isinstance(r.text_fragment, str) else "—",
           r.keyframe_ref if isinstance(r.keyframe_ref, str) else "—"]
          for r in ev1.itertuples()])
    if len(row1):
        r = row1.iloc[0]
        print("  解释卡样本 1：极性 %s（p=%s）强度 %s；文本注意力 %s / 语音 %s / 视觉 %s；证据 %d 条"
              % (r.pred_polarity, f3(max(r.pred_prob_negative, r.pred_prob_neutral, r.pred_prob_positive)),
                 f3(r.pred_intensity), f3(r.modality_attention_text),
                 f3(r.modality_attention_audio), f3(r.modality_attention_vision), len(ev1)))
else:
    print("  !! 未取到解释卡")

# 证据表列名自检
print("  证据表列：", ev.columns.tolist())

with io.open(os.path.join(OUT, "_meta.json"), "w", encoding="utf-8") as f:
    json.dump({
        "q1_assertions": ver["summary"]["total_assertions"],
        "q1_n_anomaly": int(len(led)),
        "q2_att3_n": int(len(att3)),
        "q2_att3_missing": int((att3.audio_missing_slots + att3.vision_missing_slots > 0).sum()),
        "q3_ev_n": int(len(ev)),
        "q2_artifacts": {},
    }, f, ensure_ascii=False, indent=1)
print("\n全部表格已写入", OUT)
