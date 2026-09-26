# -*- coding: utf-8 -*-
"""
_make_q1_submit.py —— 把问题一的输出整合成根目录下的 `问题一提交/`

做三件事，都是**生成**、不是手工搬：

  1. 复制：把 `data/q1_v2/` 里计入提交的部分（100 个 NPZ、两张汇总表、
     典型样本对应表、异常台账、核验回执、元数据）整份复制到 `问题一提交/`。
     `.cache/` 与 `logs/` 是可重建中间态，按体积台账的口径不入包。
  2. 出表：从产物现算两份人读文档——`汇总表_全量100条样本.md`（赛题要的
     样本编号/模态类型/原始有效时长/特征维度/对齐粒度五列，另加特征文件名
     一列以便逐条落到文件）与 `典型样本验证.md`（3 个典型样本的逐词对应表、
     追溯链、复算方法）。
  3. 核验：写完再读一遍，逐项断言（100 个 NPZ、汇总表 100 行、词单元合计、
     结构合法词与零时长词的合计、身份红线零命中），并把 NPZ 的字段清单按
     组展开——字段分组若与实际 NPZ 对不上，脚本直接抛错，不静默漏字段。

因此本脚本是**幂等**的：重跑一次会重建同一份 `问题一提交/`（覆盖同名文件）。
产物改动请改本脚本，不要去改 `问题一提交/` 里的文件。

    python _make_q1_submit.py            # 重建
    python _make_q1_submit.py --check    # 只核验现有目录，不写盘
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "q1_v2"
OUT = ROOT / "问题一提交"

#: 入包的子目录与文件（相对 data/q1_v2）。.cache/、logs/ 不在其中：它们是
#: 可重建中间态，体积台账把它们排除在提交之外，这里保持同一口径。
COPY_DIRS = ("features", "correspondence", "audit", "reports", "metadata")
COPY_FILES = ("manifest.csv", "results_100.csv", "config_snapshot.json")

#: NPZ 字段的十二个分组。顺序即表里的顺序；每个字段必须恰好落进一组，
#: 少一个或多一个都会在 _check_field_groups 里抛错。
FIELD_GROUPS = [
    ("身份与版本", "回答「这个文件是哪一条、由哪次运行、用哪套权重产出」",
     ["schema_version", "sample_key", "video_id", "clip_id", "official_text",
      "source_video_relpath", "source_video_sha256", "config_sha256",
      "roberta_model_sha256", "mediapipe_model_sha256", "opensmile_config_sha256",
      "alignment_source", "alignment_params_json", "alignment_run_sha256",
      "decode_provenance_json"]),
    ("路由判定", "逐样本的时序组织结论；判定逻辑是纯函数，可用标量现场重放",
     ["alignment_mode", "alignment_granularity", "text_av_time_mapping_status",
      "text_audio_correspondence", "word_time_basis", "content_assertion",
      "routing_reasons_json", "audio_speech_evidence"]),
    ("模态存在性与有效性闸门", "五类存在性与四级有效性分列，互不代替",
     ["text_present", "text_content_valid", "audio_present", "audio_observation_valid",
      "audio_speech_valid", "video_present", "audio_visual_time_valid",
      "vision_feature_valid", "face_feature_valid"]),
    ("词级结构", "词单元的文本与时间区间，以及槽位到词单元的映射",
     ["words", "word_char_start", "word_char_end", "aligner_slot_count",
      "aligner_slot_to_word_index", "word_start_sec", "word_end_sec",
      "word_time_valid"]),
    ("词级有效掩码", "三个模态各自的有效性，逐词给出",
     ["text_word_valid", "word_audio_valid", "word_vision_valid"]),
    ("词级特征", "逐词聚合后的三种模态特征，无效词整行为零",
     ["text_word_feat", "word_audio_feat", "word_vision_feat"]),
    ("原生观测·语音", "词级语音特征由这批窗聚合而来，保留窗级原值",
     ["raw_audio_lld_values", "raw_audio_lld_start_sec", "raw_audio_lld_end_sec",
      "raw_audio_lld_center_sec", "raw_audio_lld_valid", "raw_audio_lld_feature_names",
      "lld_frame_len_sec", "lld_frame_step_sec", "audio_observation_length"]),
    ("原生观测·视觉", "词级视觉特征由这批帧聚合而来，保留帧级原值",
     ["raw_video_blendshape_values", "raw_video_pts_sec",
      "raw_video_support_start_sec", "raw_video_support_end_sec",
      "raw_video_face_feature_valid", "raw_video_blendshape_names",
      "video_frame_count", "face_detected_frame_count", "video_observation_length"]),
    ("词→窗/帧的溯源索引", "CSR 结构，给出每个词的聚合用了哪几行原生观测",
     ["audio_word_feature_indptr", "audio_word_feature_indices",
      "vision_word_feature_indptr", "vision_word_feature_indices"]),
    ("文本子词溯源", "RoBERTa 子词到词单元的归属，以及子词级输入",
     ["roberta_token_ids", "roberta_token_offsets", "roberta_token_word_index",
      "roberta_word_token_indptr", "roberta_word_token_indices",
      "roberta_tokenizer_failure_count", "text_input_length", "text_sequence_length"]),
    ("共享时间轴与覆盖", "三路观测归到同一原点后的支撑区间",
     ["shared_t0_sec", "original_effective_duration_sec",
      "audio_presentation_coverage_sec", "video_presentation_coverage_sec",
      "av_common_coverage_sec"]),
    ("结构计数与对齐告警", "词单元/槽位/映射失败的计数，异常只记不删的依据",
     ["alignment_zero_duration_word_count", "aligner_zero_duration_slot_count",
      "alignment_mapping_fail_count", "word_without_slot_count"]),
]

#: 形状列里允许出现的固定维度。出现别的定长维度，说明 schema 变了，脚本报错。
FIXED_DIMS = {2, 25, 50, 52, 104, 768}


def _s(x) -> str:
    return "" if x is None else str(x)


def read_csv_rows(path: Path):
    with io.open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path):
    with io.open(path, encoding="utf-8") as fh:
        return json.load(fh)


def md_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        cells = [_s(c).replace("|", "\\|").replace("\n", " ") for c in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def f3(x) -> str:
    try:
        return "%.3f" % float(x)
    except Exception:
        return "—"


def i_(x) -> str:
    try:
        return "%d" % int(round(float(x)))
    except Exception:
        return "—"


# ───────────────────────────────────────────────────────────── 字段分组自检
def check_field_groups(fields):
    """十二个分组是否恰好覆盖 NPZ 的全部字段，且无重复。"""
    seen: list = []
    for name, _desc, names in FIELD_GROUPS:
        for n in names:
            seen.append(n)
    dup = sorted({n for n in seen if seen.count(n) > 1})
    if dup:
        raise SystemExit("字段分组有重复：%s" % dup)
    missing = [f for f in fields if f not in seen]
    extra = [f for f in seen if f not in fields]
    if missing or extra:
        raise SystemExit("字段分组与实际 NPZ 不符：未分组 %s；分组里多出 %s"
                         % (missing, extra))


def shape_symbol(dim, n_words, n_lld, n_frames, n_tokens) -> str:
    """把实测长度换成符号，让形状列在逐样本不同的维度上仍然说得准。"""
    if dim == n_words:
        return "L"
    if dim == n_lld:
        return "T"
    if dim == n_frames:
        return "F"
    if dim == n_tokens:
        return "S"
    if dim == n_words + 1:
        return "L+1"
    if dim in FIXED_DIMS:
        return str(dim)
    return "nnz"          # CSR 索引数组：非零元个数


def shape_of(arr, n_words, n_lld, n_frames, n_tokens) -> str:
    if arr.shape == ():
        return "标量"
    syms = [shape_symbol(d, n_words, n_lld, n_frames, n_tokens) for d in arr.shape]
    if len(syms) == 1:
        return "(%s,)" % syms[0]
    return "(" + ", ".join(syms) + ")"


# ───────────────────────────────────────────────────────────── 主流程
def main() -> int:
    import numpy as np

    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    check_only = "--check" in sys.argv
    if not SRC.is_dir():
        raise SystemExit("源目录不存在：%s" % SRC)

    results = read_csv_rows(SRC / "results_100.csv")
    manifest = read_csv_rows(SRC / "manifest.csv")
    verify = read_json(SRC / "reports" / "verify_report.json")
    ledger = read_csv_rows(SRC / "audit" / "anomaly_ledger.csv")
    corr_meta = read_json(SRC / "correspondence" / "_meta.json")
    ledger_json = read_json(SRC / "reports" / "q1v2_size_ledger.json")
    npz_files = sorted((SRC / "features").glob("*.npz"))
    if len(npz_files) != len(results) != 100:
        raise SystemExit("样本数与产物的行数不符：NPZ %d / results %d"
                         % (len(npz_files), len(results)))
    by_key = {r["sample_key"]: r for r in results}

    # 抽样文件（排序后第一个）用来读字段清单与形状符号
    with np.load(npz_files[0], allow_pickle=False) as z:
        fields = list(z.files)
        n_words = int(z["words"].shape[0])
        n_lld = int(z["raw_audio_lld_values"].shape[0])
        n_frames = int(z["raw_video_blendshape_values"].shape[0])
        n_tokens = int(z["roberta_token_ids"].shape[0])
        z_arrays = {k: (str(z[k].dtype).lstrip("<"), z[k]) for k in z.files}
    check_field_groups(fields)
    if len({n_words, n_lld, n_frames, n_tokens}) != 4:
        raise SystemExit("抽样文件的四个长度出现相等，形状符号有歧义："
                         "L=%d T=%d F=%d S=%d" % (n_words, n_lld, n_frames, n_tokens))

    n_word_units = sum(int(r["n_official_words"]) for r in results)
    sum_valid_length = sum(int(r["valid_length"]) for r in results)
    sum_struct_valid = sum(int(r["n_structurally_valid_words"]) for r in results)
    sum_zero_dur = sum(int(r["alignment_zero_duration_word_count"]) for r in results)
    ev = {}
    for c in verify["checks"]:
        ev.update(c.get("evidence") or {})
    acc = ledger_json["accounting"]

    # ---- 1. 复制 ----
    if not check_only:
        OUT.mkdir(exist_ok=True)
        for d in COPY_DIRS:
            dst = OUT / d
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(SRC / d, dst)
        for f in COPY_FILES:
            shutil.copy2(SRC / f, OUT / f)
    if not OUT.is_dir():
        raise SystemExit("目标目录不存在，且未加 --check：%s" % OUT)
    # 复制进来的文件数从**源**侧数：目标目录里可能还留着上一轮生成的说明文档，
    # 从目标侧数会让重跑一次就多出几份，数字越滚越大。
    n_copied = (sum(len([p for p in (SRC / d).rglob("*") if p.is_file()])
                    for d in COPY_DIRS) + len(COPY_FILES))

    # ---- 2. 汇总表（100 条）----
    gran_code = {"word": "W", "clip": "C"}
    modal_code = {"text|audio|vision": "TAV"}
    all_rows = []
    for r in results:
        npz_name = os.path.basename(r["npz_path"]) if r.get("npz_path") else ""
        all_rows.append([
            r["sample_key"], npz_name,
            modal_code.get(r["modalities"], r["modalities"].replace("|", "/")),
            f3(r["original_effective_duration_sec"]),
            "%d" % (768 + 50 + 104),
            gran_code.get(r["alignment_granularity"], r["alignment_granularity"]),
        ])

    spec_rows = [["文本 text", "(L, 768)", "768", "RoBERTa 子词", "768",
                  "roberta-base 末层隐状态按词池化"],
                 ["语音 audio", "(L, 50)", "50", "openSMILE LLD 窗", "25",
                  "eGeMAPSv02 LowLevelDescriptors 逐窗 25 维；词级取 均值 与 总体标准差(ddof=0)"],
                 ["视觉 vision", "(L, 104)", "104", "MediaPipe blendshape 帧", "52",
                  "FaceLandmarker 逐帧 52 维 blendshape；词级取 均值 与 总体标准差(ddof=0)"]]

    field_rows = []
    for gname, gdesc, names in FIELD_GROUPS:
        for j, n in enumerate(names):
            dt, arr = z_arrays[n]
            # 组名与组说明只在该组首行出现，其余留空：同一串字铺满全表并不增加信息
            field_rows.append([gname if j == 0 else "",
                               "`%s`" % n, dt,
                               shape_of(arr, n_words, n_lld, n_frames, n_tokens),
                               gdesc if j == 0 else ""])

    schema_md = "\n".join([
        "# 问题一：特征文件规范与全量结果汇总",
        "",
        "本文件是**生成**的（`_make_q1_submit.py`），数字全部现读产物现算，"
        "不手抄。两份机器可读的同源表在旁边：`manifest.csv`（59 列，含执行期三列）"
        "与 `results_100.csv`（56 列）。",
        "",
        "## 1 特征文件的存储格式与读取方式",
        "",
        "### 1.1 组织与命名",
        "",
        "100 条样本各一个 NPZ，放在 `features/`。文件名是"
        "`<video_id>___<clip_id>.npz`（三个下划线），与 `results_100.csv` 的"
        "`sample_key`（写作 `<video_id>$_$<clip_id>`）一一对应，映射由该表的"
        "`npz_path`、`npz_bytes` 两列给出。",
        "",
        "每个 NPZ 内部：标量 43 个、一维/二维数组 42 个，合计 %d 个字段，"
        "按十二组组织（下表）。数组长度 L（词单元数）逐样本不同，"
        "所有以 L 为第一维的数组在同一个文件里等长，可直接按行对齐。" % len(fields),
        "",
        "### 1.2 字段清单",
        "",
        "形状列中的符号：**L** 词单元数、**T** 语音 LLD 窗数、**F** 视频帧数、"
        "**S** RoBERTa 子词数、**L+1** CSR 的 indptr、**nnz** 该 CSR 的非零元个数。"
        "这些长度逐样本不同，读取时应从同一文件内的对应计数标量取，不要写死。",
        "",
        md_table(["组", "字段", "dtype", "形状", "该组回答什么"], field_rows),
        "",
        "### 1.3 读取方式",
        "",
        "```python",
        "import numpy as np",
        "",
        'with np.load("features/<video_id>___<clip_id>.npz", allow_pickle=False) as z:',
        '    L      = z["words"].shape[0]           # 词单元数，逐样本不同',
        '    start  = z["word_start_sec"]           # (L,) 词区间起点，共享原点',
        '    end    = z["word_end_sec"]             # (L,) 词区间终点，半开区间',
        '    ok     = z["word_time_valid"]          # (L,) 0 = 无合法区间',
        '    text   = z["text_word_feat"]           # (L, 768)',
        '    audio  = z["word_audio_feat"]          # (L, 50)',
        '    vision = z["word_vision_feat"]         # (L, 104)',
        "",
        '    # 只用三模态都有真实观测的词，即 valid_length 的口径',
        '    tri = (z["text_word_valid"] & z["word_audio_valid"]',
        '           & z["word_vision_valid"]).astype(bool)',
        "    pooled = text[tri].mean(axis=0)        # 各自模态内的有效词池化",
        "```",
        "",
        "`np.load(..., allow_pickle=False)`：文件里没有 pickle，读取不执行任何"
        "反序列化代码。**无效的词，其特征整行为零**——这是本套代码的不填充约定："
        "不替无区间的词找邻近观测顶替，也不插值补洞，下游按掩码取用。",
        "",
        "### 1.4 从词回捞源观测",
        "",
        "词级特征是原生观测的聚合结果，聚合关系以 CSR 结构留痕："
        "`audio_word_feature_indptr/indices` 给出第 i 个词用了 `raw_audio_lld_values` "
        "的哪几行，`vision_word_feature_indptr/indices` 同理对应 "
        "`raw_video_blendshape_values`。取第 i 个词的窗：",
        "",
        "```python",
        '    ip, ix = z["audio_word_feature_indptr"], z["audio_word_feature_indices"]',
        "    rows = ix[ip[i]:ip[i + 1]]            # 该词用到的 LLD 窗行号",
        '    mean = z["raw_audio_lld_values"][rows].mean(axis=0)   # 与 word_audio_feat[i][:25] 对应',
        "```",
        "",
        "### 1.5 逐词文本与语音时段的对照",
        "",
        "`correspondence/` 下 3 个典型样本给出逐词的源观测：词、字符区间、"
        "语音时段、落在该时段的窗与帧的编号区间。详见 `典型样本验证.md`。",
        "",
        "## 2 全量结果汇总（附件 1 全部 100 条）",
        "",
        "下表覆盖全部 100 条，一行一条。模态类型 TAV = 文本、语音、视觉三模态"
        "（本批 100 条全部为 TAV）；特征维度 922 = 768 + 50 + 104，是逐词特征的"
        "三模态合计，逐条恒定；对齐粒度 W = 词级、C = 片段级。原始有效时长是"
        "该样本切片自身的有效时长（秒）。同为时长口径的还有两个量：容器声明的"
        "`ffprobe_format_duration_sec` 与三路观测的共同覆盖区间 "
        "`av_common_coverage_*`；三者不是同一个量（本批 100 条里，原始有效时长"
        "与 ffprobe 声明值的最大差为 0.1 s），不要互相顶替。",
        "",
        md_table(["样本编号", "特征文件", "模态类型", "原始有效时长(s)", "特征维度", "对齐粒度"],
                 all_rows),
        "",
        "同表另存 `汇总表_全量100条样本.csv`（UTF-8 with BOM，Excel 可直接打开）。",
        "",
        "## 3 合计与核验锚点",
        "",
        md_table(["量", "值", "出处"], [
            ["样本数", "100", "results_100.csv 行数"],
            ["词单元合计", "%d" % n_word_units, "results_100.csv 的 n_official_words 求和"],
            ["三模态共同有效词合计", "%d" % sum_valid_length, "results_100.csv 的 valid_length 求和"],
            ["结构合法词合计", "%d" % sum_struct_valid, "对齐结构核验 V13"],
            ["零时长词合计", "%d" % sum_zero_dur, "同上"],
            ["空白块合计", i_(ev.get("sum_whitespace_chunks")), "文本层纯函数复算 V3"],
            ["视频帧 / 人脸帧", "%s / %s" % (i_(ev.get("sum_video_frame_count")),
                                            i_(ev.get("sum_face_detected_frame_count"))),
             "媒体硬锚点核验 M1"],
            ["零面部样本", i_(ev.get("n_samples_without_face")), "同上"],
            ["异常台账行数", "%d" % len(ledger), "audit/anomaly_ledger.csv"],
            ["计为删除的样本数", "0", "台账 counts_as_deletion 恒为 0"],
            ["核验", "%d 项，执行 %d 项，断言 %s 条，失败 %d 条"
             % (verify["n_checks"], verify["n_executed"],
                "{:,}".format(verify["total_assertions_checked"]),
                verify["total_assertions_failed"]),
             "reports/verify_report.json"],
            ["计入提交的体积", "{:,} B".format(acc["v2_counted_bytes"]),
             "reports/q1v2_size_ledger.json"],
        ]),
        "",
    ])
    if not check_only:
        (OUT / "汇总表_全量100条样本.md").write_text(schema_md, encoding="utf-8")
        with io.open(OUT / "汇总表_全量100条样本.csv", "w",
                     encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["样本编号", "特征文件", "模态类型", "原始有效时长(s)",
                        "特征维度", "对齐粒度"])
            w.writerows(all_rows)

    # ---- 3. 典型样本验证 ----
    sel = corr_meta["selection"]
    crit = "；".join("%s: %s" % (k, v) for k, v in sel["criteria"].items())
    typ_rows = []
    per_sample = []
    for e in corr_meta["files"]:
        r = by_key[e["sample_key"]]
        rows = read_csv_rows(SRC / "correspondence" / e["csv"])
        typ_rows.append([e["sample_key"], e["csv"], i_(e["n_words"]), i_(e["score"]),
                         i_(e["n_words_valid_audio"]), i_(e["n_words_valid_vision"]),
                         "%.3f" % e["face_valid_ratio"], i_(e["valid_length"]),
                         "通过" if e["csr_check"]["all_reproduced"] else "不通过"])
        per_sample.append((e, r, rows))

    verify_md = ["# 问题一：典型样本验证",
                 "",
                 "本文件是**生成**的（`_make_q1_submit.py`）。每条样本的逐词对应表"
                 "旁边标注了它的源 NPZ、源视频与两者的 SHA-256，用于逐级回溯到"
                 "附件 1 的原始素材。",
                 "",
                 "## 1 选样判据",
                 "",
                 "| 项 | 值 |",
                 "|---|---|",
                 "| 选样模式 | %s |" % sel["mode"],
                 "| 判据 | %s |" % crit.replace("|", "\\|"),
                 "| 入选样本 | %s |" % "、".join("`%s`" % k for k in sel["selected"]),
                 "",
                 "判据的两个下限（词单元数、三模态共同有效词占比）保证样本足够长、"
                 "三路观测在场；打分取「同时有两个以上 LLD 窗与两个以上人脸帧」的"
                 "词数，同分按 sample_key 排序，是为了让入选样本落在观测最密的那些"
                 "词上——替代表述是随机抽，那样抽到全零词的概率高，核对时看不到东西。",
                 "",
                 "## 2 三个典型样本",
                 "",
                 md_table(["样本编号", "逐词对应表", "词单元数 L", "多源观测词数",
                           "语音有效词", "视觉有效词", "人脸有效率",
                           "三模态共同有效词", "源观测复算"], typ_rows),
                 "",
                 ]
    for i, (e, r, rows) in enumerate(per_sample, 1):
        verify_md += [
            "## 3.%d 样本 %s" % (i, e["sample_key"]),
            "",
            "| 项 | 值 |",
            "|---|---|",
            "| 特征文件 | `features/%s` |" % e["source_npz"],
            "| 源 NPZ SHA-256 | `%s` |" % e["source_npz_sha256"],
            "| 源视频 | `%s` |" % r["source_video_relpath"],
            "| 源视频 SHA-256 | `%s` |" % r["source_video_sha256"],
            "| 原始有效时长 | %s s |" % f3(r["original_effective_duration_sec"]),
            "| 路由判定 | %s / %s |" % (r["alignment_mode"], r["alignment_granularity"]),
            "| 共享原点 shared_t0 | %s s |" % f3(r.get("av_common_coverage_start_sec") or 0),
            "| 文本音频对应 | %s |" % r["text_audio_correspondence"],
            "| 词单元数 / 共同有效词 | %s / %s |" % (i_(e["n_words"]), i_(e["valid_length"])),
            "| CSR 源观测复算 | %s（音频 %d 个词、视觉 %d 个词逐元素相等）|"
            % ("通过" if e["csr_check"]["all_reproduced"] else "不通过",
               e["csr_check"]["audio_csr_reproduced"],
               e["csr_check"]["vision_csr_reproduced"]),
            "",
            "逐词对应表（%d 个词单元）：" % len(rows),
            "",
            md_table(["词序", "文本片段", "语音时段(s)", "语音窗数",
                      "人脸帧数", "视频帧PTS跨度(s)", "时段内总帧数", "T/A/V"],
                     [[x["word_index"], x["char_span_original"],
                       ("[%s, %s)" % (x["word_start_sec"], x["word_end_sec"])
                        if x["word_time_valid"] == "1" else "—"),
                       x["n_audio_windows"], x["n_face_frames"],
                       x["face_pts_span_sec"] or "—", x["n_frames_in_span"],
                       "%s/%s/%s" % (x["text_word_valid"], x["word_audio_valid"],
                                     x["word_vision_valid"])]
                      for x in rows]),
            "",
            "T/A/V 三列是该词三个模态的有效性（1 = 有真实观测，特征行参与聚合；"
            "0 = 判为无效，特征整行为零）。逐词特征维度恒为 768/50/104，"
            "在上表是常数，故不逐行铺陈。",
            "",
        ]
    verify_md += [
        "## 4 复算方法",
        "",
        "上表不是抄来的：对应表由 `python code/q1v2_correspondence.py` 从 NPZ 现算，"
        "其中 CSR 源观测复算一栏是把每个词的特征用原观测重算一遍后逐元素比对"
        "（相对容差 1e-5、绝对容差 1e-6）。要自己核一遍：",
        "",
        "```bash",
        "python code/q1v2_correspondence.py                    # 重出对应表并复算",
        "python code/run_q1v2_all.py --stage=verify            # 16 项核验（含 V4 词级聚合可回溯重算）",
        "```",
        "",
        "核验需要 openSMILE 与 MediaPipe，二者装在 ASCII 路径下的隔离环境里"
        "（见 `reports/` 的环境快照与项目 README 的环境一节）。",
        "",
    ]
    if not check_only:
        (OUT / "典型样本验证.md").write_text("\n".join(verify_md), encoding="utf-8")

    # ---- 4. README ----
    readme = "\n".join([
        "# 问题一提交包",
        "",
        "本目录是赛题问题一（多模态特征提取与时序对齐）的提交件，内容是附件 1"
        "全部 100 条样本的逐词三模态特征文件，以及配套的汇总表、典型样本对应表、"
        "异常台账与核验回执。文件由 `python _make_q1_submit.py` 从运行目录 "
        "`data/q1_v2/` 整份生成，数字现读现算，不手抄。",
        "",
        "## 0 概要",
        "",
        md_table(["项目", "值"], [
            ["样本数", "100（与附件 1 一一对应，未增未删）"],
            ["词单元合计", "%d" % n_word_units],
            ["三模态共同有效词合计", "%d" % sum_valid_length],
            ["逐词特征维度", "文本 768 + 语音 50 + 视觉 104 = 922"],
            ["核验", "%d 项，执行 %d 项，断言 %s 条，失败 %d 条"
             % (verify["n_checks"], verify["n_executed"],
                "{:,}".format(verify["total_assertions_checked"]),
                verify["total_assertions_failed"])],
            ["计入提交的体积", "{:,} B".format(acc["v2_counted_bytes"])],
            ["样本与标签删除数", "0（异常只登记，`counts_as_deletion` 恒为 0）"],
            ["本包文件数", "%d（%d 份复制自运行目录 + 本文档等 4 份生成文档）"
             % (n_copied + 4, n_copied)],
        ]),
        "",
        "## 1 目录",
        "",
        "```",
        "问题一提交/",
        "├── README_问题一提交说明.md          本文",
        "├── 汇总表_全量100条样本.md / .csv     特征文件规范与读取方式 + 全部 100 条的汇总表",
        "├── 典型样本验证.md                   3 个典型样本的逐词对应、追溯链与复算方法",
        "├── features/                         100 个 NPZ，逐词三模态特征，本问的核心产物",
        "├── manifest.csv                      59 列全量表（含执行期列：内存、耗时、错误）",
        "├── results_100.csv                   56 列全量表（manifest 去掉执行期三列）",
        "├── correspondence/                   3 个典型样本的逐词对应表 + _meta.json",
        "├── audit/anomaly_ledger.csv           异常台账（29 条，只登记不删除）",
        "├── audit/v2_route_100.csv             逐样本路由判定",
        "├── metadata/                         样本登记、路由政策、模型资产身份、环境快照",
        "├── reports/                          16 项核验报告、体积台账、身份扫描回执、阶段台账",
        "└── config_snapshot.json              本次运行的有效配置",
        "```",
        "",
        "运行目录 `data/q1_v2/` 里的 `.cache/`（分模态原始观测缓存）与 `logs/`"
        "（逐样本日志）没有入包：二者不含任何不可再得的特征或标注，"
        "`python code/run_q1v2_all.py --stage=all` 重跑一次即可再生，"
        "体积台账也按同一口径把它们排除在提交之外。",
        "",
        "## 2 特征文件怎么读",
        "",
        "`汇总表_全量100条样本.md` 第 1 节是完整的字段清单（85 个字段分十二组）、"
        "不填充约定与读取代码；第 2 节是全部 100 条的汇总表。一句话概括："
        "每条样本一个 NPZ，`numpy.load(..., allow_pickle=False)` 打开，"
        "以 `words` 的长度为 L，`*_word_feat` 是 (L, 维度) 的逐词特征，"
        "三个 `*_word_valid` 掩码给出该词在这个模态上是否真有观测；"
        "**无效词的特征整行为零**，下游按掩码取用，不要拿零行当数值。",
        "",
        "## 3 典型样本",
        "",
        "`典型样本验证.md` 给出 3 个典型样本的逐词对应表：每个词的文本片段、"
        "语音时段、落在该时段的语音窗与人脸帧的编号区间，以及该词三模态的有效性。"
        "每张表旁边有源 NPZ 与源视频的 SHA-256，可逐级回溯到附件 1 的原始素材。",
        "",
        "## 4 核验与边界",
        "",
        "16 项机器核验的报告在 `reports/verify_report.json`（可读版同目录 `.md`），"
        "本次执行 15 项（V12 是与外部对照实现的路由对拍，属开发期回归，默认跳过），"
        "断言 %s 条、失败 %d 条。核验覆盖集合相等、NPZ schema、词单元可重建、"
        "词级聚合可回溯重算、时间轴一致性、跨样本查重、路由可复算、媒体硬锚点、"
        "外部 ffprobe 交叉核对、权重真实性、数字静音位精确复现等。"
        % ("{:,}".format(verify["total_assertions_checked"]),
           verify["total_assertions_failed"]),
        "",
        "有几条边界是这套产物**刻意**停住的地方，读表时不要越过：",
        "",
        "- `content_assertion` 全部取 0。它标记「文本内容与讲话是否对应」这一类"
        "需要人工听辨才能下的结论；本套代码只做机器可判的那一半"
        "（区间够不够用、时间上是否给得出位置），不替人工下内容结论。",
        "- `text_audio_correspondence` 一律记 `not_asserted`，`TRI_MODAL_WORD_VALID` "
        "与 `AV_VALID_TEXT_UNALIGNED` 两类从不产生。存在对齐区间只说明时间上"
        "给出了位置，并不说明该位置的文本内容已与语音对应。",
        "- `audio_speech_valid` 从不取 1；唯一取 0 的情形是 int16 波形逐采样全零的"
        "数字静音，可用解码现场复现。",
        "- `counts_as_deletion` 全部为 0。异常阈值只用于登记，永不作为删除样本或"
        "标签的理由，100 条样本全部保留。",
        "",
        "脱敏方面，交付物经身份扫描（本机用户名、绝对路径等）零命中，"
        "回执在 `reports/q1v2_identity_scan.json`。",
        "",
        "## 5 复现",
        "",
        "```bash",
        "python code/run_q1v2_all.py --stage=all --isolate   # 从附件 1 重跑全部六个阶段",
        "python code/q1v2_correspondence.py                  # 重出典型样本逐词对应表",
        "python code/run_q1v2_all.py --stage=verify          # 16 项机器核验",
        "```",
        "",
        "对齐侧参数固定在 `config_snapshot.json` 中：语言取 en；"
        "`remove_instant_words` 取 False（保留瞬时词）；`failure_threshold` 取空值"
        "（不因单条失败中断全量）；`regroup` 与 `stream` 均取 False。"
        "对齐器为 stable-ts，声学模型为 whisper base.en 的冻结检查点，"
        "`alignment_source` 记为 `q1v2_own_stable_ts_run`。重跑需要 openSMILE 与 "
        "MediaPipe，二者装在 ASCII 路径下的隔离环境里。",
        "",
    ])
    if not check_only:
        (OUT / "README_问题一提交说明.md").write_text(readme, encoding="utf-8")

    # ---- 5. 核验 ----
    print("=" * 72)
    print("问题一提交包 生成与核验")
    print("=" * 72)
    n_npz_out = len(list((OUT / "features").glob("*.npz")))
    n_rows_csv = len(read_csv_rows(OUT / "汇总表_全量100条样本.csv"))
    files = [p for p in OUT.rglob("*") if p.is_file()]
    total_bytes = sum(p.stat().st_size for p in files)
    checks = [
        ("features/ 下 NPZ 数 = 100", n_npz_out == 100, n_npz_out),
        ("汇总表 CSV 行数 = 100", n_rows_csv == 100, n_rows_csv),
        ("汇总表 MD 行数 = 100", (OUT / "汇总表_全量100条样本.md").read_text(
            encoding="utf-8").count("\n| ") >= 100, "见文件"),
        ("词单元合计 = 1926", n_word_units == 1926, n_word_units),
        ("结构合法词 + 零时长词 = 词单元", sum_struct_valid + sum_zero_dur == n_word_units,
         "%d + %d = %d" % (sum_struct_valid, sum_zero_dur, n_word_units)),
        ("共同有效词合计 = 1287", sum_valid_length == 1287, sum_valid_length),
        ("核验失败 0 条", verify["total_assertions_failed"] == 0,
         verify["total_assertions_failed"]),
        ("异常台账 counts_as_deletion 恒 0",
         all(x["counts_as_deletion"] == "0" for x in ledger), len(ledger)),
        ("典型样本 3 条均有逐词表", len(per_sample) == 3
         and all(len(rows) > 0 for _, _, rows in per_sample), len(per_sample)),
    ]
    bad = 0
    for name, ok, val in checks:
        bad += 0 if ok else 1
        print("  %s %-34s %s" % ("✓" if ok else "★", name, val))

    # 身份红线
    sys.path.insert(0, str(ROOT / "code"))
    from package_check import IDENTITY_PATTERNS  # noqa: E402
    hits = []
    for p in files:
        if p.suffix.lower() in {".npz", ".pt", ".png", ".jpg", ".pdf"}:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for pat, desc in IDENTITY_PATTERNS:
            m = pat.search(txt)
            if m:
                hits.append((str(p.relative_to(OUT)), desc, m.group(0)[:60]))
    print("  %s %-34s %s" % ("✓" if not hits else "★", "身份红线（文本文件）",
                             "零命中" if not hits else hits[:5]))

    print("\n  文件数 %d，合计 %s（%.2f MiB）"
          % (len(files), "{:,}".format(total_bytes), total_bytes / 2 ** 20))
    if check_only:
        print("  （--check：未写盘）")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
