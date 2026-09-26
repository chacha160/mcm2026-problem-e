# -*- coding: utf-8 -*-
"""_gen_paper_tables_q1.py —— 问题一 v2 论文表格生成器（单问稿专用）

本脚本是**唯一**的论文表格生成器：仓库已按问题一重建，读 `data/q1_delivery/` 的
旧 50 槽表生成器（原先与它分工的那一份）已随三问稿一并删除。本脚本只出问题一 v2 的
表，写 `_tables_q1/`，供 `问题一论文.md` 使用。

设计原则与旧脚本一致：**论文里出现的每个数字都必须来自 `data/q1_v2/` 的实际产物**，
正文一律用 `{TABLE:名字}` 占位符，不手写表体。每张表的来源写进 `_tables_q1/_meta.json`，
便于逐项回溯。

用法：python _gen_paper_tables_q1.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
from collections import Counter
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

# 契约常量是**冻结值**，且被 q1v2_verify 的 V2 逐样本断言过（NPZ 第二维必须等于它）。
# 从常量取而不自己写数，是为了让「论文里的维度」与「核验时的维度」同源。
from q1v2_contract import (  # noqa: E402
    FACE_MODEL_DIM,
    OPENMILE_LLD_DIM,
    ROBERTA_HIDDEN_DIM,
    SCHEMA_VERSION,
)

DATA = ROOT / "data" / "q1_v2"
OUT = ROOT / "_tables_q1"
OUT.mkdir(parents=True, exist_ok=True)
SOURCES: dict[str, str] = {}


def _esc(value) -> str:
    """单元格里的 `|` 会把 Markdown 表切断，统一转义。"""
    return str(value).replace("|", "\\|")


def save(name: str, header, rows, source: str) -> None:
    lines = ["| " + " | ".join(_esc(h) for h in header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_esc(c) for c in r) + " |")
    (OUT / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    SOURCES[name] = source
    print(f"  {name + '.md':28s} {len(rows):>4d} 行   ← {source}")


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _f3(x) -> str:
    try:
        v = float(x)
    except Exception:
        return "—"
    return "—" if v != v else f"{v:.3f}"


def _int(x) -> str:
    try:
        return f"{int(round(float(x)))}"
    except Exception:
        return "—"


# ─────────────────────────────────────────────────────────── 载入产物
results = read_csv(DATA / "results_100.csv")
manifest = read_csv(DATA / "manifest.csv")
verify = read_json(DATA / "reports" / "verify_report.json")
ledger = read_csv(DATA / "audit" / "anomaly_ledger.csv")
ledger_json = read_json(DATA / "reports" / "q1v2_size_ledger.json")
policy = read_json(DATA / "metadata" / "route_policy.json")
env = read_json(DATA / "metadata" / "environment.json")
assets = read_json(DATA / "metadata" / "model_assets.json")

import numpy as np  # noqa: E402

# 字段数取一条实测 NPZ 的真实 files 列表，不写常数——schema 若变，表要跟着变。
with np.load(sorted((DATA / "features").glob("*.npz"))[0], allow_pickle=False) as _h:
    NPZ_FIELDS = list(_h.files)

W_DIM = ROBERTA_HIDDEN_DIM            # 768
A_DIM = OPENMILE_LLD_DIM * 2          # 25 → 50
V_DIM = FACE_MODEL_DIM * 2            # 52 → 104
CORR_DIR = DATA / "correspondence"
corr_meta = read_json(CORR_DIR / "_meta.json") if (CORR_DIR / "_meta.json").is_file() else None

print("问题一 v2：")

# ─────────────────────────────────────────────── 1. 特征文件规范
save("q1v2_spec",
     ["模态", "词级特征形状", "词级维度", "原生观测", "原生维度", "提取工具与口径"],
     [["文本 text", f"(L, {W_DIM})", W_DIM, "RoBERTa 子词", W_DIM,
       "roberta-base 末层隐状态按词池化（子词归属见 roberta_word_token_*）"],
      ["语音 audio", f"(L, {A_DIM})", A_DIM, "openSMILE LLD 窗", OPENMILE_LLD_DIM,
       f"eGeMAPSv02 LowLevelDescriptors，逐窗 {OPENMILE_LLD_DIM} 维；词级取 [均值, 总体标准差(ddof=0)]"],
      ["视觉 vision", f"(L, {V_DIM})", V_DIM, "MediaPipe blendshape 帧", FACE_MODEL_DIM,
       f"FaceLandmarker 逐帧 {FACE_MODEL_DIM} 维 blendshape；词级取 [均值, 总体标准差(ddof=0)]"]],
     "code/q1v2_contract.py 常量 ROBERTA_HIDDEN_DIM / OPENMILE_LLD_DIM / FACE_MODEL_DIM；"
     "词级维度由 q1v2_verify V2 逐样本断言")

save("q1v2_schema",
     ["项", "值", "说明"],
     [["schema_version", SCHEMA_VERSION, "NPZ 内嵌，防与旧版产物混淆"],
      ["NPZ 字段数", len(NPZ_FIELDS), "含标量、掩码、特征、原生序列、CSR 溯源五类"],
      ["NPZ 读取方式", "numpy.load(..., allow_pickle=False)", "无 pickle，读取不执行任何反序列化代码"],
      ["manifest.csv 列数", len(manifest[0]), "含执行期三列（内存/耗时/错误）"],
      ["results_100.csv 列数", len(results[0]), "manifest 去掉执行期三列"],
      ["表行数", f"{len(manifest)} / {len(results)}", "两份表各恰 100 行，键集合相等（V1 断言）"],
      ["三模态共同有效词数 valid_length", "逐样本一列", "text_word_valid ∧ word_audio_valid ∧ word_vision_valid"]],
     "data/q1_v2/features/*.npz、manifest.csv、results_100.csv")

# ─────────────────────────────────────────────── 2. 全量 100 条汇总表（赛题要求）
_sum = lambda key: [float(r[key]) for r in results]  # noqa: E731
# 列只留赛题要求的五项（样本编号/模态类型/原始有效时长/特征维度/对齐粒度）。
# 词单元数、窗数、帧数、人脸有效率、路由判定同样是真数字，但其逐样本可查性由
# results_100.csv（56 列）保证，不必在论文里再铺 5 列——100 行 × 11 列吃掉篇幅的
# 份额大于它带来的信息量。
#
# 模态类型与特征维度**逐样本恒定**（已核：100 条全为三模态；逐词维度恒为 768/50/104）。
# 恒定的列若逐行写全 "text/audio/vision" 与 "768/50/104"，等于用 600 个字重复同一句
# 话；故按表注约定记成 TAV 与逐词合计 922。对齐粒度只有 W/C 两个取值，同样按表注
# 约定记成字母。三个记号的展开说明都写在论文表 A1 的表注里——这不是省掉赛题要求的
# 字段，字段仍在每一行上，只是取紧凑记法。
_MODAL_CODE = {"text|audio|vision": "TAV"}
_GRAN_CODE = {"word": "W", "clip": "C"}
all_rows = []
for r in results:
    all_rows.append([
        r["sample_key"], _MODAL_CODE.get(r["modalities"], r["modalities"].replace("|", "/")),
        _f3(r["original_effective_duration_sec"]),
        str(W_DIM + A_DIM + V_DIM),
        _GRAN_CODE.get(r["alignment_granularity"], r["alignment_granularity"]),
    ])
save("q1v2_all100",
     ["样本编号", "模态类型", "原始有效时长(s)", "特征维度", "对齐粒度"],
     all_rows,
     "data/q1_v2/results_100.csv 逐行；维度取 code/q1v2_contract.py 冻结常量")

# ─────────────────────────────────────────────── 3. 路由
mode_counts = Counter(r["alignment_mode"] for r in results)
gran_counts = Counter(r["alignment_granularity"] for r in results)
map_counts = Counter(r["text_av_time_mapping_status"] for r in results)
corr_counts = Counter(r["text_audio_correspondence"] for r in results)
basis_counts = Counter(r["word_time_basis"] for r in results)
speech_counts = Counter(r["audio_speech_valid"] for r in results)
_content_positive = sum(1 for r in results if int(r["content_assertion"]) != 0)

route_rows = []
for name in policy["vocabulary"]:
    n = mode_counts.get(name, 0)
    if name in policy.get("never_emitted", {}):
        basis = "否——含人工听辨结论"
        # route_policy.json 里这两类的完整陈述各约 90 字。论文 §5.1.5 正文已经把
        # 「只做机器可判的那一半、不替人工下内容结论」讲透，表里再铺一遍同一段话
        # 属于重复计篇幅；这里只留判据的**性质**，完整陈述的出处写在论文表 3 表注。
        note = {"TRI_MODAL_WORD_VALID":
                "要求「已确认文本内容与讲话对应」，属人工听辨结论；机器只能判区间合法",
                "AV_VALID_TEXT_UNALIGNED":
                "要求「已确认文字与讲话明显不对应」，属人工听辨结论；ASR 距离与对齐概率反推不出"
                }.get(name, "")
    else:
        basis = "是"
        note = {"EXTRACTION_OR_TIMELINE_ANOMALY": "身份/解码/共享时间轴不可靠",
                "TRI_MODAL_WITH_AUDIO_CONTENT_INVALID": "时间轴可信且 int16 波形逐采样全零",
                "UNCERTAIN_REVIEW": "其余情况（兜底）"}.get(name, "")
    route_rows.append([name, basis, n, note])
save("q1v2_route",
     ["判定类别", "机器可判", "本批条数", "判据"],
     route_rows,
     "data/q1_v2/metadata/route_policy.json + results_100.csv 的 alignment_mode 计数")

save("q1v2_route_fields",
     ["字段", "取值分布（本批 100 条）", "语义"],
     [["alignment_granularity", "、".join(f"{k} {v}" for k, v in gran_counts.most_common()),
       "该样本的时序组织粒度（词级 / 片段级）"],
      ["text_av_time_mapping_status", "、".join(f"{k} {v}" for k, v in map_counts.most_common()),
       "结构闸门（机器可判）：官方词的时间区间够不够用"],
      ["text_audio_correspondence", "、".join(f"{k} {v}" for k, v in corr_counts.most_common()),
       "内容确认：文本与实际讲话是否对应；本套代码一律 not_asserted，"
       "唯一例外是位精确可证的数字静音"],
      ["word_time_basis", "、".join(f"{k} {v}" for k, v in basis_counts.most_common()),
       "词区间来源；own_forced_alignment = 至少一个词的区间来自本次冻结对齐运行"],
      ["content_assertion", f"取 1 的条数 {_content_positive}", "uint8；1 仅当存在外部人工证据，本套恒为 0"],
      ["audio_speech_valid", "、".join(f"{k} {v}" for k, v in sorted(speech_counts.items())),
       "有语音的证据等级：0 = 已确认数字静音，−1 = 证据不足。**从不取 1**"]],
     "data/q1_v2/results_100.csv + metadata/route_policy.json 的 separate_fields")

# ─────────────────────────────────────────────── 4. 核验
save("q1v2_verify",
     ["编号", "检验内容", "断言数", "失败"],
     [(c["code"], c["name"], _int(c["checked"]), _int(c["failed"])) for c in verify["checks"]],
     "data/q1_v2/reports/verify_report.json 的 checks 数组")

# ─────────────────────────────────────────────── 5. 异常台账
# 只留「判定依据 + 结论」两端的列：中间的四个计量（结构合法词数、视频帧数、
# 检出人脸帧数、词并集覆盖率）与 results_100.csv 同源，论文里重复铺一遍只增行宽。
ledger_rows = [[r["sample_key"], r["alignment_mode"], _int(r["n_official_words"]),
                _f3(r["face_valid_ratio"]), r["notes"], r["counts_as_deletion"]]
               for r in ledger]
save("q1v2_ledger",
     ["样本编号", "路由判定", "官方词数", "人脸有效率", "异常说明", "计为删除"],
     ledger_rows,
     "data/q1_v2/audit/anomaly_ledger.csv（全量 %d 条，counts_as_deletion 恒 0）" % len(ledger))

# ─────────────────────────────────────────────── 6. 体积
acc = ledger_json["accounting"]
LOOSE, STRICT = 50 * 1024 * 1024, 50 * 10 ** 6
save("q1v2_budget",
     ["口径", "字节", "说明"],
     [["问题一 v2 计入提交", f"{acc['v2_counted_bytes']:,}",
       "features/ + 两张表 + audit/ + metadata/ + reports/；.cache/ 与 logs/ 为可重建中间态，不计入"],
      # 白名单已换向到 q1_v2，故下面这行**不再**含旧的 data/q1_delivery。
      # 行文随台账的 `v2_in_whitelist` 走，白名单哪天再换向，这张表也不会说谎。
      ["提交白名单合计（含代码）", f"{acc['whitelist_current_bytes']:,}",
       ("白名单当前指向 data/q1_v2；本行即替换后的合计，无需再加一遍 v2"
        if acc.get("v2_in_whitelist") else
        "白名单当前仍指向 data/q1_delivery，故需扣旧目录再加 v2")],
      ["其中 data/q1_delivery（不计提交）", f"{acc.get('old_q1_bytes', 0):,}",
       "v1（旧 50 槽架构）留在盘上但不计提交；仅在「叠加」一栏用到"],
      ["替换核算合计", f"{acc['replace_total_bytes']:,}",
       f"宽松读法 {'过线' if acc['replace_within_loose'] else '超线'}；"
       f"严格读法 {'过线' if acc['replace_within_strict'] else '超线'}"],
      ["叠加核算合计（反面）", f"{acc['stack_total_bytes']:,}",
       f"两份 Q1 都留着：宽松读法 {'过线' if acc['stack_within_loose'] else '超线'}；"
       f"严格读法 {'过线' if acc['stack_within_strict'] else '超线'}"],
      ["红线（宽松 50×1024²）", f"{LOOSE:,}", "—"],
      ["红线（严格 50×10⁶）", f"{STRICT:,}", "—"]],
     "data/q1_v2/reports/q1v2_size_ledger.json 的 accounting")

# ─────────────────────────────────────────────── 7. 典型样本
if corr_meta:
    typical_rows = []
    for entry in corr_meta["files"]:
        typical_rows.append([
            entry["sample_key"], _int(entry["n_words"]), entry["score"],
            _int(entry["n_words_valid_audio"]), _int(entry["n_words_valid_vision"]),
            f"{entry['face_valid_ratio']:.3f}", _int(entry["valid_length"]),
            "通过（CSR 逐元素复算相等）" if entry["csr_check"]["all_reproduced"] else "不通过",
            entry["csv"],
        ])
    save("q1v2_typical",
         ["样本编号", "词单元数 L", "多源观测词数", "语音有效词数", "视觉有效词数",
          "人脸有效率", "三模态共同有效词数", "源观测复算核验", "对应表文件"],
         typical_rows,
         "data/q1_v2/correspondence/_meta.json（选样判据见该文件 selection 字段）")

    # 逐词对应表：论文正文用，取第一个典型样本
    first = corr_meta["files"][0]
    corr_rows = read_csv(CORR_DIR / first["csv"])
    # 末列原写 "文本 768 / 语音 50 / 视觉 104"，30 行逐行重复同一串常数；改为该词
    # 三模态的**有效性三元组** T/A/V（1=该模态在此词上有真实观测、特征行参与聚合，
    # 0=判为无效、特征整行为零）。维度是常量，写在表注里一次即可；三元组才是这一列
    # 真正携带的信息，且比常量更短——无效词 `I` 的 1/0/0 一眼可见，而原写法看不出来。
    save("q1v2_corr",
         ["词序", "文本片段", "语音时段(s)", "语音窗数",
          "人脸帧数", "视频帧PTS跨度(s)", "时段内总帧数", "三模态特征行 T/A/V", "备注"],
         [[r["word_index"], r["char_span_original"],
           (f"[{r['word_start_sec']}, {r['word_end_sec']})" if r["word_start_sec"] else "—"),
           r["n_audio_windows"],
           r["n_face_frames"], r["face_pts_span_sec"] or "—", r["n_frames_in_span"],
           f"{r['text_word_valid']}/{r['word_audio_valid']}/{r['word_vision_valid']}",
           r["note"] or "—"]
          for r in corr_rows],
         f"data/q1_v2/correspondence/{first['csv']}（样本 {first['sample_key']} 全 {len(corr_rows)} 词）")

# ─────────────────────────────────────────────── 8. 环境与资产
# environment.json 是**整机快照**（12 个包）。若原样进论文，读者会以为 librosa /
# cv2 属于问题一的工具链——它们只服务问题二/三的未对齐链路。故按源码静态扫描
# 标出「本问链路是否直接 import」，并给未使用的包标出它实际被谁 import。
import ast  # noqa: E402

_DIST_TO_MODULE = {"stable-ts": "stable_whisper", "openai-whisper": "whisper"}


def _imported_modules(paths):
    mods = set()
    for p in paths:
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods.add(node.module.split(".")[0])
    return mods


LOCAL_MODULES = {p.stem for p in CODE.glob("*.py")}
CHAIN_FILES = sorted(CODE.glob("q1v2_*.py")) + [CODE / "run_q1v2_all.py"]

#: 静态扫描只能看出「谁 import 了谁」。这四个包不在本问链路的 import 名单里，
#: 但**不等于用不上**——逐项写明角色，免得读者误以为环境里多装了东西：
#:   · openai-whisper：`stable_whisper.load_model()` 加载的就是 whisper 检查点
#:     （`q1v2_align.py` 传本地路径 + SHA 校验，正是为了不触发它的联网下载）；
#:   · openpyxl：本问核验要拿 official_text 与 label-100.xlsx 逐字比对，
#:     而读 xlsx 的 `pandas.read_excel` 走的就是 openpyxl 引擎；
#:   · torchaudio：随 torch 音频栈一并装入，本问链路未直接调用；
#:   · librosa：其他链路使用，扫描结果会给出具体模块。
_TRANSITIVE_DEPS = {
    "openai-whisper": "运行期传递依赖：stable_whisper.load_model() 加载的即 whisper 检查点",
    "openpyxl": "运行期传递依赖：pandas.read_excel 读 label-100.xlsx 的引擎",
    "torchaudio": "本问链路未直接调用（随 torch 音频栈装入）",
}

env_rows = []
for name, version in sorted(env.get("packages", {}).items()):
    module = _DIST_TO_MODULE.get(name, name.replace("-", "_"))
    chain_hit = module in _imported_modules(CHAIN_FILES) and module not in LOCAL_MODULES
    if chain_hit:
        env_rows.append([name, str(version), "是", "—"])
        continue
    users = sorted(p.stem for p in CODE.glob("*.py")
                   if p not in CHAIN_FILES and p.stem not in LOCAL_MODULES
                   and module in _imported_modules([p]))
    if users:
        shown = "、".join(users[:3]) + (f" 等 {len(users)} 个模块" if len(users) > 3 else "")
        role = f"其他链路使用：{shown}"
    else:
        role = _TRANSITIVE_DEPS.get(name, "本问链路不调用；该包由整机环境快照记录")
    env_rows.append([name, str(version), "否", role])
save("q1v2_env",
     ["包", "本机实测版本", "问题一链路直接 import", "未被直接 import 时的角色"],
     env_rows,
     "data/q1_v2/metadata/environment.json 的 packages + 对 code/q1v2_*.py 与 "
     "code/run_q1v2_all.py 的 AST 静态扫描（「否」项的角色见 _gen_paper_tables_q1.py "
     "的 _TRANSITIVE_DEPS 与扫描结果）")
save("q1v2_assets",
     ["资产", "文件", "SHA-256（前 16 位）", "说明"],
     [["RoBERTa 文本编码", "model.safetensors",
       assets["roberta"]["files_sha256"]["model.safetensors"][:16] + "…",
       f'{assets["roberta"]["n_files"]} 个文件逐个校验，任一不符即报错'],
      ["whisper 强制对齐", "base.en.pt", assets["whisper"]["sha256"][:16] + "…",
       f'{assets["whisper"]["model_name"]}，与 stable-ts 配套'],
      ["MediaPipe 面部", "face_landmarker.task", assets["mediapipe"]["sha256"][:16] + "…",
       f'预取一次后离线加载（本次 downloaded_this_run={assets["mediapipe"]["downloaded_this_run"]}）'],
      ["openSMILE 配置", assets["opensmile"]["config_file"], assets["opensmile"]["config_sha256"][:16] + "…",
       f'eGeMAPSv02 LowLevelDescriptors，{assets["opensmile"]["feature_dim"]} 维，顺序已冻结']],
     "data/q1_v2/metadata/model_assets.json")

(OUT / "_meta.json").write_text(json.dumps({
    "generated_from": "data/q1_v2",
    "tables": SOURCES,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n{len(SOURCES)} 张表写入 {OUT}；来源见 _meta.json")
