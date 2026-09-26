# -*- coding: utf-8 -*-
"""
q1v2_contract.py —— 问题一 v2（对照实现架构）的冻结常量与通用 IO

本模块**刻意不导入 torch / mediapipe / opensmile / stable_whisper**。
理由：核验器（q1v2_verify.py）与体积台账（q1v2_package.py）要能在不装这些
重依赖的解释器里裸跑。凡是需要模型的判据，一律在函数内部延迟导入。

架构对照见 `Q1建模流程与逻辑框架总结.md`（对照实现）与本机的一份**外部**
参考资料目录（对照实现的权威实现，位于项目之外，故此处不写其绝对路径——
交付物里不留本机路径，这条约束对源码同样适用）。
本套代码与现有方案 B（等分 50 槽）**并存**，产物写到 data/q1_v2/，
绝不触碰 data/q1_delivery/ 等既有交付目录。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import config  # noqa: E402

# --------------------------------------------------------------------------
# 路径
# --------------------------------------------------------------------------

PROJECT_ROOT = Path(config.PROJECT_ROOT).resolve()

#: 本次全部产物落在这里。**只有这一个新目录**。
OUTPUT_ROOT = PROJECT_ROOT / "data" / "q1_v2"
#: 中间缓存与临时件（不计入提交体积）
CACHE_ROOT = OUTPUT_ROOT / ".cache"


def _resolve_work_root() -> "tuple[Path, str]":
    """解释器与原生资产的工作区。返回 `(路径, 选择理由)`。

    **为什么不能直接用 `<项目根>/work/`**：openSMILE 的 Python 绑定在
    `opensmile/core/lib.py::initialize` 里对配置路径做了
    `bytes(config_file, "ascii")`，而 `smile.py` 还会把包目录下的外部 I/O
    配置（`cExternalSource` / `cExternalSink`）一并送进原生层。于是**只要
    `opensmile` 包本身所在路径含非 ASCII 字符，构造 `Smile()` 就会抛
    `UnicodeEncodeError`**——实测本仓库根目录 `D:\\23届建模` 含中文，直接踩中。
    对照实现没踩到，是因为它的 opensmile 装在 ASCII 的 Python 目录下，
    中文只出现在数据路径上（音频以内存数组传入，不进原生层）。

    MediaPipe 的 `face_landmarker.task` 同样交给原生层，同一风险，一并挪出。

    解析顺序：
      1. 环境变量 `Q1V2_WORK_ROOT`（跨机器复现时显式指定）
      2. `<项目根>/work/q1v2`，**仅当该路径为纯 ASCII**（多数机器走这条）
      3. ASCII 兜底目录（Windows 用 `<盘符>:\\q1v2work`，其余用 `~/q1v2work`）

    工作区不随仓库提交，也不占提交体积；实际选中的路径记入
    `environment.json`，选择理由记入同文件的 `work_root_reason`。
    """
    override = os.environ.get("Q1V2_WORK_ROOT")
    if override:
        return Path(override), "env:Q1V2_WORK_ROOT"
    in_project = PROJECT_ROOT / "work" / "q1v2"
    if str(in_project).isascii():
        return in_project, "in-project (path is ASCII)"
    if os.name == "nt":
        drive = Path(PROJECT_ROOT).drive or "C:"
        return Path(drive + os.sep) / "q1v2work", "ASCII fallback outside project (project path is non-ASCII)"
    return Path.home() / "q1v2work", "ASCII fallback outside project (project path is non-ASCII)"


#: 原生资产与隔离 venv 的工作区（不随仓库提交）
WORK_ROOT, WORK_ROOT_REASON = _resolve_work_root()
#: 隔离 venv：mediapipe 会拉 opencv-contrib-python 覆盖已装 cv2，
#: 而 Q2/Q3 与现有抽帧代码都依赖 cv2。故新依赖只装在 venv 里。
VENV_PYTHON = WORK_ROOT / ".venv" / "Scripts" / "python.exe"
#: MediaPipe 权重的落点（同属原生资产，同样要求 ASCII 路径）
FACE_MODEL_PATH = WORK_ROOT / "model_assets" / "mediapipe" / "face_landmarker.task"

#: 受保护的既有产物。prepare 阶段对它们做全文件 SHA 快照，
#: 全流程结束时再算一次并断言未变——证明本次重建没有破坏已交付成果。
PROTECTED_DIRS = (
    os.path.join("data", "q1_delivery"),
    os.path.join("data", "word_align"),
    os.path.join("data", "unaligned_features"),
    os.path.join("data", "aligned"),
)

#: 受保护目录里**由各自流水线的核验器自己重写**的文件，逐条附原因。
#:
#: 为什么需要这个名单：`data/q1_delivery/verify_report.{json,md}` 是 v1 的
#: `q1_verify.py` 每次运行都会重新生成的东西——它的内容取决于「谁在什么时候
#: 跑过哪一版核验」。把这类文件当硬失败，会让**任何人重跑一次 v1 核验**都报红，
#: 而一个动不动就报红的检查会训练人忽略它，M2 就废了。
#:
#: 判据因此收紧为：差异**只**出现在这些文件里 → 记录在案并归因，不判失败；
#: 出现在**任何其他**文件里 → 一律硬失败。名单是白名单，不是通配符。
#: 键必须写成**正斜杠**形式：快照用的是 `Path.as_posix()`，
#: 用 `os.path.join` 生成的反斜杠键在 Windows 上永远匹配不上（会静默失效，
#: 表现为「名单声明了却一条都没分流」）。
VOLATILE_BY_DESIGN = {
    "data/q1_delivery/verify_report.json":
        "v1 核验器 code/q1_verify.py 每次运行都会重写",
    "data/q1_delivery/verify_report.md":
        "同上（同一份报告的 Markdown 版）",
}

# --------------------------------------------------------------------------
# 版本与资产身份（全部与对照实现的冻结值逐字符相同，见计划「已核实硬事实」）
# --------------------------------------------------------------------------

#: 与对照实现的 SCHEMA_VERSION="q1-feature-v1.0" 不同名，防止两种产物被混淆读取。
SCHEMA_VERSION = "q1v2-feature-v1.0"

ROBERTA_MODEL_ID = "FacebookAI/roberta-base"
ROBERTA_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
ROBERTA_HIDDEN_DIM = 768
#: 逐个文件核对，不只看 model.safetensors——tokenizer 换了同样会改变词向量。
ROBERTA_FILE_SHA256 = {
    "config.json": "ef0185e2aae6e06c5f105a285006952c340e20c7dbf43c86ec82601b13fc45e9",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "model.safetensors": "5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81",
    "tokenizer.json": "847bbeab6174d66a88898f729d52fa8d355fafe1bea101cf960dd404581df70e",
    "tokenizer_config.json": "994f46754c5bf4014f1aa92d34b1374319c3a6b3f702105cd5b742beaecd18ce",
    "vocab.json": "9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655",
}

#: MediaPipe FaceLandmarker 权重。**不在 pip 包里**，需一次性联网预取。
#: 该 SHA 由本机对官方 URL 做流式 SHA-256 实测得到，与对照实现的冻结值相同。
FACE_MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
FACE_MODEL_BYTES = 3_758_596
FACE_MODEL_DIM = 52

#: openSMILE eGeMAPSv02 LLD 配置文件的 SHA-256（PyPI 版 opensmile==2.6.0）。
OPENMILE_CONFIG_SHA256 = "ef451953badced2ed112ba4ffb997f9d2a7e7c444a09e8092665eadd9f24109e"
OPENMILE_LLD_DIM = 25

#: 25 维 LLD 的**名称与顺序**。顺序即身份——它决定 (L,25) 每一列是什么。
#: 取自对照实现的 `metadata/model_assets.json::opensmile_feature_names`，
#: 核验 V7(c) 要求本机 openSMILE 返回的名字与顺序逐项相同。
OPENMILE_FEATURE_NAMES = (
    "Loudness_sma3",
    "alphaRatio_sma3",
    "hammarbergIndex_sma3",
    "slope0-500_sma3",
    "slope500-1500_sma3",
    "spectralFlux_sma3",
    "mfcc1_sma3",
    "mfcc2_sma3",
    "mfcc3_sma3",
    "mfcc4_sma3",
    "F0semitoneFrom27.5Hz_sma3nz",
    "jitterLocal_sma3nz",
    "shimmerLocaldB_sma3nz",
    "HNRdBACF_sma3nz",
    "logRelF0-H1-H2_sma3nz",
    "logRelF0-H1-A3_sma3nz",
    "F1frequency_sma3nz",
    "F1bandwidth_sma3nz",
    "F1amplitudeLogRelF0_sma3nz",
    "F2frequency_sma3nz",
    "F2bandwidth_sma3nz",
    "F2amplitudeLogRelF0_sma3nz",
    "F3frequency_sma3nz",
    "F3bandwidth_sma3nz",
    "F3amplitudeLogRelF0_sma3nz",
)

#: Whisper 检查点（stable-ts 强制对齐用）。本机已存在，零下载。
WHISPER_CHECKPOINT_SHA256 = "25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead"
WHISPER_MODEL_NAME = "base.en"

#: 官方文本表的 SHA-256（与 对照实现记录一致，用于身份核对）
LABEL_XLSX_SHA256 = "827334f782b1f242c84ad944a657bc7f7c63643f71ae9c977811a3eecdd3ab33"

#: RoBERTa 本地快照目录候选，由 q1v2_assets 解析并逐文件校验 SHA。
#: 第一项是本仓库自带的工作区，其余是 HuggingFace 缓存的两种命名
#: （按短名下载是 `models--roberta-base`，带命名空间是 `models--FacebookAI--roberta-base`）。
ROBERTA_DIR_CANDIDATES = (
    PROJECT_ROOT / "work" / "roberta-base" / ROBERTA_REVISION,
    Path.home() / ".cache" / "huggingface" / "hub" / "models--roberta-base"
    / "snapshots" / ROBERTA_REVISION,
    Path.home() / ".cache" / "huggingface" / "hub" / "models--FacebookAI--roberta-base"
    / "snapshots" / ROBERTA_REVISION,
)
#: Whisper 检查点候选（stable-ts 强制对齐用），本机已存在，零下载
WHISPER_CHECKPOINT_CANDIDATES = (
    Path.home() / ".cache" / "whisper" / f"{WHISPER_MODEL_NAME}.pt",
    PROJECT_ROOT / "work" / "whisper" / f"{WHISPER_MODEL_NAME}.pt",
)

#: stable-ts 调用参数。对照实现在 `执行报告.md` 里显式写死这四个参数。
#: 注意 `regroup` 的库默认是 True，对照实现显式传 False——这是真实差异，必须照抄。
ALIGN_PARAMS: Dict[str, Any] = {
    "language": "en",
    "remove_instant_words": False,
    "failure_threshold": None,
    "regroup": False,
    "stream": False,
}
ALIGN_SOURCE = "q1v2_own_stable_ts_run"
ALIGNER_NAME = "stable-ts"

#: 词区间的半开约定。改这一条会同时影响归词判据与核验器。
INTERVAL_CONVENTION = "half-open [start,end)"
AGGREGATION_QUERY = "feature center >= word start and < word end"

#: 次采样容差（照抄对照实现）
TOL = 1e-6
#: 词级聚合重算的相对/绝对容差（核验 V4）
AGG_RTOL = 1e-5
AGG_ATOL = 1e-6

# --------------------------------------------------------------------------
# 文本词单元规则
# --------------------------------------------------------------------------

#: 纯标点块并入相邻词后的**全量**期望值，用于核验 V13 的硬断言。
#: 1932 = 100 条官方文本按 `\\S+` 切出的空白块总数；
#: 1926 = 其中含字母/数字的块数（6 个纯标点块并入相邻词，不新增行）。
EXPECTED_TOTAL_WHITESPACE_CHUNKS = 1932
EXPECTED_TOTAL_WORD_UNITS = 1926

#: RoBERTa 位置编码上限。对照实现的做法是**截断为 False**并在超限时报错，
#: 而不是静默截断——静默截断会让尾部词拿不到特征却仍被当成有效。
ROBERTA_MAX_POSITIONS_FALLBACK = 512

# --------------------------------------------------------------------------
# 五类路由词汇表（与 对照实现同名同义，便于对照）
# --------------------------------------------------------------------------

MODE_TRI_MODAL_WORD_VALID = "TRI_MODAL_WORD_VALID"
MODE_TRI_MODAL_AUDIO_CONTENT_INVALID = "TRI_MODAL_WITH_AUDIO_CONTENT_INVALID"
MODE_AV_VALID_TEXT_UNALIGNED = "AV_VALID_TEXT_UNALIGNED"
MODE_UNCERTAIN_REVIEW = "UNCERTAIN_REVIEW"
MODE_EXTRACTION_OR_TIMELINE_ANOMALY = "EXTRACTION_OR_TIMELINE_ANOMALY"

MODE_VOCABULARY = (
    MODE_TRI_MODAL_WORD_VALID,
    MODE_TRI_MODAL_AUDIO_CONTENT_INVALID,
    MODE_AV_VALID_TEXT_UNALIGNED,
    MODE_UNCERTAIN_REVIEW,
    MODE_EXTRACTION_OR_TIMELINE_ANOMALY,
)

#: 本套代码**不会发出**的两类，及原因。写进 metadata/route_policy.json，
#: 免得读者把「证据等级不足」误读成「漏跑」。
MODES_NEVER_EMITTED: Dict[str, str] = {
    MODE_TRI_MODAL_WORD_VALID: (
        "该类要求「文本内容与讲话已确认对应」（人工听辨结论）"
        "且「冻结对齐器下全部官方词区间合法」（机器可判）；"
        "本套代码只做机器可判的那一半，不因区间合法就替人工下内容结论，故不发出此类。"
        "结构合法的样本由 text_av_time_mapping_status=word_valid/word_partial 与 "
        "word_time_basis=own_forced_alignment 标注。"
    ),
    MODE_AV_VALID_TEXT_UNALIGNED: (
        "该类要求「已确认官方文字与实际讲话明显不对应」，属人工听辨结论；"
        "ASR 距离、对齐概率、置信度都只是间接量，任何阈值都反推不出该结论"
        "（机器分歧度排序与人工听辨判定不是同一件事），故本套代码不发出此类。"
    ),
}

CORRESPONDENCE_VOCABULARY = (
    "confirmed_match",
    "confirmed_mismatch",
    "no_speech",
    "not_asserted",
)
TIME_STATUS_VOCABULARY = ("word_valid", "word_partial", "clip_only", "unavailable")

#: 词时间来源。`own_forced_alignment` 表示「本样本至少有一个词的区间来自我们这次
#: 自己跑的冻结 stable-ts」。它**只声明区间的来源**，不声明文本内容与讲话对应。
WORD_TIME_BASIS_OWN = "own_forced_alignment"
WORD_TIME_BASIS_NONE = "none"
WORD_TIME_BASIS_VOCABULARY = (WORD_TIME_BASIS_OWN, WORD_TIME_BASIS_NONE)

#: 内容确认状态。本套代码**恒为 0**——我们没有人工听辨证据。
#: 设这个字段是为了让下游一眼看出「对齐区间存在」不等于「内容已确认」。
CONTENT_ASSERTION_NONE = 0
CONTENT_ASSERTION_EXTERNAL = 1

# --------------------------------------------------------------------------
# 体积红线
# --------------------------------------------------------------------------

#: 「50 MB」两种读法的松紧关系容易记反：52.43 MB > 50.00 MB，
#: 所以 1024² 那一读法**更宽松**。执行宽松口径，同时报严格口径余量。
LIMIT_BYTES = 50 * 1024 * 1024
LIMIT_STRICT_BYTES = 50 * 10 ** 6

#: 计入提交体积的子目录（其余如 .cache/ logs/ 明确排除）
DELIVERABLE_SUBDIRS = ("features", "audit", "reports", "metadata")
DELIVERABLE_FILES = ("manifest.csv", "results_100.csv")


# --------------------------------------------------------------------------
# 样本级硬失败
# --------------------------------------------------------------------------


class HardStop(RuntimeError):
    """不可继续的样本级错误。

    对照实现的 core 一遇 HardStop 就 `break` 整批；本套代码**不那样做**：
    逐样本子进程捕获它、记 `status=HARD_STOP`、继续跑其余样本。
    一条坏样本不该毁掉整批 100 条，更不该成为「删样本」的借口。
    """


# --------------------------------------------------------------------------
# 通用 IO
# --------------------------------------------------------------------------


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """原子写 JSON。先写 .tmp 再 replace，避免中途失败留下半截文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]],
              fields: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def safe_stem(sample_key: str) -> str:
    """`-aNfi7CP8vM$_$7` → `-aNfi7CP8vM___7`。文件名里不能有 `$`。"""
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sample_key).strip("_")


def norm_id(value: Any) -> str:
    """把 xlsx 里的单元格规范化成字符串。

    clip_id 在 xlsx 里可能是数字 3 也可能是字符串 '3'，同一份表里混用会让
    `f"{video_id}_{clip_id}"` 生成两种 key。统一成整数形式再转字符串。
    """
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def sample_key_of(video_id: str, clip_id: str) -> str:
    """附件2 官方口径的样本键：`video_id$_$clip_id`。

    与项目内部口径 `video_id_clip_id` 不同——后者在 video_id 自身含下划线时
    （本批 100 条里有 6 个）反解必须 `rsplit("_", 1)`。本套代码一律用官方口径，
    不用内部口径，从源头避开这个坑。
    """
    return f"{norm_id(video_id)}$_${norm_id(clip_id)}"


def config_digest(output_root: Path = OUTPUT_ROOT) -> str:
    """config_snapshot.json 的 SHA-256。NPZ 与日志都用它标「哪份配置产出」。"""
    path = Path(output_root) / "config_snapshot.json"
    if not path.is_file():
        return "config_snapshot_missing"
    return sha256(path)


def relpath_for_delivery(path: Path) -> str:
    """产出**相对路径**，绝不写绝对路径。

    对照实现的 run_q1_v1.py 把 `str(Path(args.input_root))` 写进了
    environment.json，于是产物里带上了本机用户名。本套代码不重复这个错误：
    凡落盘一律相对路径，跨机可迁移。
    """
    try:
        return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def excluded_walk(root: Path) -> Iterable[Path]:
    """遍历文件，排除缓存、版本控制与 Office 临时件。"""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".git", ".claude", ".cache")]
        for name in filenames:
            if name.startswith("~$") or name.endswith(".pyc"):
                continue
            yield Path(dirpath) / name


__all__ = [
    "PROJECT_ROOT", "OUTPUT_ROOT", "CACHE_ROOT", "WORK_ROOT", "WORK_ROOT_REASON",
    "VENV_PYTHON", "FACE_MODEL_PATH", "ROBERTA_DIR_CANDIDATES",
    "WHISPER_CHECKPOINT_CANDIDATES",
    "PROTECTED_DIRS", "VOLATILE_BY_DESIGN", "SCHEMA_VERSION",
    "ROBERTA_MODEL_ID", "ROBERTA_REVISION", "ROBERTA_HIDDEN_DIM",
    "ROBERTA_FILE_SHA256", "FACE_MODEL_SHA256", "FACE_MODEL_URL",
    "FACE_MODEL_BYTES", "FACE_MODEL_DIM", "OPENMILE_CONFIG_SHA256",
    "OPENMILE_LLD_DIM", "OPENMILE_FEATURE_NAMES",
    "WHISPER_CHECKPOINT_SHA256", "WHISPER_MODEL_NAME",
    "LABEL_XLSX_SHA256", "ALIGN_PARAMS", "ALIGN_SOURCE", "ALIGNER_NAME",
    "INTERVAL_CONVENTION", "AGGREGATION_QUERY", "TOL", "AGG_RTOL", "AGG_ATOL",
    "EXPECTED_TOTAL_WHITESPACE_CHUNKS", "EXPECTED_TOTAL_WORD_UNITS",
    "ROBERTA_MAX_POSITIONS_FALLBACK",
    "MODE_TRI_MODAL_WORD_VALID", "MODE_TRI_MODAL_AUDIO_CONTENT_INVALID",
    "MODE_AV_VALID_TEXT_UNALIGNED", "MODE_UNCERTAIN_REVIEW",
    "MODE_EXTRACTION_OR_TIMELINE_ANOMALY", "MODE_VOCABULARY",
    "MODES_NEVER_EMITTED", "CORRESPONDENCE_VOCABULARY",
    "TIME_STATUS_VOCABULARY", "WORD_TIME_BASIS_OWN", "WORD_TIME_BASIS_NONE",
    "WORD_TIME_BASIS_VOCABULARY", "CONTENT_ASSERTION_NONE",
    "CONTENT_ASSERTION_EXTERNAL", "LIMIT_BYTES", "LIMIT_STRICT_BYTES",
    "DELIVERABLE_SUBDIRS", "DELIVERABLE_FILES",
    "HardStop",
    "sha256", "sha256_text", "write_json", "write_csv", "read_csv",
    "safe_stem", "norm_id", "sample_key_of", "config_digest",
    "relpath_for_delivery", "excluded_walk",
]
