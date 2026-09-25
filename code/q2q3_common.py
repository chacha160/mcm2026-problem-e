# -*- coding: utf-8 -*-
"""
q2q3_common.py —— 问题二/问题三共用的「官方数据 + 缺失语义」层

为什么单独立这个模块
--------------------
问题二、问题三都只允许使用官方附件（赛题红线 R3），且赛题红线 R2 要求
训练/验证/专项测试使用**同一特征版本与同一输入接口**。项目早期的问题二实现
用的是自提 100 条特征 + 附件1 的 label-100，从未读过附件2、也从未对附件3 推理，
口径与赛题不符。本模块把「官方数据怎么读、缺失怎么定义」一次性钉死，
训练、评测、推理三处都从这里取数，从结构上排除口径漂移。

已实测的官方数据事实（本模块全部据此实现，不依赖任何假设）
----------------------------------------------------------
1. 附件2 `aligned_50.pkl` 顶层为 {'train','valid','test'}，样本数 3395/728/727。
   每 split 字段：raw_text, text(50,768) f32, text_bert(3,50) i64,
   audio(50,74) f64, vision(50,35) f64, id(形如 `video$_$clip`),
   classification_labels, regression_labels。
2. `text_bert` 的三行依次是 [input_ids, attention_mask, token_type_ids]（BERT 词表）。
   记 L = attention_mask.sum()，则
       槽 0        = [CLS]
       槽 L-1      = [SEP]
       槽 [1, L-1) = 词元位置（含音频/视觉对齐特征）
   实测：附件2 中 audio 的非零槽数**恰好等于 L-2**（3395 条中 3391 条成立），
   这正是「有效区 = [1, L-1)」的直接证据，而不是把 50 个槽都当有效。
3. 附件2 的 vision 自带真实缺失：110 条全零（无脸），另有约 311 条部分缺失。
   因此「零 = 缺失」在附件2 上就已是错的近似；只有把有效区先切出来才有意义。
4. 附件3 的 30 条样本，其 audio/vision 非零槽数**少于 L-2**，差额 0~14，
   零值落在有效区内部且连续——这正对应赛题第 14 行的定义
   「一个或多个模态中存在特征值全部为零的随机连续序列区间」。
5. 附件3 只提供 `{audio, text_bert, vision}`（对齐版本），**不提供 768 维 text**。
   所以问题二的文本支路只能吃 `text_bert` 词元 id；训练/推理统一如此，天然满足 R2。
6. 附件4 的对齐版本提供完整三模态（含 768 维 text）与 20 个原始 mp4。

本模块不写任何文件、不改任何数据，只做读取与掩码推导。
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import config  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

# ==================== 常量口径（全项目唯一定义，禁止在别处重复硬编码）====================

MODALITIES: Tuple[str, ...] = ("text", "audio", "vision")
MODALITY_CN = {"text": "文本", "audio": "语音", "vision": "视觉"}

SEQ_LEN = 50            # 官方对齐序列长度（附件2/3/4 对齐版本一致）
TEXT_RAW_DIM = 768      # 附件2/4 的 text 维度（附件3 不提供）
AUDIO_DIM = 74          # 附件2/3/4 的 audio 维度
VISION_DIM = 35         # 附件2/3/4 的 vision 维度
BERT_ROWS = 3           # text_bert 的三行：input_ids / attention_mask / token_type_ids
PAD_TOKEN_ID = 0        # input_ids 的填充值（BERT 系 <pad>）
CLS_TOKEN_ID = 101      # 实测首槽恒为 101，即 [CLS]
SEP_TOKEN_ID = 102      # [SEP]

# 标签口径：MOSEI 的 label 取 {-3,-2,-1,0,1,2,3}，符号即极性，绝对值即强度。
POLARITY_ORDER: Tuple[int, ...] = (-1, 0, 1)     # Negative / Neutral / Positive
POLARITY_CN = {-1: "Negative", 0: "Neutral", 1: "Positive"}
INTENSITY_MAX = 3.0


# ==================== 缺失语义：本模块的核心 ====================

@dataclass
class MissingnessInfo:
    """一个样本的三模态可用性描述。所有掩码均为 (M, SEQ_LEN) 的 bool 数组。"""
    valid_len: int                      # L = attention_mask.sum()，含 CLS/SEP
    word_region: np.ndarray             # (SEQ_LEN,) bool，[1, L-1) 为 True
    padding_mask: np.ndarray            # (SEQ_LEN,) bool，True 表示「填充/非词元槽」
    observed: Dict[str, np.ndarray]     # 模态 → (SEQ_LEN,) bool，「该槽有非零特征」
    missing: Dict[str, np.ndarray]      # 模态 → (SEQ_LEN,) bool，「有效区内该槽全零」
    available: Dict[str, bool]          # 模态 → 该样本是否整体可用（≥1 个有效观测槽）
    missing_runs: Dict[str, List[Tuple[int, int]]]  # 模态 → 连续零区间 [start, end)

    @property
    def missing_counts(self) -> Dict[str, int]:
        return {m: int(self.missing[m].sum()) for m in MODALITIES}

    @property
    def observed_counts(self) -> Dict[str, int]:
        return {m: int(self.observed[m].sum()) for m in MODALITIES}


def valid_length(text_bert: np.ndarray) -> int:
    """从 text_bert 的 attention_mask 行取出有效词元数 L。

    text_bert 形状 (3, 50) 或 (N, 3, 50)。返回 int 或 (N,) 数组。
    若 attention_mask 全零（样本级异常），退化为「input_ids 非零个数 + 1」，
    因为 input_ids 在 [CLS] 位置必为 101，全零只可能来自掩码写坏。
    """
    tb = np.asarray(text_bert)
    if tb.shape[0] == BERT_ROWS:                      # (3, 50) 单样本
        am = tb[1]
        L = int(am.sum())
        if L <= 0:
            L = int((tb[0] != PAD_TOKEN_ID).sum())
        return L
    am = tb[:, 1, :]                                   # (N, 3, 50)
    L = am.sum(axis=1).astype(np.int64)
    bad = L <= 0
    if bad.any():
        L = np.where(bad, (tb[:, 0, :] != PAD_TOKEN_ID).sum(axis=1), L)
    return L


def word_region_mask(L: int, seq_len: int = SEQ_LEN) -> np.ndarray:
    """[1, L-1) 为 True 的词元区掩码（槽 0=[CLS]、槽 L-1=[SEP] 之外的都算填充）。

    L 可能因截断等于 seq_len（此时没有 [SEP] 槽，词元区到 seq_len-1 为止）。
    """
    idx = np.arange(seq_len)
    hi = min(L - 1, seq_len)
    return (idx >= 1) & (idx < hi)


def observed_mask(feat: np.ndarray) -> np.ndarray:
    """逐槽「是否有非零特征」。feat 形状 (T, D) 或 (N, T, D)，返回 (T,) 或 (N, T)。"""
    f = np.asarray(feat)
    return np.abs(f).sum(axis=-1) != 0


def missing_runs(mask_1d: np.ndarray) -> List[Tuple[int, int]]:
    """把布尔掩码里的连续 True 归并成半开区间 [start, end)。"""
    m = np.asarray(mask_1d, dtype=bool)
    if not m.any():
        return []
    d = np.diff(np.concatenate(([0], m.view(np.int8), [0])))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def analyze_sample(feat: Dict[str, np.ndarray],
                   text_bert: np.ndarray) -> MissingnessInfo:
    """推导单样本的三模态缺失结构。

    文本可用性口径（三个附件必须一致，问题要求 R2「同一特征版本、同一输入接口」）：
    模型真正的文本输入是 `input_ids`，因此逐槽「有文本」= 词元区内出现非 pad 词元，
    即 `obs = word_region & (ids != PAD_TOKEN_ID)`。

    注意不要改用 768 维 `text` 特征的「非零即观测」：BERT 对 padding 位置同样输出
    非零上下文向量，那样会把填充槽判成有观测，进而让填充槽进入注意力 softmax，
    凭空造出既无时间区间也无原始文本片段的「关键证据」（附件4 曾因此复现）。
    该 768 维特征并非模型输入，只用于词元向量的蒸馏初始化。
    """
    L = int(valid_length(text_bert))
    wr = word_region_mask(L)
    obs: Dict[str, np.ndarray] = {}
    mis: Dict[str, np.ndarray] = {}

    for m in ("audio", "vision"):
        o = observed_mask(feat[m])
        obs[m] = o
        mis[m] = wr & ~o

    # 文本：只看 input_ids 在词元区是否有真实词元（附件2/3/4 同一规则）
    ids = np.asarray(text_bert)[0] if np.asarray(text_bert).shape[0] == BERT_ROWS \
        else np.asarray(text_bert)[0, 0]
    o = np.zeros(SEQ_LEN, dtype=bool)
    o[wr] = ids[wr] != PAD_TOKEN_ID
    obs["text"] = o
    # 文本模态「整体不可用」= 词元区没有任何真实词元；此时不算逐槽缺失，
    # 而由 available=False 表达，避免把整段缺失重复计成逐槽缺失。
    mis["text"] = wr & ~o if o.any() else np.zeros(SEQ_LEN, dtype=bool)

    avail = {m: bool(obs[m].any()) for m in MODALITIES}
    runs = {m: missing_runs(mis[m]) for m in MODALITIES}

    return MissingnessInfo(
        valid_len=L, word_region=wr,
        padding_mask=~wr,
        observed=obs, missing=mis, available=avail, missing_runs=runs,
    )


# ==================== 数据装载 ====================

@dataclass
class Split2:
    """附件2 的一个 split。数组按需惰性转成模型友好的 dtype。"""
    name: str
    ids: np.ndarray                 # (N,) <U16
    raw_text: np.ndarray            # (N,) <U...
    text_bert: np.ndarray           # (N, 3, 50) int64
    audio: np.ndarray               # (N, 50, 74) float64 → 用时就地转 float32
    vision: np.ndarray              # (N, 50, 35) float64
    text_feat: Optional[np.ndarray]  # (N, 50, 768) float32；附件2 有
    cls_label: np.ndarray           # (N,) float64，取 {-3..3}
    reg_label: np.ndarray           # (N,) float64

    def __len__(self) -> int:
        return len(self.ids)


_ATT2_CACHE: Dict[str, object] = {}


def load_attachment2(root: Optional[str] = None) -> Dict[str, Split2]:
    """装载附件2 `aligned_50.pkl`（约 1 GB，进程内缓存，重复调用不重读）。"""
    if "att2" in _ATT2_CACHE:
        return _ATT2_CACHE["att2"]  # type: ignore[return-value]
    root = root or config.attachment_dir("2")
    path = os.path.join(root, config.ATT2_ALIGNED_PKL)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"附件2 特征文件不存在：{path}")
    with open(path, "rb") as f:
        raw = pickle.load(f)

    out: Dict[str, Split2] = {}
    for sp in ("train", "valid", "test"):
        if sp not in raw:
            continue
        d = raw[sp]
        out[sp] = Split2(
            name=sp,
            ids=np.asarray(d["id"]),
            raw_text=np.asarray(d["raw_text"]),
            text_bert=np.asarray(d["text_bert"]),
            audio=np.asarray(d["audio"]),
            vision=np.asarray(d["vision"]),
            text_feat=np.asarray(d["text"]) if "text" in d else None,
            cls_label=np.asarray(d["classification_labels"], dtype=np.float64),
            reg_label=np.asarray(d["regression_labels"], dtype=np.float64),
        )
    _ATT2_CACHE["att2"] = out
    return out


def load_attachment2_labels(root: Optional[str] = None):
    """读取附件2 的 label.xlsx，返回列表字典；用于交叉核验 pkl 内标签。"""
    import pandas as pd
    root = root or config.attachment_dir("2")
    path = os.path.join(root, config.ATT2_LABEL_XLSX)
    if not os.path.isfile(path):
        return None
    return pd.read_excel(path, sheet_name="label")


@dataclass
class Sample3:
    """附件3 的一条缺失样本（对齐版本）。"""
    index: int                      # 1..30
    key: str                        # 文件名去后缀，如 'attachment3_07'
    text_bert: np.ndarray           # (3, 50)
    audio: np.ndarray               # (50, 74)
    vision: np.ndarray              # (50, 35)
    info: MissingnessInfo = field(repr=False, default=None)  # type: ignore[assignment]


def load_attachment3(version: str = "对齐版本") -> List[Sample3]:
    """装载附件3 的 30 条缺失样本。version ∈ {'对齐版本','未对齐版本'}。

    未对齐版本只有 {audio(1,500,74), raw_text, vision(1,500,35)}，
    没有 text_bert，无法与附件2 共用输入接口，故问题二默认用对齐版本。
    """
    root = os.path.join(config.attachment_dir("3"), version)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"附件3 {version} 不存在：{root}")
    files = sorted(f for f in os.listdir(root) if f.endswith(".pkl"))
    out: List[Sample3] = []
    for i, fn in enumerate(files, start=1):
        with open(os.path.join(root, fn), "rb") as f:
            d = pickle.load(f)
        d = d.get("test", d) if isinstance(d, dict) and "test" in d else d
        if "text_bert" not in d:
            raise ValueError(
                f"{fn} 缺少 text_bert，说明取的是未对齐版本；问题二需要对齐版本。")
        tb = np.asarray(d["text_bert"])[0]
        au = np.asarray(d["audio"])[0]
        vi = np.asarray(d["vision"])[0]
        rec = Sample3(index=i, key=f"attachment3_{i:02d}", text_bert=tb,
                      audio=au, vision=vi)
        rec.info = analyze_sample({"audio": au, "vision": vi}, tb)
        out.append(rec)
    return out


@dataclass
class Sample4:
    """附件4 的一条可解释专项样本（对齐版本，含原始 mp4）。"""
    index: int
    key: str                        # '01'..'20'，与 videos/01.mp4 同名
    raw_text: str
    text_bert: np.ndarray           # (3, 50)
    audio: np.ndarray               # (50, 74)
    vision: np.ndarray              # (50, 35)
    text_feat: np.ndarray           # (50, 768)
    video_path: Optional[str]       # videos/<key>.mp4
    info: MissingnessInfo = field(repr=False, default=None)  # type: ignore[assignment]


def load_attachment4(version: str = "对齐版本") -> List[Sample4]:
    root = os.path.join(config.attachment_dir("4"), version)
    if not os.path.isdir(root):
        raise FileNotFoundError(f"附件4 {version} 不存在：{root}")
    vdir = os.path.join(root, "videos")
    files = sorted(f for f in os.listdir(root) if f.endswith(".pkl"))
    out: List[Sample4] = []
    for i, fn in enumerate(files, start=1):
        with open(os.path.join(root, fn), "rb") as f:
            d = pickle.load(f)
        key = str(d.get("id", os.path.splitext(fn)[0]))
        tb = np.asarray(d["text_bert"])
        au = np.asarray(d["audio"])
        vi = np.asarray(d["vision"])
        rec = Sample4(
            index=i, key=key, raw_text=str(d.get("raw_text", "")),
            text_bert=tb, audio=au, vision=vi,
            text_feat=np.asarray(d["text"]),
            video_path=os.path.join(vdir, f"{key}.mp4"),
        )
        if not os.path.isfile(rec.video_path):
            rec.video_path = None
        rec.info = analyze_sample({"audio": au, "vision": vi}, tb)
        out.append(rec)
    return out


def load_attachment1_labels(root: Optional[str] = None):
    """读取附件1 的 label-100.xlsx（问题一用；问题二/三不使用）。"""
    import pandas as pd
    root = root or config.attachment1_root()
    path = os.path.join(root, config.LABEL_XLSX_NAME)
    return pd.read_excel(path, sheet_name=config.LABEL_SHEET_NAME)


# ==================== 标签派生 ====================

def to_polarity(label: np.ndarray) -> np.ndarray:
    """连续 label → 极性类别，编码为 0/1/2 对应 Negative/Neutral/Positive。

    用 np.sign 后查表，避免 np.sign(0)=0 被当成 Negative 的经典错误。
    """
    lab = np.asarray(label, dtype=np.float64)
    s = np.sign(lab).astype(np.int64)
    idx = np.full(s.shape, 1, dtype=np.int64)      # 默认 Neutral
    idx[s > 0] = 2
    idx[s < 0] = 0
    return idx


def to_intensity(label: np.ndarray) -> np.ndarray:
    """情感强度 = |label|，截断到 [0, INTENSITY_MAX]。"""
    return np.clip(np.abs(np.asarray(label, dtype=np.float64)), 0.0, INTENSITY_MAX)


def polarity_index_to_value(idx: np.ndarray) -> np.ndarray:
    """极性类别 0/1/2 → 连续值 -1/0/+1（用于把分类结果还原成符号）。"""
    return np.asarray(idx, dtype=np.float64) - 1.0


# ==================== 掩码打包 ====================

def build_masks(samples: Sequence[object]) -> Dict[str, np.ndarray]:
    """把一组样本的缺失结构打包成模型输入所需的批量掩码。

    返回
    ----
    obs_mask   : (N, 3, 50) bool—— **逐模态**：该模态在该槽是否有真观测。
                 刻意不做成跨模态并集——并集会让「屏蔽某模态」连带屏蔽别的
                 模态在同一槽位上的观测，受控缺失实验与留一归因都会失效。
    modal_avail: (N, 3) bool    —— 该模态在该样本上是否整体可用
    miss_flag  : (N, 3, 50) bool—— 有效区内该槽是否为缺失（供缺失感知注意力使用）
    padding    : (N, 50) bool   —— 该槽是否为填充（非词元区）
    """
    n = len(samples)
    obs_mask = np.zeros((n, len(MODALITIES), SEQ_LEN), dtype=bool)
    modal_avail = np.zeros((n, len(MODALITIES)), dtype=bool)
    miss_flag = np.zeros((n, len(MODALITIES), SEQ_LEN), dtype=bool)
    padding = np.zeros((n, SEQ_LEN), dtype=bool)

    for i, s in enumerate(samples):
        info: MissingnessInfo = s.info  # type: ignore[attr-defined]
        padding[i] = info.padding_mask
        for j, m in enumerate(MODALITIES):
            obs_mask[i, j] = info.observed[m]
            miss_flag[i, j] = info.missing[m]
            modal_avail[i, j] = info.available[m]

    return {"obs_mask": obs_mask, "modal_avail": modal_avail,
            "miss_flag": miss_flag, "padding": padding}


# ==================== 命令行：缺失结构审计 ====================

def _audit_main() -> int:
    """打印附件2/3/4 的缺失结构清单，作为问题二建模前的口径核验证据。"""
    import json as _json

    a2 = load_attachment2()
    report: Dict[str, object] = {"attachment2": {}, "attachment3": [], "attachment4": []}

    for sp, ds in a2.items():
        infos = []
        for i in range(len(ds)):
            tb = ds.text_bert[i]
            rec = analyze_sample({"audio": ds.audio[i], "vision": ds.vision[i]}, tb)
            infos.append(rec)
        mc = {m: np.array([r.missing_counts[m] for r in infos]) for m in MODALITIES}
        avail = {m: np.array([r.available[m] for r in infos]) for m in MODALITIES}
        L = np.array([r.valid_len for r in infos])
        report["attachment2"][sp] = {
            "n": len(ds),
            "valid_len": {"min": int(L.min()), "median": int(np.median(L)), "max": int(L.max())},
            "modality_available_ratio": {m: round(float(avail[m].mean()), 4) for m in MODALITIES},
            "samples_with_missing": {m: int((mc[m] > 0).sum()) for m in MODALITIES},
            "missing_slot_total": {m: int(mc[m].sum()) for m in MODALITIES},
        }
        print(f"[附件2/{sp}] N={len(ds)}  L(min/med/max)="
              f"{L.min()}/{int(np.median(L))}/{L.max()}")
        for m in MODALITIES:
            print(f"    {m:7s} 可用率={avail[m].mean():.3f}  "
                  f"含缺失样本={int((mc[m]>0).sum())}  缺失槽合计={int(mc[m].sum())}")

    a3 = load_attachment3()
    for s in a3:
        mc = s.info.missing_counts
        runs = {m: s.info.missing_runs[m] for m in MODALITIES}
        report["attachment3"].append({
            "key": s.key, "L": s.info.valid_len,
            "missing_slots": mc, "missing_runs": {m: runs[m] for m in ("audio", "vision")},
        })
    print(f"\n[附件3] N={len(a3)}")
    for s in a3:
        print(f"    {s.key}  L={s.info.valid_len:2d}  "
              f"audio缺失{ s.info.missing_counts['audio']:2d}  "
              f"vision缺失{s.info.missing_counts['vision']:2d}  "
              f"runs(audio)={s.info.missing_runs['audio']}")

    a4 = load_attachment4()
    for s in a4:
        report["attachment4"].append({
            "key": s.key, "L": s.info.valid_len,
            "missing_slots": s.info.missing_counts,
            "has_video": s.video_path is not None,
        })
    print(f"\n[附件4] N={len(a4)}  有视频={sum(1 for s in a4 if s.video_path)}")
    for s in a4:
        print(f"    {s.key}  L={s.info.valid_len:2d}  "
              f"audio缺失{s.info.missing_counts['audio']:2d}  "
              f"vision缺失{s.info.missing_counts['vision']:2d}")

    out_dir = os.path.join(config.PROJECT_ROOT, "data", "q2q3")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "missingness_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        _json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n缺失结构审计已写出：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_audit_main())
