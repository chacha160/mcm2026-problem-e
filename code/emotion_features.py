# -*- coding: utf-8 -*-
"""
emotion_features.py —— 情感判定模型的「特征装配」层

职责边界：
    本文件只负责把问题1 的对齐产物（(50, D) 张量 + 逐模态有效掩码）装配成
    机器学习可直接吃的二维设计矩阵，并把监督标签从 xlsx 关联进来。
    **不做任何模型训练/评估**——那是 emotion_model.py 的事。

为什么单独一层：
    1) 「只在有效槽上池化」是一条容易写错、写错了又不会报错的规则（见下），
       集中在一处实现，训练与消融共用同一份逻辑，避免两边口径漂移；
    2) 消融实验需要按模态切片取子矩阵，切片边界必须与装配时一致，
       故由 build_design_matrix 通过 meta 显式返回。

池化口径（实测依据）
--------------------
    · 文本：**仅均值**。有效槽只占 43%（每条中位 20/50），句内词向量 std 是噪声。
      实测改为仅均值后 文本单模态 acc 0.590→0.620、r +0.359→+0.421。
    · 音频/视觉：**均值 + std**。二者是真时间序列（20 Hz / 15 Hz），
      沿时间轴的波动本身含信息，std 有意义。
    · 所有模态一律**只在 *_valid 为真的槽上统计**：
      文本空槽是精确零填充，若不剔除会把 768 维向量稀释过半；
      音频/视觉空槽是插值值（非零），不剔除会把构造值当真值统计。

⚠️ `*_valid` 的语义边界（实测，容易误用）
----------------------------------------
    `valid=True` 只表示「该槽被分配到了采样单元」，**不表示该单元是真观测**。
    实测反例：`-mJ2ud6oKI8_1` 的 `vision_valid` 为 50/50 全 True，但其视觉特征
    能量恰为 0.0（MTCNN 一帧人脸都没检出，`_fill_missing_faces` 保持零向量）。
    即插值/零填充的槽同样会被标成 valid。
    因此**不能用 `*_valid` 判断模态是否真的可用**——那要用原始 `face_ratio`
    （见 `load_availability`）。本模块的池化对这类样本会自然得到零向量，
    语义上恰好正确（"该模态对此样本不可用"），但调用方若需要区分
    「因不可用而零」与「模型学出来是零」，必须另读 availability 标志。
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

# 控制台按 UTF-8 输出，避免 GBK 终端下中文日志乱码（与既有模块同一写法）
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

import config  # noqa: E402
import utils  # noqa: E402
from utils import LOGGER  # noqa: E402

MODALITIES: Tuple[str, ...] = ("text", "audio", "vision")

# 各模态池化是否附带 std（见模块 docstring 的实测依据）
POOL_WITH_STD: Dict[str, bool] = {"text": False, "audio": True, "vision": True}


def project_root() -> str:
    """项目根目录（code/ 的上一层）。"""
    return os.path.dirname(_CODE_DIR)


def aligned_npz_path(align_dir: Optional[str] = None) -> str:
    """对齐汇总 npz 的路径，默认 data/aligned/aligned_50.npz。"""
    if align_dir:
        return os.path.join(align_dir, f"aligned_{config.ALIGN_SEQ_LEN}.npz")
    return os.path.join(project_root(), "data", "aligned",
                        f"aligned_{config.ALIGN_SEQ_LEN}.npz")


# ==================== 监督标签 ====================

def load_supervision(label_xlsx: Optional[str] = None,
                     sheet_name: Optional[str] = None) -> "object":
    """
    读取监督标签，返回 pandas.DataFrame，列：
        sample_id, video_id, clip_id, label, annotation, text

    为什么不用 unaligned_common.list_samples()：
        它返回的 dict **不含 annotation**（分类标签），只含 label（回归标签）。
        本任务需要同时给出「强度」与「状态」，故直接走 utils.load_label_table()，
        它是取标签的唯一可靠入口，且列名已做 strip 规范化。

    sample_id 的构造口径与问题1 严格一致：f"{video_id}_{clip_id}"，
    已实测与 aligned_50.npz 的 sample_id 集合相等、顺序一致、零缺失。
    """
    import pandas as pd

    if label_xlsx is None:
        label_xlsx = os.path.join(config.ATTACHMENT1_DIR, config.LABEL_XLSX_NAME)
    if not os.path.isfile(label_xlsx):
        raise FileNotFoundError(
            f"找不到标签表：{label_xlsx}\n"
            f"（该路径由 config.ATTACHMENT1_DIR + config.LABEL_XLSX_NAME 拼出，"
            f"若数据位置变动请改 config.py）")

    df = utils.load_label_table(label_xlsx, sheet_name or config.LABEL_SHEET_NAME)
    for col in ("video_id", "clip_id", "label", "annotation"):
        if col not in df.columns:
            raise ValueError(f"标签表缺少列 {col!r}；实际列：{list(df.columns)}")

    df = df.copy()
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["clip_id"] = df["clip_id"].astype(str).str.strip()
    df["sample_id"] = df["video_id"] + "_" + df["clip_id"]
    df["label"] = df["label"].astype(float)
    df["annotation"] = df["annotation"].astype(str).str.strip()
    if "text" not in df.columns:
        df["text"] = ""
    LOGGER.info("[装配] 监督标签：%d 条，label 范围 [%.3f, %.3f]，状态分布 %s",
                len(df), df["label"].min(), df["label"].max(),
                dict(df["annotation"].value_counts()))
    return df


def label_to_state(label: np.ndarray) -> np.ndarray:
    """连续 label → 三分类状态编码（1=Positive / 0=Neutral / -1=Negative）。

    依据：实测 annotation 是 sign(label) 的确定性函数、零例外，
    故状态标签可由 label 无歧义导出，不需要独立标注。
    """
    y = np.asarray(label, dtype=np.float64)
    return np.where(y > 1e-9, 1, np.where(y < -1e-9, -1, 0)).astype(np.int64)


STATE_TO_NAME: Dict[int, str] = {1: "Positive", 0: "Neutral", -1: "Negative"}


# ==================== 池化 ====================

def pool_modality(F: np.ndarray, V: np.ndarray, with_std: bool) -> np.ndarray:
    """
    按有效掩码池化单模态特征：(N, L, D) + (N, L) → (N, D) 或 (N, 2D)。

    只在 V 为真的槽上做均值（with_std=True 时再拼一维 std）。
    某条样本该模态一个有效槽都没有时整块置零——这不是 bug 而是明确语义：
    「该模态对此样本不可用」，与视觉 face_ratio=0 的样本一致，
    由调用方（问题2 的缺失模态分析）据此识别。
    """
    F = np.asarray(F, dtype=np.float64)
    V = np.asarray(V, dtype=bool)
    if F.ndim != 3 or V.ndim != 2 or F.shape[:2] != V.shape:
        raise ValueError(f"形状不匹配：F{F.shape} V{V.shape}")
    n, _, d = F.shape
    P = np.zeros((n, d * (2 if with_std else 1)), dtype=np.float64)
    n_empty = 0
    for i in range(n):
        vi = np.flatnonzero(V[i])
        if len(vi) == 0:
            n_empty += 1
            continue
        w = F[i, vi]
        P[i, :d] = w.mean(axis=0)
        if with_std:
            P[i, d:] = w.std(axis=0)
    if n_empty:
        LOGGER.warning("[装配] 有 %d 条样本该模态无任何有效槽，已整块置零", n_empty)
    return P


# ==================== 设计矩阵 ====================

def build_design_matrix(modalities: Sequence[str] = MODALITIES,
                        align_dir: Optional[str] = None,
                        label_xlsx: Optional[str] = None,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    装配设计矩阵。

    返回 (X, y, ycls, groups, meta)：
        X      (N, D_total) float64 设计矩阵
        y      (N,)          float64 连续情感强度 label
        ycls   (N,)          int64   状态编码 1/0/-1
        groups (N,)          <U*    视频 ID，供分组 CV（同一视频的片段必须同折）
        meta   dict          含 modalities / slices / n_groups / n_effective 等溯源信息

    slices 记录每个模态在 X 中的列区间，消融时按区间取子矩阵，
    保证与装配口径完全一致（不会出现「消融时列错位」这类静默错误）。
    """
    z = np.load(aligned_npz_path(align_dir), allow_pickle=True)
    sids = np.array([str(x) for x in z["sample_id"]])
    sup = load_supervision(label_xlsx)

    # 以标签表为准做左连接式对齐：逐条查表，任何缺失都当场报错（不静默丢弃）
    lut = dict(zip(sup["sample_id"], zip(sup["label"], sup["annotation"],
                                         sup["video_id"], sup["clip_id"])))
    missing = [s for s in sids if s not in lut]
    if missing:
        raise ValueError(f"{len(missing)} 条对齐样本在标签表中找不到，例：{missing[:5]}")

    y = np.array([lut[s][0] for s in sids], dtype=np.float64)
    ann = np.array([lut[s][1] for s in sids], dtype=object)
    vid = np.array([lut[s][2] for s in sids], dtype=object)
    cid = np.array([lut[s][3] for s in sids], dtype=object)
    ycls = label_to_state(y)

    # 标签与状态自洽性硬检查：annotation 必须等于 sign(label)
    name_from_label = np.array([STATE_TO_NAME[int(c)] for c in ycls], dtype=object)
    bad = np.flatnonzero(name_from_label != ann)
    if len(bad):
        raise ValueError(
            f"{len(bad)} 条样本的 annotation 与其 label 符号不一致，"
            f"例：{[(sids[i], y[i], ann[i]) for i in bad[:3]]}")

    blocks: List[np.ndarray] = []
    slices: Dict[str, Tuple[int, int]] = {}
    cur = 0
    for m in modalities:
        if m not in MODALITIES:
            raise ValueError(f"未知模态 {m!r}，可选 {MODALITIES}")
        P = pool_modality(z[f"{m}_features"], z[f"{m}_valid"], POOL_WITH_STD[m])
        blocks.append(P)
        slices[m] = (cur, cur + P.shape[1])
        cur += P.shape[1]
        LOGGER.info("[装配] 模态 %-6s → %d 维（%s 池化）", m, P.shape[1],
                    "均值+std" if POOL_WITH_STD[m] else "仅均值")

    X = np.hstack(blocks) if blocks else np.zeros((len(sids), 0))
    groups = np.array([s.rsplit("_", 1)[0] for s in sids], dtype=object)

    # 模态可用性（原始 face_ratio 口径，非 valid 掩码）——供问题2 缺失模态分析
    try:
        avail = load_availability(sids.tolist())
        avail_map = dict(zip(avail["sample_id"], avail["vision_available"]))
        zero_map = dict(zip(avail["sample_id"], avail["vision_all_zero"]))
        fr_map = dict(zip(avail["sample_id"], avail["face_ratio"]))
    except Exception as exc:
        LOGGER.warning("[装配] 可用性读取失败（不影响建模）：%s", exc)
        avail_map, zero_map, fr_map = {}, {}, {}

    meta = {
        "sample_id": sids.tolist(),
        "video_id": vid.tolist(),
        "clip_id": cid.tolist(),
        "annotation": ann.tolist(),
        "modalities": list(modalities),
        "slices": {k: list(v) for k, v in slices.items()},
        "pool_with_std": {m: POOL_WITH_STD[m] for m in modalities},
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_groups": int(len(set(groups.tolist()))),
        "n_state": {STATE_TO_NAME[k]: int((ycls == k).sum()) for k in (1, 0, -1)},
        "vision_available": {s: bool(avail_map.get(s, True)) for s in sids.tolist()},
        "vision_all_zero": {s: bool(zero_map.get(s, False)) for s in sids.tolist()},
        "face_ratio": {s: fr_map.get(s, float("nan")) for s in sids.tolist()},
    }
    check_alignment(X, y, ycls, groups)
    LOGGER.info("[装配] 设计矩阵 X%s，%d 个视频分组，状态分布 %s",
                X.shape, meta["n_groups"], meta["n_state"])
    return X, y, ycls, groups, meta


def check_alignment(X: np.ndarray, y: np.ndarray, ycls: np.ndarray,
                    groups: np.ndarray) -> None:
    """装配后的硬自检。任何一条不过就直接抛错，绝不把坏数据喂给模型。"""
    n = X.shape[0]
    if not (len(y) == len(ycls) == len(groups) == n):
        raise ValueError(f"样本数不一致：X{n} y{len(y)} ycls{len(ycls)} groups{len(groups)}")
    if np.isnan(X).any():
        raise ValueError("设计矩阵含 NaN")
    if np.isinf(X).any():
        raise ValueError("设计矩阵含 inf")
    if not np.array_equal(ycls, label_to_state(y)):
        raise ValueError("ycls 与 y 的符号不一致")
    n_groups = len(set(groups.tolist()))
    LOGGER.info("[装配] 自检通过：%d 条样本 / %d 维 / %d 组 / 无 NaN", n, X.shape[1], n_groups)


def load_availability(sids: Sequence[str],
                      face_ratio_threshold: float = 0.5) -> "object":
    """
    按 sample_id 读取「原始信号级」的模态可用性，返回 pandas.DataFrame：
        sample_id, face_ratio, vision_energy, vision_available, vision_all_zero

    为什么不能用 `vision_valid` 判断（见模块 docstring 的实测反例）：
        valid 掩码只回答"这槽有没有分配到单元"，插值与零填充同样被标 True。
        真正的可用性要看 MTCNN 的 face_ratio（出自 unaligned 阶段的原始 meta，
        与标签无关，不构成泄漏）。

    vision_available = face_ratio >= threshold。实测该阈值恰好命中 6 条退化样本。
    另外单独给出 vision_all_zero（对齐后视觉能量 ≈ 0），用于交叉核验：
    若某条 available=True 却 all_zero=True，说明两种口径不一致，需人工复核。
    """
    import glob
    import json
    import pandas as pd

    vdir = os.path.join(project_root(), "data", "unaligned_features", "vision")
    ratios: Dict[str, float] = {}
    for p in glob.glob(os.path.join(vdir, "*.npz")):
        sid = os.path.basename(p)[:-4]
        try:
            m = np.load(p, allow_pickle=True)["meta"].item()
            if isinstance(m, str):
                m = json.loads(m)
            ratios[sid] = float(m.get("face_ratio") or 0.0)
        except Exception as exc:
            LOGGER.warning("[可用性] 读取 %s 失败：%s", sid, exc)
            ratios[sid] = float("nan")

    z = np.load(aligned_npz_path(), allow_pickle=True)
    zsids = [str(x) for x in z["sample_id"]]
    energy = {zsids[i]: float((z["vision_features"][i].astype(np.float64) ** 2).sum())
              for i in range(len(zsids))}

    rows = []
    for s in sids:
        fr = ratios.get(s, float("nan"))
        en = energy.get(s, float("nan"))
        rows.append({
            "sample_id": s, "face_ratio": round(fr, 4) if fr == fr else float("nan"),
            "vision_energy": round(en, 4) if en == en else float("nan"),
            "vision_available": bool(fr == fr and fr >= face_ratio_threshold),
            "vision_all_zero": bool(en == en and en < 1e-9),
        })
    df = pd.DataFrame(rows)
    n_un = int((~df["vision_available"]).sum())
    LOGGER.info("[可用性] 视觉不可用（face_ratio<%.2f）%d 条；其中特征全零 %d 条",
                face_ratio_threshold, n_un, int(df["vision_all_zero"].sum()))
    # 口径互证：不可用但没有全零 = 插值借来的特征（如实记录，不擅自丢弃）
    borrowed = df[(~df["vision_available"]) & (~df["vision_all_zero"])]
    if len(borrowed):
        LOGGER.info("[可用性] 其中 %d 条为「低人脸率但特征非零」（插值借用），"
                    "已保留其原始特征不做丢弃：%s", len(borrowed),
                    borrowed["sample_id"].tolist())
    return df


def slice_modalities(X: np.ndarray, meta: Dict, modalities: Sequence[str]) -> np.ndarray:
    """按 meta['slices'] 取指定模态的子矩阵（消融用）。"""
    cols: List[int] = []
    for m in modalities:
        a, b = meta["slices"][m]
        cols.extend(range(a, b))
    return X[:, cols]


if __name__ == "__main__":
    # 自检入口：直接跑本文件即可验证装配链路
    X, y, ycls, groups, meta = build_design_matrix()
    print(f"X={X.shape}  y={y.shape}  组数={meta['n_groups']}")
    print(f"各模态切片={meta['slices']}")
    print(f"状态分布={meta['n_state']}")
    for m in MODALITIES:
        Xm = slice_modalities(X, meta, [m])
        print(f"  {m:6s} 子矩阵={Xm.shape}")
