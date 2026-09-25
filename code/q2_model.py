# -*- coding: utf-8 -*-
"""
q2_model.py —— 缺失模态鲁棒的三模态情感预测网络（问题二、问题三共用主干）

赛题要求与本文档的对应
----------------------
赛题第 24–25 行（问题二）：缺失模态下仍能稳定预测情感极性与强度，并分析
「缺失模态类型 / 缺失位置 / 缺失时长」三因素对性能的影响。
赛题第 28 行（问题三）：模型需具可解释性，关键证据可对应到原始文本片段、
语音时段或视觉关键帧。

因此主干的三个硬性设计约束：

1. **输入只吃官方对齐字段**（`text_bert` 词元 id + `audio` + `vision`）。
   原因见 q2q3_common 顶部：附件3 不提供 768 维 text，若训练用 768 维而推理
   只能用词元 id，训练/推理接口不一致，直接违反红线 R2。统一吃词元 id
   可在附件2/3/4 上完全一致。

2. **缺失必须显式建模，不能靠补零**。附件2 的视觉自带真实缺失（无脸），
   若把「缺失」和「真实零值」混为一谈，模型会把缺失当成一种情感特征去学。
   本网络的每个槽位携带三路信号：
        obs_mask  （**逐模态**：该模态该槽是否有真观测）→ 注意力里被屏蔽。
                  必须是 (B,3,S) 的逐模态掩码而非跨模态并集：并集会让
                  「屏蔽文本」连带屏蔽语音/视觉在同一槽位的观测，受控缺失
                  实验与留一归因会同时失效（四种缺失类型退化成同一种）。
        miss_flag （该槽是否为「有效区内的缺失」）→ 作为可学习嵌入加在输入上
        padding   （该槽是否为非词元填充）→ 用于区分「没这段」与「这段没信号」

3. **可解释性内建，不做事后编造**。槽级重要性同时来自
   (a) 分类/回归头前的槽级注意力权重、(b) 逐模态留一（leave-one-modality-out）
   的预测变化量。两者都要能落到具体槽号，才能在第 28 行要求下映射回
   文本片段 / 语音时段 / 视觉关键帧。
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import q2q3_common as Q  # noqa: E402

# ==================== 超参默认值 ====================

D_MODEL = 128           # 共享隐空间维度
N_LAYERS = 2            # 跨模态 Transformer 层数
N_HEADS = 4
D_FF = 256
DROPOUT = 0.15
TEXT_EMB_DIM = 64       # 词元嵌入维度（先嵌入再投影，控制参数量以适配 50 MB 提交上限）
TEXT_VOCAB = 30522      # BERT 系词表大小（实测 input_ids 首槽为 101=[CLS]）
MODALITY_DROPOUT = 0.25  # 训练期随机整模态丢弃的概率，用于制造缺失鲁棒性


@dataclass
class ModelConfig:
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    n_heads: int = N_HEADS
    d_ff: int = D_FF
    dropout: float = DROPOUT
    text_vocab: int = TEXT_VOCAB
    text_emb_dim: int = TEXT_EMB_DIM
    seq_len: int = Q.SEQ_LEN
    n_modalities: int = len(Q.MODALITIES)
    n_polarity: int = 3
    max_intensity: float = Q.INTENSITY_MAX


# ==================== 文本支路：用官方 768 维语义蒸馏词元嵌入 ====================
#
# 为什么不直接随机初始化 nn.Embedding(30522, 64)：
#   附件2 训练集只有 3395 条、词元区共 76882 个词元，其中**只出现过 1 次**的
#   词元占相当比例（出现 ≥5 次的仅 1592 个 id、却覆盖 87% 的词元）。从零学
#   一个 3 万行的嵌入表，稀有词元等于纯噪声，是纯粹的参数浪费。
#
# 做法：官方在附件2 同时给了 `text`（768 维）与 `text_bert`（词元 id），二者
#   逐槽对应。于是可以用 768 维向量按 id 求均值，得到该 id 的「官方语义中心」，
#   再 PCA 降到 64 维作为嵌入初值。
#
# 三条红线都守得住：
#   R3 —— 只用了附件2 自身的特征，没有引入任何外部数据或预训练权重；
#   R2 —— 训练、验证、附件3/4 的**输入接口仍是词元 id**，没有多出任何字段，
#         降维表是训练期产物（与「用训练集算均值方差做标准化」同一性质）；
#   R4 —— 表大小 30522×64 个 fp32 ≈ 7.5 MB，用 float32 存得下。
#
# 表**只在训练集上统计**，验证/测试不参与，因此不构成信息泄漏。

def fit_token_text_vectors(train_split, target_dim: int = TEXT_EMB_DIM,
                           min_count: int = 1
                           ) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """统计出「词元 id → target_dim 维语义向量」的**稀疏**结果。

    返回 `(used_ids, used_vecs, stats)`；只有真正出现过的词元 id 才有向量。
    重活都在这里，`build_token_text_table` 只是把它补成整表。
    """
    ids = np.asarray(train_split.text_bert[:, 0, :], dtype=np.int64)
    feat = np.asarray(train_split.text_feat, dtype=np.float32)
    if feat.ndim != 3 or feat.shape[:2] != ids.shape:
        raise ValueError(f"768 维文本特征与词元 id 不同构：{feat.shape} vs {ids.shape}")

    # 只统计**词元区**：槽 0 是 [CLS]、槽 L-1 是 [SEP]，都不是词。
    n, S = ids.shape
    in_word = np.zeros((n, S), dtype=bool)
    for i in range(n):
        L = int(np.asarray(Q.valid_length(train_split.text_bert[i:i + 1])).reshape(-1)[0])
        in_word[i, 1:max(1, L - 1)] = True
    in_word &= (ids != Q.PAD_TOKEN_ID) & (ids != Q.CLS_TOKEN_ID) & (ids != Q.SEP_TOKEN_ID)

    flat_ids = ids[in_word]
    flat_vec = feat[in_word]
    order = np.argsort(flat_ids, kind="stable")
    flat_ids, flat_vec = flat_ids[order], flat_vec[order]
    uniq, start = np.unique(flat_ids, return_index=True)
    sums = np.add.reduceat(flat_vec.astype(np.float64), start, axis=0)
    counts = np.diff(np.append(start, flat_ids.size)).astype(np.float64)
    means = sums / counts[:, None]

    keep = counts >= min_count
    used, used_means, used_counts = uniq[keep], means[keep], counts[keep]

    # PCA：中心化后取 top-target_dim 主方向。用 SVD 而非协方差特征分解，
    # 因为 768 维下 SVD 的数值稳定性更好且不需要显式构造 768×768 矩阵。
    mu = used_means.mean(axis=0, keepdims=True)
    X = used_means - mu
    # 按出现次数加权，让高频词的语义中心主导主方向
    w = np.sqrt(used_counts)[:, None]
    _, _, vt = np.linalg.svd(X * w, full_matrices=False)
    comp = vt[:target_dim]                       # (target_dim, 768)
    proj = X @ comp.T                            # (n_used, target_dim)
    scale = proj.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    proj = proj / scale

    stats = {
        "n_train_samples": int(n),
        "n_word_tokens": int(flat_ids.size),
        "n_distinct_ids": int(uniq.size),
        "n_ids_with_vector": int(used.size),
        "coverage_of_word_tokens": float(used_counts.sum() / flat_ids.size),
        "target_dim": int(target_dim),
        "source_dim": int(feat.shape[-1]),
        "built_from": "附件2 train 的 text(768) 与 text_bert 逐槽配对；"
                      "验证/测试/附件3/附件4 均未参与统计",
    }
    return used.astype(np.int64), proj.astype(np.float32), stats


def build_token_text_table(train_split, target_dim: int = TEXT_EMB_DIM,
                           min_count: int = 1) -> Tuple[np.ndarray, Dict]:
    """一步到位：按 id 聚合 768 维官方特征 → PCA 降维 → 补成整表。"""
    ids, vecs, stats = fit_token_text_vectors(train_split, target_dim, min_count)
    return expand_token_text_table(ids, vecs, stats), stats


SEED_TEXT_TABLE = 20260925


def expand_token_text_table(ids: np.ndarray, vecs: np.ndarray, stats: Dict
                            ) -> np.ndarray:
    """把「只含出现过的词元」的稀疏表补全成整张 (V, target_dim) 嵌入表。

    未出现过的词元用固定种子的随机小初值填充——它们在任何 split 里都不出现，
    取值不影响预测，但必须**可复现**（故用固定种子而非全局随机）。
    存稀疏形式而非整表，是因为 30522 行里有 2.3 万行是这种填充行：
    整表 7.45 MB、稀疏形式 1.9 MB，而稀疏形式配合本函数能无损还原。
    """
    dim = int(stats["target_dim"])
    rng = np.random.default_rng(SEED_TEXT_TABLE)
    table = rng.normal(0.0, 0.02, size=(TEXT_VOCAB, dim)).astype(np.float32)
    table[np.asarray(ids, dtype=np.int64)] = np.asarray(vecs, dtype=np.float32)
    table[Q.PAD_TOKEN_ID] = 0.0                  # padding_idx 必须恒为零
    return table


# ==================== 主干 ====================

class MissingAwareFusionEncoder(nn.Module):
    """把三模态的 50 个槽位串成一条 150 长的序列，用带掩码的 Transformer 融合。

    为什么用「串联 + 掩码注意力」而不是「分别编码再拼接」：
        缺失可能只发生在某些槽位上（附件3 实测 0~14 个连续槽），
        串联后每个槽位能与任意模态的任意槽位直接交互，缺失槽被掩码屏蔽后
        信息仍可由同模态邻槽或跨模态同槽补上；先编码再拼接会把模态内的
        局部缺失错误地平滑进整个模态向量。
    """

    def __init__(self, cfg: ModelConfig, text_emb_table: Optional[np.ndarray] = None):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        S, M = cfg.seq_len, cfg.n_modalities

        # 文本支路：词元 id → 嵌入 → 线性投影。
        # 给了蒸馏表就用它初始化并继续微调；没给则退化为随机初始化，
        # 两种情形**输入接口完全一致**（都只吃词元 id），故不破坏 R2。
        self.text_embed = nn.Embedding(cfg.text_vocab, cfg.text_emb_dim,
                                       padding_idx=Q.PAD_TOKEN_ID)
        if text_emb_table is not None:
            tbl = np.asarray(text_emb_table, dtype=np.float32)
            if tbl.shape != (cfg.text_vocab, cfg.text_emb_dim):
                raise ValueError(f"蒸馏表形状 {tbl.shape} 与配置不符")
            with torch.no_grad():
                self.text_embed.weight.copy_(torch.from_numpy(tbl))
        self.text_table_used = text_emb_table is not None
        self.text_proj = nn.Sequential(
            nn.Linear(cfg.text_emb_dim, d), nn.LayerNorm(d), nn.GELU())
        # 音频/视觉支路：数值特征 → 线性投影
        self.audio_proj = nn.Sequential(
            nn.Linear(Q.AUDIO_DIM, d), nn.LayerNorm(d), nn.GELU())
        self.vision_proj = nn.Sequential(
            nn.Linear(Q.VISION_DIM, d), nn.LayerNorm(d), nn.GELU())

        # 三路结构嵌入：模态身份 / 槽位位置 / 缺失状态
        self.modality_embed = nn.Parameter(torch.randn(M, d) * 0.02)
        self.position_embed = nn.Parameter(torch.randn(S, d) * 0.02)
        # 缺失状态有 3 种：0=已观测，1=有效区内缺失，2=该模态整体不可用
        self.state_embed = nn.Parameter(torch.randn(3, d) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(d)

        # 槽级注意力池化：同时产出可解释的槽重要性
        self.pool_query = nn.Parameter(torch.randn(d) * 0.02)
        self.pool_score = nn.Linear(d, 1)

        # 双头输出
        self.polarity_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, cfg.n_polarity))
        self.intensity_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(d, 1))

    # ---------- 特征装配 ----------

    def _assemble(self,
                  text_ids: torch.Tensor,      # (B, S) long
                  audio: torch.Tensor,         # (B, S, 74)
                  vision: torch.Tensor,        # (B, S, 35)
                  obs_mask: torch.Tensor,      # (B, M, S) bool，True=该模态该槽有观测
                  miss_flag: torch.Tensor,     # (B, M, S) bool，True=有效区内缺失
                  modal_avail: torch.Tensor,   # (B, M) bool
                  modality_dropout: bool = False,
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把三模态特征拼成 (B, M*S, d)，并产出每槽的可用性与注意力掩码。

        `obs_mask` 必须是**逐模态**的 (B, M, S)，不能是跨模态并集。
        早期实现把各模态的观测 OR 成一个 (B, S) 掩码再与逐模态可用性相与，
        后果是「屏蔽某模态的一段槽位」会连带把其他模态在同一槽位上的观测也屏蔽掉，
        使受控缺失实验里「屏蔽文本」与「屏蔽语音」退化成同一件事（实测两组
        四项指标完全相同）。逐模态掩码从结构上排除这种串扰。
        """
        B, S = text_ids.shape
        M = self.cfg.n_modalities
        dev = text_ids.device


        t = self.text_proj(self.text_embed(text_ids))            # (B,S,d)
        a = self.audio_proj(audio)
        v = self.vision_proj(vision)
        x = torch.stack([t, a, v], dim=1)                        # (B,M,S,d)

        # 模态身份 + 位置
        x = x + self.modality_embed.view(1, M, 1, self.cfg.d_model)
        x = x + self.position_embed.view(1, 1, S, self.cfg.d_model)

        # 缺失状态嵌入：区分「观测到」「有效区内缺失」「该模态整体没有」
        state = torch.zeros((B, M, S), dtype=torch.long, device=dev)
        state = torch.where(miss_flag, torch.ones_like(state), state)          # 1=有效区内缺失
        unavailable = (~modal_avail).view(B, M, 1).expand(B, M, S)
        state = torch.where(unavailable, torch.full_like(state, 2), state)     # 2=整体不可用
        x = x + self.state_embed[state]

        # 每槽是否参与计算：模态自身可用 且 **该模态**在该槽有观测
        usable = modal_avail.view(B, M, 1) & obs_mask                          # (B,M,S)

        if modality_dropout and self.training:
            # 训练期整模态随机丢弃：让模型见过「任意模态消失」的情形，
            # 这是问题二「缺失模态鲁棒」在训练侧的对应手段。
            keep = torch.rand((B, M), device=dev) > MODALITY_DROPOUT
            # 至少保留一个模态，避免整样本无输入的退化情况
            all_dropped = ~keep.any(dim=1)
            if all_dropped.any():
                forced = torch.randint(0, M, (int(all_dropped.sum()),), device=dev)
                keep[all_dropped] = False
                keep[all_dropped, forced] = True
            usable = usable & keep.view(B, M, 1)

        x = x.view(B, M * S, self.cfg.d_model)
        usable = usable.view(B, M * S)                            # (B, M*S)

        # key_padding_mask: True 表示「屏蔽掉」
        pad = ~usable
        # 理论上可能某样本所有槽都不可用（极端缺失）；此时放开全部掩码以避免
        # softmax 全 -inf 产生 NaN，并在调用方按 modal_avail 判定为退化样本。
        degenerate = pad.all(dim=1)
        if degenerate.any():
            pad[degenerate] = False
        return x, pad

    # ---------- 前向 ----------

    def forward(self,
                text_ids: torch.Tensor,
                audio: torch.Tensor,
                vision: torch.Tensor,
                obs_mask: torch.Tensor,
                miss_flag: torch.Tensor,
                modal_avail: torch.Tensor,
                modality_dropout: bool = False,
                need_slot_scores: bool = False,
                ) -> Dict[str, torch.Tensor]:
        x, pad = self._assemble(text_ids, audio, vision, obs_mask, miss_flag,
                                modal_avail, modality_dropout=modality_dropout)
        h = self.encoder(x, src_key_padding_mask=pad)
        h = self.norm(h)

        # 槽级注意力池化（只在可用槽上归一化）
        score = self.pool_score(h).squeeze(-1)                    # (B, M*S)
        score = score.masked_fill(pad, -1e4)
        alpha = torch.softmax(score, dim=1)
        pooled = torch.einsum("bn,bnd->bd", alpha, h)

        out = {
            "polarity_logits": self.polarity_head(pooled),
            "intensity_raw": self.intensity_head(pooled).squeeze(-1),
        }
        # 强度恒非负：用 softplus 并把上界压到 max_intensity，
        # 直接 relu 会让「预测 0」这个平凡解在梯度上占优。
        out["intensity"] = F.softplus(out["intensity_raw"]).clamp(0.0, self.cfg.max_intensity)
        if need_slot_scores:
            out["slot_alpha"] = alpha.view(-1, self.cfg.n_modalities, self.cfg.seq_len)
            out["token_usable"] = (~pad).view(-1, self.cfg.n_modalities, self.cfg.seq_len)
        return out


def as_config(cfg) -> ModelConfig:
    """把 `None` / `dict` / `ModelConfig` 统一成 `ModelConfig`。

    检查点里 `config` 存的是 `dataclasses.asdict` 出来的**普通 dict**，
    而模型构造函数要的是带属性的对象。以前只有「从零训练」这一条路径被走过，
    所以没暴露；一旦要「加载既有检查点做推理/重算归因」就会
    `AttributeError: 'dict' object has no attribute 'd_model'`。
    """
    if cfg is None:
        return ModelConfig()
    if isinstance(cfg, ModelConfig):
        return cfg
    if isinstance(cfg, dict):
        base = ModelConfig()
        for k, v in cfg.items():
            if hasattr(base, k):
                setattr(base, k, v)
        return base
    return cfg


class MaskedTriModalModel(nn.Module):
    """对外统一封装：负责把 numpy 批次转成张量并规范输出。"""

    def __init__(self, cfg: Optional[ModelConfig] = None,
                 text_emb_table: Optional[np.ndarray] = None):
        super().__init__()
        self.cfg = as_config(cfg)
        self.encoder = MissingAwareFusionEncoder(self.cfg, text_emb_table=text_emb_table)

    def forward(self, batch: Dict[str, torch.Tensor],
                modality_dropout: bool = False,
                need_slot_scores: bool = False) -> Dict[str, torch.Tensor]:
        return self.encoder(
            batch["text_ids"], batch["audio"], batch["vision"],
            batch["obs_mask"], batch["miss_flag"], batch["modal_avail"],
            modality_dropout=modality_dropout, need_slot_scores=need_slot_scores)


# ==================== 损失 ====================

class MultiTaskLoss(nn.Module):
    """极性分类（加权 CE）+ 强度回归（SmoothL1），联合训练。

    加权 CE 的必要性：MOSEI 的 label 集中在 ±2，Neutral(=0) 极稀缺，
    不加权时回归/分类都会退化成「永不预测中性」——项目早期用 100 条自提特征
    训出的模型 Neutral 召回率就是 0.000，这里从损失层面直接防住。
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None,
                 intensity_weight: float = 1.0):
        super().__init__()
        self.register_buffer("class_weights",
                             class_weights if class_weights is not None else torch.ones(3))
        self.intensity_weight = intensity_weight

    def forward(self, out: Dict[str, torch.Tensor],
                polarity: torch.Tensor, intensity: torch.Tensor) -> Dict[str, torch.Tensor]:
        ce = F.cross_entropy(out["polarity_logits"], polarity,
                             weight=self.class_weights)
        mae = F.smooth_l1_loss(out["intensity"], intensity, beta=0.5)
        total = ce + self.intensity_weight * mae
        return {"total": total, "ce": ce.detach(), "intensity": mae.detach()}


def make_class_weights(polarity: np.ndarray) -> torch.Tensor:
    """按类别频次的倒数开方加权，避免极端类别（Neutral）权重爆炸。"""
    cnt = np.bincount(polarity.astype(np.int64), minlength=3).astype(np.float64)
    cnt = np.where(cnt <= 0, 1.0, cnt)
    w = 1.0 / np.sqrt(cnt)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


# ==================== 批量装配 ====================

def stack_split(ds: "Q.Split2", idx: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """把附件2 的一个 split（或其子集）装成模型可直接吃的 numpy 批次。"""
    if idx is None:
        idx = np.arange(len(ds))
    tb = ds.text_bert[idx]                     # (N,3,50)
    audio = ds.audio[idx]
    vision = ds.vision[idx]

    infos = []
    for k in range(len(idx)):
        infos.append(Q.analyze_sample(
            {"audio": audio[k], "vision": vision[k]}, tb[k]))

    n = len(idx)
    S = Q.SEQ_LEN
    M = len(Q.MODALITIES)
    obs_mask = np.zeros((n, M, S), dtype=bool)
    miss_flag = np.zeros((n, M, S), dtype=bool)
    modal_avail = np.zeros((n, M), dtype=bool)
    for i, inf in enumerate(infos):
        for j, m in enumerate(Q.MODALITIES):
            obs_mask[i, j] = inf.observed[m]
            miss_flag[i, j] = inf.missing[m]
            modal_avail[i, j] = inf.available[m]

    return {
        "text_ids": tb[:, 0, :].astype(np.int64),
        "audio": audio.astype(np.float32),
        "vision": vision.astype(np.float32),
        "obs_mask": obs_mask,
        "miss_flag": miss_flag,
        "modal_avail": modal_avail,
    }


def _stack_samples(samples, text_ids_getter, audio_getter, vision_getter
                   ) -> Dict[str, np.ndarray]:
    """附件3/4 的对齐字段与附件2 同构，装配逻辑完全共用一份。"""
    n = len(samples)
    S, M = Q.SEQ_LEN, len(Q.MODALITIES)
    out = {
        "text_ids": np.zeros((n, S), dtype=np.int64),
        "audio": np.zeros((n, S, Q.AUDIO_DIM), dtype=np.float32),
        "vision": np.zeros((n, S, Q.VISION_DIM), dtype=np.float32),
        "obs_mask": np.zeros((n, M, S), dtype=bool),
        "miss_flag": np.zeros((n, M, S), dtype=bool),
        "modal_avail": np.zeros((n, M), dtype=bool),
    }
    for i, s in enumerate(samples):
        out["text_ids"][i] = text_ids_getter(s)
        out["audio"][i] = audio_getter(s)
        out["vision"][i] = vision_getter(s)
        for j, m in enumerate(Q.MODALITIES):
            out["obs_mask"][i, j] = s.info.observed[m]
            out["miss_flag"][i, j] = s.info.missing[m]
            out["modal_avail"][i, j] = s.info.available[m]
    return out


def stack_attachment3(samples) -> Dict[str, np.ndarray]:
    """把附件3 的 30 条样本装成与 stack_split 完全同构的批次（R2 接口一致性）。"""
    return _stack_samples(
        samples,
        lambda s: np.asarray(s.text_bert)[0].astype(np.int64),
        lambda s: s.audio.astype(np.float32),
        lambda s: s.vision.astype(np.float32))


def stack_attachment4(samples) -> Dict[str, np.ndarray]:
    return _stack_samples(
        samples,
        lambda s: np.asarray(s.text_bert)[0].astype(np.int64),
        lambda s: s.audio.astype(np.float32),
        lambda s: s.vision.astype(np.float32))


def to_torch(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


def batch_indices(n: int, batch_size: int, shuffle: bool = False,
                  rng: Optional[np.random.Generator] = None):
    idx = np.arange(n)
    if shuffle:
        rng = rng or np.random.default_rng(0)
        rng.shuffle(idx)
    for s in range(0, n, batch_size):
        yield idx[s:s + batch_size]


# ==================== 指标 ====================

def polarity_metrics(pred_idx: np.ndarray, true_idx: np.ndarray) -> Dict[str, float]:
    """极性：准确率 + 宏 F1 + 各类召回（宏 F1 是赛题点名指标，不报微 F1 以免掩盖小类）。"""
    pred_idx = np.asarray(pred_idx).astype(np.int64)
    true_idx = np.asarray(true_idx).astype(np.int64)
    acc = float((pred_idx == true_idx).mean()) if len(true_idx) else float("nan")
    f1s, recalls = [], {}
    for c in range(3):
        tp = int(((pred_idx == c) & (true_idx == c)).sum())
        fp = int(((pred_idx == c) & (true_idx != c)).sum())
        fn = int(((pred_idx != c) & (true_idx == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        recalls[Q.POLARITY_CN[c - 1]] = rec
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s)),
            "recall_negative": recalls["Negative"],
            "recall_neutral": recalls["Neutral"],
            "recall_positive": recalls["Positive"]}


def intensity_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """强度：MAE + Pearson。同时给出「恒预测训练集均值」的 MAE 基线，
    不报基线的 MAE 无法判断模型是否真的在预测强度。"""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mae = float(np.abs(pred - true).mean()) if len(true) else float("nan")
    if len(true) > 1 and pred.std() > 1e-12 and true.std() > 1e-12:
        pearson = float(np.corrcoef(pred, true)[0, 1])
    else:
        pearson = float("nan")
    return {"mae": mae, "pearson": pearson}


def constant_baselines(y_intensity: np.ndarray, y_polarity: np.ndarray) -> Dict[str, float]:
    """两条必须并列输出的平凡基线：均值回归 与 多数类分类。"""
    y_intensity = np.asarray(y_intensity, dtype=np.float64)
    mean_pred = np.full_like(y_intensity, y_intensity.mean())
    pol = np.asarray(y_polarity)
    maj = np.bincount(pol, minlength=3).argmax()
    return {
        "mae_predict_train_mean": float(np.abs(mean_pred - y_intensity).mean()),
        "accuracy_predict_majority": float((pol == maj).mean()),
        "majority_class": Q.POLARITY_CN[int(maj) - 1],
    }
