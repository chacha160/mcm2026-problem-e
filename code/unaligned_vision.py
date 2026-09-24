# -*- coding: utf-8 -*-
"""
unaligned_vision.py —— 视觉模态「未对齐」特征提取

职责边界：
    本文件只负责「视频 → 逐帧人脸深度特征 (T, 35) + 每帧实测时间戳 pts」，
    **不做任何时间槽分配**。对齐见 align_multimodal.py。

与已删除的上一版提取器（原 extractors/vision_extractor.py，已随 extractors/ 目录一并移除）
的差异
--------------------------------------------------------------------------------
  1) 抽帧：流式逐帧 + 真实 PTS 触发（内存与视频长度无关），而不是
     `utils.safe_read_video_frames()` 的「先把整段视频所有帧读进内存再按名义 fps 抽」。
     上一版做法在 720p 下每片段要吃约 0.7 GB，且名义 fps 与实际时间轴不一致。
  2) 帧率：15 Hz。依据是附件2 的 500 帧上限 ÷ 最长片段 25 s ≈ 20 Hz，
     但实测 18 条与官方重叠的样本，视觉帧数中位数对应 14.7 Hz，故取 15 Hz。
  3) 每帧带实测 pts，对齐时按「秒」重采样而非按数组下标。
  4) 缺失人脸：按相邻有效帧线性插值补齐，并逐帧记录 face_flags 供核对。
     「补帧」是视觉模态自身的属性（人脸检测会漏），不属于对齐步骤，
     因此放在本模块完成，使对齐模块对模态无感知。

特征口径（与既有流水线严格一致，保证「未对齐 → 对齐」可与「直接对齐」对照）
    MTCNN(image_size=160) 裁出人脸 → PIL Resize(224) → ToTensor → ImageNet 归一化
    → ResNet-50 全局平均池化 2048 维 → 固定随机投影降到 35 维

    随机投影矩阵 W ∈ R^{2048×35}，元素 ~ N(0, 1/√2048)，seed = config.VISION_PROJECTION_SEED(42)。
    该缩放使 E||Wᵀx||² = ||x||²，配合 JL 引理近似保距；矩阵 md5 随文件落盘，
    任何一次运行都能验证「用的是不是同一个投影」。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import config  # noqa: E402
import unaligned_common as U  # noqa: E402
from utils import LOGGER  # noqa: E402

MODALITY = "vision"

# 官方视觉采样率：500 帧上限 ÷ 25 s 最长片段 = 20 Hz；实测与官方重叠样本的中位数为 14.7 Hz
OFFICIAL_RATE_HZ = 15.0


def build_projection_matrix(raw_dim: int, out_dim: int, seed: int) -> np.ndarray:
    """
    构造固定随机投影矩阵（与已删除的上一版 vision_extractor.py 同式同种子，
    因此两版产出的 35 维特征在数值上可比）。

    行列式确定性由 seed 保证：同一 seed 在任何机器、任何 numpy 版本下都会得到
    完全相同的 default_rng 序列，从而保证 100 条样本共用同一个投影、结果可复现。
    """
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(raw_dim)
    return rng.normal(0.0, scale, size=(raw_dim, out_dim)).astype(np.float32)


def matrix_md5(mat: np.ndarray) -> str:
    """投影矩阵指纹：用于证明两次运行用的是同一个矩阵（而不是靠嘴保证可复现）。"""
    return hashlib.md5(np.ascontiguousarray(mat, dtype=np.float32).tobytes()).hexdigest()


def _fill_missing_faces(features: np.ndarray, face_flags: np.ndarray) -> np.ndarray:
    """
    对未检测到人脸的帧，按相邻有效帧线性插值补齐。

    只在「有效帧 ≥ 2」时插值；只有 1 帧有效则整体复制该帧（保住该模态的尺度）；
    一帧都没有则保持零向量，由 meta.face_ratio=0 明确标记该样本视觉不可用。
    """
    valid = np.flatnonzero(face_flags)
    if len(valid) == 0 or len(valid) == features.shape[0]:
        return features
    idx = np.arange(features.shape[0])
    if len(valid) == 1:
        features[:] = features[valid[0]]
        return features
    for d in range(features.shape[1]):
        features[:, d] = np.interp(idx, valid, features[valid, d])
    return features


class UnalignedVisionExtractor:
    """逐帧 35 维人脸特征提取器（不对齐）。"""

    def __init__(self, fps: float = OFFICIAL_RATE_HZ,
                 feature_dim: Optional[int] = None,
                 raw_dim: Optional[int] = None,
                 projection_seed: Optional[int] = None,
                 face_size: Optional[int] = None,
                 device: Optional[str] = None) -> None:
        import torch
        from facenet_pytorch import MTCNN
        from torchvision.models import ResNet50_Weights, resnet50
        from torchvision import transforms as T

        self.torch = torch
        self.fps = float(fps)
        self.raw_dim = int(raw_dim or config.VISION_RAW_DIM)
        self.feature_dim = int(feature_dim or config.VISION_FEATURE_DIM)
        self.face_size = int(face_size or config.VIDEO_FACE_SIZE)
        self.projection_seed = int(projection_seed if projection_seed is not None
                                   else config.VISION_PROJECTION_SEED)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.mtcnn = MTCNN(image_size=self.face_size, margin=0, min_face_size=20,
                           thresholds=[0.6, 0.7, 0.7], factor=0.709,
                           post_process=False, keep_all=False, device=self.device)

        res = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.body = torch.nn.Sequential(*list(res.children())[:-1]).to(self.device).eval()
        # 与上一版 vision_extractor 一致的预处理链（MTCNN(160) 裁剪 → Resize(224)
        # → ImageNet 归一化 → ResNet-50 池化），保证跨版本特征可比
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.projection = build_projection_matrix(self.raw_dim, self.feature_dim, self.projection_seed)
        self.projection_md5 = matrix_md5(self.projection)
        LOGGER.info("[视觉] 逐帧 %d 维（ResNet-50 %d 维 → 随机投影，seed=%d, md5=%s）",
                    self.feature_dim, self.raw_dim, self.projection_seed, self.projection_md5[:8])
        LOGGER.info("[视觉] 采样率 %.1f Hz，人脸尺寸 %d，device=%s",
                    self.fps, self.face_size, self.device)

    # ---------- 特征 ----------
    def extract_frames(self, video_path: str
                       ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object], Dict[str, object]]:
        """
        视频 → (features (T,35) float32, pts (T,), meta, aux)

        aux 含逐帧 face_flags（该帧是否检出人脸），供 QC 与时间轴可视化核对；
        它属于「附带记录」而非特征本身，故走 extra 通道而不是塞进 features。

        逐帧流式处理：任何时刻内存中只有一帧 + 一个 2048 维向量。
        """
        import cv2

        pts_list: List[float] = []
        raw_list: List[np.ndarray] = []
        flags: List[bool] = []
        probs: List[float] = []

        for pts, frame_bgr in U.iter_video_frames(video_path, target_fps=self.fps):
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            with self.torch.no_grad():
                out = self.mtcnn(rgb, return_prob=True)
            face, prob = out if isinstance(out, tuple) else (out, None)

            pts_list.append(float(pts))
            if face is None:
                flags.append(False)
                probs.append(0.0)
                raw_list.append(np.zeros(self.raw_dim, np.float32))
                continue

            face_img = face.permute(1, 2, 0).cpu().numpy()
            face_img = np.clip(face_img, 0, 255).astype(np.uint8)
            x = self.transform(face_img).unsqueeze(0).to(self.device)
            with self.torch.no_grad():
                v = self.body(x).flatten().cpu().numpy().astype(np.float32)
            flags.append(True)
            probs.append(float(prob) if prob is not None else 1.0)
            raw_list.append(v)

        T_ = len(pts_list)
        if T_ == 0:
            return (np.zeros((0, self.feature_dim), np.float32), np.zeros((0,), np.float32),
                    {"num_frames": 0, "empty_video": True}, {"face_flags": []})

        raw = np.stack(raw_list, axis=0) if raw_list else np.zeros((0, self.raw_dim), np.float32)
        face_flags = np.array(flags, dtype=bool)
        raw = _fill_missing_faces(raw, face_flags)
        feats = (raw @ self.projection).astype(np.float32)
        pts = np.asarray(pts_list, dtype=np.float32)

        meta = {
            "num_frames": int(T_),
            "target_fps": self.fps,
            "realized_fps": round(T_ / max(float(pts[-1]), 1e-6), 3),
            "raw_dim": self.raw_dim,
            "feature_dim": self.feature_dim,
            "face_size": self.face_size,
            "face_detected_frames": int(face_flags.sum()),
            "face_ratio": round(float(face_flags.mean()), 4),
            "face_prob_mean": round(float(np.mean(probs)), 4) if probs else 0.0,
            "missing_face_policy": "按相邻有效帧线性插值补齐；face_ratio=0 的样本保持零向量",
            "projection": {
                "method": "fixed_random_projection (Johnson-Lindenstrauss)",
                "seed": self.projection_seed,
                "matrix_shape": [self.raw_dim, self.feature_dim],
                "matrix_md5": self.projection_md5,
                "scale": round(1.0 / float(np.sqrt(self.raw_dim)), 10),
            },
            "frame_pts_source": "cv2 CAP_PROP_POS_MSEC（解码器实际时间戳，非等间隔）",
            "duration_last_pts": round(float(pts[-1]), 4),
        }
        return feats, pts, meta, {"face_flags": [int(b) for b in face_flags]}


# ==================== 单样本抽取 ====================

def extract_one(extractor: UnalignedVisionExtractor, sample: Dict[str, object]
                ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object], Dict[str, object]]:
    """对一条样本抽视觉特征。返回 (features, pts, meta, extra)。"""
    video_path = str(sample["video_path"])
    probe = U.probe_streams(video_path)
    feats, pts, meta, aux = extractor.extract_frames(video_path)

    meta = {
        "modality": MODALITY,
        "unit_level": "frame",
        "alignment_rule": "无（本文件不做时间槽分配，对齐见 align_multimodal.py）",
        "time_basis": "measured_pts",
        "time_basis_desc": "pts 由解码器逐帧给出（CAP_PROP_POS_MSEC），允许不均匀",
        "duration_used": round(float(probe.get("video_duration") or 0.0), 4),
        "container_video_duration": round(float(probe.get("video_duration") or 0.0), 4),
        "container_audio_duration": round(float(probe.get("audio_duration") or 0.0), 4),
        "width": probe.get("width"),
        "height": probe.get("height"),
        "nominal_fps": probe.get("nominal_fps"),
        "video_codec": probe.get("video_codec"),
        "pipeline_version": config.PIPELINE_VERSION,
        "code_fingerprint": U.module_fingerprint(UnalignedVisionExtractor.extract_frames),
        **meta,
    }
    # 帧间隔：供 QC 核对抽帧是否均匀（VFR 视频会体现在这里）
    gaps = np.diff(pts).tolist() if len(pts) > 1 else []
    extra = {
        "frame_gaps": [round(float(g), 5) for g in gaps],
        "face_flags": aux.get("face_flags", []),
    }
    return feats, pts, meta, extra


def run(samples: List[Dict[str, object]], out_dir: str, overwrite: bool = False,
        fps: float = OFFICIAL_RATE_HZ, feature_dim: Optional[int] = None,
        device: Optional[str] = None) -> List[Dict[str, object]]:
    """批量抽取视觉未对齐特征。返回汇总行列表。"""
    os.makedirs(out_dir, exist_ok=True)
    extractor = UnalignedVisionExtractor(fps=fps, feature_dim=feature_dim, device=device)

    rows: List[Dict[str, object]] = []
    n_ok = n_skip = n_fail = 0
    for i, s in enumerate(samples, 1):
        sid = str(s["sample_id"])
        out_path = os.path.join(out_dir, f"{sid}.npz")
        if os.path.exists(out_path) and not overwrite:
            n_skip += 1
            continue
        if not os.path.isfile(str(s["video_path"])):
            LOGGER.warning("[视觉] (%d/%d) 找不到视频: %s", i, len(samples), s["video_path"])
            n_fail += 1
            continue
        try:
            feats, pts, meta, extra = extract_one(extractor, s)
            U.save_unaligned(out_path, feats, pts, MODALITY, meta, extra=extra)
            n_ok += 1
            LOGGER.info("[视觉] (%d/%d) %s -> (T=%d, D=%d) 人脸率=%.2f 实测帧率=%.2f 时长=%.2fs",
                        i, len(samples), sid, feats.shape[0], feats.shape[1],
                        float(meta.get("face_ratio") or 0.0),
                        float(meta.get("realized_fps") or 0.0),
                        float(meta.get("duration_last_pts") or 0.0))
            rows.append({
                "sample_id": sid, "official_id": s["official_id"], "modality": MODALITY,
                "n_units": int(feats.shape[0]), "feature_dim": int(feats.shape[1]),
                "duration_sec": round(float(meta.get("duration_last_pts") or 0.0), 3),
                "face_ratio": float(meta.get("face_ratio") or 0.0),
                "time_basis": "measured_pts", "out_path": out_path, "status": "ok",
            })
        except Exception as exc:
            n_fail += 1
            LOGGER.error("[视觉] (%d/%d) 失败 %s: %s: %s", i, len(samples), sid, type(exc).__name__, exc)

    LOGGER.info("[视觉] 完成：成功 %d / 跳过 %d / 失败 %d（输出目录 %s）", n_ok, n_skip, n_fail, out_dir)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="视觉模态未对齐特征提取（逐帧人脸特征，不做时间对齐）")
    p.add_argument("--only", default=None, help="只处理 sample_id 含该子串的样本")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 条")
    p.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出")
    p.add_argument("--out", default=None, help="输出目录，默认 data/unaligned_features/vision")
    p.add_argument("--fps", type=float, default=OFFICIAL_RATE_HZ, help="采样帧率，默认 15")
    p.add_argument("--dim", type=int, default=None, help="输出维度，默认 35（与附件2 一致）")
    p.add_argument("--threads", type=int, default=None, help="torch 线程数（CPU 推理提速用）")
    args = p.parse_args(argv)

    if args.threads:
        import torch
        torch.set_num_threads(int(args.threads))

    samples = U.list_samples(only=args.only, limit=args.limit)
    LOGGER.info("[视觉] 待处理样本 %d 条", len(samples))
    out_dir = args.out or U.unaligned_dir(MODALITY)
    run(samples, out_dir, overwrite=args.overwrite, fps=args.fps, feature_dim=args.dim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
