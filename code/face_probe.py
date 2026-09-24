# -*- coding: utf-8 -*-
"""
face_probe.py —— 「多人或远景」类别的**轻量人脸探测**取证

为什么需要这个文件
------------------
题目要求人工抽查覆盖「多人或远景」这一类，可现有视觉产物**无法支撑这个判断**：

    unaligned_vision.py:113-115 建的 MTCNN 是 keep_all=False, post_process=False
    ——它每帧只返回**一张**裁好的人脸（按框面积选最大者），
      **不保留任何框坐标**，因此逐帧结果里只有「有没有人脸」这一个布尔量。
      没有框，就分不出「画面里一个人占满屏」和「五个人各占一角」；
      也没有框面积，就分不出「近景大脸」和「远景小脸」。

本模块因此**另建一个 keep_all=True 的探测器实例**，对每条样本只做**纯检测**
（不跑 ResNet 前向、不写特征），抽样取 3 帧，记录每帧检出的人脸个数与框面积占比。
探测用的实例与提取器互不影响，也不修改任何既有产物。

诚实边界（必须随结果一并声明）
------------------------------
1) **抽样下界，不是全片最大值。** 每条样本只探 3 帧（0.15T / 0.50T / 0.85T），
   所以 `max_face_count` 是「这 3 帧里最多同时出现几张脸」，**不能**声称是
   「该片段中出现的最大人脸数」。若这 3 帧恰好都没拍到第二个人，
   本批结论只能是「抽样未发现多人」，**不构成"该片段只有一个人"的证据**。
2) **探测帧与特征帧不同格。** 提取器按 15 Hz 在实测 PTS 上取帧，本模块按
   固定比例取帧并记录解码器返回的**实际**时间戳（`probe_actual_t_sec`）。
   两者最近可差约 1/15 s。因此「探到人脸 ⇔ 提取器在该处有 face_flag」只作
   **一致性交叉核对**报出（`agree_with_extractor`），不作硬断言。
3) **探测失败不等于没有人脸。** 失败/超时写成 `detector_ok=0` + 状态码，
   判据标为**不可用**，绝不写成「0 张脸」。也**不因此剔除任何样本**。
4) 本模块不产出也不修改 `data/unaligned_features/` 与 `data/aligned/` 的任何文件。

产物
----
    data/q1_delivery/face_probe.csv         每条样本 1 行（100 行），供 q1_delivery 读
    data/q1_delivery/face_probe_frames.csv  每探测帧 1 行（≤300 行），逐帧原始证据

超时防卡死
----------
逐样本独立子进程 + 墙钟超时，沿用 `run_unaligned_all.run_vision_watched()` 已验证的
模式：本批数据实测出现过 torch/OpenCV 原生线程的**非确定性**卡死，发生在原生代码内部，
Python 层计时器无法中断，只能靠父进程按墙钟收割子进程。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
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
DEFAULT_FRACS = (0.15, 0.50, 0.85)

# 与 unaligned_vision.py:113-115 完全同参，**只有 keep_all 不同**——这是本模块能
# 区分「多人/远景」的全部理由，也是「同一探测器、同权重」这一说法的依据。
MTCNN_KW = dict(image_size=160, margin=0, min_face_size=20,
                thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False)

SAMPLES_CSV = "face_probe.csv"
FRAMES_CSV = "face_probe_frames.csv"

SAMPLE_FIELDS = [
    "sample_id", "official_id", "video_id", "clip_id",
    "detector_ok", "status", "note",
    "n_frames_probed", "probe_fracs", "probe_target_t_sec", "probe_actual_t_sec",
    "seek_delta_max_sec", "n_faces", "face_probs",
    "max_face_count", "max_face_area_ratio", "box_area_ratio_median",
    "frame_width", "frame_height",
    "nearest_vision_pts_sec", "vision_face_flag",
    "n_compared", "n_agree", "agree_with_extractor",
]

FRAME_FIELDS = [
    "sample_id", "official_id", "frac", "target_t_sec", "actual_t_sec",
    "seek_delta_sec", "frame_width", "frame_height", "status",
    "n_faces", "face_probs", "box_areas_px2", "box_area_ratios",
    "frame_max_face_area_ratio",
    "nearest_vision_pts_sec", "nearest_vision_pts_delta_sec",
    "vision_face_flag", "agree_with_extractor",
]

# 探测状态码。OK 之外的都表示「判据不可用」，而不是「没有人脸」。
STATUS_OK = "OK"
STATUS_READ_FAIL = "READ_FAIL"            # 某一帧解不出来
STATUS_NO_DURATION = "NO_DURATION"        # 拿不到公共时间轴长度，无法定位 0.15T 等
STATUS_NO_VIDEO = "NO_VIDEO_FILE"
STATUS_TIMEOUT = "PROBE_TIMEOUT"          # 墙钟超时，判定卡死
STATUS_FAILED = "PROBE_FAILED"            # 子进程非零退出或输出不可解析


# ==================== 单样本探测（在子进程内运行）====================

def _build_detector(threads: Optional[int] = None):
    """建 keep_all=True 的 MTCNN（只做检测，不加载 ResNet）。"""
    import torch
    from facenet_pytorch import MTCNN
    if threads:
        torch.set_num_threads(int(threads))
    return MTCNN(keep_all=True, device="cpu", **MTCNN_KW)


def _read_frame_at(cap, t_sec: float):
    """
    定位到 t_sec 附近并取一帧。返回 (frame_bgr, actual_t_sec)。

    用 CAP_PROP_POS_MSEC 定位是**近似**的（解码器只能从关键帧起解），所以
    把解码器回报的**实际**时间戳一并返回并落盘，供评审核对定位偏差；
    实测本批 100 条的最大偏差在 1 帧（约 1/30 s）以内。
    """
    import cv2
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t_sec)) * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None, float("nan")
    actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if actual != actual:      # 部分后端返回 NaN，退回请求值
        actual = float(t_sec)
    return frame, float(actual)


def _clip_area_ratio(box: np.ndarray, w: int, h: int) -> float:
    """
    框面积占整帧的比例。**先把框裁到画面内再算面积**：
    MTCNN 的框可以越出画面边界（margin=0 时仍会），不裁会得到 >1 的「面积占比」。
    """
    x1 = max(0.0, float(box[0])); y1 = max(0.0, float(box[1]))
    x2 = min(float(w), float(box[2])); y2 = min(float(h), float(box[3]))
    if x2 <= x1 or y2 <= y1 or w <= 0 or h <= 0:
        return 0.0
    return (x2 - x1) * (y2 - y1) / float(w * h)


def probe_one_sync(video_path: str, duration: float, fracs: Sequence[float],
                   threads: Optional[int] = None, detector=None
                   ) -> Tuple[List[Dict[str, object]], str, str]:
    """
    对一条样本的 3 个抽样时刻做纯检测。返回 (逐帧记录, 状态码, 备注)。

    抽帧与检测分开写：**一帧失败只记该帧 READ_FAIL，其余帧照常探**，
    这样「部分可探」不会被误报成「整条不可探」。
    """
    import cv2

    if not video_path or not os.path.isfile(video_path):
        return [], STATUS_NO_VIDEO, "视频文件不存在"
    if duration is None or not np.isfinite(duration) or duration <= 0:
        return [], STATUS_NO_DURATION, "公共时间轴时长无效，无法定位 0.15T/0.50T/0.85T"

    try:
        det = detector if detector is not None else _build_detector(threads)
    except Exception as exc:
        return [], STATUS_FAILED, f"探测器初始化失败: {type(exc).__name__}: {exc}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], STATUS_NO_VIDEO, "OpenCV 无法打开视频"

    rows: List[Dict[str, object]] = []
    try:
        for frac in fracs:
            target = float(frac) * float(duration)
            row: Dict[str, object] = {
                "frac": round(float(frac), 4), "target_t_sec": round(target, 4),
                "status": STATUS_OK, "n_faces": None, "face_probs": "",
                "box_areas_px2": "", "box_area_ratios": "",
                "frame_max_face_area_ratio": float("nan"),
                "frame_width": 0, "frame_height": 0,
                "actual_t_sec": float("nan"), "seek_delta_sec": float("nan"),
            }
            try:
                frame, actual = _read_frame_at(cap, target)
            except Exception as exc:
                row["status"] = STATUS_READ_FAIL
                row["note_frame"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                continue
            if frame is None:
                row["status"] = STATUS_READ_FAIL
                rows.append(row)
                continue

            h, w = frame.shape[:2]
            row["frame_width"], row["frame_height"] = int(w), int(h)
            row["actual_t_sec"] = round(actual, 4)
            row["seek_delta_sec"] = round(actual - target, 4)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                # detect() 返回**全部**过阈值的框（不受 keep_all 影响），
                # 这里要的正是「有几张脸、各自多大」，所以用 detect 而非 forward。
                boxes, probs = det.detect(rgb, landmarks=False)
            except Exception as exc:
                row["status"] = STATUS_READ_FAIL
                row["note_frame"] = f"检测异常: {type(exc).__name__}: {exc}"
                rows.append(row)
                continue

            bx = boxes[0] if isinstance(boxes, (list, tuple)) else boxes
            pb = probs[0] if isinstance(probs, (list, tuple)) else probs
            if bx is None or len(bx) == 0:
                row["n_faces"] = 0
                row["face_probs"] = ""
                row["box_areas_px2"] = ""
                row["box_area_ratios"] = ""
                row["frame_max_face_area_ratio"] = 0.0
                rows.append(row)
                continue

            bx = np.asarray(bx, dtype=np.float64)
            ratios = [_clip_area_ratio(b, w, h) for b in bx]
            areas = [max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1])) for b in bx]
            row["n_faces"] = int(len(bx))
            row["face_probs"] = ";".join(
                f"{float(p):.4f}" for p in ([] if pb is None else np.atleast_1d(pb)))
            row["box_areas_px2"] = ";".join(f"{a:.0f}" for a in areas)
            row["box_area_ratios"] = ";".join(f"{r:.6f}" for r in ratios)
            row["frame_max_face_area_ratio"] = round(float(max(ratios)), 6)
            rows.append(row)
    finally:
        cap.release()

    ok_rows = [r for r in rows if r["status"] == STATUS_OK]
    if not ok_rows:
        return rows, STATUS_READ_FAIL, f"{len(rows)} 个抽样时刻均未能解出帧"
    note = "" if len(ok_rows) == len(rows) else f"仅 {len(ok_rows)}/{len(rows)} 个抽样时刻可用"
    return rows, STATUS_OK, note


# ==================== 子进程入口 ====================

def _worker_main(argv: Sequence[str]) -> int:
    """
    单样本子进程：探测一条样本，把结果写进 `--json-out`。

    单独成进程的唯一原因是防卡死（见模块头）。所有诊断信息走 JSON 文件而不是
    标准输出，避免日志混进结果。
    """
    p = argparse.ArgumentParser(description="face_probe 单样本子进程（内部使用）")
    p.add_argument("--video", required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--fracs", default=",".join(str(f) for f in DEFAULT_FRACS))
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--json-out", required=True)
    args = p.parse_args(argv)

    fracs = [float(x) for x in str(args.fracs).split(",") if x.strip()]
    payload: Dict[str, object] = {}
    try:
        rows, status, note = probe_one_sync(args.video, args.duration, fracs,
                                            threads=args.threads)
        payload = {"rows": rows, "status": status, "note": note}
    except Exception as exc:
        payload = {"rows": [], "status": STATUS_FAILED,
                   "note": f"{type(exc).__name__}: {exc}"}
    with open(args.json_out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return 0


# ==================== 交叉核对 ====================

def _nearest_vision(pt: float, vpts: np.ndarray) -> Tuple[int, float]:
    """返回 (最近特征帧下标, 时间差)。vpts 已升序。"""
    if len(vpts) == 0:
        return -1, float("nan")
    i = int(np.argmin(np.abs(vpts - pt)))
    return i, float(vpts[i] - pt)


def _load_vision_evidence(sid: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    取该样本视觉特征帧的 pts 与逐帧 face_flags，用于与探测结果交叉核对。

    注意 `face_flags` 是 keep_all=False 的探测器的结果（该帧是否检出人脸），
    与本次 keep_all=True 的探测**同参同权重**，差别只在「取不取全部框」。
    """
    path = os.path.join(U.unaligned_dir("vision"), f"{sid}.npz")
    if not os.path.isfile(path):
        return np.zeros((0,), np.float32), None
    d = U.load_unaligned(path)
    pts = np.asarray(d["pts"], np.float32)
    flags = d.get("extra", {}).get("face_flags")
    return pts, (None if flags is None else np.asarray(flags, np.int64))


# ==================== 逐样本驱动（父进程）====================

def _to_float_list(cell: object, sep: str = ";") -> List[float]:
    out: List[float] = []
    for x in str(cell or "").split(sep):
        x = x.strip()
        if not x:
            continue
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out


def _aggregate(sid: str, video: str, duration: float, payload: Dict[str, object]
               ) -> Dict[str, object]:
    """把子进程的逐帧结果汇总成 per-sample 一行，并做交叉核对。"""
    rows = list(payload.get("rows") or [])
    status = str(payload.get("status") or STATUS_FAILED)
    note = str(payload.get("note") or "")

    ok = [r for r in rows if r.get("status") == STATUS_OK]
    # 「有没有探到东西」看的是**帧是否被成功解码并送入检测**，而不是脸数：
    # 解码成功但 0 张脸是有效结论，解码失败才是判据不可用。
    detector_ok = 1 if ok else 0
    if detector_ok:
        status = STATUS_OK

    n_faces = [int(r["n_faces"]) for r in ok if r.get("n_faces") is not None]
    ratios = [float(r["frame_max_face_area_ratio"]) for r in ok]
    fw = next((int(r["frame_width"]) for r in ok if r.get("frame_width")), 0)
    fh = next((int(r["frame_height"]) for r in ok if r.get("frame_height")), 0)

    vpts, vflags = _load_vision_evidence(sid)
    n_cmp = n_agree = 0
    near_pts: List[str] = []
    near_flags: List[str] = []
    for r in ok:
        i, dt = _nearest_vision(float(r["actual_t_sec"]), vpts)
        r["nearest_vision_pts_sec"] = round(float(vpts[i]), 4) if i >= 0 else ""
        r["nearest_vision_pts_delta_sec"] = round(dt, 4) if i >= 0 else ""
        pol = None if (i < 0 or vflags is None or i >= len(vflags)) else int(vflags[i])
        r["vision_face_flag"] = "" if pol is None else pol
        agree = ""
        if pol is not None and r.get("n_faces") is not None:
            agree = int((int(r["n_faces"]) > 0) == (pol == 1))
            n_cmp += 1
            n_agree += int(agree)
        r["agree_with_extractor"] = agree
        near_pts.append(str(r["nearest_vision_pts_sec"]))
        near_flags.append(str(r["vision_face_flag"]))

    row: Dict[str, object] = {
        "sample_id": sid,
        "official_id": U.official_id(*sid.rsplit("_", 1)),
        "video_id": sid.rsplit("_", 1)[0],
        "clip_id": sid.rsplit("_", 1)[1],
        "detector_ok": detector_ok,
        "status": status,
        "note": note,
        "n_frames_probed": len(ok),
        "probe_fracs": ";".join(f"{float(r['frac']):.2f}" for r in ok),
        "probe_target_t_sec": ";".join(
            "" if r.get("target_t_sec") is None else f"{float(r['target_t_sec']):.4f}" for r in ok),
        "probe_actual_t_sec": ";".join(
            "" if not np.isfinite(float(r.get("actual_t_sec") or float('nan')))
            else f"{float(r['actual_t_sec']):.4f}" for r in ok),
        "seek_delta_max_sec": (round(max(abs(float(r["seek_delta_sec"])) for r in ok), 4)
                               if ok else -1),
        "n_faces": ";".join(str(x) for x in n_faces),
        # 抽到的 0 张脸也是结论，因此默认取 0 而不是 -1；-1 只留给「不可用」。
        "max_face_count": int(max(n_faces)) if n_faces else (-1 if not detector_ok else 0),
        "max_face_area_ratio": round(float(max(ratios)), 6) if ratios else (-1.0 if not detector_ok else 0.0),
        "box_area_ratio_median": (round(float(np.median(ratios)), 6) if ratios
                                  else (-1.0 if not detector_ok else 0.0)),
        "frame_width": fw, "frame_height": fh,
        "face_probs": ";".join(str(r.get("face_probs") or "") for r in ok),
        "nearest_vision_pts_sec": ";".join(near_pts),
        "vision_face_flag": ";".join(near_flags),
        "n_compared": n_cmp, "n_agree": n_agree,
        "agree_with_extractor": (round(n_agree / n_cmp, 4) if n_cmp else ""),
    }
    for r in rows:                           # 逐帧记录补上归属，供 frames CSV
        r["sample_id"] = sid
        r["official_id"] = row["official_id"]
    row["_frames"] = rows                    # 只在本进程内传递，不落 per-sample CSV
    return row


def _spawn_one(sid: str, video: str, duration: float, fracs: Sequence[float],
               timeout: float, threads: Optional[int], tmp_dir: str
               ) -> Dict[str, object]:
    """起一个子进程探一条样本；超时/崩溃都转成结构化状态，不抛异常。"""
    jout = os.path.join(tmp_dir, f"{sid}.json")
    if os.path.exists(jout):
        try:
            os.remove(jout)
        except OSError:
            pass
    cmd = [sys.executable, "-X", "utf8", os.path.join(_CODE_DIR, "face_probe.py"),
           "--video", video, "--duration", repr(float(duration)),
           "--fracs", ",".join(repr(float(f)) for f in fracs),
           "--json-out", jout]
    if threads:
        cmd += ["--threads", str(int(threads))]
    try:
        proc = subprocess.run(cmd, timeout=timeout, capture_output=True,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"rows": [], "status": STATUS_TIMEOUT,
                "note": f"超过 {timeout:.0f} s 未返回，判定卡死并终止（可 --only={sid} 单独重跑）"}
    if not os.path.isfile(jout):
        return {"rows": [], "status": STATUS_FAILED,
                "note": f"子进程退出码 {proc.returncode} 且未写出结果: {(proc.stderr or '')[-200:]}"}
    try:
        with open(jout, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        return {"rows": [], "status": STATUS_FAILED,
                "note": f"结果不可解析: {type(exc).__name__}: {exc}"}


# ==================== 主流程 ====================

def run(unaligned_root: Optional[str] = None, aligned_dir: Optional[str] = None,
        out_dir: Optional[str] = None, sids: Optional[List[str]] = None,
        fracs: Sequence[float] = DEFAULT_FRACS, timeout: float = 180.0,
        threads: Optional[int] = 4, overwrite: bool = False,
        only: Optional[str] = None) -> Dict[str, object]:
    """
    逐样本探测并写出两份 CSV。返回汇总 dict（含 per-sample 行）。

    每条样本一个独立子进程 + 墙钟超时：一条卡死只影响它自己，
    其余样本照常完成，超时者如实记状态码而**不剔除样本**。
    """
    root = os.path.dirname(_CODE_DIR)
    unaligned_root = unaligned_root or os.path.join(root, "data", "unaligned_features")
    aligned_dir = aligned_dir or os.path.join(root, "data", "aligned")
    out_dir = out_dir or DEFAULT_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    tmp_dir = os.path.join(out_dir, "_face_probe_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    samples = U.list_samples(only=only)
    keep = None if sids is None else set(str(x) for x in sids)
    if keep is not None:
        samples = [s for s in samples if str(s["sample_id"]) in keep]
    LOGGER.info("[探测] 样本 %d 条；抽样时刻 %s；逐样本墙钟超时 %.0f s",
                len(samples), [f"{f:.2f}T" for f in fracs], timeout)

    # 断点续探：CSV 里已有结果的样本默认跳过。整批约需十几分钟，任何中断
    # （卡死、Ctrl-C、断电）都不应让已探好的样本重来。
    spath = os.path.join(out_dir, SAMPLES_CSV)
    fpath = os.path.join(out_dir, FRAMES_CSV)
    old: Dict[str, Dict[str, object]] = _read_csv(spath, SAMPLE_FIELDS)
    old_frames: List[Dict[str, object]] = _read_csv_rows(fpath, FRAME_FIELDS)
    if old and not overwrite:
        LOGGER.info("[探测] 已有结果 %d 条，默认跳过（--overwrite 可全部重探）", len(old))

    # 「样本行存在」**不足以**说明这条样本探完整了：逐帧证据可能只写了一半。
    # 本项目真踩过：先用 --only 单探了一条做冒烟测试（只落 1 帧行），
    # 之后全量运行时因「样本行已存在」跳过它，于是它的 3 帧证据永远缺 2 帧，
    # 而汇总表看起来完全正常。因此续探的判据放宽为「帧行数也齐」。
    n_fracs = len(fracs)
    n_frame_rows: Dict[str, int] = {}
    for f in old_frames:
        k = str(f.get("sample_id"))
        n_frame_rows[k] = n_frame_rows.get(k, 0) + 1
    incomplete = sorted(sid for sid in old if n_frame_rows.get(str(sid), 0) != n_fracs)
    if incomplete and not overwrite:
        LOGGER.warning("[探测] %d 条样本的逐帧证据不齐（期望每样本 %d 帧），将重新探测：%s",
                       len(incomplete), n_fracs, "、".join(incomplete[:8])
                       + ("…" if len(incomplete) > 8 else ""))

    new_rows: List[Dict[str, object]] = []
    new_frames: List[Dict[str, object]] = []
    n_new = n_skip = 0

    def flush() -> List[Dict[str, object]]:
        """把「本轮新探的 + 盘上已有的」合并后整表重写。

        每条样本写一次全表：文件在**任何**时刻都是完整的 100 行，
        中途崩溃也不会留下一个缺行或被截断的交付物（CSV 只有几十 KB，代价可忽略）。
        """
        rows = dict(old)
        for r in new_rows:
            rows[str(r["sample_id"])] = r
        merged = [rows[k] for k in sorted(rows)]
        fresh = set(str(r["sample_id"]) for r in new_rows)
        frames = [f for f in old_frames if str(f.get("sample_id")) not in fresh] + new_frames
        frames.sort(key=lambda f: (str(f.get("sample_id")), str(f.get("frac"))))
        _write_csvs(spath, fpath, merged, frames)
        return merged

    for i, s in enumerate(samples, 1):
        sid = str(s["sample_id"])
        video = str(s["video_path"])
        if sid in old and not overwrite and n_frame_rows.get(sid, 0) == n_fracs:
            n_skip += 1
            continue

        # 抽样时刻按**公共时间轴**的时长定比例，与对齐用的 duration 同源
        # （aligned npz 的 duration 已按 D1 修复存为 float64）。
        apath = os.path.join(aligned_dir, f"{sid}.npz")
        duration = float("nan")
        if os.path.isfile(apath):
            with np.load(apath, allow_pickle=True) as z:
                duration = float(np.asarray(z["duration"], np.float64))
        else:
            LOGGER.warning("[探测] (%d/%d) %s 缺少对齐文件，无法确定公共时间轴",
                           i, len(samples), sid)

        t0 = time.time()
        payload = _spawn_one(sid, video, duration, fracs, timeout, threads, tmp_dir)
        row = _aggregate(sid, video, duration, payload)
        if not (duration > 0):
            # 时长无效时 0.15T/0.50T/0.85T 无意义，判据一律标为不可用
            row["detector_ok"] = 0
            if row["status"] == STATUS_OK:
                row["status"] = STATUS_NO_DURATION
        dt = time.time() - t0
        n_new += 1
        new_rows.append(row)
        new_frames.extend(row.pop("_frames", []))
        merged = flush()
        LOGGER.info("[探测] (%d/%d) %s → %s 脸数=%s 最大框面积比=%s（%.1f s，累计 %d/%d 条）",
                    i, len(samples), sid, row["status"], row["n_faces"] or "-",
                    row["max_face_area_ratio"] if row["detector_ok"] else "-",
                    dt, len(merged), len(samples))

    all_rows = flush() if n_new else [old[k] for k in sorted(old)]
    # 清掉子进程的 JSON 中转目录。**必须用 rmtree**：这里放的是每个样本一份
    # 探测结果（约 100 个文件），`os.rmdir` 只能删空目录、遇到非空会抛 OSError，
    # 而上面又把它吞掉了——结果是这个临时目录被留在交付目录里，
    # 下一次跑 q1_delivery 出图后就会被算进「交付物合计」（实测 0.087 MB）。
    # 它不是交付物，不该占交付体积，也不该混进交付清单。
    shutil.rmtree(tmp_dir, ignore_errors=True)

    bad = [r for r in all_rows if str(r.get("detector_ok")) not in ("1", "1.0")]
    LOGGER.info("[探测] 完成：新增 %d / 跳过 %d；不可用 %d 条（已如实记状态码，未剔除样本）",
                n_new, n_skip, len(bad))

    # 收尾自检：再数一遍逐帧证据是否每样本都齐。**报出来而不是修掉**——
    # 若这里非空，说明续探逻辑没救回来，交付里的「逐帧证据」就有洞，
    # 必须让调用方看到，而不是留一个看起来正常的 100 行汇总表。
    final_frames = _read_csv_rows(fpath, FRAME_FIELDS)
    fin: Dict[str, int] = {}
    for f in final_frames:
        k = str(f.get("sample_id"))
        fin[k] = fin.get(k, 0) + 1
    # 注意：`all_rows` 是**行字典**的列表，不是 sample_id 的列表——
    # 写成 `for sid in all_rows` 会把整行字典当成 id，`sorted()` 立刻抛 TypeError
    # （本项目真踩过：这道自检第一次跑就在这一行崩了，而且因为命令接了 `| tail`
    #  管道，退出码被管道吃掉，看起来像正常结束）。
    gap = sorted(str(r.get("sample_id")) for r in all_rows
                 if fin.get(str(r.get("sample_id")), 0) != n_fracs)
    if gap:
        LOGGER.error("[探测] 逐帧证据仍不齐的样本 %d 条：%s（请 --overwrite 重探）",
                     len(gap), "、".join(gap[:8]))
    else:
        LOGGER.info("[探测] 逐帧证据自检通过：%d 条样本 × %d 帧 = %d 行",
                    len(all_rows), n_fracs, len(final_frames))

    return {"out_dir": out_dir, "n_samples": len(all_rows),
            "rows": all_rows,
            # 本轮新探的帧（new_frames）与盘上全部帧（n_frames_total）不是一回事，
            # 两个都给出，免得调用方把「本轮新增」误当成「交付物里的总数」。
            "frames": new_frames, "n_frames_total": len(final_frames),
            "samples_csv": spath, "frames_csv": fpath,
            "n_unavailable": len(bad), "incomplete_frames": gap}


def _read_csv_rows(path: str, fields: Sequence[str]) -> List[Dict[str, object]]:
    """
    读回已落盘的**逐行**结果；缺列补空，避免旧版本 CSV 少列时读取崩溃。

    ⚠️ 逐帧表（`face_probe_frames.csv`）**必须**用这个函数，不能用下面的
    `_read_csv()`——后者按 `sample_id` 建字典，而逐帧表每个 sample_id 有 3 行，
    一建字典就只剩最后一行。本项目真踩过：续探时 `old_frames` 由 297 行缩成 100 行，
    后果有两个——(1) 每条已有样本的逐帧证据在内存里丢了 2/3，
    若当轮没有重新落盘，文件就被写少；(2) 「样本行已存在」的续探判据因此恒不成立，
    整批 100 条被无谓重探。**一张表能不能按主键建字典，取决于主键在表里唯不唯一。**
    """
    out: List[Dict[str, object]] = []
    if not os.path.isfile(path):
        return out
    try:
        with open(path, encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                out.append({k: r.get(k, "") for k in fields})
    except Exception as exc:
        LOGGER.warning("[探测] 已有结果 %s 不可读（将从零开始）：%s", path, exc)
        return []
    return out


def _read_csv(path: str, fields: Sequence[str]) -> Dict[str, Dict[str, object]]:
    """
    读回已落盘的**单行样本表**结果，按 `sample_id` 建字典。
    只适用于 `face_probe.csv`（一行一样本、主键唯一）；逐帧表请用 `_read_csv_rows()`。
    """
    out: Dict[str, Dict[str, object]] = {}
    for r in _read_csv_rows(path, fields):
        sid = str(r.get("sample_id") or "")
        if sid:
            out[sid] = r
    return out


def _write_csvs(spath: str, fpath: str, rows: List[Dict[str, object]],
                frames: List[Dict[str, object]]) -> None:
    """整表重写两份 CSV。`extrasaction="ignore"` 保证内部字段（如 _frames）不会漏进交付物。"""
    if rows:
        with open(spath, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=SAMPLE_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in SAMPLE_FIELDS})
    if frames:
        with open(fpath, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=FRAME_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in frames:
                w.writerow({k: r.get(k, "") for k in FRAME_FIELDS})


def main(argv: Optional[Sequence[str]] = None) -> int:
    # 子进程分支必须在解析前拦下，避免父进程的参数集与之混淆。
    if "--json-out" in (argv if argv is not None else sys.argv[1:]):
        return _worker_main((argv if argv is not None else sys.argv[1:]))

    p = argparse.ArgumentParser(description="轻量人脸探测（3 帧抽样，仅供『多人/远景』判据取证）")
    p.add_argument("--out", default=None, help="输出目录，默认 data/q1_delivery")
    p.add_argument("--only", default=None, help="只处理 sample_id 含该子串的样本")
    p.add_argument("--fracs", default=",".join(str(f) for f in DEFAULT_FRACS),
                   help="抽样时刻，按公共时间轴时长 T 的比例给出，默认 0.15,0.5,0.85")
    p.add_argument("--timeout", type=float, default=180.0, help="每条样本墙钟超时秒数")
    p.add_argument("--threads", type=int, default=4, help="torch 线程数（纯检测，4 足够）")
    p.add_argument("--overwrite", action="store_true", help="忽略已有结果，全部重探")
    args = p.parse_args(argv)

    fracs = [float(x) for x in str(args.fracs).split(",") if x.strip()]
    res = run(out_dir=args.out, fracs=fracs, timeout=args.timeout,
              threads=args.threads, overwrite=args.overwrite, only=args.only)
    # 报**盘上的实际行数**（n_frames_total），不是本轮新探的帧数（len(res["frames"])）。
    # 两者在「一条没探、全部跳过」时差得很远：本轮是 0 行，盘上是 300 行。
    # 日志里写 0 行会让人以为交付物的逐帧证据空了——日志说错话比不说更糟。
    LOGGER.info("[探测] 写出 %s（%d 行）与 %s（%d 行）",
                res["samples_csv"], res["n_samples"],
                res["frames_csv"], res["n_frames_total"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
