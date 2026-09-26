# -*- coding: utf-8 -*-
"""
run_unaligned_all.py —— 一键跑通「三模态分别提取（不对齐）→ 统一对齐 → 时间轴可视化」

流水线五步（每一步都能单独重跑，产物各自独立）：
    1) unaligned_text.py    文本 → 词级 RoBERTa 向量 (W, 768)，无时间槽
    2) unaligned_audio.py   语音 → 逐帧 74 维 @ 20 Hz + 实测 pts
    3) unaligned_vision.py  视觉 → 逐帧 35 维 @ 15 Hz + 实测 pts（人脸特征）
    4) word_align.py        官方文本 → stable-ts 强制对齐 → 实测词时间 + 证据路由
    5) align_multimodal.py  上述产物（含第 4 步的路由结论）→ 统一秒轴 → (50, D) 定长张量
                            + 槽↔时间映射
    6) timeline_visualize.py（可选）画共享时间轴图 + 逐槽对应表
    7) 问题一的验收（**可选、默认关闭**）：face_probe.py / q1_delivery.py / q1_verify.py
       只读 1~5 步的产物，另写 data/q1_delivery/，不改动任何特征文件。

用法：
    python run_unaligned_all.py                 # 全量 100 条
    python run_unaligned_all.py --limit 3       # 先跑 3 条验证链路
    python run_unaligned_all.py --skip-vision   # 跳过最慢的视觉步骤
    python run_unaligned_all.py --timeline      # 跑完顺带出图
    python run_unaligned_all.py --verify        # 跑完执行 13 项机器核验
    python run_unaligned_all.py --deliver       # 跑完生成交付物（含人脸探测，约 +10 分钟）

耗时参考（CPU，8 线程，本机实测）：
    文本 ~1.5 s/条；语音 ~1.5 s/条；视觉 ~135 ms/帧（全量约 11800 帧 ≈ 27 分钟）；
    词对齐 ~0.8 s/条（本机 3 条实测；模型加载一次性约 6 s，首次另需下 base.en 权重约 140 MB）；
    三模态对齐 <1 s/条。全量合计约 36 分钟，视觉占绝大部分。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Sequence

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import unaligned_common as U  # noqa: E402
from utils import LOGGER  # noqa: E402


def run_vision_watched(samples, out_dir: str, overwrite: bool, sample_timeout: float,
                       threads: int) -> int:
    """
    逐样本跑视觉提取，每个样本套一层「墙钟超时看门狗」。

    为什么需要它：视觉环节依赖 torch 与 OpenCV 的原生线程，实测出现过
    **非确定性**卡死——某条样本在批量运行中 8 核满载空转数小时不返回，
    而同一条样本单独重跑只需不到 1 秒。这类卡死发生在原生代码内部，
    Python 层的进程内计时器无法中断它，唯一可靠的办法是把每个样本
    放进独立子进程，由父进程按墙钟超时收割。

    代价是每个样本多一次解释器与模型加载（约 2 s），换来的是：
      · 任何一条样本卡死都只影响它自己，其余 99 条照常完成；
      · 超时的样本被明确记入日志，不会拖着一整批任务无声停摆；
      · 因为提取器默认跳过已存在的产物，重跑本步即是断点续跑。

    返回超时的样本数。
    """
    import subprocess

    os.makedirs(out_dir, exist_ok=True)
    n_timeout = 0
    pending = []
    for s in samples:
        sid = str(s["sample_id"])
        if os.path.exists(os.path.join(out_dir, f"{sid}.npz")) and not overwrite:
            continue
        pending.append(sid)
    LOGGER.info("[视觉] 待处理 %d 条，逐样本看门狗超时 %.0f s", len(pending), sample_timeout)

    for i, sid in enumerate(pending, 1):
        cmd = [sys.executable, "-X", "utf8", os.path.join(_CODE_DIR, "unaligned_vision.py"),
               f"--only={sid}", "--threads", str(threads)]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, timeout=sample_timeout, capture_output=True,
                                  encoding="utf-8", errors="replace")
            if proc.returncode != 0:
                LOGGER.error("[视觉] (%d/%d) %s 退出码 %d: %s", i, len(pending), sid,
                             proc.returncode, (proc.stderr or "")[-300:])
            else:
                LOGGER.info("[视觉] (%d/%d) %s 完成（%.1f s）", i, len(pending), sid, time.time() - t0)
        except subprocess.TimeoutExpired:
            n_timeout += 1
            LOGGER.error("[视觉] (%d/%d) %s 超过 %.0f s 未返回，判定卡死，已终止并跳过。"
                         "可稍后单独重跑该样本：python unaligned_vision.py --only=%s",
                         i, len(pending), sid, sample_timeout, sid)
    if n_timeout:
        LOGGER.warning("[视觉] 共 %d 条超时被跳过，请对上述样本单独重跑后再执行对齐", n_timeout)
    return n_timeout


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="一键跑通未对齐提取 + 对齐 + 可视化")
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 条（调试用）")
    p.add_argument("--only", default=None, help="只处理 sample_id 含该子串的样本")
    p.add_argument("--overwrite", action="store_true", help="覆盖已有产物")
    p.add_argument("--skip-text", action="store_true")
    p.add_argument("--skip-audio", action="store_true")
    p.add_argument("--skip-vision", action="store_true")
    p.add_argument("--skip-align", action="store_true")
    p.add_argument("--skip-word-align", action="store_true",
                   help="跳过 stable-ts 强制对齐：文本模态一律用均匀假设时间戳"
                        "（无 stable-ts/openai-whisper 环境时的降级路径）")
    p.add_argument("--timeline", action="store_true", help="跑完自动为典型样本出时间轴图")
    p.add_argument("--threads", type=int, default=8, help="视觉推理线程数")
    p.add_argument("--sample-timeout", type=float, default=900.0,
                   help="视觉环节每个样本的墙钟超时秒数（默认 900）。"
                        "设为 0 则整批在一个进程里跑（更快，但一条卡死会拖停整批）")
    p.add_argument("--slots", type=int, default=None, help="对齐槽数，默认取 config.ALIGN_SEQ_LEN")
    # 验收环节：默认关闭，见下方注释
    p.add_argument("--verify", action="store_true",
                   help="跑完后执行 q1_verify.py 的机器核验（默认关闭）")
    p.add_argument("--deliver", action="store_true",
                   help="跑完后生成 data/q1_delivery/ 下的四项交付物与五类典型样本（默认关闭）")
    p.add_argument("--skip-face-probe", action="store_true",
                   help="配合 --deliver：跳过人脸探测（该步约 10 分钟），"
                        "「多人/远景」判据将标为不可用而不是留空")
    p.add_argument("--no-figures", action="store_true",
                   help="配合 --deliver：不出五类典型样本的图与对应表")
    args = p.parse_args(argv)

    import config
    root = os.path.join(os.path.dirname(_CODE_DIR), "data")
    uroot = os.path.join(root, "unaligned_features")
    adir = os.path.join(root, "aligned")

    samples = U.list_samples(only=args.only, limit=args.limit)
    LOGGER.info("=" * 72)
    LOGGER.info("流水线启动：%d 条样本；未对齐输出 %s；对齐输出 %s", len(samples), uroot, adir)
    LOGGER.info("=" * 72)

    t_all = time.time()
    steps = []
    if not args.skip_text:
        import unaligned_text as M
        steps.append(("文本（RoBERTa 词级，不对齐）",
                      lambda: M.run(samples, os.path.join(uroot, "text"), overwrite=args.overwrite)))
    if not args.skip_audio:
        import unaligned_audio as M2
        steps.append(("语音（74 维 @ 20 Hz，不对齐）",
                      lambda: M2.run(samples, os.path.join(uroot, "audio"), overwrite=args.overwrite)))
    if not args.skip_vision:
        vout = os.path.join(uroot, "vision")
        if args.sample_timeout and args.sample_timeout > 0:
            steps.append(("视觉（35 维人脸特征 @ 15 Hz，不对齐；逐样本看门狗）",
                          lambda: run_vision_watched(samples, vout, args.overwrite,
                                                     float(args.sample_timeout), int(args.threads))))
        else:
            import torch
            torch.set_num_threads(int(args.threads))
            import unaligned_vision as M3
            steps.append(("视觉（35 维人脸特征 @ 15 Hz，不对齐）",
                          lambda: M3.run(samples, vout, overwrite=args.overwrite)))
    # 词对齐必须排在「三模态对齐」之前：后者要读前者的路由结论来决定
    # 文本模态用实测词时刻还是均匀假设。
    wadir = os.path.join(root, "word_align")
    if not args.skip_word_align and not args.skip_align:
        import word_align as WA
        steps.append(("文本强制对齐（stable-ts → 实测词时间 + 证据路由）",
                      lambda: WA.run(samples, wadir, overwrite=args.overwrite)))
    if not args.skip_align:
        import align_multimodal as M4
        slots = int(args.slots or config.ALIGN_SEQ_LEN)
        steps.append(("三模态对齐（统一秒轴 → (50,D)）",
                      lambda: M4.run(samples, uroot, adir, n_slots=slots,
                                     word_align_dir=wadir if not args.skip_word_align else None,
                                     word_align_enabled=not args.skip_word_align)))

    for i, (name, fn) in enumerate(steps, 1):
        t0 = time.time()
        LOGGER.info("-" * 72)
        LOGGER.info("[%d/%d] 开始：%s", i, len(steps), name)
        fn()
        LOGGER.info("[%d/%d] 完成：%s（用时 %.1f s）", i, len(steps), name, time.time() - t0)

    if args.timeline:
        import timeline_visualize as TV
        TV.setup_chinese_font()
        sid = TV.pick_typical(uroot, samples)
        if sid:
            odir = os.path.join(root, "timeline")
            TV.make_figure(sid, uroot, adir, out_dir=odir)
            TV.export_correspondence(sid, uroot, adir, odir)

    # ---- 问题一的验收环节，**默认关闭** ----
    # 这两步不产生任何特征、也不修改 data/unaligned_features 与 data/aligned，
    # 只读它们并另写 data/q1_delivery/。默认关闭是为了避免「跑一次流水线」
    # 顺带触发十几分钟的人脸探测与出图——它们有自己的入口，可随时单独跑：
    #     python q1_verify.py       （13 项机器核验）
    #     python q1_delivery.py     （四项交付物 + 台账 + 体积 + 五类典型样本）
    if args.verify or args.deliver:
        q1_out = os.path.join(root, "q1_delivery")
        sids = [str(s["sample_id"]) for s in samples]
        # 人脸探测是交付物里「多人/远景」判据的输入，只有要交付时才跑
        if args.deliver and not args.skip_face_probe:
            import face_probe as FP
            FP.run(unaligned_root=uroot, aligned_dir=adir, out_dir=q1_out,
                   sids=sids, threads=args.threads)
        if args.deliver:
            import q1_delivery as D
            D.run_all(unaligned_root=uroot, aligned_dir=adir, out_dir=q1_out,
                      with_figures=not args.no_figures)
        # 核验放在最后：交付物刚生成，这一步才覆盖得到它们的内容与新鲜度
        import q1_verify as V
        rep = V.verify_all(unaligned_root=uroot, aligned_dir=adir, out_dir=q1_out)
        LOGGER.info("[验收] 机器核验 all_passed=%s", rep.get("all_passed"))

    LOGGER.info("=" * 72)
    LOGGER.info("全部完成，总用时 %.1f 分钟", (time.time() - t_all) / 60)
    LOGGER.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
