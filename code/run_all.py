# -*- coding: utf-8 -*-
"""
run_all.py —— 问题一（v1 旧链路）一键复现入口

按依赖顺序串起全部环节，任一步失败立即停止并报出是哪一步
（而不是继续跑后面几步、最后给出一堆看似完成实则残缺的产物）。

    python code/run_all.py                # 全量
    python code/run_all.py --only q1raw   # 只跑提取与对齐

各步与赛题的对应：
    q1raw   问题一：三模态提取 → 词对齐 → 统一对齐（重投入，结果已有时默认跳过）
    q1      问题一：交付物 + 13 项机器核验 + 文档 + 融合增强层
    pack    提交体积核算 + 红线扫描 + 就地脱敏

本仓已按问题一重建（v2 链路见 `run_q1v2_all.py`），问题二、问题三的数据与
代码不在库内，故原来串在这条链上的 audit / q2 / q2abl / q3 四步（以及只服务
它们的 --quick、--epochs、--force-q2abl 三个开关）已一并去掉。

关于 q1raw 的取舍（重要）：
    提取与对齐是全流程唯一的重投入环节（约 35 分钟起步，视觉环节更久），
    而它只依赖附件 1 的原始素材、不依赖任何超参。因此默认策略是
    「**产物在则跳过，缺失则补跑**」——只要 data/q1_delivery/ 与
    data/aligned/ 齐全，就不重跑；`--force-q1raw` 强制重跑。
    这样 `python code/run_all.py` 在一个已有产物的仓库上是幂等的，
    在一个干净仓库上则是自足的，两种情形都不会静默地少跑一段。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional, Sequence, Tuple

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import config  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def _run(script: str, extra: Sequence[str], label: str) -> Tuple[bool, float]:
    cmd = [sys.executable, "-u", os.path.join(_CODE_DIR, script), *extra]
    print(f"\n{'#' * 72}\n# {label}\n#   {' '.join(cmd)}\n{'#' * 72}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=config.PROJECT_ROOT)
    dt = time.time() - t0
    ok = r.returncode == 0
    print(f"[{label}] {'完成' if ok else '失败'}，耗时 {dt:.1f}s，退出码 {r.returncode}",
          flush=True)
    return ok, dt


def _q1raw_ready() -> bool:
    """提取与对齐是否已有可用产物——决定 q1raw 是跑还是跳过。"""
    need = [
        os.path.join(config.PROJECT_ROOT, "data", "aligned", "aligned_50.npz"),
        os.path.join(config.PROJECT_ROOT, "data", "unaligned_features", "text"),
        os.path.join(config.PROJECT_ROOT, "data", "unaligned_features", "audio"),
        os.path.join(config.PROJECT_ROOT, "data", "unaligned_features", "vision"),
        # 词对齐结论已写进 data/aligned 的 routing 块。缺了它，文本模态会静默退回
        # 均匀假设——形状照样是 (50,768)，核验也照样过，但实测词时间不再生效。
        # 故把它列为「产物齐备」的必要条件之一，宁可重跑也不静默降级。
        os.path.join(config.PROJECT_ROOT, "data", "word_align"),
    ]
    return all(os.path.exists(p) for p in need)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="问题一（v1 旧链路）一键复现")
    ap.add_argument("--only", default="", help="逗号分隔：q1raw,q1,pack")
    ap.add_argument("--force-q1raw", action="store_true",
                    help="即使已有提取/对齐产物也重跑 q1raw（默认产物在则跳过）")
    ap.add_argument("--no-sanitize", action="store_true",
                    help="打包步不做就地脱敏（默认执行 --sanitize）")
    args = ap.parse_args(argv)

    want = {s.strip() for s in args.only.split(",") if s.strip()}
    def _on(step: str) -> bool:
        return (not want) or (step in want)

    results: List[Tuple[str, bool, float]] = []

    steps = []
    # ---- 问题一：提取与对齐（重投入，产物在则跳过）----
    if _on("q1raw") or _on("q1"):
        if _on("q1raw") and (args.force_q1raw or not _q1raw_ready()):
            steps.append(("q1raw", "run_unaligned_all.py",
                          ["--deliver", "--verify"],
                          "问题一：三模态提取 → 词对齐 → 统一对齐 → 交付 → 核验"))
        elif _on("q1raw"):
            print("[跳过] q1raw：data/unaligned_features 与 data/aligned/aligned_50.npz "
                  "均已存在（需重跑请加 --force-q1raw）", flush=True)

    # ---- 问题一：交付物、核验、文档、融合增强层 ----
    if _on("q1"):
        q1_out = os.path.join(config.PROJECT_ROOT, "data", "q1_delivery")
        if os.path.isdir(q1_out):
            steps.append(("q1", "q1_readme.py", [], "问题一：生成交付说明文档"))
            steps.append(("q1", "q1_fusion.py", [], "问题一：融合增强层（三层语义 + 对应三态）"))
        else:
            steps.append(("q1", "q1_delivery.py", [], "问题一：生成交付物"))
            steps.append(("q1", "q1_verify.py", [], "问题一：13 项机器核验"))
            steps.append(("q1", "q1_readme.py", [], "问题一：生成交付说明文档"))
            steps.append(("q1", "q1_fusion.py", [], "问题一：融合增强层（三层语义 + 对应三态）"))

    if _on("pack"):
        extra = [] if args.no_sanitize else ["--sanitize"]
        steps.append(("pack", "package_check.py", extra, "提交体积核算、红线扫描与脱敏"))

    for step, script, extra, label in steps:
        if not os.path.isfile(os.path.join(_CODE_DIR, script)):
            print(f"[跳过] {script} 未找到", flush=True)
            continue
        ok, dt = _run(script, extra, label)
        results.append((label, ok, dt))
        if not ok:
            print(f"\n>>> 在「{label}」处中断；后续步骤未执行。", flush=True)
            break

    print("\n" + "=" * 72)
    print("执行汇总")
    for label, ok, dt in results:
        print(f"  {'✅' if ok else '❌'} {label}  ({dt:.1f}s)")
    allok = all(ok for _, ok, _ in results) and len(results) == len(steps)
    print("=" * 72)
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
