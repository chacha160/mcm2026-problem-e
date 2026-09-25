# -*- coding: utf-8 -*-
"""
run_all.py —— 三问一键复现入口

按依赖顺序串起全部环节，任一步失败立即停止并报出是哪一步
（而不是继续跑后面几步、最后给出一堆看似完成实则残缺的产物）。

    python code/run_all.py                # 全量
    python code/run_all.py --quick        # 快速自检（少轮次、跳过 48 组实验和图件）
    python code/run_all.py --only q2      # 只跑某一问

各步与赛题的对应：
    audit   缺失结构审计（口径核验，非交付物但决定后续一切）
    q1raw   问题一：三模态提取 → 对齐（重投入，结果已有时默认跳过）
    q1      问题一：交付物 + 12 项机器核验 + 文档 + 融合增强层
    q2      问题二：训练 + 评测 + 48 组受控缺失实验 + 附件3 推理
    q3      问题三：可解释训练 + 附件4 证据落地 + 图件
    pack    提交体积核算 + 红线扫描 + 就地脱敏

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
    ]
    return all(os.path.exists(p) for p in need)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="三问一键复现")
    ap.add_argument("--quick", action="store_true", help="快速自检模式")
    ap.add_argument("--only", default="",
                    help="逗号分隔：audit,q1raw,q1,q2,q2abl,q3,pack")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--force-q1raw", action="store_true",
                    help="即使已有提取/对齐产物也重跑 q1raw（默认产物在则跳过）")
    ap.add_argument("--force-q2abl", action="store_true",
                    help="即使已有 structure_ablation.csv 也重跑结构消融")
    ap.add_argument("--no-sanitize", action="store_true",
                    help="打包步不做就地脱敏（默认执行 --sanitize）")
    args = ap.parse_args(argv)

    want = {s.strip() for s in args.only.split(",") if s.strip()}
    def _on(step: str) -> bool:
        return (not want) or (step in want)

    epochs = 3 if args.quick else args.epochs
    results: List[Tuple[str, bool, float]] = []

    steps = []
    if _on("audit"):
        steps.append(("audit", "q2q3_common.py", [], "缺失结构审计（附件2/3/4）"))

    # ---- 问题一：提取与对齐（重投入，产物在则跳过）----
    if _on("q1raw") or _on("q1"):
        if _on("q1raw") and (args.force_q1raw or not _q1raw_ready()):
            steps.append(("q1raw", "run_unaligned_all.py",
                          ["--deliver", "--verify"],
                          "问题一：三模态提取 → 对齐 → 交付 → 核验"))
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
            steps.append(("q1", "q1_verify.py", [], "问题一：12 项机器核验"))
            steps.append(("q1", "q1_readme.py", [], "问题一：生成交付说明文档"))
            steps.append(("q1", "q1_fusion.py", [], "问题一：融合增强层（三层语义 + 对应三态）"))

    if _on("q2"):
        extra = ["--epochs", str(epochs)]
        if args.quick:
            extra.append("--skip-experiment")
        steps.append(("q2", "q2_train.py", extra, "问题二：训练/评测/受控实验/附件3 推理"))
    # ---- 问题二：结构消融（论文 5.2.6 表 15 的对照实验）----
    # 单独一步、单独产物：它要训 5 个模型，是整条链路里最慢的一段；且它**不改动**
    # 交付模型与任何既有产物，故可以与 q2 主步互不影响地单独重跑。
    if _on("q2abl"):
        q2_out = os.path.join(config.PROJECT_ROOT, "data", "q2")
        q2abl_csv = os.path.join(q2_out, "structure_ablation.csv")
        if args.quick:
            # --quick 把 epochs 压到 3。用 3 轮训出的消融表数值无意义，若落盘还会
            # 让后续正式跑因「产物已存在」而跳过，等于用一份废数据顶替论文的表 15。
            # 故 quick 模式直接不产出该步，而不是产出再靠人去发现。
            print("[跳过] q2abl：--quick 模式下不产出结构消融"
                  "（3 轮训出的消融数值无意义，会顶替正式结果）", flush=True)
        elif os.path.isfile(q2abl_csv) and not args.force_q2abl:
            print("[跳过] q2abl：data/q2/structure_ablation.csv 已存在"
                  "（需重跑请加 --force-q2abl）", flush=True)
        else:
            steps.append(("q2abl", "q2_train.py",
                          ["--out", q2_out, "--structure-ablation-only",
                           "--epochs", str(epochs)],
                          "问题二：结构消融（5 变体 × 同一受控缺失探针）"))

    if _on("q3"):
        extra = ["--epochs", str(epochs)]
        if args.quick:
            extra.append("--no-figures")
        steps.append(("q3", "q3_explain.py", extra, "问题三：可解释预测与证据落地"))
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
