# -*- coding: utf-8 -*-
"""
run_all.py —— 三问一键复现入口

按依赖顺序串起全部环节，任一步失败立即停止并报出是哪一步
（而不是继续跑后面几步、最后给出一堆看似完成实则残缺的产物）。

    python code/run_all.py                # 全量
    python code/run_all.py --quick        # 快速自检（少轮次、跳过 48 组实验和图件）
    python code/run_all.py --only q2      # 只跑某一问

各步与赛题的对应：
    audit  缺失结构审计（口径核验，非交付物但决定后续一切）
    q1     问题一融合增强层（三层语义 + 对应三态）
    q2     问题二：训练 + 评测 + 48 组受控缺失实验 + 附件3 推理
    q3     问题三：可解释训练 + 附件4 证据落地 + 图件
    pack   提交体积核算与红线扫描
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="三问一键复现")
    ap.add_argument("--quick", action="store_true", help="快速自检模式")
    ap.add_argument("--only", default="", help="逗号分隔：audit,q1,q2,q3,pack")
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args(argv)

    want = {s.strip() for s in args.only.split(",") if s.strip()}
    def _on(step: str) -> bool:
        return (not want) or (step in want)

    epochs = 3 if args.quick else args.epochs
    results: List[Tuple[str, bool, float]] = []

    steps = []
    if _on("audit"):
        steps.append(("audit", "q2q3_common.py", [], "缺失结构审计（附件2/3/4）"))
    if _on("q1"):
        steps.append(("q1", "q1_fusion.py", [], "问题一：融合增强层"))
    if _on("q2"):
        extra = ["--epochs", str(epochs)]
        if args.quick:
            extra.append("--skip-experiment")
        steps.append(("q2", "q2_train.py", extra, "问题二：训练/评测/受控实验/附件3 推理"))
    if _on("q3"):
        extra = ["--epochs", str(epochs)]
        if args.quick:
            extra.append("--no-figures")
        steps.append(("q3", "q3_explain.py", extra, "问题三：可解释预测与证据落地"))
    if _on("pack"):
        steps.append(("pack", "package_check.py", [], "提交体积核算与红线扫描"))

    for step, script, extra, label in steps:
        if not os.path.isfile(os.path.join(_CODE_DIR, script)):
            print(f"[跳过] {script} 尚未实现", flush=True)
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
