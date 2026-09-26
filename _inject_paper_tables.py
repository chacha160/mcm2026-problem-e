# -*- coding: utf-8 -*-
"""
_inject_paper_tables.py —— 把表格片段注入论文 Markdown

论文正文 Markdown 里留下占位符：

  {TABLE:name}      → 替换为 `_tables_q1/name.md` 的全部内容

这样正文只写论述、不夹带几十行的表格数据，表格始终由**已验证产物**生成，
不会出现"论文里的数字与交付物不一致"这种最难查的错误。

默认的源稿与表目录是问题一正文那一套；别的稿子用 --src/--out/--tables 覆盖
（`_build_q1_paper.py` 走 问题一论文.md → 问题一论文_展开.md，注入规则同一套）。

仓库已按问题一重建，问题二、三的产物不在库内，故原来那条
`{STRUCT_ABLATION}` → `data/q2/structure_ablation.csv` 的渲染分支已去掉；
该占位符若出现在稿子里，会落进下面的「未替换占位符」告警，不会静默通过。

用法：
    python _inject_paper_tables.py                 # 写到 问题一论文正文_展开.md
    python _inject_paper_tables.py --check         # 只报告占位符解析情况
"""

from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MD_SRC = os.path.join(ROOT, "问题一论文正文.md")
MD_OUT = os.path.join(ROOT, "问题一论文正文_展开.md")
TABLES = os.path.join(ROOT, "_tables_q1")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查占位符，不写文件")
    ap.add_argument("--src", default=MD_SRC, help="源稿 Markdown（默认 问题一论文正文.md）")
    ap.add_argument("--out", default=MD_OUT, help="展开稿输出路径")
    ap.add_argument("--tables", default=TABLES, help="表格片段目录")
    args = ap.parse_args(argv)
    src, out, tables = args.src, args.out, args.tables

    with open(src, encoding="utf-8") as f:
        text = f.read()

    names = re.findall(r"\{TABLE:([A-Za-z0-9_]+)\}", text)
    missing, used = [], []
    for n in names:
        p = os.path.join(tables, n + ".md")
        (used if os.path.isfile(p) else missing).append(n)

    print(f"[占位符] TABLE 共 {len(names)} 处，涉及 {len(set(names))} 张表")
    if missing:
        print(f"  ⚠ 缺表 {len(set(missing))} 张：{sorted(set(missing))}")

    def sub(m):
        p = os.path.join(tables, m.group(1) + ".md")
        if not os.path.isfile(p):
            return f"> **（表 {m.group(1)} 未生成）**\n"
        with open(p, encoding="utf-8") as fh:
            return fh.read().rstrip("\n")

    text = re.sub(r"\{TABLE:([A-Za-z0-9_]+)\}", sub, text)

    # STRUCT_ABLATION 已不再渲染（它的数据源 data/q2 不在库内），但保留在告警网里：
    # 稿子里若还留着这个占位符，应当报出来，而不是静默通过。
    leftover = re.findall(r"\{[A-Z_]+:[A-Za-z0-9_]+\}|\{STRUCT_ABLATION\}", text)
    if leftover:
        print(f"  ⚠ 仍有未替换占位符：{set(leftover)}")

    if args.check:
        return 0
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"[写出] {out}（{len(text.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
