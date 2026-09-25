# -*- coding: utf-8 -*-
"""
_inject_paper_tables.py —— 把表格片段注入论文 Markdown

论文正文 `论文_完整三问.md` 里留下两类占位符：

  {TABLE:name}      → 替换为 `_tables/name.md` 的全部内容
  {STRUCT_ABLATION} → 由 `data/q2/structure_ablation.csv` 现场渲染成表格

这样正文只写论述、不夹带几十行的表格数据，表格始终由**已验证产物**生成，
不会出现"论文里的数字与交付物不一致"这种最难查的错误。

用法：
    python _inject_paper_tables.py                 # 写到 论文_正文_展开.md
    python _inject_paper_tables.py --check         # 只报告占位符解析情况
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
MD_SRC = os.path.join(ROOT, "论文_完整三问.md")
MD_OUT = os.path.join(ROOT, "论文_正文_展开.md")
TABLES = os.path.join(ROOT, "_tables")
STRUCT_CSV = os.path.join(ROOT, "data", "q2", "structure_ablation.csv")

# 结构消融表的行序与列序（与 q2_train.STRUCTURE_LABELS 对齐）
STRUCT_ORDER = ("full", "no_state_emb", "no_obs_mask", "union_mask", "no_md")


def f3(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    return "—" if v != v else "%.3f" % v


def render_structure_ablation() -> str:
    """把结构消融 CSV 渲染成 Markdown 表；缺失时给出明确的未完成标记。"""
    if not os.path.isfile(STRUCT_CSV):
        return ("> **（结构消融表待填）** 运行以下命令生成：\n>\n"
                "> `python code/q2_train.py --out data/q2 --structure-ablation-only "
                "--epochs 60 --patience 12`\n>\n"
                "> 该实验训练 5 个结构变体，探针条件为「文本缺失 · 中部 · 40%」。\n")
    with open(STRUCT_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by = {r["structure"]: r for r in rows}

    # 表头分两段：无缺失（对照）与探针条件（结构信号该起作用的地方）
    # 「无缺失 Pearson」必须单列：5.2.6 的解读里要引用它（"是全部变体中该指标最差的一项"），
    # 若表内没有这一列，读者就无法在表内复核那句话。
    header = ("| 结构变体 | 说明 | 最优轮 | "
              "无缺失 macro-F1 | 无缺失 MAE | 无缺失 Pearson | "
              "探针 macro-F1 | 探针 MAE | 探针 Pearson | "
              "Δmacro-F1 | ΔMAE |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|")
    ref = by.get("full")
    ref_f1 = float(ref["probe_valid_macro_f1"]) if ref else float("nan")
    ref_mae = float(ref["probe_valid_mae"]) if ref else float("nan")

    lines = [header]
    for st in STRUCT_ORDER:
        r = by.get(st)
        if r is None:
            lines.append(f"| `{st}` | — | — | — | — | — | — | — | — | — | — |")
            continue
        d_f1 = float(r["probe_valid_macro_f1"]) - ref_f1
        d_mae = float(r["probe_valid_mae"]) - ref_mae
        label = r.get("structure_label", st)
        if st == "full":
            d_f1s = d_maes = "—（基准）"
        else:
            d_f1s, d_maes = "%+.3f" % d_f1, "%+.3f" % d_mae
        lines.append(
            f"| `{st}` | {label} | {r['best_epoch']} | "
            f"{f3(r['clean_valid_macro_f1'])} | {f3(r['clean_valid_mae'])} | "
            f"{f3(r['clean_valid_pearson'])} | "
            f"{f3(r['probe_valid_macro_f1'])} | {f3(r['probe_valid_mae'])} | "
            f"{f3(r['probe_valid_pearson'])} | {d_f1s} | {d_maes} |")
    probe = rows[0].get("probe", "?") if rows else "?"
    return ("\n".join(lines) +
            f"\n\n> 探针条件：`{probe}`（文本缺失 · 中部 · 40%）。五变体共用同一划分、同一种子、"
            "同一最大轮数与同一早停准则、同一蒸馏初值，唯一差异是结构信号这一路；"
            "早停独立触发，故最优轮列并不全同。\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查占位符，不写文件")
    args = ap.parse_args(argv)

    with open(MD_SRC, encoding="utf-8") as f:
        text = f.read()

    names = re.findall(r"\{TABLE:([A-Za-z0-9_]+)\}", text)
    missing, used = [], []
    for n in names:
        p = os.path.join(TABLES, n + ".md")
        (used if os.path.isfile(p) else missing).append(n)

    print(f"[占位符] TABLE 共 {len(names)} 处，涉及 {len(set(names))} 张表")
    if missing:
        print(f"  ⚠ 缺表 {len(set(missing))} 张：{sorted(set(missing))}")

    def sub(m):
        p = os.path.join(TABLES, m.group(1) + ".md")
        if not os.path.isfile(p):
            return f"> **（表 {m.group(1)} 未生成）**\n"
        with open(p, encoding="utf-8") as fh:
            return fh.read().rstrip("\n")

    text = re.sub(r"\{TABLE:([A-Za-z0-9_]+)\}", sub, text)
    n_struct = text.count("{STRUCT_ABLATION}")
    if n_struct:
        print(f"[占位符] STRUCT_ABLATION 共 {n_struct} 处；"
              f"结构消融 CSV {'已就绪' if os.path.isfile(STRUCT_CSV) else '尚未生成'}")
        text = text.replace("{STRUCT_ABLATION}", render_structure_ablation().rstrip("\n"))

    leftover = re.findall(r"\{[A-Z_]+:[A-Za-z0-9_]+\}|\{STRUCT_ABLATION\}", text)
    if leftover:
        print(f"  ⚠ 仍有未替换占位符：{set(leftover)}")

    if args.check:
        return 0
    with open(MD_OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"[写出] {MD_OUT}（{len(text.splitlines())} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
